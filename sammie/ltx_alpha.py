# sammie/ltx_alpha.py
"""
LTX-2.5 IC-LoRA Alpha-Gen wrapper.

The model takes an RGB clip and returns an alpha matte; it has no mask input. Sammie
steers it at the selected subject by premultiplying the grown + feathered SAM2 mask
onto black before inference (see LTXAlphaManager in matting.py).

Fully offline: the empty text prompt is pre-encoded into EMBED_CACHE, so the Gemma 4
text encoder is never loaded or shipped. Weights live in checkpoints/ltx_alpha/.
"""

import contextlib
import gc
import json
import os

import av
import numpy as np
import torch

LTX_DIR = os.path.join("checkpoints", "ltx_alpha")
TRANSFORMER = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
LORA = "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"
# Built once by tools/build_ltx_fp8_fused.py: the transformer with LORA fused in, stored
# fp8. Preferred when present; otherwise TRANSFORMER + LORA are fused at load time.
FUSED_FP8 = "ltx-2.5-22b-distilled-alpha-gen-0.9-fp8-fused.safetensors"
# The same weights expanded to bf16 (tools/build_ltx_bf16_fused.py, exact) for GPUs without
# fp8 support such as Apple Silicon; preferred there over TRANSFORMER + LORA.
FUSED_BF16 = "ltx-2.5-22b-distilled-alpha-gen-0.9-bf16-fused.safetensors"
EMBED_CACHE = "empty_prompt_embeddings.safetensors"
LEGACY_EMBED_CACHE = "empty_prompt_embeddings.pt"  # older builds; loaded with weights_only
COMMON_FILES = (VIDEO_VAE, AUDIO_VAE, EMBED_CACHE)

# Below this much VRAM the transformer is streamed from system RAM block by block.
_FULL_GPU_VRAM = 30 * 2**30
# Keep transformer weights cached in RAM between batches only with this much headroom.
_RAM_HEADROOM = 16 * 2**30


def has_fused(models_dir=LTX_DIR, fp8=True):
    return os.path.exists(os.path.join(models_dir, FUSED_FP8 if fp8 else FUSED_BF16))


def fp8_supported(device):
    """fp8 weights are upcast per layer on the GPU; only verified on NVIDIA (CUDA).

    Elsewhere (Apple MPS, Intel XPU) the bf16 fused build, or bf16 transformer + LoRA, is used.
    """
    return torch.device(device).type == "cuda"


def missing_code():
    """LTX-2 packages that are not importable (installed separately, see README)."""
    import importlib.util

    return [m for m in ("ltx_core", "ltx_pipelines") if importlib.util.find_spec(m) is None]


def embed_cache_path(models_dir=LTX_DIR):
    """The empty-prompt cache, preferring safetensors over the legacy pickle."""
    path = os.path.join(models_dir, EMBED_CACHE)
    legacy = os.path.join(models_dir, LEGACY_EMBED_CACHE)
    return legacy if not os.path.exists(path) and os.path.exists(legacy) else path


def missing_files(models_dir=LTX_DIR, fp8=True):
    """Files still needed; with fp8=False (non-NVIDIA GPUs) the bf16 fused build is used."""
    fused = FUSED_FP8 if fp8 else FUSED_BF16
    needed = COMMON_FILES + ((fused,) if has_fused(models_dir, fp8) else (TRANSFORMER, LORA))
    missing = [f for f in needed if not os.path.exists(os.path.join(models_dir, f))]
    if EMBED_CACHE in missing and os.path.exists(embed_cache_path(models_dir)):
        missing.remove(EMBED_CACHE)
    if TRANSFORMER in missing:
        hint = "" if fp8 or not has_fused(models_dir) else (
            f"; make it from {FUSED_FP8} with tools/build_ltx_bf16_fused.py")
        missing[missing.index(TRANSFORMER)] = f"{fused} (or {TRANSFORMER} + {LORA}){hint}"
        if LORA in missing:
            missing.remove(LORA)
    return missing


def snap_up_8n1(n):
    """Smallest 8n+1 frame count >= n (the LTX temporal grid)."""
    return ((max(n, 1) - 1 + 7) // 8) * 8 + 1


def _available_ram():
    try:
        import psutil  # installed with accelerate; works on Linux, Windows and macOS

        return psutil.virtual_memory().available
    except Exception:
        return 0


def _no_fp16_autocast(device):
    """Sammie enables float16 autocast on MPS and pre-Ampere CUDA; LTX is a bf16 model and
    can overflow in float16, so switch autocast off for it there."""
    kind = device.type
    try:
        fp16 = torch.is_autocast_enabled(kind) and torch.get_autocast_dtype(kind) == torch.float16
    except (RuntimeError, TypeError):
        fp16 = False
    return torch.autocast(kind, enabled=False) if fp16 else contextlib.nullcontext()


class _CachedPromptEncoder:
    """Stands in for ltx's PromptEncoder: returns the pre-encoded empty prompt."""

    def __init__(self, path, device):
        from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput

        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            c = load_file(path)
        else:
            c = torch.load(path, map_location="cpu", weights_only=True)
        to = lambda t: None if t is None else t.to(device)
        self._out = EmbeddingsProcessorOutput(
            to(c["video_encoding"]), to(c.get("audio_encoding")), to(c["attention_mask"])
        )

    def __call__(self, prompts, **_kwargs):
        return [self._out for _ in prompts]


@contextlib.contextmanager
def _step_hook(callback):
    """Report denoising steps through `callback(step, total)` by wrapping ltx's tqdm.

    The callback may raise (e.g. RuntimeError("USER_CANCELLED")) to abort mid-batch.
    """
    from ltx_pipelines.utils import samplers

    original = samplers.tqdm
    if callback is None:
        yield
        return

    def hooked(iterable, *args, **kwargs):
        total = len(iterable) if hasattr(iterable, "__len__") else 0
        for i, item in enumerate(original(iterable, *args, **kwargs)):
            callback(i, total)
            yield item
        callback(total, total)

    samplers.tqdm = hooked
    try:
        yield
    finally:
        samplers.tqdm = original


def _write_reference_video(frames_rgb, path, fps):
    """Lossless FFV1/Matroska; ltx reads reference videos from a file path."""
    container = av.open(path, mode="w")
    stream = container.add_stream("ffv1", rate=max(1, int(round(fps))))
    stream.width, stream.height = frames_rgb[0].shape[1], frames_rgb[0].shape[0]
    stream.pix_fmt = "bgr0"
    for f in frames_rgb:
        for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _to_alpha(tensors):
    """Pipeline output (F, H, W, 3) in [0, 1] -> list of float32 (H, W) alphas."""
    alphas = []
    for t in tensors:
        t = t.detach().float().cpu()
        if t.dim() == 5:
            t = t.squeeze(0)
        if t.shape[-1] != 3:
            raise ValueError(f"unexpected LTX output layout {tuple(t.shape)}")
        alphas.extend(g.clamp(0, 1).numpy() for g in t.mean(-1))
    return alphas


class LTXAlphaPipeline:
    """Loads the Alpha-Gen pipeline once; `run` mattes one 8n+1-frame batch."""

    def __init__(self, device, models_dir=LTX_DIR):
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_core.loader.registry import ModelRegistry
        from ltx_core.quantization.fp8_cast import build_policy
        import ltx_pipelines.ic_lora as ic_lora
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode

        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        path = lambda name: os.path.abspath(os.path.join(models_dir, name))
        self.device = torch.device(device)
        self.video_vae_path = path(VIDEO_VAE)

        fp8 = fp8_supported(self.device)
        self.fused = has_fused(models_dir, fp8)
        if self.fused:
            transformer, loras = path(FUSED_FP8 if fp8 else FUSED_BF16), []
        else:
            transformer = path(TRANSFORMER)
            loras = [LoraPathStrengthAndSDOps(path(LORA), 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]

        if self.device.type == "cuda":
            vram = torch.cuda.get_device_properties(self.device).total_memory
            self.offload = OffloadMode.NONE if vram >= _FULL_GPU_VRAM else OffloadMode.CPU
        elif self.device.type == "mps":
            # Apple unified memory: GPU and "CPU" are the same RAM, so streaming gains nothing.
            self.offload = OffloadMode.NONE
        else:
            self.offload = OffloadMode.CPU
        # Standard (non-streaming) builds re-read the transformer for every batch unless
        # the registry keeps its weights; streaming builds manage their own CPU cache.
        cache_weights = (
            self.offload == OffloadMode.NONE
            and self.device.type != "mps"  # a RAM copy would double unified-memory use
            and _available_ram() > os.path.getsize(transformer) + _RAM_HEADROOM
        )
        registry = ModelRegistry(cache_models=True, cache_weights=cache_weights)

        # ICLoraPipeline builds a Gemma PromptEncoder in __init__; swap in the cached
        # empty-prompt encoder so no text encoder path is needed at all.
        cached = _CachedPromptEncoder(os.path.abspath(embed_cache_path(models_dir)), self.device)
        original = ic_lora.PromptEncoder
        ic_lora.PromptEncoder = lambda *a, **k: cached
        try:
            self.pipeline = ic_lora.ICLoraPipeline(
                model_paths=ModelPaths.from_split(
                    transformer_path=transformer,
                    video_vae_path=path(VIDEO_VAE),
                    audio_vae_path=path(AUDIO_VAE),
                ),
                spatial_upsampler_path="",  # stage 2 is always skipped
                loras=loras,
                device=self.device,
                quantization=build_policy(transformer) if fp8 else None,
                registry=registry,
                offload_mode=self.offload,
            )
        finally:
            ic_lora.PromptEncoder = original
        if self.fused:
            # The pipeline reads the reference scale from LoRA metadata; with the LoRA
            # fused in, take it from the fused checkpoint's build record instead.
            from safetensors import safe_open
            with safe_open(transformer, framework="pt") as f:
                record = json.loads((f.metadata() or {}).get("ltx_alpha_fused", "{}"))
            self.pipeline.reference_downscale_factor = int(record.get("lora_reference_downscale_factor", 1))
        self.cache_weights = cache_weights

    @torch.inference_mode()
    def run(self, frames_rgb, fps, work_dir, seed=1234, step_callback=None):
        """Matte one batch.

        Args:
            frames_rgb: list of uint8 (H, W, 3) premultiplied frames; H and W divisible
                by 32 and len() of the form 8n+1.
            fps: clip frame rate (conditions the model's motion prior).
            work_dir: scratch dir for the lossless reference video.
            step_callback: optional callable(step, total), may raise to cancel.

        Returns:
            list of float32 (H, W) alpha arrays in [0, 1], one per input frame.
        """
        n = len(frames_rgb)
        h, w = frames_rgb[0].shape[:2]
        if n != snap_up_8n1(n):
            raise ValueError(f"LTX needs 8n+1 frames, got {n}")
        if h % 32 or w % 32:
            raise ValueError(f"LTX needs sizes divisible by 32, got {w}x{h}")

        # ltx plans the decode tiling for the requested (2x) size, but stage 1 only decodes
        # h x w; planning at 4x the pixels needlessly fails the VRAM budget. Plan it here.
        from ltx_pipelines.utils.helpers import tiling_config_for_vae

        tiling = tiling_config_for_vae(
            self.video_vae_path,
            height=h,
            width=w,
            num_frames=n,
            diffvae_optimization=self.pipeline.video_decoder.diffvae_optimization,
            device=self.device,
        )

        ref_path = os.path.join(work_dir, "ltx_reference.mkv")
        _write_reference_video(frames_rgb, ref_path, fps)
        try:
            with _step_hook(step_callback), _no_fp16_autocast(self.device):
                # Stage 1 renders at half the requested size, so ask for 2x.
                result = self.pipeline(
                    prompt="",
                    seed=seed,
                    height=h * 2,
                    width=w * 2,
                    num_frames=n,
                    frame_rate=float(fps),
                    images=[],
                    video_conditioning=[(ref_path, 1.0)],
                    tiling_config=tiling,
                    skip_stage_2=True,
                )
                alphas = _to_alpha(result.video)
        finally:
            if os.path.exists(ref_path):
                os.remove(ref_path)
        return alphas[:n]

    def close(self):
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
