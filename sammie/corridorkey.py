# sammie/corridorkey.py
"""
CorridorKey 2.0 wrapper.

CorridorKey predicts an alpha matte and a clean foreground colour per frame. Its three
foundation models (C-RADIOv4-SO400M, MoGe-2 ViT-B, RVM small) are frozen feature
extractors that the network was trained on, so all three are required. The subject is
chosen by a hint matte: Sammie passes its tracked SAM masks as a full per-frame hint
(see CorridorKeyManager in matting.py).

The code (Corridor Key Licence) is not part of this repository; it is expected as a
checkout in CODE_DIR. Weights live in checkpoints/corridorkey/ in the upstream layout.
Runs fully offline. The upstream runtime is driven unchanged; only its ffmpeg video
writer is swapped for one that hands the frames back to Sammie.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

CODE_DIR = "CorridorKey2.0"
CK_DIR = os.path.join("checkpoints", "corridorkey")
WEIGHTS = os.path.join("weights", "corridorkey_v2.safetensors")
STATS = os.path.join("weights", "foundation_stats.safetensors")
RVM = os.path.join("vfms", "rvm", "pretrain_rvm_small16_256_204031069.npz")
MOGE = os.path.join("vfms", "moge", "model.pt")
CRADIO = os.path.join("vfms", "cradio", "C-RADIOv4-SO400M")
CRADIO_WEIGHTS = os.path.join(CRADIO, "model.safetensors")
REQUIRED = (WEIGHTS, STATS, RVM, MOGE, CRADIO_WEIGHTS, os.path.join(CRADIO, "hf_model.py"))


# utils3d unpacked here by the downloader when it is not installed in the Python environment
UTILS3D_DIR = os.path.join(CODE_DIR, "third_party", "utils3d")


def _add_code_path():
    for path in (os.path.abspath(CODE_DIR), os.path.abspath(UTILS3D_DIR)):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def missing_code():
    """Download groups (see model_downloader.MODEL_GROUPS) for code that is not in place."""
    import importlib

    _add_code_path()
    importlib.invalidate_caches()
    missing = []
    if not os.path.isfile(os.path.join(CODE_DIR, "corridorkey", "runtime.py")):
        missing.append("corridorkey_code")
    if importlib.util.find_spec("utils3d") is None:
        missing.append("utils3d")
    return missing


def missing_files(models_dir=CK_DIR):
    return [f for f in REQUIRED if not os.path.exists(os.path.join(models_dir, f))]


def despill(fg, strength):
    """CorridorKey's average despill on sRGB colour (HxWx3, RGB or BGR order, green in the middle)."""
    if strength <= 0:
        return fg
    f = fg.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    spill = np.maximum(g - (r + b) * 0.5, 0.0)
    full = np.stack([r + spill * 0.5, g - spill, b + spill * 0.5], axis=-1)
    out = full if strength >= 1 else f * (1.0 - strength) + full * strength
    return out.round().clip(0, 255).astype(np.uint8)


class _FrameSink:
    """Stands in for the runtime's ThreadedVideoWriter; passes uint16 NHWC batches to a callback."""

    on_frames = None

    def __init__(self, path, fps, size, cfg, *, channels=3, color_trc=None, **_):
        stem = os.path.splitext(os.path.basename(path))[0]
        self.kind = "alpha" if stem.endswith("_alpha") else "fg"

    def write_many(self, frames):
        _FrameSink.on_frames(self.kind, frames)

    def release(self):
        pass


class CorridorKeyPipeline:
    def __init__(self, device, models_dir=CK_DIR):
        _add_code_path()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from corridorkey import runtime

        if torch.device(device).type != "cuda":
            raise RuntimeError("CorridorKey needs an NVIDIA GPU (CUDA).")
        root = Path(models_dir).resolve()
        self.runtime = runtime
        self.paths = runtime.RuntimePaths(
            package_root=Path(CODE_DIR).resolve(),
            weights=root / WEIGHTS,
            foundation_stats=root / STATS,
            rvm_ckpt=root / RVM,
            moge_checkpoint=root / MOGE,
            cradio_dir=root / CRADIO,
        )
        self.model = runtime.load_model(self.paths, device="cuda")

    def close(self):
        self.model = None

    def run(self, frame_dir, hint_dir, work_dir, on_frames, *, fps=24.0, hint_quality=0.75,
            carry_hint=True, chunk=0, model_scale=1.0):
        """Matte every frame in frame_dir, using the same-named masks in hint_dir.

        on_frames(kind, frames) receives "alpha" (NxHxWx3, uint16) and "fg" (NxHxWx3 BGR, uint16
        sRGB) batches in frame order. chunk=0 picks the Hann window from free VRAM.
        """
        rt = self.runtime
        cfg = rt.InferenceConfig(
            hann_chunk=chunk or 80,
            hann_stride=max(1, (chunk or 80) // 2),
            auto_vram=chunk == 0,
            carry_hint=carry_hint,
            hint_quality=hint_quality,
            carry_hint_quality=0.95,
            model_scale=model_scale,
            outputs={"alpha", "fg"},
            temp_dir=Path(work_dir),
            frame_dir_fps=float(fps),
            output_format="video",
        )
        original = rt.ThreadedVideoWriter
        rt.ThreadedVideoWriter = _FrameSink
        _FrameSink.on_frames = on_frames
        try:
            rt.run_inference(
                self.model, str(frame_dir), str(work_dir), cfg, rt.WriterConfig(),
                initial_hint_path=None, hint_video_path=str(hint_dir),
            )
        finally:
            rt.ThreadedVideoWriter = original
            _FrameSink.on_frames = None
