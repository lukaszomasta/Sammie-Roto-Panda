"""
Pre-encode the empty text prompt for LTX Alpha (run once).

The Alpha-Gen IC-LoRA is always run with an empty prompt, so Sammie loads this cached
embedding instead of the 26 GB Gemma 4 text encoder (sammie/ltx_alpha.py). This needs
the Gemma 4 file and the bf16 distilled transformer (its embeddings connector) once;
afterwards the Gemma file can be deleted.

Runs on CPU by default (~23 GB RAM, ~6 min); --device cuda needs a large GPU.

usage: python tools/encode_ltx_empty_prompt.py --gemma /path/to/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
"""

import argparse
import os
import sys
import time

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sammie.ltx_alpha import AUDIO_VAE, EMBED_CACHE, LTX_DIR, TRANSFORMER, VIDEO_VAE


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gemma", required=True, help="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    p.add_argument("--dir", default=os.path.join(ROOT, LTX_DIR),
                   help="folder holding the bf16 transformer and VAEs; the cache is written here")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    from ltx_pipelines.utils.blocks import PromptEncoder
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import OffloadMode

    transformer = os.path.join(args.dir, TRANSFORMER)
    for path in (args.gemma, transformer):
        if not os.path.exists(path):
            raise SystemExit(f"not found: {path}")

    t0 = time.time()
    device = torch.device(args.device)
    encoder = PromptEncoder(
        ModelPaths.from_split(
            transformer_path=transformer,
            text_encoder_path=os.path.abspath(args.gemma),
            video_vae_path=os.path.join(args.dir, VIDEO_VAE),
            audio_vae_path=os.path.join(args.dir, AUDIO_VAE),
        ),
        torch.bfloat16,
        device,
        # Streaming offload is a GPU-only trick; on CPU the weights already live in RAM.
        offload_mode=OffloadMode.NONE if device.type == "cpu" else OffloadMode.CPU,
    )
    (out,) = encoder([""])
    tensors = {
        "video_encoding": out.video_encoding,
        "audio_encoding": out.audio_encoding,
        "attention_mask": out.attention_mask,
    }
    tensors = {k: v.detach().cpu().contiguous() for k, v in tensors.items() if v is not None}
    path = os.path.join(args.dir, EMBED_CACHE)
    save_file(tensors, path, metadata={"prompt": "", "gemma": os.path.basename(args.gemma), "transformer": TRANSFORMER})
    print(f"Saved {path} in {time.time() - t0:.0f}s")
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
