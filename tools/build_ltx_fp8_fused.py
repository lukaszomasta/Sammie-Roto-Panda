"""
Build a single fp8 LTX-2.5 transformer with the Alpha-Gen IC-LoRA fused in.

Build-machine tool (run once, not shipped as part of the app logic). Produces the file
sammie/ltx_alpha.py prefers at runtime, so each batch loads ~half the bytes of
bf16 transformer + LoRA and the tarball shrinks accordingly.

Output layout mirrors what ltx's fp8-cast policy produces at runtime:
  * the Linears ltx fp8-casts (transformer_blocks.*.{to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2},
    weight and bias) are stored as float8_e4m3fn, with no *_scale siblings (plain cast);
  * every other tensor keeps its original dtype;
  * key names and the checkpoint metadata (model config) are unchanged.
LoRA deltas, (B * strength) @ A as in ltx's fuse_loras, are added in fp32 before the
single fp8 rounding. At runtime ltx rounds twice (fp8 cast, then fuse + re-round).

usage: python_env/bin/python tools/build_ltx_fp8_fused.py
"""

import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ltx_core.quantization.fp8_cast import _is_fp8_cast_linear  # same matcher the runtime uses
from sammie.ltx_alpha import FUSED_FP8, LORA, LTX_DIR, TRANSFORMER

PREFIX = "model.diffusion_model."
LORA_PREFIX = "diffusion_model."
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=os.path.join(ROOT, LTX_DIR))
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    src = os.path.join(args.dir, TRANSFORMER)
    lora_path = os.path.join(args.dir, LORA)
    dst = os.path.join(args.dir, FUSED_FP8)
    tmp = dst + ".part"
    device = torch.device(args.device)
    t0 = time.time()

    with safe_open(lora_path, framework="pt", device="cpu") as lf:
        lora_meta = lf.metadata() or {}
        lora = {}
        for k in lf.keys():
            target, kind = k.removeprefix(LORA_PREFIX).rsplit(".lora_", 1)
            lora.setdefault(target, {})[kind.split(".")[0]] = lf.get_tensor(k)
    print(f"LoRA: {len(lora)} targets, strength {args.strength}")

    out = {}
    stats = {"fp8": 0, "kept": 0, "fused": 0, "overflow": 0}
    with safe_open(src, framework="pt", device="cpu") as tf:
        meta = dict(tf.metadata() or {})
        keys = list(tf.keys())
        for i, key in enumerate(keys):
            t = tf.get_tensor(key)
            param = key.removeprefix(PREFIX)
            module, _, leaf = param.rpartition(".")

            if leaf == "weight" and module in lora:
                ab = lora.pop(module)
                w = t.to(device, torch.float32)
                w += (ab["B"].to(device, torch.float32) * args.strength) @ ab["A"].to(device, torch.float32)
                t = w
                stats["fused"] += 1

            if leaf in ("weight", "bias") and _is_fp8_cast_linear(module):
                t = t.to(device, torch.float32)
                stats["overflow"] += int((t.abs() > FP8_MAX).sum())
                out[key] = t.to(torch.float8_e4m3fn).cpu()
                stats["fp8"] += 1
            else:
                out[key] = t.cpu()
                stats["kept"] += 1

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(keys)} tensors ({time.time() - t0:.0f}s)")

    if lora:
        raise SystemExit(f"LoRA targets not found in transformer: {sorted(lora)[:5]}...")
    if stats["overflow"]:
        print(f"WARNING: {stats['overflow']} values exceed fp8 range ({FP8_MAX}); "
              "the runtime fp8-cast policy would hit the same values")

    meta["ltx_alpha_fused"] = json.dumps({
        "base": TRANSFORMER,
        "lora": LORA,
        "lora_strength": args.strength,
        "lora_reference_downscale_factor": lora_meta.get("reference_downscale_factor", "1"),
        "quantization": "float8_e4m3fn plain cast of ltx fp8-cast Linears (no scales); LoRA fused in fp32 first",
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"tensors: {stats['fused']} LoRA-fused, {stats['fp8']} fp8, {stats['kept']} kept as-is; saving...")
    save_file(out, tmp, metadata=meta)
    os.replace(tmp, dst)
    print(f"wrote {dst} ({os.path.getsize(dst) / 2**30:.2f} GiB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
