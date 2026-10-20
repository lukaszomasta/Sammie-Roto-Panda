"""
Expand the fp8 LTX Alpha transformer (LoRA already fused) into a bf16 file for GPUs
without fp8 support, such as Apple Silicon (MPS).

Every float8_e4m3fn value is exactly representable in bfloat16, so this adds no loss:
the weights are the ones the fp8 build upcasts to at runtime anyway. All other tensors,
the key names and the metadata are copied unchanged; the "ltx_alpha_fused" build record
gets a note.

The output (~40 GiB) is written tensor by tensor, so the tool needs little RAM.

usage: python tools/build_ltx_bf16_fused.py
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import torch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sammie.ltx_alpha import FUSED_BF16, FUSED_FP8, LTX_DIR

_DTYPE_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int64: "I64",
                torch.int32: "I32", torch.uint8: "U8", torch.bool: "BOOL"}
_ITEM_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "U8": 1, "BOOL": 1}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=os.path.join(ROOT, LTX_DIR))
    args = p.parse_args()

    src = os.path.join(args.dir, FUSED_FP8)
    dst = os.path.join(args.dir, FUSED_BF16)
    tmp = dst + ".part"
    t0 = time.time()

    with safe_open(src, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        keys = sorted(f.keys())

        # Header first: dtypes and shapes after conversion, back-to-back offsets.
        header, offset, expanded = {}, 0, 0
        for key in keys:
            sl = f.get_slice(key)
            shape = list(sl.get_shape())
            dtype = sl.get_dtype()
            if dtype == "F8_E4M3":
                dtype, expanded = "BF16", expanded + 1
            if dtype not in _ITEM_BYTES:
                raise SystemExit(f"unexpected dtype {dtype} for {key}")
            nbytes = math.prod(shape) * _ITEM_BYTES[dtype]
            header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
            offset += nbytes

        record = json.loads(meta.get("ltx_alpha_fused", "{}"))
        record["expanded"] = f"bf16 from {FUSED_FP8} ({expanded} fp8 tensors, exact)"
        record["expanded_built"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta["ltx_alpha_fused"] = json.dumps(record)
        header["__metadata__"] = meta
        blob = json.dumps(header, separators=(",", ":")).encode()
        blob += b" " * (-len(blob) % 8)

        with open(tmp, "wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            for i, key in enumerate(keys):
                t = f.get_tensor(key)
                if t.dtype == torch.float8_e4m3fn:
                    t = t.to(torch.bfloat16)
                if _DTYPE_NAMES.get(t.dtype) != header[key]["dtype"]:
                    raise SystemExit(f"dtype mismatch for {key}: {t.dtype}")
                out.write(t.contiguous().view(torch.uint8).numpy().tobytes())
                if (i + 1) % 1000 == 0:
                    print(f"  {i + 1}/{len(keys)} tensors ({time.time() - t0:.0f}s)", flush=True)

    os.replace(tmp, dst)
    print(f"wrote {dst} ({os.path.getsize(dst) / 2**30:.2f} GiB, {expanded} tensors expanded) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
