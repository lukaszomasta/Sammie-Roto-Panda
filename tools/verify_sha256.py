"""
Check files against a `sha256sum`-format list (SHA256SUMS, SHA256SUMS.windows).

Works anywhere Python does, including Windows without `sha256sum`.

usage (from the Sammie folder):
  Windows:  python-3.12.8-embed-amd64\\python.exe tools\\verify_sha256.py SHA256SUMS.windows
  Linux:    python_env/bin/python tools/verify_sha256.py SHA256SUMS
"""

import hashlib
import os
import sys


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    listing = sys.argv[1]
    root = os.path.dirname(os.path.abspath(listing))
    with open(listing, encoding="utf-8") as f:
        entries = [line.rstrip("\r\n").split("  ", 1) for line in f if line.strip()]

    bad = 0
    for i, (digest, name) in enumerate(entries, 1):
        path = os.path.join(root, os.path.normpath(name))
        if not os.path.isfile(path):
            print(f"MISSING  {name}")
            bad += 1
        elif sha256(path) != digest:
            print(f"FAILED   {name}")
            bad += 1
        if i % 2000 == 0:
            print(f"  {i}/{len(entries)} checked", flush=True)

    print(f"{len(entries) - bad}/{len(entries)} files OK" + ("" if not bad else f", {bad} problems"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
