"""Upload a run directory (checkpoint + probability maps) as a PRIVATE Kaggle dataset.

Creates <owner>/<slug> the first time and adds a new version afterwards. Used by the
Colab notebook so results reach the local machine with `kaggle datasets download`.

Usage:
  python scripts/kaggle_upload.py --dir /content/out/unet1024_r34 --owner nguyenchicuong \
      --slug filseg-unet1024-r34 --message "30 epochs"
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--message", default="update")
    args = ap.parse_args()

    ref = f"{args.owner}/{args.slug}"
    (args.dir / "dataset-metadata.json").write_text(json.dumps(
        {"title": args.slug, "id": ref, "licenses": [{"name": "other"}]}, indent=2))
    exists = subprocess.run(["kaggle", "datasets", "status", ref], capture_output=True, text=True)
    if exists.returncode == 0 and "ready" in exists.stdout.lower():
        cmd = ["kaggle", "datasets", "version", "-p", str(args.dir), "-m", args.message, "--dir-mode", "zip"]
    else:
        # private by default (no --public flag)
        cmd = ["kaggle", "datasets", "create", "-p", str(args.dir), "--dir-mode", "zip"]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"uploaded -> https://www.kaggle.com/datasets/{ref}")


if __name__ == "__main__":
    main()
