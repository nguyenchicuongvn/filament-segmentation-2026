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

    # One tar per probs_* directory: Kaggle unpacks zips into one file per map, which then has to
    # be downloaded file by file (whole-dataset download returns 404 for these datasets).
    import shutil
    import tarfile

    for d in sorted(args.dir.glob("probs_*")):
        if d.is_dir():
            with tarfile.open(d.with_suffix(".tar"), "w") as tf:
                tf.add(d, arcname=d.name)
            shutil.rmtree(d)
            print("packed", d.with_suffix(".tar").name, flush=True)

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
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    # the CLI exits 0 even when Kaggle refuses the dataset, so check what it said
    if r.returncode != 0 or "error" in (r.stdout + r.stderr).lower():
        raise SystemExit(f"upload of {ref} FAILED (see above). A slug already used by a notebook cannot be "
                         f"reused; pass another --slug, e.g. {args.slug}-ds")
    print(f"uploaded -> https://www.kaggle.com/datasets/{ref}")


if __name__ == "__main__":
    main()
