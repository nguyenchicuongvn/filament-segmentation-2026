"""Build a self-contained Kaggle kernel that trains on GPU.

The repo's src/ and scripts/ are embedded as a base64 tarball, so no extra
dataset upload is needed. The kernel links the competition data into the
layout the code expects, trains, then writes val/test probability maps.

Usage:
  python kaggle/make_kernel.py --slug filseg-unet1024 --args "--res 1024 --bs 8 --epochs 30"
  kaggle kernels push -p kaggle/build/filseg-unet1024
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = '''import base64, io, os, subprocess, sys, tarfile, glob
REPO = "/kaggle/working/repo"
os.makedirs(REPO, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE))).extractall(REPO)
ann = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)[0]
data_root = os.path.dirname(os.path.dirname(ann))
os.makedirs(f"{{REPO}}/data", exist_ok=True)
if not os.path.exists(f"{{REPO}}/data/MAGFiLO_1.0_Kaggle_2026"):
    os.symlink(data_root, f"{{REPO}}/data/MAGFiLO_1.0_Kaggle_2026")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0"], check=True)
os.chdir(REPO)
def run(cmd):
    print(">>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
run("nvidia-smi -L")
run("python scripts/train_unet.py --workers 4 --name {name} {train_args}")
run("cp outputs/models/{name}.* /kaggle/working/")
run("python scripts/infer_probs.py --ckpt outputs/models/{name}.pt --which test --tta --out /kaggle/working/probs_test")
{val_cmd}
run("rm -rf /kaggle/working/repo")
'''


def code_tarball() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in ("src", "scripts"):
            tar.add(ROOT / d, arcname=d, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--args", default="", help="extra args for train_unet.py")
    ap.add_argument("--machine", default="NvidiaL4")
    ap.add_argument("--user", default="nguyenchicuong")
    a = ap.parse_args()

    has_val = "--split all" not in a.args
    val_cmd = (f'run("python scripts/infer_probs.py --ckpt outputs/models/{a.slug}.pt --which val '
               f'--tta --out /kaggle/working/probs_val")') if has_val else ""
    body = TEMPLATE.format(name=a.slug, train_args=a.args, val_cmd=val_cmd)
    out = ROOT / "kaggle" / "build" / a.slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(f'CODE = "{code_tarball()}"\n' + body)
    meta = {
        "id": f"{a.user}/{a.slug}",
        "title": a.slug,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": a.machine,
        "dataset_sources": [],
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("built", out)


if __name__ == "__main__":
    main()
