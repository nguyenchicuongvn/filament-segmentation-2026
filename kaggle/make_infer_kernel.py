"""Write notebooks/final_inference.ipynb and a Kaggle kernel that runs it end to end.

The notebook rebuilds both final submissions from the 15 checkpoints and the test H-alpha images only:
8x-TTA maps per checkpoint (scripts/infer_probs.py), the fold averages and the family averages
(scripts/ensemble_probs.py), then post-processing + FP filter (scripts/make_submission.py) with the
configs in configs/. It also writes SHA256 hashes of every map and CSV so a run can be checked
against the maps the submissions were made from (scripts/map_hashes.py --compare).

In the repo the notebook runs from the repo root with weights in weights/. The Kaggle build adds one
cell that unpacks src/, scripts/ and configs/ (embedded) and points at the private weights dataset.

  python kaggle/make_infer_kernel.py              # notebook + kaggle/build/filseg-final-infer/
  kaggle kernels push -p kaggle/build/filseg-final-infer --accelerator NvidiaTeslaT4   # T4 x2
"""
from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "filseg-final-infer"
SCRIPTS = ["infer_probs.py", "ensemble_probs.py", "make_submission.py", "map_hashes.py"]
CONFIGS = ["fpfilter_ens_n1n2n9p_orient.pkl", "post_ens_n1n2n9p_hyst_orient.json",
           "fpfilter_ens_n1n9p_orient.pkl"]

CELLS = [
    ("markdown", """# Filament segmentation: final inference

Rebuilds the two final submissions from 15 U-Net checkpoints (5 folds each of three families) using only
the test H-alpha images:

* N1: EfficientNet-B4, whole image @1024
* N2: EfficientNet-B4, 1024 crops @2048
* N9': EfficientNetV2-S, 512 crops @2048

* **final 1**: N1, N2 and N9' family maps averaged 1:1:1 at 2048, hysteresis (t 0.5, seed 0.99, area 200),
  oriented fragment grouping (r 6, line 16), FP filter (`configs/fpfilter_ens_n1n2n9p_orient.pkl`).
* **final 2**: N1 + N9' averaged 1:1 at 2048, same post-processing as final 1, FP filter
  (`configs/fpfilter_ens_n1n9p_orient.pkl`).

Every map is 8x TTA (4 rotations x flip), saved as uint8. On a 2-GPU machine N1 + N9' and N2 run in parallel."""),
    ("code", """# parameters
import os, sys
from pathlib import Path
ON_KAGGLE = os.path.isdir('/kaggle/input')
REPO = Path('/kaggle/working/repo') if ON_KAGGLE else Path.cwd()
WEIGHTS = Path(os.environ.get('FILSEG_WEIGHTS', REPO / 'weights'))
OUT = Path('/kaggle/working') if ON_KAGGLE else REPO / 'outputs' / 'final'
N1 = [f'effb4_s1024_f{k}' for k in range(5)]
N2 = [f'effb4_c2048_f{k}' for k in range(5)]
N9 = [f'effv2s_c512_f{k}' for k in range(5)]"""),
    ("code", """import glob, subprocess
os.chdir(REPO)
if ON_KAGGLE:
    imgs = glob.glob('/kaggle/input/**/test_images', recursive=True)
    assert imgs, 'attach the competition data'
    data = REPO / 'data'
    data.mkdir(exist_ok=True)
    link = data / 'MAGFiLO_1.0_Kaggle_2026'
    if not link.exists():
        link.mkdir()
        (link / 'test').mkdir()
        (link / 'test' / 'test_images').symlink_to(imgs[0])
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.5.0'], check=True)
import torch, segmentation_models_pytorch as smp
print('torch', torch.__version__, 'smp', smp.__version__, 'gpus', torch.cuda.device_count())
missing = [n for n in N1 + N2 + N9 if not (WEIGHTS / f'{n}.pt').exists()]
assert not missing, f'missing checkpoints in {WEIGHTS}: {missing}'"""),
    ("code", """# 8x-TTA test maps per checkpoint; N1 + N9' on GPU 0, N2 on GPU 1 when there are two
import time
t0 = time.time()
ngpu = max(torch.cuda.device_count(), 1)
def chain(names, gpu):
    cmds = [f'{sys.executable} scripts/infer_probs.py --ckpt {WEIGHTS}/{n}.pt --which test --tta '
            f'--out {OUT}/maps/{n}' for n in names]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu % ngpu))
    return subprocess.Popen(' && '.join(cmds), shell=True, env=env)
procs = [chain(N1 + N9, 0), chain(N2, 1)] if ngpu > 1 else [chain(N1 + N2 + N9, 0)]
assert all(p.wait() == 0 for p in procs), 'inference failed'
print(f'maps done in {(time.time() - t0) / 60:.1f} min')"""),
    ("code", """def run(cmd):
    print('>>', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
m = OUT / 'maps'
run(f'{sys.executable} scripts/ensemble_probs.py --inputs ' + ' '.join(str(m / n) for n in N1) + f' --out {m}/n1')
run(f'{sys.executable} scripts/ensemble_probs.py --inputs ' + ' '.join(str(m / n) for n in N2) + f' --out {m}/n2')
run(f'{sys.executable} scripts/ensemble_probs.py --inputs ' + ' '.join(str(m / n) for n in N9) + f' --out {m}/n9p')
run(f'{sys.executable} scripts/ensemble_probs.py --inputs {m}/n1 {m}/n2 {m}/n9p --out {m}/n1n2n9p')
run(f'{sys.executable} scripts/ensemble_probs.py --inputs {m}/n1 {m}/n9p --out {m}/n1n9p')
run(f'{sys.executable} scripts/make_submission.py --probs {m}/n1n2n9p --filter configs/fpfilter_ens_n1n2n9p_orient.pkl '
    f'--out {OUT}/submission_final1.csv')
run(f'{sys.executable} scripts/make_submission.py --probs {m}/n1n9p --filter configs/fpfilter_ens_n1n9p_orient.pkl '
    f'--out {OUT}/submission_final2.csv')"""),
    ("code", """# hashes for the reproducibility check, then drop the maps (several GB)
import hashlib, shutil
dirs = ' '.join(f'{n}={m / n}' for n in N1 + N2 + N9 + ['n1', 'n2', 'n9p', 'n1n2n9p', 'n1n9p'])
run(f'{sys.executable} scripts/map_hashes.py --dirs {dirs} --out {OUT}/map_hashes.json')
for f in ('submission_final1.csv', 'submission_final2.csv'):
    print(f, hashlib.sha256((OUT / f).read_bytes()).hexdigest())
if ON_KAGGLE:
    shutil.copy(OUT / 'submission_final1.csv', OUT / 'submission.csv')
    shutil.rmtree(m)
    shutil.rmtree(REPO)"""),
]


def notebook(cells: list[tuple[str, str]]) -> dict:
    return {
        "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                   **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def code_tarball() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(ROOT / "src", arcname="src", filter=lambda ti: None if "__pycache__" in ti.name else ti)
        for s in SCRIPTS:
            tar.add(ROOT / "scripts" / s, arcname=f"scripts/{s}")
        for c in CONFIGS:
            tar.add(ROOT / "configs" / c, arcname=f"configs/{c}")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    nb_path = ROOT / "notebooks" / "final_inference.ipynb"
    nb_path.parent.mkdir(exist_ok=True)
    nb_path.write_text(json.dumps(notebook(CELLS), indent=1))
    unpack = ("code", f"""# Kaggle only: the repo code (src/, scripts/, configs/ used here), embedded at build time
import base64, glob, io, os, tarfile
os.environ['FILSEG_WEIGHTS'] = os.path.dirname(glob.glob('/kaggle/input/**/effb4_s1024_f0.pt', recursive=True)[0])
os.makedirs('/kaggle/working/repo', exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode('{code_tarball()}'))).extractall('/kaggle/working/repo')""")
    out = ROOT / "kaggle" / "build" / SLUG
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{SLUG}.ipynb").write_text(json.dumps(notebook([CELLS[0], unpack] + CELLS[1:]), indent=1))
    meta = {
        "id": f"nguyenchicuong/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["nguyenchicuong/filseg-final-weights"],
        "competition_sources": ["filament-segmentation-2026"], "kernel_sources": [], "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", nb_path, "and", out)


if __name__ == "__main__":
    main()
