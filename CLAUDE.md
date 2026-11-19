# CLAUDE.md

Kaggle "Solar Filament Segmentation Challenge 2026": instance segmentation of solar filaments,
scored by Panoptic Quality. Read `docs/PROGRESS.md` first (status, submissions, log), then
`docs/PLAN.md` (next steps, deadlines) and `docs/EXPERIMENTS.md` (what was tried, results).

The public LB top is contaminated (test labels leak via public MAGFiLO 1.0; a 0.55 payload);
honest best is ~0.40 public / ~0.46 OOF — see `docs/RESEARCH.md`. Never use MAGFiLO 1.0 public
data, never submit overlapping masks, keep final weights inside the repo (host requirement).

## Conventions

- Python 3.12 venv at `.venv` (created with `uv`; deleted on 2026-11-19 to save space, recreate with
  the two `uv` lines in README); run everything as `.venv/bin/python ...` from the repo root.
  Tests: `.venv/bin/python -m pytest -q tests`.
- Choose models and parameters by **val PQ** (`filseg.data.split`, by image, seed 42), never by
  the public LB. The metric in `src/filseg/metric.py` mirrors the host's notebook; keep it that way.
- Train JSON has several annotators per image: an entry is an annotator-image (`TrainImage.key`),
  and splitting must stay by `image_id`.
- Rules: masks only (no bbox/spine/chirality labels for training); inference uses only test
  H-alpha images; open-source deps must be OSI-licensed.
- `data/`, `outputs/`, `build/` are git-ignored. Since 2026-11-19 (project archived) `data/`, `build/`
  and the heavy `outputs/` folders (`final`, `ens`, `cmp`, `colab`, `models`, `probs`, `wf_*`) live on
  the external exFAT SSD at `/Volumes/PortableSSD/kaggle/filament-segmentation-2026` behind symlinks
  (readable only while the SSD is plugged in; internal symlinks such as `*_oof` folds are listed in
  `SYMLINKS.tsv` there, the old worktrees are in `claude_worktrees.tar`). The Mac keeps code, docs,
  `weights/`, `.git` and the small files in `outputs/` (submission CSVs, `final_kernel`). Tuned configs go in `configs/`; cleaned logs of
  experiments worth keeping go in `docs/logs/`.
- Accounts: GitHub `nguyenchicuongvn`, Kaggle `nguyenchicuong` (different!).
- The GitHub repo is **public since the competition closed (2026-11-16)**; keep it public through the winner announcement (11-30), no force-push or history rewrite. Max 5 Kaggle submissions/day.
- GPU: Kaggle quota may be exhausted; GPU runs go through `colab/train_unet.ipynb` (code bundle from
  `scripts/package_colab.py`, results come back as private Kaggle datasets
  `nguyenchicuong/filseg-<run>`). Small runs fit on the Mac (MPS).

## After every experiment or submission

1. Add a row/section to `docs/EXPERIMENTS.md` (config, val PQ raw and tuned, LB if submitted).
2. Update the status table, submission history and log in `docs/PROGRESS.md`.
3. Update `docs/PLAN.md` if priorities changed; update the results table in `README.md`.
4. Commit and push.
