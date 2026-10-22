# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha images (MAGFiLO dataset).
Competition: https://www.kaggle.com/competitions/filament-segmentation-2026 (PQ metric, deadline 2026-11-15).
Prize conditions: [PRIZE_RULES.md](PRIZE_RULES.md).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
kaggle competitions download filament-segmentation-2026 -p data && (cd data && unzip -q *.zip && rm *.zip)
```

## Layout

| Path | Purpose |
|---|---|
| `src/filseg/data.py` | image/annotation loading, COCO RLE helpers |
| `src/filseg/metric.py` | Panoptic Quality, same rules as the host's self-evaluation notebook |
| `src/filseg/classical.py` | classical (no-learning) baseline |
| `scripts/eval_classical.py` | evaluate / random-search baseline params on a fixed 70/30 train/val split |
| `scripts/predict_classical.py` | write `outputs/submission.csv` for the test set |
| `tests/` | metric and RLE unit tests |

## Classical baseline

Disk fit → radial limb-darkening flattening → median-background ratio → hysteresis threshold
on dark pixels → morphology, small-blob and sunspot removal → fragment grouping by dilation →
per-instance dilation → upsample to 2048².

```bash
.venv/bin/python scripts/eval_classical.py --n 160 --search 80          # tune on train split
.venv/bin/python scripts/eval_classical.py --params outputs/best_params.json --split val
.venv/bin/python scripts/predict_classical.py --params outputs/best_params.json
```

Tuned parameters are versioned in `configs/classical_best.json`.

### Results (local, fixed split seed 42)

| Version | train PQ (160 img) | val PQ (213 img) | Test inference |
|---|---|---|---|
| classical, tuned (`configs/classical_best.json`) | 0.246 | 0.235 | 13 s / 180 images, CPU |
