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

## Environment

| | |
|---|---|
| Python | 3.12 (tested 3.12.14), packages pinned in `requirements.txt` |
| Hardware | CPU only, no GPU needed for the classical baseline |
| Reference machine | Apple M5, 10 cores, 32 GB RAM, macOS 27 |
| Memory | < 1 GB per worker process (2048² images processed at 1024²) |

Measured runtimes on the reference machine:

| Step | Time |
|---|---|
| Segment + RLE-encode one image, single core | ~0.22 s |
| Test set prediction (180 images, 10 processes, incl. image loading) | ~13 s |
| Validation evaluation (213 images, 10 processes) | ~31 s |
| One random-search iteration (160 images) | ~8–10 s |

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

## License

Code: [MIT](LICENSE). The MAGFiLO data is not included and is subject to the competition's non-commercial data terms.
