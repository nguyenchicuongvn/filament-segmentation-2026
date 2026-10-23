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

### Results

Validation: fixed split by image (seed 42), 213 val images / 337 annotator-image entries,
PQ pooled over all entries as in the host's notebook.

| Version | val PQ | Public LB | Test inference |
|---|---|---|---|
| classical, tuned (`configs/classical_best.json`) | 0.242 | 0.21 | 13 s / 180 images, CPU |
| U-Net ResNet34 @512, 8x TTA, `configs/post_unet512.json` | 0.354 | 0.29 | 80 s / 180 images, Apple M5 MPS |

## U-Net pipeline

```bash
.venv/bin/python scripts/train_unet.py --res 512 --epochs 30 --name unet512_r34       # ~1.5 h on Apple M5 (MPS)
.venv/bin/python scripts/infer_probs.py --ckpt outputs/models/unet512_r34.pt --which val --tta
.venv/bin/python scripts/infer_probs.py --ckpt outputs/models/unet512_r34.pt --which test --tta
.venv/bin/python scripts/tune_post.py --probs outputs/probs/unet512_r34_val --out configs/post_unet512.json
.venv/bin/python scripts/make_submission.py --probs outputs/probs/unet512_r34_test \
    --post configs/post_unet512.json --out outputs/submission_unet512.csv
```

### GPU training on Colab

`colab/train_unet.ipynb` trains the runs listed in its parameters cell, writes 8x-TTA val/test
probability maps and uploads each finished run to a private Kaggle dataset
`nguyenchicuong/filseg-<run>`. Training state is saved to Drive every epoch (`--resume`).

```bash
.venv/bin/python scripts/package_colab.py          # then upload build/colab/filseg-colab.zip to MyDrive/filseg/
# ... run the notebook on a GPU runtime (Colab secret KAGGLE_API_TOKEN) ...
kaggle datasets download nguyenchicuong/filseg-unet1024-r34 -p outputs/colab/unet1024_r34 --unzip
.venv/bin/python scripts/tune_post.py --probs outputs/colab/unet1024_r34/probs_val --out configs/post_unet1024.json
```

`kaggle/make_kernel.py` does the same on a Kaggle GPU kernel (code embedded, no dataset upload).

## License

Code: [MIT](LICENSE). The MAGFiLO data is not included and is subject to the competition's non-commercial data terms.
