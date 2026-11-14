# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha full-disk images (MAGFiLO dataset), scored by
Panoptic Quality (PQ). Competition: https://www.kaggle.com/competitions/filament-segmentation-2026.
The 4-page technical report: [`report/report.pdf`](report/report.pdf) (LaTeX source in [`report/`](report/), host template).

## Result

| | Main pipeline (selected Kaggle final) | Alternative: + N2 (submitted, not selected) |
|---|---|---|
| Networks | N1 + N9': 10 U-Nets (2 families × 5 folds) | N1 + N2 + N9': 15 U-Nets |
| Out-of-fold PQ, 707 train images (95% CI) | **0.4382** [0.429, 0.447] | 0.4372 |
| Public leaderboard PQ | 0.38 | 0.38 |
| Mean Dice per annotator entry (union mask) / mean IoU of TPs | 0.669 / 0.678 | – |
| Submission file | `outputs/submission_ens_n1n9p_orient.csv` | `submission_final1.csv` (Kaggle kernel) |

Out-of-fold (OOF) PQ is the selection metric throughout: every model is trained once per fold of a
5-fold split by image, and its held-out predictions cover all 707 training images. The public
leaderboard (~90 images, two decimals) was used only to check reproducibility. The rubric metrics
(Dice/IoU distributions, one-to-many and many-to-one counts) are in
[`docs/logs/final/rubric_final.json`](docs/logs/final/rubric_final.json), made by
`scripts/rubric_metrics.py`.

## Pipeline

```
test H-alpha image (2048², gray)
 ├─ N1:  U-Net EfficientNet-B4, whole image resized to 1024, 5 folds, 8x TTA ─┐  average 5 folds,
 └─ N9': U-Net EfficientNetV2-S, trained on 512 crops at 2048, 5 folds, 8x TTA ┘  upsample N1, average 1:1
      → hysteresis: components of p > 0.5 with area ≥ 200 px and peak p > 0.99
      → oriented grouping: merge collinear fragments (disk r 6 ∪ ±16 px segment along the major axis)
      → false-positive filter: gradient boosting on 17 instance features, drop score < 0.275
      → upsample instances to 2048, COCO RLE → submission.csv (masks never overlap)
```

Training targets are soft (fraction of an image's annotators covering a pixel); loss BCE + soft Dice;
AdamW, one-cycle schedule, last checkpoint. Only masks are used for training (no bounding boxes, spines
or chirality labels) and only the test H-alpha images at inference. Details and ablations: the report.

## Reproduce

### 1. Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
kaggle competitions download filament-segmentation-2026 -p data && (cd data && unzip -q *.zip && rm *.zip)
.venv/bin/python -m pytest -q tests          # metric, rubric metrics, grouping and loss tests
```

Tested with Python 3.12.14 and the versions pinned in `requirements.txt`, on an Apple M5 (MPS),
Kaggle T4 ×2 and Colab L4 GPUs.

### 2. Inference from the weights in this repo

All 15 checkpoints are in [`weights/`](weights/) (fp32, verify with `cd weights && shasum -a 256 -c SHA256SUMS`).
[`notebooks/final_inference.ipynb`](notebooks/final_inference.ipynb) runs the whole pipeline on the test
images and writes `outputs/final/submission_final2.csv` (main pipeline, the selected final) and `submission_final1.csv`
(the N1 + N2 + N9' alternative). The competition scores one selected submission; the file names keep the
labels used during development:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/final_inference.ipynb
```

`kaggle/make_infer_kernel.py` builds the same notebook as a Kaggle kernel. On Kaggle (T4 ×2) all 15
models took 90 min, of which N2 took 75; the main pipeline alone (10 models) takes about 25 min on a T4.
Post-processing and the filter run on the CPU. The committed notebook output was made on a laptop (Apple M5, maps
reused from an earlier run): ensembling and post-processing of both pipelines took 38 min there, mostly reading
maps from an external drive. Its CSVs have the same filaments as the submitted ones; 45 of 1038 masks differ by
a few pixels (IoU ≥ 0.997) because some maps were recomputed on MPS instead of CUDA.
Maps are computed per checkpoint with `scripts/infer_probs.py`, averaged with `scripts/ensemble_probs.py`
and turned into a CSV by `scripts/make_submission.py`, so each step can also be run by hand.

### 3. Training

Each family is five runs of `scripts/train_unet.py`, one per held-out fold (`--fold 0..4`). We trained on
Colab / Kaggle GPUs with [`colab/train_unet.ipynb`](colab/train_unet.ipynb) (code bundle from
`scripts/package_colab.py`); the notebook calls exactly these commands:

| Family | Command (`k` = 0..4) | GPU time per fold |
|---|---|---|
| N1 | `train_unet.py --res 1024 --bs 4 --encoder efficientnet-b4 --targets soft --epochs 30 --val-every 10 --fold k` | 2.0 h T4 |
| N9' | `train_unet.py --res 2048 --crop 512 --bs 16 --lr 3e-4 --encoder tu-tf_efficientnetv2_s.in21k_ft_in1k --targets soft --epochs 184 --val-every 184 --fold k` | 1.3 h L4 |
| N2 (alternative only) | `train_unet.py --res 2048 --crop 1024 --bs 4 --encoder efficientnet-b4 --targets soft --epochs 30 --val-every 10 --fold k` | 1.8 h T4 |

Each run writes its checkpoint and the 8x-TTA maps of its held-out images. Then, on the CPU:

```bash
# OOF maps of a family = the held-out maps of its 5 runs in one folder (707 images);
# the pool is the 1:1 average of the two family folders
.venv/bin/python scripts/ensemble_probs.py --inputs outputs/oof/n1 outputs/oof/n9p --out outputs/oof/n1n9p
# hysteresis + oriented grouping grid on the OOF maps
.venv/bin/python scripts/tune_post.py --probs outputs/oof/n1n9p --out configs/post_n1n9p.json
# instance filter, 5-fold CV by image, threshold = argmax of OOF PQ
.venv/bin/python scripts/fp_filter.py --probs outputs/oof/n1n9p --post configs/post_n1n9p.json \
    --out configs/fpfilter_n1n9p.pkl
# paired bootstrap over images against another pipeline, post-processing cross-tuned
.venv/bin/python scripts/compare.py n1n9p=outputs/oof/n1n9p,tune:hyst other=outputs/oof/<other>,tune:hyst
```

The filters used by the finals are versioned in `configs/` and carry their post-processing config:
`fpfilter_ens_n1n9p_orient.pkl` (main, threshold 0.275) and `fpfilter_ens_n1n2n9p_orient.pkl`
(alternative, threshold 0.375); both use `post_ens_n1n2n9p_hyst_orient.json`.

## Code layout

| Path | Purpose |
|---|---|
| `src/filseg/data.py` | image and annotation loading, folds by image, COCO RLE helpers |
| `src/filseg/metric.py` | Panoptic Quality, same rules as the host's self-evaluation notebook |
| `src/filseg/dl.py` | U-Net builder, soft targets, datasets and augmentation, TTA inference, post-processing (hysteresis, oriented grouping) |
| `src/filseg/instfilter.py` | instance features of the false-positive filter |
| `scripts/train_unet.py` | training (resumable); writes held-out maps |
| `scripts/infer_probs.py` | 8x-TTA probability maps of a checkpoint (stored as uint8) |
| `scripts/ensemble_probs.py` | average maps of several models / families |
| `scripts/tune_post.py` | grid search of post-processing on OOF maps |
| `scripts/fp_filter.py` | train and cross-validate the instance filter |
| `scripts/make_submission.py` | maps + post config + filter → submission CSV |
| `scripts/compare.py` | paired-bootstrap comparison of two pipelines with cross-tuning |
| `scripts/rubric_metrics.py`, `scripts/report_figures.py` | rubric metrics and report figures |
| `notebooks/final_inference.ipynb` | end-to-end inference of the main pipeline and the alternative |
| `colab/train_unet.ipynb` | GPU training queue (Colab) |
| `configs/` | fitted post-processing and filters (also of earlier experiments) |
| `weights/` | the 15 final checkpoints |
| `tests/` | unit tests |

Not part of the final pipeline, kept because the experiment log refers to them: the classical
baseline (`src/filseg/classical.py`, `scripts/eval_classical.py`, `scripts/predict_classical.py`),
Mask R-CNN (`scripts/train_maskrcnn.py`, `scripts/infer_detections.py`), and the analyses in
`scripts/research/` and `scripts/post_research.py`, `scripts/res_ceiling.py`, `scripts/analyze_errors.py`.
Operational helpers: `scripts/package_colab.py`, `scripts/kaggle_upload.py`, `scripts/watch_colab.py`,
`scripts/kaggle_gpu.py`, `kaggle/`.

## Development history

| Version | Validation PQ | Public LB |
|---|---|---|
| classical (no learning), tuned | 0.242 (val split) | 0.21 |
| U-Net ResNet34 @768, 8x TTA, FP filter | 0.389 (val split) | 0.33 |
| N1 alone, 5 folds, hysteresis | 0.4229 (OOF) | 0.37 |
| N1 + N2 | 0.4284 (OOF) | 0.37 |
| N1 + N2 + N9', oriented grouping (alternative) | 0.4372 (OOF) | 0.38 |
| **N1 + N9' (main, selected final)** | **0.4382 (OOF)** | **0.38** |

Every experiment, with its pre-registered acceptance rule and outcome, is in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). The working documents in `docs/` (`PROGRESS.md`, `PLAN.md`,
`EXPERIMENTS.md`, `RESEARCH.md`) are the team's day-to-day log and are written in Vietnamese; this README
and the report are the English reference. `docs/RESEARCH.md` explains why the top of the public
leaderboard is not comparable (test labels are available in the public MAGFiLO 1.0 release, which this
work does not use).

## License

Code and weights: [MIT](LICENSE). The MAGFiLO data is not included and is subject to the competition's
non-commercial data terms.
