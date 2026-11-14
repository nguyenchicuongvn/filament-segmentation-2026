# Defending the solution

The rules ask winners to understand, explain, defend and reproduce every major component; relying on
AI-generated code that cannot be explained can lead to disqualification. This page is the study sheet
for that check. For each component it gives **what it does**, **why it is there**, **the evidence**
and **where the code is**, then lists the questions a reviewer is likely to ask, with honest answers.
Numbers come from `docs/logs/final/` and Table 2 of the report. All OOF values are on the 707 training
images (1,154 annotator entries), and every Δ is a paired bootstrap over images (2,000 resamples, 95% CI).

Selected Kaggle submission: ref 58269004, `outputs/submission_ens_n1n9p_orient.csv` (N1 + N9' + filter),
OOF PQ 0.4382 [0.429, 0.447], public LB 0.38.

## 1. The metric and the selection protocol

- **What.** PQ = Σ IoU(TP) / (TP + ½FP + ½FN), with a TP when IoU > 0.5. It is computed against every
  annotator entry separately, and the counts are pooled over all entries before taking the ratio.
  Implementation: `src/filseg/metric.py` (`PQAccumulator.add`), checked against the host's notebook in
  `tests/test_metric.py`.
- **Why it shapes everything.** A filament split in two costs 1 TP and adds 2 FP, so how instances are
  separated matters as much as pixel accuracy. Annotators disagree strongly: one annotator scored
  against another gets PQ 0.340.
- **Protocol.** There are 5 folds, split by *image* (seed 42) so that all annotators of an image land
  in the same fold. An earlier split by entry leaked images between train and validation; we found it
  and changed it. Every family is trained 5 times, and the held-out maps give OOF predictions for all
  707 images. A change is accepted only if its paired CI is above 0, with the rule written down before
  the run (`docs/PLAN.md`). When two pipelines are compared, post-processing is *cross-tuned*: chosen
  on 4 folds and scored on the 5th (`scripts/compare.py`, `evaluate_crosstuned`).
- **Why not the leaderboard.** It has about 90 images and shows two decimals, so it resolves only
  about 0.01 (bootstrap SD 0.014). The public top is contaminated (see `docs/RESEARCH.md`), so it says
  nothing about method quality.

## 2. Networks: two U-Net families

| | N1 | N9' |
|---|---|---|
| Encoder | EfficientNet-B4 (ImageNet) | EfficientNetV2-S (ImageNet-21k, timm) |
| Input when training | whole image resized to 1024 | random 512 crops of the native 2048 image |
| Input at test time | whole image at 1024, map upsampled to 2048 | whole 2048 image (fully convolutional) |
| Batch, LR, epochs | 4, 1e-3, 30 | 16, 3e-4, 184 |

- **Shared recipe** (`scripts/train_unet.py`, `src/filseg/dl.py`):
  - Model: `build_model`, a U-Net from segmentation-models-pytorch with 1 output channel.
  - Loss: BCE + soft Dice (`bce_dice_loss`).
  - Optimiser: AdamW with weight decay 1e-4 and a one-cycle schedule with 5% warm-up, mixed precision,
    no EMA.
  - Augmentation (`SoftDataset.__getitem__`): random 90° rotation, horizontal flip, gamma 0.7–1.4,
    gain 0.8–1.2 and offset ±0.08.
  - Crops (N9'): centred near a filament pixel with probability 0.7, otherwise uniform.
- **Soft target** (`soft_cached`). Each pixel's target is the fraction of the image's annotators who
  drew it (0, 1/3, 1/2, 2/3, 1).
  - Why: the metric averages over annotators, so the best prediction is the "average annotator", not
    one person's outline.
  - Sampling: images are drawn with probability proportional to their annotator count
    (`WeightedRandomSampler`), because the metric weights an image by its number of entries.
- **Why two families.** They see the image at different scales. N1 sees global context at half
  resolution; N9' sees native-resolution detail. Averaging maps from different scales removes false
  positives.
- **Evidence.**
  - N1 + N2 over N1 alone: +0.011 [+0.007, +0.014], with 222 fewer FP and almost no change in TP.
  - N9' added on top: +0.006 [+0.003, +0.008].
  - The leave-one-family-out test then showed N2 no longer contributes: N1 + N9' is +0.002
    [−0.001, +0.004] over all three families, because N9' also sees native crops.
  - So the main pipeline keeps 10 models instead of 15.
- **Inference** (`predict_prob`). 8× TTA (4 rotations × flip, averaged). Maps are stored as uint8 and
  averaged over the 5 folds, then 1:1 over the two families (`scripts/ensemble_probs.py`). Unequal
  family weights did not help.

## 3. Hysteresis (map → components)

- **What** (`instances_lowres`, mode `hyst`). Take 8-connected components of p > 0.5. Keep a component
  if its area is ≥ 200 px *and* its peak p > 0.99.
- **Why.** A faint region is real only when it is attached to a confident core, the same idea as
  Canny's hysteresis. A plain threshold keeps many faint blobs that are false positives.
- **Evidence.** This is the largest post-processing effect. Removing hysteresis costs −0.025
  [−0.029, −0.020] on N1 + N2 and 0.02 on the LB; the LB sees it. A plain threshold of 0.5 (no
  hysteresis, no grouping) gives −0.037 [−0.043, −0.031].

## 4. Oriented grouping (fragments → instances)

- **What** (`_group_oriented`).
  - Orientation: for each kept component, compute it from second moments,
    θ = ½·atan2(2μ11, μ20 − μ02).
  - Dilation: dilate the component with a kernel made of a disk of radius 6 and a ±16 px line along θ.
  - Merge: components whose dilations touch become one instance.
  - Masks: the instance keeps only its original pixels, so masks never overlap.
- **Why.** Filaments are long and thin and often come out broken into collinear pieces. The line
  kernel bridges gaps *along* the filament without merging neighbours side by side, which an isotropic
  disk would do.
- **Evidence.**
  - No grouping: −0.006 [−0.011, −0.002] OOF and −0.01 on the LB.
  - Plain disk of radius 12: −0.001 on three families, and −0.002 [−0.004, −0.001] on N1 + N2.
  - Parameters come from the grid in `scripts/tune_post.py`.
  - Test: `tests/test_group_oriented.py`.

## 5. False-positive filter

- **What** (`src/filseg/instfilter.py`, `scripts/fp_filter.py`).
  - Features: each instance gets 17 features (size, shape, network confidence, contrast, position on
    the disk).
  - Classifier: HistGradientBoosting (scikit-learn) predicts whether the instance matches *some*
    annotator at IoU > 0.5.
  - Cut-off: instances scoring below 0.275 are dropped. Stored with its post config in
    `configs/fpfilter_ens_n1n9p_orient.pkl`.
- **Why.** Annotators skip filaments beyond about 70° from the central meridian, plus small or faint
  ones. These patterns are learnable from instance-level features but not from pixels.
- **How we avoid leaking.** The filter is trained only on OOF maps, and its scores are cross-validated
  by image (each image's fold is held out). Only the threshold was chosen on those scores.
- **Evidence.** +0.001 [+0.000, +0.002] on N1 + N9': small, but the CI does not go below 0. It was
  larger on weaker pools (+0.003 on three families).
- **Detail a reviewer may spot.** One of the 17 columns, `group_mean`, is the mean of a "group map" that
  none of the final models has, so it is the constant −1 and the trees never use it. In effect there
  are 16 features.

## 6. Output

Instances are upsampled to 2048 with nearest-neighbour (`upsample_labels`) and encoded as COCO RLE
(`encode_masks`); see `scripts/make_submission.py`. Because every pixel belongs to at most one label,
masks cannot overlap.

## 7. Questions to be ready for

1. **"Did you use AI to write the code?"**
   - Answer: yes, an AI coding assistant helped write and refactor code, under our direction.
   - Every design decision was made by us from OOF tests that we pre-registered and logged
     (`docs/EXPERIMENTS.md`, `docs/PLAN.md`).
   - Be able to walk through sections 1–6 without notes, and to open each function named above and
     explain it line by line.
2. **"Did you use MAGFiLO 1.0 or any test labels?"** No. Training uses only the competition's train
   masks, and inference uses only the test H-alpha images. The public LB top (0.70) comes from that
   leak, which is why our honest 0.38 is far below it (`docs/RESEARCH.md`).
3. **"Bounding boxes, spines, chirality?"** Not used; the dataset class reads only the masks.
4. **"How was the checkpoint chosen?"**
   - `train_unet.py` saves the checkpoint with the best PQ on its held-out fold among the validated
     epochs.
   - N9' validates only at the last epoch, so its checkpoint is the last one.
   - N1 validated at epochs 10/20/30, and for all 5 folds epoch 30 was best, so these are also the last
     checkpoints. The main pipeline is therefore free of checkpoint selection on the OOF fold.
   - Exception, outside the selected submission: in N2 fold 4 (used only in the alternative
     N1 + N2 + N9') epoch 20 scored higher (0.362 vs 0.351), so that checkpoint was picked on its
     held-out fold. That is a mild optimistic bias, on one fold of one family.
5. **"Why is the LB 0.05 below OOF?"**
   - Every team sees this gap.
   - The test set likely has a different annotator mix.
   - Our operating point does not move under annotator shift: within 0.0005 of the best of 8 variants
     whether scored on all entries, one random annotator per image, or the two hardest annotator groups
     (`docs/logs/final/hedge_final2_n1n9p.txt`).
   - A much looser setting (+27% instances) still gives 0.38.
6. **"Why not submit the three-family pool?"** Same OOF (0.4372 vs 0.4382) and the same LB, with a
   third more models and 3× the inference time; the leave-one-family-out test shows N2 is redundant.
7. **"Can you reproduce the submission?"**
   - Run `notebooks/final_inference.ipynb` from the repo and the test images.
   - On a Kaggle kernel the 15-model version gave the same LB, and the N1 maps were byte-identical to
     ours.
   - From a fresh clone on a Mac the CSV has the same 1,038 filament ids. 45 masks differ by a few
     pixels (IoU ≥ 0.997) because some maps were recomputed on MPS instead of CUDA.
   - The selected CSV itself was made locally with the same code after our Kaggle GPU quota ran out.
8. **"What did not work?"** A second N9' seed, ResNet34, BCE-only and ConvNeXt families; blob-Dice
   and false-component losses; watershed splitting; Mask R-CNN fusion; multi-scale TTA; size-dependent
   mask growth (report § 3.3).
9. **"Where is the remaining error?"**
   - Before the filter, 1 − PQ splits into 37% boundary, 36% FN and 27% FP.
   - 34% of our FP are drawn by another annotator of the same image.
   - 39% of FN were drawn by only one annotator.
   - The ceiling is set mostly by annotator disagreement (`docs/logs/final/error_decomp_n1n9p.txt`).

## 8. How to rehearse

Open the files below in this order and explain each one out loud:

1. `metric.py`
2. `data.py` (`fold_split`)
3. `dl.py` (`soft_cached`, `SoftDataset`, `bce_dice_loss`, `predict_prob`, `instances_lowres`,
   `_group_oriented`)
4. `instfilter.py`
5. `make_submission.py`

Then run section 4 of the notebook on one test image and explain each panel.
