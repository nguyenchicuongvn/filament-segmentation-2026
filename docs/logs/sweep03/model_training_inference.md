# Sweep 03/11 — model_training_inference (agent report, verbatim)

# Model, training and inference options: ranked plan for the last 6 days before freeze

**Scope.** I read RESEARCH (§ 6 and § 10), PLAN, the EXPERIMENTS ledger, the 30/10 training ideation and judge reports, `scripts/train_unet.py`, `src/filseg/dl.py`, `scripts/compare.py` and the Colab queue. I also ran two local checks. I edited no repo files. The only scratch file is `scripts/research/sweep03/sweep_model/bench_mps.py`.

## Five findings that change the plan

1. **The GPU budget is tighter than the plan assumes.**
   - E9a on an L4 took 109 s/epoch for 810 entries at bs 6 (0.81 s/step).
   - A soft-target fold samples about 923 entries per epoch, so one fold-run is about 70 min on an L4.
   - At your estimated 2x slowdown, a T4 needs about **2.3 h for an N1 run**, about **2.6 h for a v2 run** (raw-weight validation is doubled, plus the CPU connected-components step in `false_component_loss`), and about **2.8 h for an N2 run** (8x TTA maps at 2048).
   - So 20 h on one GPU per session is about **7–8 fold-runs**. The 2026-11-01 queue (9 runs) alone would take about 21 h.
2. **"T4 x2" may nearly double the budget.** Kaggle reportedly charges quota by session time, not per GPU, and T4 x2 and P100 draw on one pool (third-party sources only, not Kaggle docs).
   - If that holds, run two independent trainings per session with `CUDA_VISIBLE_DEVICES=0` and `=1`. There is no need for DDP.
   - The cost is 4 vCPUs shared between two loaders. Set `--workers 2` each and expect about 10–15% slowdown.
   - **Check before planning:** a 10-minute T4 x2 session, then read the quota counter. Plans A and B below depend on the answer.
   - P100 has 2.3x the memory bandwidth of a T4. EfficientNet's depthwise convolutions are memory-bound, so a P100 may beat a single T4 even with AMP. A 50-step benchmark on each settles it.
3. **The pre-registered `tune:hyst` grid cannot fairly score soft-target maps.**
   - `compare.py` GRIDS["hyst"] has `seed_thresh` [0.9, 0.95, 0.97, 0.99] and `thresh` ≥ 0.3. That grid was tuned on hard-target E9a maps.
   - With soft targets, p estimates the fraction of annotators. A filament drawn by 2 of 3 annotators peaks near 0.67, so a seed ≥ 0.9 always drops it.
   - By F26 that filament should be kept: mean IoU is about 2×0.7/3 ≈ 0.47, well above PQ/2 ≈ 0.21.
   - v2 (EMA plus instance losses) will shift calibration differently from N1, so a clipped grid can flip the G2 verdict.
   - **Fix before any maps arrive (CPU only):** pre-register `seed_thresh` [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99] and `thresh` [0.2, 0.25, 0.3, 0.4, 0.5] for both arms. Point 6 in § 4 of the PLAN ("widen the grid if the best combo is at the edge") already allows this; doing it up front avoids a grid chosen after seeing the results.
4. **This explains ConvNeXt's loss and rules out stride-4 backbones.** I instantiated the smp encoders locally: `tu-convnext_tiny` has `out_channels [1, 0, 96, 192, 384, 768]`, so it gives **no stride-2 skip** to the U-Net decoder. For 5–10 px-wide filaments at 1024 that is a plausible cause of E9b's −0.020 [−0.031, −0.010]. MiT/SegFormer, Swin and the FPN/DeepLabV3+ heads (1/4-resolution output) share the problem. EfficientNet-B4/B5, EfficientNetV2-S, RegNetY and ResNet all have a stride-2 skip.
5. **Inference experiments can run on the Mac for free.** E9a inference on MPS takes about 1.0–1.2 s per forward at 1024 and about 2.1 s at 2048 (fp32; fp16 on MPS is slower at 2048). A full 8x TTA pass over the 213 holdout images is about 35 min. Using the local `outputs/colab/unet1024_effb4/unet1024_effb4.pt`, TTA, multi-scale and averaging variants cost no Kaggle GPU.

Two smaller points:
- **Runs are not reproducible.** `torch.manual_seed(0)` is set, but `np.random.default_rng()` in the datasets is unseeded, so "same seed" pairs still carry seed noise. That is why the two-fold rule is right.
- **The decoder starts from the same weights in every run** (manual_seed 0). That makes weight soups slightly more plausible than usual (rejected below anyway).

## (a) Ranked ideas (top 10)

Error buckets are from F28 (FP) and F29 (FN). GPU hours are T4 hours for one fold, by the estimate in finding 1.

| # | Idea | Mechanism / bucket | Evidence | Expected Δ PQ (conservative) | T4 GPU-h | Effort in our code | Minimal decisive experiment |
|---|---|---|---|---|---|---|---|
| 1 | **N3' recipe v2** (blob Dice, false-component penalty, EMA), already built | Equal weight per instance → fewer near-miss "too thin" FN/FP (14.5% / 18%) and fragments (18.6%); false-component term → the ~120 orphan FP (21% in-disk) | S11a +0.009 (90% CI, a bundle); S10b +0.0095 on 4/5 folds; F32 (a better encoder did not move near-miss or merge errors) | **+0.004 to +0.008.** The pre-registered bar (+0.010, CI > 0, positive on both folds) will probably say "reject" even if the effect is real. Accept that and keep N1 | 2.6 per fold, ×2 folds (plus matching N1 controls) | Done (flags) | `effb4_v2_s1024_f0/f1` vs `effb4_s1024_f0/f1`, same bs. Score with the **widened** hyst grid. Also run the fp_kinds diagnostics |
| 2 | **Widen the post grid for soft maps** (finding 3) | Stops soft maps losing majority-supported filaments to a seed ≥ 0.9 | F26 arithmetic; the current grid was tuned on hard-target E9a | 0 to +0.01 for N1 versus a clipped grid; mainly it keeps G2 fair | 0 | 1 line in GRIDS; re-run time is CPU only (about 8 min per model per 213 images, longer per mode) | Pre-register now; run on N1 f0 when it lands. If the best combo is at the new edge, widen once for both arms |
| 3 | **5-fold ensemble of the G2 winner** (H1) | Lower variance: fewer isolated look-alike FP; smooths the mid-run instability (F35) | Every top team does it; our own measurements are weak: E5+E7 +0.003, E6 −0.003, ens6 −0.001 (members weaker) | +0.003 to +0.008 on test, **not measurable on OOF** | 3 more folds of the winner: 6.9–7.8 | Done (watcher) | None possible on OOF. Adopt on the prior. Use #4/#5 as the measured proxy |
| 4 | **Measure ensemble gain on folds 0/1 from the G2 pair** | Both arms predict the same held-out fold images, so avg(N1, v2) vs the better single is a real 2-member, same-data ensemble | – | +0.000 to +0.005 | **0** (a by-product of #1) | `ensemble_probs.py` + `compare.py` | Δ(avg − best single), paired bootstrap, cross-tuned, over 282 images. Adds a "loser" member to the final only if CI > 0 (this answers the heterogeneous-member question) |
| 5 | **Second seed of N1 f0** (`effb4_s1024_f0_s1`, only with Plan A) | Measures seed noise directly (we have none) and gives a 2-seed ensemble gain on OOF, the best proxy for #3 | – | Ensemble +0.002 to +0.005; mainly sets the noise floor for G2 | 2.3 | Add a `--seed` flag and seed the np rng (about 5 lines) | \|PQ(seed0) − PQ(seed1)\| on fold 0, and Δ(avg of the two seeds − single) |
| 6 | **Mac-side TTA and multi-scale tests on E9a** | Scale jitter at test time may recover borderline extent (near-miss "too fat/thin") | none on this task | −0.003 to +0.003 (the model is trained at one scale; may hurt) | **0** (MPS, about 35–70 min per variant) | A short script around `predict_prob` (resize to 1280/1536, predict, resize back, average) | `compare.py` on the 213-image holdout: 8x D4 @1024 vs 8x @1024 + 8x @1280; also 4x vs 8x TTA (useful to drop TTA cost if equal) |
| 7 | **N2: 1024 crops from 2048 images** | Native resolution for small filaments and boundaries | F14 ceiling: IoU 0.894 → 0.909; S10a and S15 use native resolution. Against: S13 −0.07, F30 (small misses are mostly annotator disagreement), F20 | **0 ± 0.02**, prior slightly negative for us | 2.8 per fold; needs 2 folds plus 3 more for rollout (about 14) | Done (`--res 2048 --crop 1024`); inference at 2048 is fully convolutional and fits in 16 GB, so no tiling is needed | Only under Plan A: `effb4_<winner>_c2048_f0/f1` vs the winner's f0/f1 at 1024. The N2 card's error-bucket condition stays |
| 8 | **EfficientNetV2-S (`tu-tf_efficientnetv2_s`, 22 M params) or B5 (31 M)** as encoder or diversity member | A better encoder cut look-alike FP (F32: isolated FP 291 → 257) | F20: R34 → B4 +0.013 [+0.003, +0.023]; Dariush used 5×R34 + 5×V2-S (OOF 0.450, no ablation) | +0.000 to +0.006 | V2-S about 2.0 (fused MBConv is faster than B4); B5 about 3.3 at bs 4 (a bs change breaks pairing) | One flag (`--encoder`); B5 fp16 weights are about 62 MB, under the 100 MB limit | Same-fold pair vs the winner on f0 + f1. Realistically there is **no budget** unless Plan A holds and N2 is skipped |
| 9 | **Fixed last-epoch selection for rollout folds and the report** | Removes best-of-3 selection on the held-out fold (about +0.002 optimism) | E9a: ep25 0.3811 vs ep30 0.3809, so OneCycle annealing already converges | 0 (honesty) | 0 | Use the `.last.pt` weights / ep30 | Report OOF both ways; the final models are ep30 anyway |
| 10 | **Store maps as fp16 instead of uint8 (logit-ready)** | uint8 saturates at 255 for >99.8%, so the hysteresis seed cannot rank the top 0.2%. F34 found 75% of predictions peak at exactly 1.0; soft targets fix most of this | F34 | ≈0 (soft maps are less saturated) | 0 for new runs (twice the map disk space) | `infer_probs` / notebook: `np.float16` | Only if the hyst seed on N1 ends up ≥ 0.97 |

**Logit vs probability averaging, calibration before averaging (S9b).**
- For U-Nets trained on the same soft targets, the outputs are calibrated fractions. Probability averaging is the principled choice, and the cross-tuned threshold absorbs any global shift.
- S9b's −0.021 concerns detector *scores* fused with semantic maps, which does not apply here.
- If v2 turns out sharper than N1 (EMA plus Dice-type losses), test one per-member temperature fit on OOF inside #4 before calling the ensemble negative. Do not bring in isotonic calibration.

**Member count.**
- 5 fold members × 8x TTA @1024 is about 6 s per image on a T4, about 18 min for the test set. That is acceptable for S6b.
- Go beyond 5 only for members that pass #4 with CI > 0. "One recipe × 5 folds" is the explainable default.

## (b) GPU schedule, 2026-11-03 to 11-08 (freeze 11-09)

Rules for every pair:
- Same fold, same `--bs`, epochs, `--val-every` and checkpoint rule.
- First do a 20-step memory smoke test of **v2 at bs 6 on a T4**. If it OOMs, every arm, the N1 control included, drops to bs 4. Never compare a bs 4 run with a bs 6 run.
- The G2 decision needs all four runs (N1 f0/f1, v2 f0/f1). A fold-0 pair alone is not decisive.

**Plan B (conservative: one GPU per session, 2 sessions)**

| When | Session 1 | Session 2 | Quota used |
|---|---|---|---|
| 11-03 evening → 11-04 | N1 f0 → N1 f1 (4.6 h) | v2 f0 → v2 f1 (5.2 h) | 9.8 h |
| 11-04 to 11-05 (CPU) | G2 with the widened grid; #4 ensemble measurement; #6 on the Mac (can start today) | | – |
| 11-05 → 11-06 | winner f2 → f3 | winner f4 | +6.9 to 7.8 → **≈17.5 h** |
| 11-06 to 11-08 (CPU) | OOF assembly and tuning, submissions, fp16 weight check | | – |
| Reserve | about 2.5 h for one crashed or re-run fold. **No N2, no encoder change** | | ≈20 h |

**Plan A (if T4 x2 is charged as one session, 2 runs per session)**

| Wave | Session 1 (GPU0 / GPU1) | Session 2 (GPU0 / GPU1) | Wall | Quota |
|---|---|---|---|---|
| W1, 11-03 → 11-04 | N1 f0 / N1 f1 | v2 f0 / v2 f1 | ≈3 h | ≈6 h |
| G2 (CPU), 11-04 | | | | |
| W2, 11-04 → 11-05 | winner f2 / winner f3 | winner f4 / **N1 f0 seed 1** (#5) | ≈3 h | ≈6 h |
| W3, 11-05 → 11-06, optional | N2 with the winner's recipe: c2048 f0 / c2048 f1 (#7) | spare / V2-S f0 (#8, only if the pair can still roll out) | ≈3.2 h | ≈6.5 h |
| W4, 11-06 → 11-07, only if W3 passes its card | N2 f2 / f3 | N2 f4 | ≈3.2 h | ≈6.5 h (over budget, so run W3 only if W1–W2 came in under estimate) |

- W3 must finish by 11-06 so a W4 rollout can end by 11-08. Otherwise skip W3.
- An N2 member that does not get 5 folds is useless at test time, apart from what #4 measures.

## (c) Rejected ideas

| Idea | Reason |
|---|---|
| Mask R-CNN / any detector member (N4) | Our 512 proxy scored 0.25 alone and −0.006 in fusion. The S9a gain is relative to a U-Net weaker than ours. The 28×28 RoI mask head is poor for long thin objects. About 2 days of code plus calibration plus overlap resolution goes against S6b, and G3 (11-05) cannot be met |
| YOLO11-seg / RT-DETR (ultralytics) | AGPL-3.0. Same complexity argument as above |
| SegFormer/MiT, Swin, ConvNeXt as U-Net encoder, MaxViT; FPN/DeepLabV3+ heads | Stride-4 first features (convnext stride-2 channels = 0, checked); E9b −0.020 with CI below 0; 1/4-resolution output; MaxViT memory at 1024 on 16 GB |
| UNet++ / MAnet / HRNet | No evidence on this task beyond being one member of tushhaaarrrr's ensemble. UNet++ doubles the high-resolution decoder cost; not affordable within 20 h |
| Panoptic-DeepLab center/offset, affinity or watershed-energy, boundary/"group" heads | Targets merge/split errors (about 25% of FN), but E8's group map gave +0.003 (indistinguishable), F23 shows rule-based splitting trades one error for another, and F31/S18 say much of the merge/split is annotator disagreement. New targets plus a new decoder are not testable in 6 days |
| Lovász / Jaccard losses | JML1 alone +0.0003 (umerkang66) |
| Tversky / focal-Tversky | Pixel FP/FN asymmetry is absorbed by the cross-tuned threshold; theainerd has no ablation |
| Boundary / Hausdorff losses | Near-miss errors are mostly disagreement (F31: 59% are TP for another annotator); ill-defined on soft targets |
| clDice | Failed (S9e) |
| Deep supervision and skeleton/core heads (rest of S11a) | No native smp support; would make v2 a second untested bundle; no budget for another pair |
| scSE decoder attention (theainerd) | One flag, but an effect far below the ±0.009 two-fold sensitivity; no ablation anywhere |
| Longer training (40–60 epochs) | E10 R34 60 epochs was inconclusive; T4 cost doubles |
| Shorter training to save budget | ep20 < ep30 on E9a (not annealed); changing epochs breaks the controls |
| Lower or different LR to fix the mid-run instability | The final annealed checkpoints are stable (ep25 ≈ ep30); only final weights matter |
| SWA / checkpoint averaging beyond EMA | EMA (in v2) already averages over about 3 epochs; annealed OneCycle gives the same; no evidence of a gain |
| Model soups across folds | Can't be validated OOF: every fold model has seen the other folds' validation images. Only testable between two seeds of fold 0 (#5) as a curiosity |
| Sliding window with Hann blending at 2048 | Whole-image inference at 2048 fits in 16 GB and avoids seams; tiling only adds edge effects |
| Isotonic calibration before averaging | For detector-score fusion (S9b), not for same-target U-Net probabilities; at most a per-member temperature inside #4 |
| A full-data model without folds | Cannot be validated; the 5-fold ensemble already covers 100% of the data |
| Annotator-conditioned training | Judged 1.5 and rejected on 30/10; test annotators are unknown (S7) |
| Test pseudo-labels, SimSiam, CLAHE, neighbouring GONG frames | S9e and S9d negative |

## Bottom line

The decisive work is the 4-run G2 block, scored with a grid that is fair to soft maps, then a 5-fold rollout of the winner. Everything else is either free analysis on the Mac (#4, #6, #9, #10) or depends on T4 x2 actually doubling the budget (#5, #7, #8). Honest expectation for the final relative to N1: +0.005 to +0.012 OOF, mostly from the fold ensemble and possibly v2. Nothing here promises the +0.03 or more needed to reach 0.44 or above.

Sources (Kaggle quota is third-party, not Kaggle's own docs):
- [Free Cloud GPUs in 2026 (gpuperhour)](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- [ND course page on Kaggle quota counting session time](https://www3.nd.edu/~dchiang/teaching/nlp/2023/kaggle.html)
- [Kaggle 29 GB RAM / 4 CPU update](https://lilys.ai/en/notes/google-tpu-20251128/free-kaggle-gpu-29gb-4-cpu-update)
- [Kaggle quota summary (resourify)](https://resourify.com/resources/kaggle)
- [aimultiple free cloud GPU](https://aimultiple.com/free-cloud-gpu)