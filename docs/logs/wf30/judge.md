# Workflow 30/10 — judge (verbatim)

## Ranking

- **6 · build_now · training** — Instance-normalised loss: closed-form blob Dice + BiCC-style predicted-component term + EMA, on the N1 soft-target recipe (replaces the limb-input part of N3)
  Best training-side bet, but the expected gain is modest. Checked: S11a +0.0092 (90% CI [+0.006,+0.012]) is a bundle (blob + skeleton/core heads + deep supervision + JML1), not blob alone, and JML1 alone gave +0.0003. S10b BiCC per-fold +0.008/+0.005/+0.023/+0.0135/-0.002 (mean +0.0095, no CI) matches RESEARCH.md. Halving each external number gives about +0.004-0.005 per component, so a realistic bundle expectation is +0.006-0.008, under the one-fold CI (+-0.012). Fold 1 will probably be needed (the PLAN rule already allows it). The mechanism is backed by two independent anatomies: near-miss/partial/fragment buckets total 44-56% of FP, near-miss FN 22-24%, and the encoder upgrade did not move near-miss (301->289) or merge (66->67). The arithmetic (+0.012 per 50 near-miss conversions) re-checks. It costs no more GPU than the planned N3. Changes vs the investigator: instances come from connected components of the soft map itself (no per-annotator label cache, so we avoid repeating the E12-E14 cache bugs), and limb input is dropped.

- **6.5 · later · training** — Do not spend GPU on limb-corrected input or small-filament recall; re-scope N3 and treat N2 as an SQ/near-miss test
  ACCEPT as a finding (doc/plan change, nothing to build). The errors lens replicates it independently: location/limb/sunspot buckets hold 3/794 FP and 1/2194 predictions lie beyond 70 deg. The none-matched rate is flat with r/R (0.25-0.31). The human-vs-model small recall (0.33 vs 0.32) is pooled over different image sets, which is a caveat, but the 63-co-drawn-miss ceiling (+0.011) from the errors lens points the same way. This removes the only N3 component with zero evidence. It feeds directly into build_gpu.

- **6 · later · decision** — Do NOT build a learned multi-threshold decision layer (H6 rejected on our maps)
  ACCEPT as a measured negative result (record in EXPERIMENTS/RESEARCH H6, no build). Two investigators replicated it independently with different features and code. Decision lens: tree r4 -0.004 [-0.013,+0.005] on 123 images. Errors lens: -0.0043 [-0.011,+0.003] on all 213. The keep/drop variant reproduces ledger R5b (-0.002). Oracle +0.125 vs learned about 0, explained by predictor R2 0.26 and the winner's curse in the tree DP. The S15 +0.06 was against fixed thresholds, not tuned strict hysteresis, so it does not transfer. The decision lens's 123-image subset is a weakness, but the errors lens's 213-image run covers it; no rerun needed.

- **5.5 · later · errors** — Stop post-processing work and drop location/sunspot priors (re-prioritisation)
  ACCEPT (doc-only); duplicates the two findings above. Adds the single-feature AUCs (location/darkness/contrast 0.40-0.51) and is consistent with F21 (all post tweaks +0.000..+0.008, non-additive). Exception: a post/filter change that adds a genuinely new input signal (see the disagreement filter) is still allowed, under a strict CPU-only budget.

- **5.5 · build_now · errors** — Use near-miss/extent and partition buckets as mechanism diagnostics for N3 acceptance
  Folded into build_gpu as secondary diagnostics; it costs nothing because collect.py/analyze.py and fp_kinds.py already exist. It supports a report claim of why a bundle works, which judges value. It is not the acceptance criterion: acceptance stays PQ delta with CI. Caveat noted: 59% of near-miss predictions on multi-annotator images already satisfy another annotator, so widening can be zero-sum.

- **3.5 · build_now · decision** — Cross-model disagreement features for the keep/drop filter (4 other holdout maps already on disk)
  It is the only CPU-measurable idea with a non-zero prior, and it adds a new signal (epistemic disagreement) rather than another post-processing grid on the same map. The prior is low: E-ens6 mean-ensembling gave -0.001 [-0.009,+0.007], the models share the same 70% split, and the ceiling for removing FP alone is +0.058 while the current predictor's R2 is 0.26. Expect +0.003. It costs about 3 dev hours and 25 CPU-minutes, and a cheap kill gate is pre-registered. Deployment caveat: at test time it needs the 4 holdout models' test maps (they exist), which adds pipeline weight. It cannot be re-derived from N1 folds, because OOF images see only one fold model. Accept only under the strict rule below.

- **5 · later · errors** — Gate N2 (2048 crops) hard: small-filament misses are mostly annotator-idiosyncratic
  Apply at G1 (no build). The gate (delta >= +0.012, CI > 0) already exists in PLAN. What this adds is the prediction to check: N2 must move the co-drawn/near-fat FN buckets, and if it does not, no 2048 roll-out (saves 6-7 GPU h). The numbers check: 63 co-drawn misses give a +0.011 ceiling, and F14 shows the resolution ceiling barely moves.

- **3 · later · decision** — Re-run the oracle/learned tree analysis on N1 soft-target maps (is extent learnable after calibration?)
  0 GPU, about 1 CPU hour, but the most likely outcome is again 0 (annotator lottery). Run only if the CPU is idle after N1 lands. It is a diagnostic for the report, not a gain.

- **3.5 · later · training** — Mask R-CNN detector member for grouping errors (N4)
  The grouping buckets are real (merge+fragment about 11% of FP, merged+split about 25% of FN; the oracle for partition fixes is about +0.05, overlapping). But the evidence is a single external number (S9a +0.033) against a weaker U-Net (0.396 vs our 0.408), and voting reached CI > 0 in only 3/5 splits (S9c). Halved and discounted for our stronger baseline, that is about +0.005-0.008. Dev about 20 h, GPU 7.5 h, judge-complexity penalty, and the E8 group channel (a learned grouping signal) gave only +0.003. Keep the existing G3 (05/11) go/no-go. Spend GPU only if build_gpu leaves at least 7.5 h plus a 2 h reserve.

- **3 · later · errors** — Instance-partition errors need an instance-aware signal: N4 detector or a boundary/affinity head
  Same target and evidence as N4; merged into that card. An affinity/boundary head is new, untested code with no external number. E8 (group channel, +0.003) is the closest internal analogue and was null.

- **1.5 · reject · training** — Annotator-conditioned training with condition dropout (style head)
  No external or internal anchor (the investigator says so). In expectation the 'unknown' condition learns the same soft fraction as N1, so the gain is plausibly exactly 0. Test annotators are unknown (S7). With 37 annotators it risks overfitting. Not worth 1.5 GPU h of a 16 h budget.

## build_cpu: Cross-model disagreement features for the instance keep/drop filter, fully cross-tuned (new compare.py spec 'tune:hyst,filter:<set>')

Goal: test whether disagreement between the 5 holdout models (effb4 + r34, r34_grp, r34_e60, cnxt; maps at outputs/colab/<run>/probs_val, 213 images) gives the instance filter a signal the 17 existing features lack. All parameters are pre-registered here; no feature selection after seeing results.

1) src/filseg/instfilter.py: add AGREE_FEATURES = ['agree_mean_m1','agree_mean_m2','agree_mean_m3','agree_mean_m4','agree_std5','agree_min4','agree_vote3','agree_iou4'] and a function agreement_features(inst: np.ndarray, base_prob: np.ndarray, other_probs: list[np.ndarray]) -> np.ndarray of shape (n_inst, 8), all at 1024. For each instance label k of the effb4 hysteresis output: agree_mean_mj = mean of other model j's prob inside instance k (fixed order r34, r34_grp, r34_e60, cnxt; for r34_grp use channel 0 from load_maps). agree_std5 = std over the 5 per-model instance means (effb4 included). agree_min4 = min of the 4 other means. agree_vote3 = fraction of instance pixels where >= 3 of the 5 models have p > 0.5. agree_iou4 = mean over the 4 other models of the best IoU between instance k and any connected component (8-connectivity, min_area 50, of that model's p > 0.5 map). Do NOT append these to FEATURES: saved filters select columns by name, and existing pickles must load unchanged.

2) scripts/compare.py: add a spec form name=probs_dir,tune:<mode>,filter:<set>, where <set> is 'base' (the 17 FEATURES) or 'base+agree' (17 + 8), plus a CLI flag --agree-maps (comma-separated probs dirs, required only for base+agree). Existing spec forms and their outputs must be byte-identical. Procedure inside each of the existing 3 random half-splits x 2 directions (same seeds as evaluate_crosstuned): (a) tune the <mode> post grid on half A exactly as tune:<mode> does now; (b) build instances on all images with the half-A params; (c) fit the classifier (fp_filter.make_model('clf'), label = matches at least one annotator at IoU > 0.5, as fp_filter._image_table does) on half-A instances only; (d) choose the drop threshold on half A by the PQ sweep fp_filter already uses; (e) score half B with half-A params, model and threshold. Aggregate per-image stats exactly as evaluate_crosstuned does, so the paired bootstrap works unchanged. Nothing is fitted on the images it scores.

3) Cheap kill gate (diagnostic only, it never selects): a 5-fold-by-image OOF ROC-AUC of the clf label for 'base' vs 'base+agree' on the hyst instances (configs/post_unet1024_effb4_hyst.json). Print both. If AUC rises by < 0.01, still run the full comparison once (it is cheap), but expect a null.

4) Tests (tests/test_agreement.py): synthetic 64x64 maps with two instances; agree_mean, std, vote3 and iou4 match a hand computation; base_prob identical to the others gives agree_std5 = 0 and agree_iou4 = 1 for an instance equal to the thresholded component. A compare.py test (or assertion) checks that the tune:hyst output without a filter is unchanged.

Resource rules: set OMP_NUM_THREADS=1, --jobs <= 4, wall time <= 25 min. Cache the per-image feature tables in a scratch pickle so the bootstrap re-runs do not recompute them.

**Measurement**: Command (repo root):
OMP_NUM_THREADS=1 .venv/bin/python scripts/compare.py base=outputs/colab/unet1024_effb4/probs_val,tune:hyst filt17=outputs/colab/unet1024_effb4/probs_val,tune:hyst,filter:base agree=outputs/colab/unet1024_effb4/probs_val,tune:hyst,filter:base+agree --agree-maps outputs/colab/unet1024_r34/probs_val,outputs/colab/unet1024_r34_grp/probs_val,outputs/colab/unet1024_r34_e60/probs_val,outputs/colab/unet1024_cnxt/probs_val --jobs 4

Control: the current best, strict hysteresis cross-tuned on the 213 holdout images (spec 'base', must reproduce PQ 0.4083 +- 0.001; if it does not, stop and debug before reading anything else). Secondary control: filt17, the same harness with the existing 17 features, which isolates the new signal from the already-known filter gain (about +0.004).

Pre-registered acceptance: delta(agree - base) >= +0.010 AND 95% paired-bootstrap CI lower bound > 0 (2000 resamples by image), AND delta(agree - filt17) > 0 with P(>0) >= 0.90. If both hold, add it to the E9a-holdout final candidate only (test maps exist in outputs/colab/<run>/probs_test). Record that it needs 4 extra holdout models at inference, and that it cannot be ported to the N1 fold ensemble without new OOF evidence. Otherwise record it in the EXPERIMENTS ledger as a negative result with its delta and CI, and drop it. Also record in the ledger the two replicated findings that need no build: the learned tree decision layer gave -0.004 [-0.011,+0.003] on 213 images against an oracle of +0.125 (H6 rejected), and location/limb/sunspot errors are under 1% of errors.

## build_gpu: N3' 'recipe v2': instance-normalised loss (closed-form blob Dice + BiCC-style false-component term) + EMA on the N1 soft-target recipe; limb-corrected input dropped

Why it replaces the planned N3 bundle: EMA and blob loss stay. The limb-corrected input is dropped because two independent analyses measured location/limb/sunspot errors at under 1% of errors (3/794 FP; none-matched rate flat with r/R 0.25-0.31), so it has no measurable target and adds an input-pipeline change to the test path. It is replaced by a predicted-component term, the 'other half' of BiCC (S10b, +0.0095 mean, 4/5 folds). That term targets the about 120 FP that touch no GT and the fragment FP, which blob loss does not penalise per component. The bundle stays one pre-registered card, as PLAN § 3.3 requires.

All new behaviour sits behind new flags whose defaults reproduce today's code path exactly (same loss call, same dataset tuple, same checkpoint/meta/history contents), because the Colab Q3 runs use the current flags.

src/filseg/dl.py:
- instance_labels(soft: np.ndarray, min_area: int) -> np.ndarray[int32]: 8-connected components of soft > 0. Components smaller than min_area get label -1: they still count as 'other instance' pixels for everyone else's blob term but contribute no blob term of their own. Background is 0.
- SoftDataset.__init__ gains instances: bool = False and inst_min_area: int = 10. With instances=False, __getitem__ is unchanged and returns the same 2-tuple. With True, labels are computed on the final (cropped, rotated, flipped) soft mask, so they stay aligned, and a third int64 tensor of shape (1,H,W) is returned. No new disk cache.
- blob_dice_loss(logits, target, labels, eps=1.0) -> Tensor, closed form with scatter_add per sample. p = sigmoid(logits). For sample b and instance k >= 1: Pin_k = sum over lab==k of p; A_k = sum over lab==k of p*s; S_k = sum over lab==k of s; F = sum over lab!=0 of p (instances incl. -1); Pt = sum of p. Dice_k = (2*A_k + eps) / (Pt - F + Pin_k + S_k + eps), i.e. each instance's Dice is computed over the whole image with all other GT instances masked out (blob loss). Loss = mean over all valid k in the batch of (1 - Dice_k). With no instances, return logits.sum()*0 (finite, graph intact).
- false_component_loss(logits, target, thresh=0.5, min_area=10) -> Tensor: binarise the detached sigmoid at thresh, cv2.connectedComponentsWithStats (8-connectivity) per sample on CPU uint8, and keep components with area >= min_area and zero overlap with target > 0. Loss = mean over those components (batch-pooled) of the mean non-detached p inside the component; logits.sum()*0 if none.
- bce_dice_loss is untouched. New wrapper: train_loss(logits, target, labels, w_blob, w_fpc, fpc_thresh, min_area) = bce_dice_loss + w_blob*blob + w_fpc*fpc, which also returns the components for logging.

scripts/train_unet.py, new flags with backward-compatible defaults:
--blob-weight (float, 0.0)
--fpc-weight (float, 0.0)
--fpc-thresh (float, 0.5)
--inst-min-area (int, 0 = auto: round(10*(res_of_training_input/1024)^2), with crops using --res)
--ema (float, 0.0 = off; e.g. 0.998)
parser.error if blob or fpc weight > 0 without --targets soft. Only when a weight > 0: SoftDataset(instances=True), the loader yields 3-tuples, and the loss is train_loss. Otherwise the loop calls bce_dice_loss exactly as now.

EMA (only if --ema > 0): ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True). ema.update_parameters(model) runs after each optimizer step (after scaler.step/update). validate() and the best-checkpoint save use ema.module. The checkpoint key 'model' holds the EMA weights, so predict_prob, the Colab TTA map writer and make_submission need no change; a 'model_raw' key holds the raw weights for a possible later free ablation. last.pt additionally stores 'ema' and is restored on --resume (re-initialised from the model if absent).

meta/history: new keys enter meta, and the history columns loss_blob and loss_fpc are added, only when non-default, so files from default runs stay identical.

Tests (tests/test_instance_loss.py):
(1) the closed-form blob_dice_loss equals a brute-force per-instance loop that builds each masked image explicitly (random logits, soft target with 4 instances incl. one below min_area), loss and autograd gradients within 1e-5;
(2) a zero-instance crop gives loss 0, finite, and backward works;
(3) false_component_loss on a synthetic case with one GT-overlapping and one isolated predicted blob equals the mean sigmoid of the isolated blob; with no isolated blob it is 0;
(4) SoftDataset(instances=True): (labels != 0) == (mask > 0) after rot/flip/crop for many seeds; SoftDataset(instances=False) returns a 2-tuple;
(5) the argparse defaults give blob/fpc 0 and ema 0 and the code path builds no AveragedModel (factor the setup into a small helper so it can be tested);
(6) an EMA checkpoint 'model' state_dict loads into build_model(...) and gives the same output as ema.module.
The whole suite must pass: .venv/bin/python -m pytest -q tests.

Mac smoke (MPS, about 15 min):
.venv/bin/python scripts/train_unet.py --res 256 --bs 2 --epochs 2 --encoder resnet34 --targets soft --fold 0 --blob-weight 1.0 --fpc-weight 0.3 --ema 0.998 --val-every 1 --workers 0 --name smoke_v2 --out-dir outputs/smoke
Then check finite losses, loss_blob/loss_fpc in the history and val PQ > 0. Then run --resume with --epochs 3 to check EMA restore. Then run the same command without the new flags (name smoke_v1) to confirm the old path still runs and its meta/history columns match the old set. Use the atomic-write pattern already in place; no new cache.

colab/train_unet.ipynb (parameters cell): add after the third-queue loop, without touching RUN_NAMES:
EXPERIMENTS['effb4_v2_s1024_f0'] = f'--res 1024 --bs 6 --encoder efficientnet-b4 {FOLD} --fold 0 --blob-weight 1.0 --fpc-weight 0.3 --ema 0.998'
with a comment pointing to PLAN card N3. If G1 selects 2048, the builder adds the analogous 'effb4_v2_c2048_f0' with '--res 2048 --crop 1024 --bs 6'. Re-run scripts/package_colab.py so the bundle carries the new code. Update the PLAN.md N3 card: drop limb input, and move BiCC-style from 'Không đưa vào' (not included) to the bundle, with the evidence above.

**Measurement**: Control: N1 fold 0 (effb4_s1024_f0, or effb4_c2048_f0 if G1 picks 2048): same encoder, resolution, 30 epochs, --val-every 10, same best-of-3-checkpoints rule, same fold-0 images (~141). Treatment: effb4_v2_s1024_f0. Command:
.venv/bin/python scripts/compare.py n1=outputs/colab/effb4_s1024_f0/probs_val,tune:hyst v2=outputs/colab/effb4_v2_s1024_f0/probs_val,tune:hyst
Paired bootstrap by image, cross-tuned. Report tune:cc as a secondary check. If the best combo of either arm sits at a grid edge, widen the grid for both arms identically and rerun.

Pre-registered rule (PLAN § 3, card N3): delta >= +0.012 with 95% CI > 0 on fold 0: accept. 0 < delta < +0.012: run fold 1 (effb4_v2_s1024_f1 vs effb4_s1024_f1, +1.3 GPU h) and accept if the CI pooled over folds 0+1 is > 0. Otherwise reject and keep N1. Acceptance is never decided on the LB.

Mechanism diagnostics, reported but not used for acceptance: run fp_kinds.py and errors/analyze.py on both arms, recording near-miss+partial FP count, isolated (no-GT) FP count, merged/split FN, mean TP IoU (SQ) and the 1:n/n:1 rates. The report may credit the instance-level loss only if near-miss+partial FP fall by >= 10% or isolated FP by >= 20%. Report it as a bundle, not per component.

## Roadmap

GPU budget: about 25 h L4 total, Q3 about 8 h, so about 16 h left for everything new. Keep a 2 h reserve for one failed or cut run, which leaves 14 h of spendable time.

30-31/10 (CPU, no GPU):
- Build build_cpu (disagreement filter, cross-tuned). Decide on the 213 holdout images with the pre-registered rule.
- Record in EXPERIMENTS/RESEARCH: H6 rejected (learned tree -0.004 [-0.011,+0.003] vs oracle +0.125, replicated twice), and location/limb errors under 1% (so limb input is removed from N3).
- Implement build_gpu (N3') on the Mac with unit tests and smoke, and re-package the Colab bundle.
- When Q3 lands: N1 OOF baseline B1, and G1 (N2 vs N1 fold 0) with the stricter diagnostic. N2 must also move the co-drawn/near-fat FN buckets, else no 2048 roll-out.

31/10, Q4: N3' fold 0 at the resolution G1 chose, 1.3 h (@1024) or about 1.6 h (@2048 crops). Spent: 1.3-1.6 h.

01-02/11, G2:
- delta >= +0.012 with CI > 0: accept.
- 0 < delta < 0.012: fold 1, +1.3 h (spent up to 2.9 h), decide on the pooled CI.
- Accepted: run the remaining folds, 3.9-5.2 h (spent up to about 8 h). The v2 5-fold ensemble replaces N1 as the main U-Net. If N2 also won G1, run v2 at 2048 instead of rolling out N2 and v2 separately (never both: that would cost about 13 h).
- Rejected: N1 stays the main member, and about 13 h remain.

02-05/11, N4 (Mask R-CNN), only if:
- its code and smoke test are done by 02/11;
- at least 7.5 h + 2 h reserve remain, which is true if N3' was rejected, or accepted after only fold 0;
- fold 0 is trained by 03/11 (1.5 h).
G3 on 05/11: the ensemble must reach delta >= +0.012 with CI > 0 and partition errors (merge+split+fragment) must fall by >= 25%, else N4 is dropped. Folds 1-4 cost 6 h.
Worst case if everything passes: 1.6 + 1.3 + 3.9 + 1.5 + 6 = 14.3 h. That is inside 14 h only if N3' needs no fold 1. If both N3' needs fold 1 and N4 passes G3, cut N4 to fold 0-2 members or drop it; the reserve is not touched.

Deferred, CPU only, run if the box is idle:
- Decision-lens D3 (oracle/learned tree on N1 or v2 OOF maps, as a report diagnostic).
- N5 rubric metrics on the final OOF.
Rejected: annotator-conditioned training. Merged into N4: the affinity/boundary head idea.

06-08/11: assemble the finals from OOF cross-tune.
- Final 1: the highest OOF pipeline, v2 or N1 5-fold, plus the disagreement filter only if build_cpu passed and it ports honestly. Otherwise strict hysteresis.
- Final 2: the N1 5-fold ensemble, lowest risk.
Every component listed in the report carries its measured delta and CI.

09/11: model freeze. Weights go into the repo. Masks are asserted non-overlapping. The repo stays private until 15/11.
