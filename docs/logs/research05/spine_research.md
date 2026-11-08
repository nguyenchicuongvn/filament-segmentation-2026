# Spine / bbox / area as training signals (research 05/11)

Trigger: host reply (discussion 755241, 04/11): per-annotation `spine`, `bbox`, `area` MAY be used for
training; inference only on the test image. Until now the team banned them (PRIZE_RULES.md L41 still
says so: update it).

Scripts (all read-only on `data/MAGFiLO_1.0_Kaggle_2026`, CPU):
- `scripts/research/spine_stats.py` -> `spine_stats.txt` (all 8,199 annotations)
- `scripts/research/spine_vs_skel.py` -> `spine_vs_skel.txt` (1,500 random annotations)
- `scripts/research/spine_group_oracle.py` -> `spine_group_oracle.txt` (N1+N2 OOF maps, 707 images)
- `scripts/research/spine_pair_group.py` -> `spine_pair_group.txt` (learned pairwise grouping, 707 images)

## 1. What the fields contain (measured)

| Field | Content | Numbers |
|---|---|---|
| `spine` | one flat polyline `[x0,y0,x1,y1,...]` (2048 px coords) per annotation | present in 8,199/8,199; never nested/multi-part; 2–122 vertices (median 10); length p10 49 / median 111 / p90 285 px |
| relation to mask | lies inside its own mask | median 99.7% of rasterised spine samples inside; >= 90% inside for 99.5% of annotations; >= 5 px outside for 6/1,500 |
| vs. `skeletonize(mask)` | essentially the mask skeleton without spurs | 99.7% of spine pixels within width/2 of the morphological skeleton; skeleton spur pixels (> 1 width from spine) median 0%, p90 6.5%; spine length / Feret 1.03 median |
| `segmentation` | always ONE polygon | 8,199/8,199 single polygon; 30 rasterise to 2 pieces (thin necks), of which the spine crosses both in 3 |
| `area` | = decoded mask area | identical for 100% (|diff| <= 1%) |
| `bbox` | = mask bbox (sub-pixel) | mask-bbox offset median 0, |offset| < 0.5 px p1..p99 |
| (`category_id`) | chirality | still not allowed (host allowed spine/bbox/area only) |

Per annotator: every annotator-image entry has its own spines (one per filament it drew), so a
"spine soft target" = fraction of annotators whose spine passes a pixel is constructible exactly like
the mask soft target.

**Consequences.**
1. One spine = one instance, and instances are always one connected polygon. Spines therefore carry
   **no grouping information beyond the masks**: "which fragments are one filament" is already fully
   determined by the instance masks we have always used (e.g. the E8 group channel, F23, and the
   pairwise labels below come from masks).
2. `bbox` and `area` are deterministic functions of the mask: **zero new information**.
3. The only new signal is a *clean, annotator-drawn centreline* (no barb spurs, no tip artefacts).
   Since a mask skeleton is within width/2 of it 99.7% of the time, a "spine head" ~= the
   skeleton head of S11a, which was legal all along. Expected marginal value of JSON spine over
   `skeletonize(mask)`: small (cleaner target on the ~10% of filaments with barbs).

## 2. How much PQ is in instance grouping at all? (measured, N1+N2 OOF, 707 images)

Hysteresis components of final-1 config (`configs/post_ens_n1n2_hyst.json`: thresh 0.4, seed 0.99,
min_area 200, group_radius 12 @2048), regrouped. Oracle = consensus GT relation (a component's GT
per annotator entry = GT covering >= 50% of it; same / different if >= half the entries say so).
Paired bootstrap over images (500).

| Grouping | PQ | Δ vs radius | 95% CI | TP | FP | FN | gt_split | pred_merge |
|---|---|---|---|---|---|---|---|---|
| none (radius 0) | 0.4269 | −0.0026 | [−0.0061, +0.0010] | 5017 | 2669 | 3182 | 478 | 165 |
| **radius 12 (current)** | **0.4294** | 0 | – | 4881 | 2292 | 3318 | 282 | 289 |
| oracle join only | 0.4414 | +0.0120 | [+0.0094, +0.0149] | 4886 | 1860 | 3313 | 14 | 349 |
| oracle cut only | 0.4395 | +0.0101 | [+0.0074, +0.0132] | 5036 | 2266 | 3163 | 291 | 195 |
| oracle join + cut | 0.4523 | **+0.0228** | [+0.0190, +0.0266] | 5049 | 1826 | 3150 | 23 | 257 |

(radius 0.4294 reproduces the recorded in-sample final-1 number, so the script is consistent.)
Ceiling for ANY grouping step on existing components is +0.023 (cannot fix merges inside one
connected component: pred_merge 257 remain). History warning (F27): decision-layer oracle +0.058 was
learned to −0.004. A learned model typically recovers a fraction of a ceiling.

## 3. Learned pairwise grouping (option b, CPU) — measured

`spine_pair_group.py`: candidate pairs = component pairs with gap <= 48 px @2048 (519 pairs in 707
images, 0.73 per image: grouping decisions are rare); 428 labelled (247 same, 181 different) by the
consensus GT relation from **masks** (spines are not needed for the labels). 17 geometric/map features
(gap, areas, min/mean p along the gap segment, PCA-axis angles, gap-to-axis angles, elongation,
endpoint-ness, radius-rule flag). HistGBM fitted 5-fold by image (OOF), join threshold cross-tuned.

| | value |
|---|---|
| radius rule accuracy on labelled pairs | 0.621 |
| learned AUC / accuracy@0.5 | 0.767 / 0.701 |
| PQ in-sample over thresholds 0.2..0.8 | 0.4318–0.4331 (flat) vs radius 0.4294 |
| **cross-tuned Δ** (3 splits x 2 halves) | **+0.0030** (halves +0.0017, +0.0042, −0.0022, +0.0070, +0.0025, +0.0046) |
| at t = 0.8 (most picked), paired bootstrap 300 | +0.0036 [−0.0001, +0.0071] |

Recovers ~15% of the +0.023 oracle. Zero GPU. Not yet acceptable by F48 (CI touches 0, threshold
picked from the grid).

## 4. Ranking (expected PQ per GPU hour)

| # | Option | Evidence | Expected Δ OOF | Cost | Verdict |
|---|---|---|---|---|---|
| 1 | (b) learned pairwise grouping on existing OOF maps; labels from masks | measured +0.0030 cross-tuned, +0.0036 [−0.0001, +0.0071]; ceiling +0.023 | +0.003 to +0.006 (with better features) | CPU only, ~1–2 h (machine load-dependent) | **do it** |
| 2 | (a) aux spine head (2nd channel) | spine ~= mask skeleton (99.7%); S11a whole package +0.009 (90% CI, R34 proxy), skeleton head share unknown; E8 group channel +0.003 no CI; v2 (F49) −0.05 from untuned weights | solo 0 to +0.005 (undetectable: 2-fold CI ±0.009); ~+0.004 as an extra ensemble member, but that is the generic member effect (F51), not the spine | 2.3 h T4 per fold, needs 2 folds = 4.6 h + Mac pre-test | only if GPU is left after BCE f2–f4; then make its spine map a feature of (b) |
| 3 | (c) centre/offset/direction field (Omnipose-like) | E8 +0.003; F23 rule-based splitting swaps errors; remaining merges inside one component: pred_merge 257 | unknown, likely <= +0.005 in 10 days | new head + new post-processing + ≥2 folds GPU | no |
| – | bbox / area | identical to the mask | 0 | – | nothing to use |

## 5. Plan for option (b)

Pre-registered (write before the run):
- Arms: A = final-1 (N1+N2 OOF maps, `configs/post_ens_n1n2_hyst.json`, radius 12); B = same components,
  grouping by the pair GBM. Same hysteresis params in both arms.
- B's GBM is fitted out-of-fold by `fold_split` (as in the script); threshold cross-tuned on halves.
- Feature set v2 decided BEFORE scoring: v1 + skeleton-endpoint tangents of each component (angle
  between the end tangent and the gap vector, extrapolation error of the tangent line to the other end),
  gap <= 64 px. One run of v1 and v2; take v2 only if its cross-tuned Δ > v1's.
- Accept if cross-tuned Δ > 0 with paired bootstrap (1000) 95% CI > 0 on 707 images AND Δ > 0 on >= 4/5
  folds. Else ship nothing (radius stays).
- Ship: GBM refitted on all 707 OOF pairs -> `configs/pairgroup_n1n2.pkl` + threshold (median of the
  cross-tune picks); apply in `make_submission.py` after hysteresis. Risk: test maps come from the
  5-fold ensemble (sharper than OOF maps); check pair counts/feature means on test vs OOF.

