# Workflow 30/10: analysis scripts (snapshot)

Analysis scripts written by the three investigator agents of the 2026-10-30 multi-agent workflow
(see `docs/RESEARCH.md` § 10). They are kept so every number in RESEARCH § 10 can be reproduced;
they are research code, not part of the pipeline.

| Folder | What | Outputs (in `docs/logs/wf30/`) |
|---|---|---|
| `decision/` | component trees over 0.1–0.98 levels on the E9a holdout maps; oracle and learned (5-fold OOF GBM + tree DP, cross-tuned) selection | `decision_*.log`, `ideation_decision.md` |
| `errors/` | per-object FP/FN buckets, PQ-if-fixed counterfactuals, tree oracle vs learned selection on 213 images | `errors_*.out`, `ideation_errors.md` |
| `training/` | FP/FN anatomy (size, limb, peak probability, human vs model on small filaments), FP/FN kinds for E9a and R34-e60 | `training_*.txt`, `ideation_training.md` |

Run from the repo root with `.venv/bin/python scripts/research/wf30/<folder>/<script>.py` and
`OMP_NUM_THREADS=1`. Intermediate pickles are written next to the scripts (git-ignored).
