# PhysioNet-comprehensive deep-learning comparison: results

Date: 2026-10-15. Kaggle kernel `sweety611/eeg-physionet-deep-sweep` (Tesla T4), version 4,
completed successfully (~27700s / 7.7h total runtime for all 5 architectures). Addresses TMLR
review Critical Issue B for the self-supervised / geometric-deep-learning families, using the
reversed-pretraining-direction protocol described in `reports/data_leakage_audit.md`: each encoder
pretrained on BCI-IV 2a's 9 subjects (`source="bnci"`), then zero-shot LOSO-evaluated across
PhysioNet's full subject population. No held-out PhysioNet subject's data ever touches its own
encoder's pretraining (BCI-IV 2a and PhysioNet are disjoint populations by construction) or its own
linear/MLP probe's training (fit on the other PhysioNet subjects only) -- strict zero-shot
throughout, identical protocol to every other Family 1/2 method in this project.

## Results

| Method | Accuracy | Kappa | n subjects | Pretrain+eval wall time |
|---|---|---|---|---|
| MAE-only | 33.99% +/- 4.75% | 0.008 +/- 0.070 | 109/109 | 7.68h (initial Kaggle run) + local CPU recovery |
| MAE+GRL | 36.80% +/- 7.71% | 0.052 +/- 0.115 | 109/109 | 1.19h |
| SupCon | 33.04% +/- 4.53% | -0.008 +/- 0.068 | 109/109 | 0.88h |
| SPDNet | **49.40% +/- 14.25%** | **0.241 +/- 0.214** | 109/109 | 0.91h |
| Riemannian Transformer v2 | 42.60% +/- 10.45% | 0.139 +/- 0.157 | 109/109 | 1.13h |

Per-subject fold results: `paper/results/physionet_loso_{mae_only,mae_grl,supcon,spdnet,riemannian_transformer_v2}.csv`.
Summary: `paper/results/physionet_loso_deep_summary.csv`.

**MAE-only initially used only 95/109 subjects, later recovered to 109/109 (update: 2026-10-18).**
`load_physionet_zeroshot` skipped 14 subjects during the original Kaggle run due to transient
physionet.org server errors (`Read timed out`, `502 Bad Gateway`) while downloading their raw EDF
files -- the same shared loader's built-in, pre-existing skip-and-continue behavior used by every
PhysioNet-internal experiment in this project. The 4 other architectures, run later in the same
kernel session (so retrying the same downloads), all succeeded on the full 109/109 at the time,
confirming the failures were transient rather than permanent. **Recovery**: local network access
to the exact 14 failed subject IDs was tested directly this session and succeeded immediately, so
MAE-only was retrained from scratch locally (CPU, identical config/seed) and re-evaluated zero-shot
across the full 109-subject population with one single, internally-consistent encoder (the original
incomplete 95-subject Kaggle result is preserved at
`paper/results/physionet_loso_mae_only_INCOMPLETE_95subj_kaggle.csv.bak` for provenance, not
deleted, but is no longer the reported number). New complete result: **33.99% +/- 4.75%, kappa
0.008 +/- 0.070, n=109/109** -- close to the original 95-subject estimate (33.74%/0.006), as
expected for a near-chance result, confirming the missing subjects were not a biased subsample.

## Interpretation

- **SPDNet is the standout**: 49.40% on PhysioNet, matching the classical EA+CSP(8)+LDA baseline's
  PhysioNet result (49.40%, see `reports/` comprehensive classical comparison) almost exactly, and
  far above its own BCI-IV 2a zero-shot number (44.12%, pretrained the *original*, PhysioNet
  direction). This is a genuinely interesting, disclosed finding: staying on the SPD manifold
  (BiMap/ReEig/LogEig) seems to transfer unusually well to PhysioNet's larger, more homogeneous
  population even when pretrained on only 9 source subjects -- worth a sentence in the Results/
  Discussion, not just a table entry.
- **MAE-only, MAE+GRL, SupCon all remain near chance** (33-37% on the project's 3-class PhysioNet
  label space, where chance = 33.3%), consistent with this project's existing BCI-IV 2a findings
  for the same raw-EEG self-supervised family (27-28% on BCI-IV 2a's 4-class chance=25%) -- the
  representation-quality bottleneck identified elsewhere in this project appears population- and
  pretraining-direction-independent, not an artifact specific to the original PhysioNet-source
  setup.
- **SupCon's near-zero/negative kappa (-0.008) is consistent with this project's own documented,
  unresolved SupCon collapse** (`final_results_table.csv` rows on "SimCLR-style... collapse
  persists") -- the reversed pretraining direction does not fix it, which is itself informative
  (rules out "wrong pretraining population" as the cause of the collapse, leaving the root cause
  exactly as unresolved as the original project left it).
- **Riemannian Transformer v2 (42.60% here) is close to its BCI-IV 2a number (42.38%, from the
  *original* PhysioNet-direction pretraining)** -- note the two numbers are not a controlled
  comparison (different pretraining population as well as different eval population, see caveat
  below), but the similar magnitude suggests a modest, unremarkable transfer, unlike SPDNet's much
  larger jump.

## Caveat on this comparison's scope (see `data_leakage_audit.md` for full discussion)

This is a population swap (pretrain on BCI-IV 2a's 9 subjects, evaluate on PhysioNet's 109), not a
literal increase in LOSO fold count for the same pretraining corpus used in the manuscript's
BCI-IV 2a headline numbers (which pretrain on PhysioNet's full 109 subjects). The two sets of
numbers -- BCI-IV 2a zero-shot (PhysioNet-pretrained) and PhysioNet zero-shot (BCI-IV-2a-
pretrained) -- are each internally valid (no leakage within either), but are not a controlled,
single-factor comparison of "same encoder, different eval population" since the pretraining
population also changed. This must be stated explicitly wherever both are shown side by side.
