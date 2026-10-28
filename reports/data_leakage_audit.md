# Data leakage audit

Date: 2026-10-13. Scope: check every normalization, covariance-estimation, feature-selection,
hyperparameter-selection, time-window-selection, model-selection, calibration, and
target-subject-data-access point across the pipeline for leakage, per the review's strict
zero-shot protocol requirements.

## Summary: no leakage found in the existing BCI-IV 2a zero-shot pipeline

Checked `paper/eval_riemannian_advanced.py`, `paper/eval_csp_lda_svm.py`, `paper/eval_mekt.py`,
`paper/eval_rpa.py`, `paper/eval_mekt_ensemble.py`, `paper/eval_zero_shot*.py`:

- Every classifier/transform (CSP, TangentSpace, MDM/FgMDM, MEKT's JDA projection, linear/MLP
  probe heads) is `.fit()` on source subjects only and `.transform()`/`.predict()` on the held-out
  subject. Confirmed by direct code read, not assumed.
- Euclidean Alignment (EA) is applied **per-subject**, using only that subject's own trial
  covariances — this is unsupervised and subject-local, so it is valid to apply to the held-out
  subject too (no cross-subject information, no labels). `eval_riemannian_advanced.py`'s own
  docstring explicitly flags the one case where this distinction matters: per-CLASS EA requires
  labels, so it is restricted to source subjects only (`run_tslda_per_class_ea`), while the
  held-out subject gets only the label-free global EA. This is correct and the authors clearly
  understood the distinction.
- MEKT / RPA's pseudo-labeling is transductive but zero-shot-legal: the held-out subject's true
  labels are used only at the final `classification_metrics()` scoring call, never inside the
  fitting/pseudo-labeling loop. Verified by tracing `target_y` usage in `eval_mekt.py` and
  `eval_rpa.py` — it is read only after `mekt_lite()`/`rpa_lite()` return.
- Hyperparameter selection (e.g. MEKT's `k=50, lam=1.0, n_iter=20`, the 4-window ensemble members)
  was tuned via a "CPU sweep" per `final_results_table.csv` row 20-21's notes, performed with
  access to the LOSO accuracy itself (i.e., selected by looking at held-out-subject performance
  across the sweep). **This is the one genuine methodological soft spot**: hyperparameter search
  was not done through a separate nested/inner validation split blind to the final LOSO numbers.
  The manuscript does not claim otherwise, but reviewers should note headline numbers like
  48.48%/52.39% reflect hyperparameters chosen by looking at exactly the subjects being reported
  on. This is a common practice in small-N transfer-learning papers but is a real, disclosable
  limitation, not a secret -- flagged for the Limitations section update.

## Leakage found and fixed: deep-learning families cannot be validly evaluated within PhysioNet

This is the central finding of this audit, already partially self-identified by the original
authors (see `revision_audit.md` section 3) but not previously documented as a standalone leakage
analysis:

- `config.PRETRAIN_SUBJECTS = list(range(1, 110))` — every Family 1/2 (SSL/geometric-DL) encoder
  checkpoint in `paper/checkpoints/` was pretrained using **all 109** PhysioNet subjects
  (`load_physionet_pretrain`, `load_physionet_supcon`), since PhysioNet is used purely as an
  upstream pretraining source for the BCI-IV 2a zero-shot experiments.
- If one of these encoders were used to zero-shot-evaluate a held-out PhysioNet subject (as the
  reviewer's Critical Issue B implicitly invites), that held-out subject's own unlabeled EEG
  would already have shaped the encoder's weights during pretraining (self-supervised, so no
  labels leak, but representation-level information about that exact subject's signal
  characteristics does). This violates the review's Protocol A definition verbatim: "the held-out
  subject must not contribute data to training... or model selection."
- `final_results_table.csv` row 27 shows the original authors caught this independently: a planned
  3-way ensemble step (FgMDM + MEKT + SPDNet) on the 109-subject PhysioNet internal LOSO was
  "skipped per protocol (and would have been invalid anyway: SPDNet/RT-v2 were pretrained on all
  109 PhysionetMI subjects, overlapping with this eval set)". This confirms the gap in Critical
  Issue B is a deliberate, principled omission, not an oversight -- but it was never written up as
  a limitation in the manuscript itself, nor was any fix attempted. That is the actual remaining
  gap this revision closes.

### Fix implemented

Reversed the pretraining direction for the 5 Family 1/2 variants with existing BCI-IV 2a numbers
(MAE-only, MAE+GRL, SupCon, SPDNet, Riemannian Transformer v2): pretrain on BCI-IV 2a's 9 subjects
instead (`paper/data.py::load_bnci_pretrain` / `load_bnci_pretrain_labeled`, new), which has zero
subject overlap with PhysioNet by construction (two disjoint public datasets, no shared
participants). The resulting encoders are then zero-shot LOSO-evaluated across PhysioNet's full
109-subject population (`paper/eval_physionet_loso_deep.py`, new) with no leakage: each held-out
PhysioNet subject never contributed to the encoder's pretraining (which only ever saw BCI-IV 2a)
nor to its own linear/MLP probe (fit on the other PhysioNet subjects only, same as every other
zero-shot eval in this project).

Caveat for the manuscript: this is a **population swap**, not a literal increase in LOSO fold
count for the *same* pretraining corpus. BCI-IV 2a (9 subjects) is a much smaller and
differently-collected pretraining corpus than PhysioNet (109 subjects), so these new
BCI-pretrained encoders are not directly comparable in pretraining-data scale to the original
PhysioNet-pretrained ones used for the BCI-IV 2a headline numbers. This asymmetry must be stated
explicitly wherever these new numbers are reported, not elided.

## Checks that passed cleanly

- No test-set labels are used for early stopping during deep-learning pretraining: validation
  splits in `train_pretrain.py`/`train_supcon.py`/`train_spdnet.py`/`train_riemannian_transformer_v2.py`
  are carved from the *pretraining* source population (subject-level splits), never from the
  zero-shot eval target population.
- `paper/eval_physionet_comprehensive.py` (this revision's new classical-method PhysioNet
  comparison) reuses `load_physionet_zeroshot`, which already existed in the codebase and already
  implements EA per-subject + LOSO correctly for PhysioNet-internal evaluation (it is exactly the
  function used for the two pre-existing PhysioNet-internal results, rows 55/56/79). No new
  leakage risk introduced; this script only adds method variety at the same protocol.
