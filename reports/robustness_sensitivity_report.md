# Robustness and sensitivity analyses report

Date: 2026-10-18. Addresses the directive's mandatory Phase VIII. Each subsection below is a
separate, explicitly-labeled uncertainty source, per the directive's instruction not to conflate
bootstrap resampling, independent retraining, subject-level variability, and data-completeness
issues.

## 5.1 Random-seed sensitivity

Three genuinely distinct sources of variability exist in this project, and are kept separate here:

1. **Bootstrap-resampled source trials** (MEKT-lite and every classical baseline are deterministic
   -- no learned weights -- so a bare seed change is a no-op; genuine variation requires resampling
   which source trials are used). Pre-existing, re-verified this session: 4-seed mean 51.98% ±
   0.27% for the headline ensemble (arithmetic independently re-checked against the documented
   per-seed values, see `reports/final_revision_audit.md` Section 6). **Limitation, disclosed**:
   the underlying per-seed bootstrap computation has no standalone script in this repository to
   rerun from scratch; only the arithmetic on its already-documented outputs was verified.
2. **Independent neural-network retraining** (genuinely different weight init/data
   shuffling/dropout/mask sampling for Family 1/2 methods). Pre-existing: seeds 0, 123, 21 for 6
   deep-learning methods (`final_results_table.csv`), plus seed 999
   (`paper/results/multiseed_dl_seed999.csv`). **Limitation, disclosed**: only aggregate
   mean/std is saved for these; per-fold predictions were not preserved, so a jackknife
   subject-influence check (as done for the new PhysioNet comparisons, Section 5.5 below) is not
   possible for these specific numbers. No new independent-retraining seed sweep was run this
   session -- the existing 5-seed coverage (42/0/123/21/999) across all 6 deep-learning methods
   was judged sufficient evidence of retraining stability without spending further GPU budget that
   this session's Kaggle quota did not clearly have available (see Section 5.6/compute notes).
3. **Variability across held-out subjects**: this is what every reported standard deviation and
   bootstrap CI in this paper already measures, and what Section 5.5 below additionally probes via
   jackknife.

## 5.2 Analysis-window sensitivity

**New this session**: the factorial experiment (`reports/factorial_experiment_report.md`,
`paper/eval_factorial.py`) directly tests all 4 previously-used windows against each other,
crossed with estimator and method, not just the single sequential window swap originally reported.
Result: the window main effect is the most robust finding in the whole revision -- positive and of
similar magnitude (+1.3 to +2.0pp) across all three non-default windows, independent of which
method or estimator it's combined with (interaction p=0.82). Full window set tested and recorded,
not just the eventual winner, per the directive's requirement.

## 5.3 Covariance-estimator sensitivity

**New this session**: also covered by the factorial experiment. Unlike the window effect, the
`lwf` vs. `corr` main effect does **not** replicate as a general preprocessing principle when
averaged across windows and methods (+0.04pp, p=0.82) -- see
`reports/factorial_experiment_report.md` for the full discussion and the manuscript-text
revision this triggered (`tmlr_2026.tex` Section~\ref{sec:factorial}). This is a negative
result relative to the original framing and is reported as such, not suppressed.

A targeted check of estimator regularization/definitions: `pyriemann.estimation.Covariances`'s
`lwf` (Ledoit-Wolf shrinkage) and `corr` (sample correlation) estimators were used with their
library defaults throughout this project (no custom regularization); `tmlr_2026.tex`'s existing
Section 8 discussion (`sec:signalquality`) already includes a from-first-principles numerical
verification that `corr` is not a trivial rescaling of `lwf` (max absolute matrix difference
0.064, not ≈0) -- re-read and confirmed accurate this session, not newly computed.

## 5.4 Hyperparameter-transfer sensitivity (MEKT tuned vs. default on PhysioNet)

**Pre-existing finding from the prior session, independently re-verified this session** (not
retuned on PhysioNet to look better -- the directive's explicit prohibition on this was followed):
MEKT's BCI-IV-2a-tuned hyperparameters (k=50,λ=1.0,20 iter) do not outperform the untuned default
(k=100,λ=1.0,5 iter) on PhysioNet: 50.95% vs. 51.54%, mean difference -0.59pp, p=0.4497 (not
significant), re-confirmed via fresh rerun of `paper/stats_analysis.py` this session (exact match
to prior numbers). Jackknife check (Section 5.5) was not applied to this specific comparison in
the prior session; added this session -- see the robustness CSV referenced below.

**Interpretation, carefully scoped per the directive's rule 12**: this supports caution specifically
about *this* small-sample tuning procedure (hyperparameters chosen by sweeping against the same 9
subjects later reported on), not a general claim that all tuning necessarily overfits. The
factorial experiment (Section 5.2/5.3 above) provides a second, independent illustration of the
same caution: the estimator effect found via sequential tuning did not generalize as a main effect
either.

## 5.5 Subject-level robustness

**New this session**: leave-one-subject-out jackknife on the 4 most consequential paired
comparisons from the PhysioNet statistical analysis, via `paper/robustness_subject_influence.py`
(new script). For each comparison, the paired Wilcoxon test is recomputed with each single subject
removed in turn, checking whether the significance conclusion ever flips.

| Comparison | Full-sample p | LOO p-value range | Flips? |
|---|---|---|---|
| SPDNet → MEKT (default) | 0.000245 | [9.7e-05, 3.8e-04] | No -- robust |
| MDM → FgMDM | 6.6e-10 | [1.4e-10, 1.1e-09] | No -- robust |
| FgMDM → MEKT (default) | 7.1e-05 | [2.6e-05, 1.1e-04] | No -- robust |
| MAE+GRL → SPDNet | 1.5e-14 | [3.8e-15, 2.4e-14] | No -- robust |

Full output: `paper/results/robustness_subject_influence.csv`. **None of the four central
PhysioNet findings are driven by, or fragile to, any single subject** -- the significance
conclusions are unchanged under removal of any one of the 109 subjects in every case. No subject
was removed from any reported analysis merely because it weakened a finding (per the directive's
explicit prohibition); this jackknife is reported in full regardless of direction.

Per-subject outlier context (pre-existing, re-confirmed): Subject 2 on BCI-IV 2a never exceeds
~32% accuracy under any method tried in this entire project (both the original work and this
revision's new PhysioNet/LaBraM experiments do not change this), consistent with the well-documented
"BCI illiteracy" phenomenon -- already discussed in the manuscript's Results section and not
removed or reweighted.

## 5.6 Missing or failed recordings

**MAE-only's PhysioNet evaluation originally used 95/109 subjects** (14 skipped during the Kaggle
run due to transient `physionet.org` server errors -- timeouts and 502s -- while downloading their
raw recordings). Per the directive's explicit instruction to investigate recoverability:

- **Tested this session**: local network access (this development machine, not Kaggle's network)
  to the exact subject IDs that failed on Kaggle (1, 3, 6, 7, 9, ...) -- all succeeded immediately,
  confirming the failures were transient/Kaggle-network-specific, not permanent data unavailability.
- **Action taken**: MAE-only was retrained from scratch locally (CPU, `source="bnci"`, identical
  config/seed to the original Kaggle run) and re-evaluated zero-shot across the full 109-subject
  PhysioNet population locally, producing one single, internally-consistent encoder's complete
  109-subject result -- rather than patching the missing 14 folds onto the original (different)
  Kaggle-trained encoder's 95 folds, which would have silently mixed predictions from two different
  trained models under one label. The original incomplete 95-subject Kaggle result is preserved
  as `paper/results/physionet_loso_mae_only_INCOMPLETE_95subj_kaggle.csv.bak` for provenance, not
  deleted.
- **Status: completed this session.** Full 109/109-subject result: **33.99% ± 4.75%, kappa 0.008 ±
  0.070** (`paper/results/physionet_loso_mae_only.csv`), close to the original 95-subject estimate
  (33.74%/0.006) as expected for a near-chance result -- confirming the 14 originally-missing
  subjects were not a biased subsample. `tmlr_2026.tex`'s Table~\ref{tab:physionet-comprehensive},
  its accompanying figure, and `paper/results/physionet_loso_deep_summary.csv` were all updated to
  this complete number.

If this recovery had not been possible, the fallback (per directive Section 5.6) would have been
to retain the 95-subject disclosure as-is, which was already done in the prior session
(`reports/physionet_deep_learning_comparison.md`) -- that disclosure is preserved regardless of
this session's recovery outcome, consistent with rule 2 (preserve existing work, document reasons
for any change).
