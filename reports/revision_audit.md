# Revision audit: TMLR review response (Paper #11562)

Date: 2026-10-13. Scope: reconstruct the actual end-to-end pipeline behind every headline claim
in `tmlr_2026.tex`, before making any methodological changes in response to reviewer ajnW.

## 1. What this repository actually is

A real, previously-executed research project, not a stub. Evidence:

- `data/` contains real cached data: all 9 BCI-IV 2a subjects as both raw `.gdf` (`data/BCICIV_2a_gdf/`)
  and preprocessed `.npy` (`data/bnci2014_001/`), and all 109 PhysioNet subjects preprocessed
  (`data/PhysioNet/`, 4.4 GB).
- `paper/checkpoints/` contains 20+ real `.pt` files (~70-75 MB each) for every deep-learning
  variant claimed in the manuscript (MAE-only, MAE+GRL, MAE+CLUB+SCIN+VIB, SPDNet, Riemannian
  Transformer v1/v2, SupCon), plus a `checkpoints_seed999/` set for the multi-seed robustness check.
- `paper/results/` contains 37 CSVs of real per-experiment output (not just summary tables).
- `paper/results/final_results_table.csv` (80 rows) is a genuine lab notebook: every row documents
  one experiment actually run, including ~15 explicitly negative results, with per-row provenance
  notes (hyperparameters, what was tried, why it didn't work). This is the single most informative
  file in the repo for reconstructing protocol history and is treated as authoritative below.
- `kaggle_kernel/` contains the actual Kaggle kernel scripts, metadata, and downloaded output logs
  used to run the GPU-bound and large-population experiments (this project's pretraining and the
  109-subject PhysioNet replication both ran there, not locally).

## 2. Reconstructed pipeline per headline claim

| Claim (manuscript) | Script | Verified how |
|---|---|---|
| EA+FgMDM baseline, 47.45% | `paper/eval_riemannian_advanced.py` | Reproduced locally this session (see `current_results_reproduction.md`) |
| MEKT-lite default, 48.48% | `paper/eval_mekt.py` | Matches `final_results_table.csv` row 21 |
| 8-member MEKT ensemble (headline), 52.39% | `paper/eval_mekt_ensemble.py` | Config provenance documented in script docstring + `paper/results/mekt_ensemble_verified_reference.csv`; README already flags the separate 52.53% confidence-weighted variant as **not reproducible** (original implementation lost) |
| PhysioNet replication, 51.95% | historical Kaggle run, `kaggle_kernel/physionet_full_pipeline_push/run_kernel.py` | Script present and inspected; logic matches `eval_mekt_ensemble.py` with PhysioNet substituted; output `paper/results/physionet_full_pipeline_109.csv` has 109 real per-fold rows |
| Statistical analysis (paired bootstrap, Wilcoxon, rank-biserial) | `paper/writeup/generate_manuscript_figures.py` | Already present; **this already matches much of what review Critical Issue C asks for** — the manuscript (Section "statanalysis", `tmlr_2026.tex` lines ~370-396) already reports that no individual preprocessing step reaches p<0.05 at n=9, only the cumulative FgMDM→ensemble comparison does. The reviewer's concern is partially already addressed in-text; see `current_results_reproduction.md` for what's genuinely missing. |

## 3. What the review's Critical Issues A/B actually point at

Checked directly: of the ~15 classical/Riemannian method variants compared on BCI-IV 2a
(`final_results_table.csv` rows 2-44), only **two** (`EA+FgMDM` and `MEKT-lite default`, rows
55-56) plus the final winning ensemble (row 79) had been replicated on PhysioNet before this
revision. No deep-learning family member (MAE/SupCon/SPDNet/Riemannian Transformer) had ever been
evaluated zero-shot *within* PhysioNet. This is a real, confirmed gap — not an exaggeration by the
reviewer.

Critically, row 27 of `final_results_table.csv` shows the **original authors already identified
why** the deep-learning families can't simply be added to that comparison: those encoders were
pretrained on all 109 PhysioNet subjects (`config.PRETRAIN_SUBJECTS = list(range(1, 110))`), so
evaluating them zero-shot *within* PhysioNet would leak each held-out subject's own (unlabeled)
data into its own encoder's pretraining corpus. The 3-way ensemble step there was explicitly
skipped "per protocol" for this reason. See `data_leakage_audit.md` for the full analysis and the
fix implemented in this revision (reversed-direction pretraining).

## 4. Actions taken this revision (see individual experiment logs for full detail)

1. `paper/eval_physionet_comprehensive.py` (new): fills the classical/Riemannian family gap on
   PhysioNet — CSP+LDA, CSP+SVM, FBCSP+SVM, MDM, FgMDM, RPA, MEKT-lite (default + tuned) — all
   full 109-subject strict LOSO, CPU, run locally. See `paper/results/physionet_comprehensive_*.csv`.
2. `paper/data.py::load_bnci_pretrain` / `load_bnci_pretrain_labeled` (new), plus additive
   `source="bnci"` parameters on `train_pretrain.py`, `train_supcon.py`, `train_spdnet.py`,
   `train_riemannian_transformer_v2.py`: pretrains these 5 deep-learning variants on BCI-IV 2a
   instead of PhysioNet, so they can be validly zero-shot evaluated *within* PhysioNet.
   `paper/eval_physionet_loso_deep.py` + `paper/run_physionet_deep_sweep.py` (new): the
   corresponding PhysioNet-wide LOSO eval, pushed to Kaggle (GPU) as kernel
   `sweety611/eeg-physionet-deep-sweep`. All new checkpoints/logs are suffixed `_srcbnci` so the
   original PhysioNet-pretrained checkpoints (used for every existing manuscript number) are
   untouched.
3. Every new script was smoke-tested locally (1-epoch, tiny-subject-count) before being run at
   full scale or pushed to Kaggle, to catch integration bugs without spending GPU quota or CPU
   hours on a broken pipeline.

## 5. Explicitly not attempted / scoped down

- **Full 109-fold strict pretrain-per-fold LOSO** for the deep-learning families (pretrain
  excluding literally one subject at a time, 109 separate pretraining runs per architecture) was
  not attempted — compute-prohibitive even on a single Kaggle GPU account. The reversed-direction
  fix (pretrain once on BCI-IV 2a's 9 subjects, which never overlaps with PhysioNet) is used
  instead; this is methodologically clean (zero leakage) but is a different population-swap
  design, not a literal LOSO-on-the-same-109-subject-pretraining-corpus design. This is disclosed
  as a protocol choice, not hidden.
- **EEG foundation-model comparison**: environment has no local GPU; attempted as a lightweight,
  CPU/Kaggle-feasible linear-probe-only comparison (no fine-tuning) — see separate report once
  complete, or the manuscript limitations section if infeasible.
