# Compute inventory

Date: 2026-10-13.

## Local environment

- CPU: 12 logical cores, Windows 11.
- RAM: not GPU-constrained; sufficient for all classical/Riemannian LOSO experiments (confirmed:
  109-subject comprehensive classical sweep ran without memory issues).
- GPU: **none**. `torch==2.12.1+cpu`, `torch.cuda.is_available() == False`. No `nvidia-smi`.
- Disk: 168 GB free on the working volume at audit time.
- Data already cached locally: PhysioNet (4.4 GB, all 109 subjects as preprocessed `.npy`),
  BCI-IV 2a (872 MB, both raw `.gdf` and preprocessed `.npy`). No re-download needed for any
  CPU-side experiment.

## Implication for experiment design

Classical/Riemannian methods (CSP, MDM/FgMDM, RPA, MEKT) are CPU-tractable at full 109-subject
LOSO scale -- confirmed empirically: a 10-subject timing smoke test of 3 representative methods
ran in 10-20s each; the full 109-subject run of all 8 classical method variants was launched
locally (`paper/eval_physionet_comprehensive.py`) and completed its first method (EA+CSP(8)+LDA)
in ~17 minutes.

Deep-learning methods (MAE-family, SupCon, SPDNet, Riemannian Transformer) require GPU for
tractable runtime at the scale needed (pretraining + 109-fold zero-shot eval) -- the original
project's own pretraining already ran on GPU via Kaggle (`kaggle_kernel/run_kernel.py`'s torch
install comments reference Kaggle's P100/T4 GPU assignment), consistent with this being a hard
requirement, not a convenience.

## External compute used: Kaggle

- Kaggle API CLI (`kaggle==1.7.4.5`) was already installed and configured in this environment from
  prior project work, with several historical account credentials present
  (`~/.kaggle/kaggle_*_active.json` / `*_quota_exhausted.json` / `*_leaked_rotated.json` -- the
  latter two explicitly retired, not used).
- Per user instruction, this revision uses a new account (`sweety611`) for all new Kaggle work.
  Verified authenticated (empty `datasets --mine` / `kernels --mine` listings, consistent with a
  fresh account rather than an auth failure).
- Pushed: code dataset `sweety611/eeg-mae-paper-code` (11 MB, the `paper/` source tree, no
  checkpoints or raw data -- those are downloaded/regenerated inside the kernel, matching the
  project's existing convention in `kaggle_kernel/run_kernel.py` and
  `kaggle_kernel/physionet_full_pipeline_push/run_kernel.py`).
- Pushed: kernel `sweety611/eeg-physionet-deep-sweep` (GPU enabled, internet enabled) running
  `paper/run_physionet_deep_sweep.py` -- pretrains 5 deep-learning variants on BCI-IV 2a then
  zero-shot LOSO-evaluates each across PhysioNet's full population. Status and results tracked
  separately as the kernel completes (Kaggle sessions are asynchronous; this report will be
  updated with final runtime once it finishes).

## Explicitly out of scope given this compute budget

- A literal 109-fold-per-architecture pretrain-from-scratch LOSO for the deep-learning families
  (i.e., excluding exactly one PhysioNet subject's data per pretraining run) -- would require
  ~109x the pretraining compute per architecture on top of what already ran for the manuscript's
  existing numbers. Not attempted; see `data_leakage_audit.md` for the population-swap alternative
  used instead.
- Fine-tuning any EEG foundation model on this task -- a lightweight, no-fine-tuning (linear-probe
  only) comparison was attempted instead, scoped to what the available compute (Kaggle GPU, no
  guaranteed large quota on a fresh account) can support within a single or few kernel sessions.
