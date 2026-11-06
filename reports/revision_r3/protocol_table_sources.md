# Protocol table sources (Table~\ref{tab:protocol}, Table~\ref{tab:protocol-full})

Every cell was verified against the script below, not from memory or manuscript prose alone.

| Row | Script(s) | Config constants checked |
|---|---|---|
| MAE-only, MAE+GRL | `paper/train_pretrain.py` (`run(variant=...)`) | `paper.config.PRETRAIN_SUBJECTS` (PhysioNet, 109), `PRETRAIN_EPOCHS`; unsupervised reconstruction loss, no class labels used in the objective |
| MAE+CLUB+SCIN+VIB | `paper/train_club_scin.py` | same pretraining population; CLUB/VIB/SCIN losses added, still unsupervised w.r.t. class label |
| MAE+EA | `paper/train_pretrain.py` with `align="ea"` | EA applied to pretraining data before the MAE objective |
| SupCon | `paper/train_supcon.py` | `load_physionet_supcon` — contrastive grouping uses PhysioNet's own class labels during pretraining (`"Supervised Contrastive (SupCon) pretraining on PhysioNet"`, confirmed in the script's own docstring) |
| Riemannian Transformer v1/v2 | `paper/train_riemannian_transformer.py`, `paper/train_riemannian_transformer_v2.py` | `load_physionet_supcon` + `align_per_subject(..., "ea")` + `CrossEntropyLoss` on PhysioNet's 3 shared classes — supervised, not self-supervised |
| SPDNet (original protocol) | `paper/train_spdnet.py` | `prepare_covariances(source="physionet"|"bnci")`, EA + Ledoit-Wolf covariance, `CrossEntropyLoss` — population-swap pretraining, frozen encoder used downstream |
| SPDNet (matched protocol) | `paper/eval_spdnet_matched_protocol.py` (new this revision) | Reuses `paper/run_dl_loso_sweep.py::load_window_subjects`/`set_all_seeds`/`WINDOWS`/`EA_OPTIONS`/`SEEDS` directly; covariances computed per fold via `pyriemann.estimation.Covariances(estimator="lwf")` after EA; `SPDNetModel` trained end-to-end with `paper.config.SPDNET_DIMS`/`SPDNET_LR`/`SPDNET_WEIGHT_DECAY`/`SPDNET_BATCH_SIZE`/`SPDNET_EPOCHS`, patience 20 (matching `run_dl_loso_sweep.py`'s `PATIENCE`) |
| EEGNetv4, ShallowFBCSPNet, ATCNet | `paper/run_dl_loso_sweep.py` | `build_model`, `train_fold`, `SEEDS = [42,0,123,21,7]`, `WINDOWS`, `EA_OPTIONS`, `MAX_EPOCHS=150`, `PATIENCE=20`, `BATCH_SIZE=64`, `LR=1e-3`, `WEIGHT_DECAY=1e-4` |
| EEGNetv4 (full PhysioNet) | `paper/eval_e1_eegnet_physionet_full.py` | `SEEDS=[42,0,123]`, `WINDOW=(0.0,3.0)`, EA on only, reuses `run_dl_loso_sweep.py`'s `train_fold`/`load_window_subjects` |
| LaBraM (frozen) | `paper/eval_foundation_model_labram.py` | `braindecode.models.Labram`, HuggingFace checkpoint `braindecode/Labram-Braindecode`, backbone frozen, linear probe fit per fold on 8 source subjects |
| LaBraM (fine-tuned, full/last2) | `paper/eval_labram_source_finetune.py` | `--mode full` / `--mode last2`; source-subjects-only fine-tuning, early stopping on one held-out source subject, target never touched during training or selection |
| CSP+LDA/SVM, FBCSP+SVM, MDM, FgMDM | `paper/eval_csp_lda_svm.py`, `paper/eval_riemannian_advanced.py` | EA applied per-subject (unsupervised); deterministic fit, no seed dependence |
| MEKT-lite | `paper/eval_mekt.py` | `k=50, lambda=1.0`, grid-searched against BCI-IV 2a's own 9-subject LOSO accuracy (Section~\ref{sec:signalquality}) — disclosed as not source-validated |
| RPA | `paper/eval_rpa.py` | EA + orthogonal Procrustes rotation on class-mean tangent vectors using iteratively pseudo-labeled target means |
| 8-member ensemble, in-sample | `paper/eval_mekt_ensemble.py` | Window/estimator/ensemble choice selected by comparing accuracy on the same 9 subjects reported |
| 8-member ensemble, nested (primary) | `paper/eval_e4_nested_selection.py` | Per-outer-fold inner-8-fold search over source subjects only, no reference to the in-sample candidate set |

## Not independently re-verified this pass

The Cho2017 source-subject counts for EEGNetv4/classical methods ("up to 51") and the exact
per-dataset subject availability after any download retries were taken from the existing
manuscript text and result CSVs (`paper/results/e5_*cho2017*.csv`) rather than re-read from the
Cho2017-specific training scripts line by line in this pass, since Cho2017 was not part of
Reviewer c1um's requested comparison (SPDNet matched protocol was not run on Cho2017, per the
task's own "if EEGNetv4 was run there with the same structure" condition — EEGNetv4 on Cho2017
uses a 5-seed per-subject average on the full 52-subject population, not a fixed per-fold source
count in the same sense as the other datasets, so the matched-protocol comparison was scoped to
BCI-IV 2a and PhysioNet only).
