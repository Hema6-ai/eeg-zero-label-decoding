# Zero-Label Cross-Subject Motor Imagery Decoding

Code and results for **"What Generalizes in Zero-Label Cross-Subject Motor Imagery Decoding? A
Controlled Study of Alignment, Preprocessing and Architecture"**, Anonymous Authors, under
double-blind review at *Transactions on Machine Learning Research (TMLR)*.

*This repository is anonymized for double-blind peer review. Author names and contact information
will be restored after the review process concludes.*

## Research problem

Motor-imagery brain-computer interfaces (BCIs) typically require subject-specific calibration
before a decoder is usable. This project studies **zero-label cross-subject** decoding: no
labeled target-subject data is used to fit any model. We compare four method families under one
shared zero-label evaluation framework, across three independent populations, and find that
Euclidean Alignment (domain adaptation) is the lever that generalizes most consistently, not
preprocessing choices or architectural complexity in general.

## Terminology

- **Zero-label**: no *labeled* target-subject data is used to fit any model.
- **Transductive**: a method (Euclidean/Riemannian Alignment, MEKT's pseudo-labeling) additionally
  uses the held-out subject's own *unlabeled* trials, accessed collectively, for alignment or
  adaptation.
- **Zero-shot**: reserved for the strictly inductive setting with no batch access to the target
  subject at all (the LaBraM comparison) and for quoting other papers' own terminology.

## Datasets

All three are public and loaded through [MOABB](https://github.com/NeuroTechX/moabb), which
downloads and caches them automatically on first use:

- **BCI Competition IV Dataset 2a** (`BNCI2014_001`, 9 subjects, 4-class motor imagery) — the
  primary evaluation dataset.
- **PhysioNet EEG Motor Movement/Imagery Database** (up to 109 subjects) — used both as a
  pretraining source for the self-supervised methods and as a second, larger-population dataset
  for the full method-family comparison and factorial replication.
- **Cho2017** (52 subjects, 2-class, different hardware/protocol) — a third, independent
  population.

## Preprocessing / common protocol

PhysioNet (64 ch, 160 Hz), BCI-IV 2a (22 ch, 250 Hz), and Cho2017 are loaded through the same
MOABB `MotorImagery` paradigm, restricted to BCI-IV 2a's 22-channel montage and resampled to a
shared 128 Hz, giving every method the same input contract regardless of source dataset
(`paper/data.py::get_paradigm`). All trials are z-scored per-channel, per-trial. Unless a method
explicitly sweeps it, the analysis window is `t ∈ [0, 3]`s post-cue. Every zero-label evaluation
uses leave-one-subject-out (LOSO): for each held-out subject, all models/classifiers are fit only
on the source subjects, with seed 42 throughout unless a seed sweep is explicit. Methods using
Euclidean Alignment or MEKT's pseudo-labeling additionally access the held-out subject's
*unlabeled* trials collectively (never its true labels) — see the manuscript's Section 3.1 for the
precise zero-label/transductive/zero-shot distinction.

## Methods (four families)

1. **Self-supervised representation learning on raw EEG** (`paper/models/mae*.py`,
   `paper/train_pretrain.py`, `paper/train_club_scin.py`, `paper/train_supcon.py`) — masked
   autoencoding, plus subject-disentanglement variants (gradient-reversal, and a CLUB+SCIN+VIB
   combination).
2. **Geometric deep learning on covariance features** (`paper/models/riemannian_transformer.py`,
   `paper/models/spdnet.py`) — a Riemannian tangent-space transformer and SPDNet.
3. **Classical Riemannian geometry and domain adaptation** (`paper/eval_csp_lda_svm.py`,
   `paper/eval_riemannian_advanced.py`, `paper/eval_mekt.py`, `paper/eval_rpa.py`) — CSP, MDM,
   FgMDM, Manifold Embedded Knowledge Transfer (MEKT-lite), Riemannian Procrustes Analysis.
4. **Supervised deep architectures, trained directly on each dataset's own subjects**
   (`paper/run_dl_loso_sweep.py`, `paper/eval_e1_eegnet_physionet_full.py`) — EEGNetv4,
   ShallowFBCSPNet, ATCNet (braindecode), trained from scratch, source-subjects-only, not via
   population-swapped pretraining.

## Main experiments and results

| Result | Accuracy | Script | Status |
|---|---|---|---|
| Strongest classical baseline (EA+FgMDM), BCI-IV 2a | 47.45% | `paper/eval_riemannian_advanced.py` | reproducible |
| MEKT-lite, default window/estimator | 48.48% | `paper/eval_mekt.py` | reproducible |
| 8-member MEKT ensemble (4 windows × 2 estimators), BCI-IV 2a, **in-sample selection** | 52.39% ± 14.92% (κ=0.365) | `paper/eval_mekt_ensemble.py` | reproducible, verified below |
| **Fully nested re-search (primary generalization estimate)**, BCI-IV 2a | **49.94% ± 14.41%** (κ=0.333) | `paper/eval_e4_nested_selection.py` | reproducible |
| 8-member ensemble + confidence-weighted fusion | 52.53% ± 14.95% | `paper/eval_e7_confidence_weighted_fusion.py` | reproducible |
| Full method-family comparison, PhysioNet (109 subjects) | see manuscript Table 6 | `paper/eval_physionet_comprehensive.py`, `paper/run_physionet_deep_sweep.py` (Kaggle GPU) | reproducible given GPU for the deep-learning rows |
| Time-window/covariance-estimator factorial, replicated on PhysioNet at n=30 and n=109 | see manuscript Table 5 | `paper/eval_e3_physionet_factorial.py`, `paper/stats_e3_physionet_factorial.py` | reproducible, resumable per-cell |
| Supervised architectures (EEGNetv4, ShallowFBCSPNet, ATCNet) directly on BCI-IV 2a and PhysioNet30; EEGNetv4 additionally on full PhysioNet (n=109) | see manuscript Table 14 | `paper/run_dl_loso_sweep.py`, `paper/eval_e1_eegnet_physionet_full.py` (Kaggle GPU) | reproducible given GPU |
| Third population, Cho2017 (52 subjects): classical methods, ensemble, EEGNetv4 | see manuscript Table 7 | `paper/eval_e5_third_dataset.py`, `paper/eval_e5_best_cnn.py`, `paper/eval_e5_best_cnn_noea.py` | reproducible |
| LaBraM foundation model: frozen zero-shot, positive control, source-only fine-tuning (2 regimes) | all at chance | `paper/eval_foundation_model_labram.py`, `paper/eval_labram_source_finetune.py` | reproducible, CPU-feasible for the frozen probe; GPU for fine-tuning |
| CBraMod attempt (documented upstream loading bug); BIOT exclusion (channel-scheme incompatibility) | not available | `paper/eval_foundation_model_cbramod.py` | fails reproducibly with the exact traceback described in the manuscript's Limitations; not a result to reproduce |
| Euclidean Alignment ablation (EA on/off), PhysioNet + Cho2017, FgMDM/MEKT | see manuscript Section 13 | `paper/eval_e9_alignment_ablation.py`, `paper/stats_e9_alignment_ablation.py` | reproducible |
| Statistical analysis (all populations, Holm-Bonferroni corrected) | see `paper/results/stats_*.csv`, `paper/results/statistical_tests_*.csv` | `paper/stats_analysis.py` and the per-experiment `stats_*.py` scripts | reproducible, deterministic |
| Independent claim verification (every aggregate number vs. raw per-subject CSVs) | 16/16 verified | `paper/verify_all_claims.py` | reproducible, deterministic |
| Machine-readable claim-to-evidence audit (every manuscript number mapped to script+data) | 98/98 verified | `paper/build_claim_evidence_audit.py`, `paper/build_claim_evidence_audit_r2.py` → `reports/claim_evidence_audit.csv` | reproducible, deterministic |

**On the two headline accuracy figures:** 52.39% is the in-sample, optimistically selected
8-member ensemble result (the window/estimator/ensemble configuration was chosen by comparing
accuracy on the same 9 subjects used to report it). 49.94% is this paper's primary generalization
estimate, from a fully nested re-search that rebuilds the entire window/estimator/method
selection from scratch per outer fold using only source subjects. Both numbers are reported
throughout the manuscript, each explicitly labeled; see Section 5.1 (Nested cross-validation) for
the full derivation, including a second, restricted-candidate nested check (selecting only among
the 8 individual configurations plus the pre-built ensemble) that reproduces 52.39% and rules out
leakage in that narrower sense without being the primary estimate.

**Provenance of `paper/eval_mekt_ensemble.py`:** this script's configuration (4 analysis windows,
2 covariance estimators, MEKT hyperparameters) was recovered from a historical deployment run and
independently verified against that run's saved per-fold output
(`paper/results/mekt_ensemble_verified_reference.csv`) before being added here — see the script's
docstring for details. Nothing in it was tuned or reverse-engineered to hit the reported number.

**PhysioNet download reliability:** PhysioNet downloads go through MOABB/`pooch`; `physionet.org`
occasionally times out on individual subjects. Affected subjects were retried individually and
recovered; every reported PhysioNet result uses the full declared subject count. If downloads are
flaky in your environment, mirror the dataset (ODC-By licensed) locally first rather than retrying
indefinitely — see `kaggle_kernel/spdnet_corrcov_push/run_kernel.py` for a worked example of
pre-populating MOABB's cache from a local mirror.

## Repository structure

```
paper/
├── models/                # Architecture definitions (MAE, GRL, CLUB, SCIN, VIB, SPDNet,
│                           # Riemannian Transformer; EEGNetv4/ShallowFBCSPNet/ATCNet via braindecode)
├── data.py, align.py, augment*.py, config.py, utils.py   # Shared data loading / preprocessing
├── train_*.py              # Training entry points for each method
├── eval_*.py                # Zero-label LOSO evaluation entry points for each method/experiment
├── eval_mekt_ensemble.py    # In-sample 8-member ensemble (52.39%) -- see table above
├── eval_e4_nested_selection.py  # Primary generalization estimate (49.94%) -- see table above
├── run_*.py                 # Kaggle/experiment orchestration scripts
├── stats_*.py                # Paired Wilcoxon / Holm-Bonferroni statistical analysis per experiment
├── verify_all_claims.py, build_claim_evidence_audit*.py   # Independent numerical verification
├── representational_analysis.py, diagnose_in_distribution_probe.py, visualize_latent.py
│                           # Representation-quality diagnostics (t-SNE, linear probing, feature stats)
├── results/                 # Per-experiment CSV outputs (accuracy, std, kappa per method/fold)
└── writeup/
    ├── references.bib        # Bibliography (all DOIs verified live against Crossref/arXiv)
    ├── figures/               # All manuscript figures
    └── generate_manuscript_figures.py, make_pipeline_overview.py   # Regenerate figures from results/
```

Trained model checkpoints and the raw datasets are not included in this repository (checkpoints
are large binaries re-derivable via the `train_*.py` scripts; the datasets are public and
MOABB-downloaded — see **Datasets** above).

## Setup

```bash
pip install torch numpy scipy scikit-learn moabb mne pyriemann matplotlib seaborn braindecode
```

(Exact pinned versions were not preserved from the original run; any reasonably recent release of
each package should reproduce the reported results within the reported variance. Pretraining and
the supervised-architecture training in Family 4 ran on a single Kaggle GPU; all classical and
domain-adaptation methods, including both headline ensemble numbers, run on CPU only.)

## Reproducing results

```bash
python paper/eval_riemannian_advanced.py   # classical baselines, incl. EA+FgMDM (47.45%)
python paper/eval_mekt.py                  # MEKT-lite, single default window/estimator (48.48%)
python paper/eval_mekt_ensemble.py         # in-sample 8-member ensemble (52.39%, see table above)
python paper/eval_e4_nested_selection.py   # primary generalization estimate (49.94%, see table above)
python paper/eval_e7_confidence_weighted_fusion.py  # confidence-weighted refinement (52.53%)
```

Methods requiring pretraining (Families 1/2) are trained first via the corresponding `train_*.py`
/ `run_*.py` script, then evaluated zero-label with the encoder frozen via the matching `eval_*.py`.
Family 4's supervised architectures are trained end-to-end via `run_dl_loso_sweep.py`. Every run is
seeded at 42 unless a seed sweep is explicitly being performed.

### Script-to-table mapping

| Manuscript row | Script | Output |
|---|---|---|
| In-sample 8-member ensemble (52.39%, Table 1/2) | `eval_mekt_ensemble.py` | per-fold CSV under `paper/results/` |
| Primary nested generalization estimate (49.94%, Table 1) | `eval_e4_nested_selection.py` | per-fold CSV under `paper/results/` |
| Confidence-weighted refinement (52.53%, Section "negative-result sweep") | `eval_e7_confidence_weighted_fusion.py` | per-fold CSV under `paper/results/` |

Each script writes one per-fold CSV; every reported mean/std/kappa is independently recomputed
from those CSVs by `paper/verify_all_claims.py`, which can be re-run to confirm every number in
the manuscript against its source file.

## Citation

Citation details are withheld during anonymous review and will be added once the review process
concludes.

```bibtex
@article{anonymous2026zerolabel,
  title   = {What Generalizes in Zero-Label Cross-Subject Motor Imagery Decoding? A Controlled
             Study of Alignment, Preprocessing and Architecture},
  author  = {Anonymous},
  note    = {Under double-blind review at Transactions on Machine Learning Research (TMLR)},
  year    = {2026}
}
```

## License

MIT (code). See `LICENSE`.

## Contact

Withheld during double-blind review.
