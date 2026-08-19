# Signal Quality, Not Model Complexity

Code and results for **"Signal Quality, Not Model Complexity: A Systematic Study of Zero-Shot
Cross-Subject Motor Imagery Decoding"**, Anonymous Authors, under double-blind review at
*Transactions on Machine Learning Research (TMLR)*.

*This repository is anonymized for double-blind peer review. Author names and contact information
will be restored after the review process concludes.*

## Research problem

Motor-imagery brain-computer interfaces (BCIs) typically require subject-specific calibration
before a decoder is usable. This project studies **zero-shot cross-subject** decoding: can a
decoder trained only on other subjects work on a new subject with no labeled calibration data
from that subject at all? We compare three increasingly different classes of method under one
fixed, strict evaluation protocol, and report a largely negative-result-driven finding: gains came
from signal-quality preprocessing choices, not from model complexity.

## Datasets

Both are public and loaded through [MOABB](https://github.com/NeuroTechX/moabb), which downloads
and caches them automatically on first use:

- **BCI Competition IV Dataset 2a** (`BNCI2014_001`, 9 subjects, 4-class motor imagery) — the
  primary evaluation dataset.
- **PhysioNet EEG Motor Movement/Imagery Database** (109 subjects) — used both as the pretraining
  source for the self-supervised methods (Family 1/2) and as a second, larger-population dataset
  to replicate the headline result.

## Preprocessing / common protocol

PhysioNet (64 ch, 160 Hz) and BCI-IV 2a (22 ch, 250 Hz) are loaded through the same MOABB
`MotorImagery` paradigm, restricted to BCI-IV 2a's 22-channel montage and resampled to a shared
128 Hz, giving every method the same input contract regardless of source dataset
(`paper/data.py::get_paradigm`). All trials are z-scored per-channel, per-trial. Unless a method
explicitly sweeps it, the analysis window is `t ∈ [0, 3]`s post-cue. Every zero-shot evaluation
uses leave-one-subject-out (LOSO): for each held-out subject, all models/classifiers are fit only
on the other 8 (or 108, for PhysioNet), with seed 42 throughout unless a seed sweep is explicit.

## Methods (three families)

1. **Self-supervised representation learning on raw EEG** (`paper/models/mae*.py`,
   `paper/train_pretrain.py`, `paper/train_club_scin.py`, `paper/train_supcon.py`) — masked
   autoencoding, plus subject-disentanglement variants (gradient-reversal, and a CLUB+SCIN+VIB
   combination).
2. **Geometric deep learning on covariance features** (`paper/models/riemannian_transformer.py`,
   `paper/models/spdnet.py`) — a Riemannian tangent-space transformer and SPDNet.
3. **Classical Riemannian geometry and domain adaptation** (`paper/eval_csp_lda_svm.py`,
   `paper/eval_riemannian_advanced.py`, `paper/eval_mekt.py`, `paper/eval_rpa.py`) — CSP, MDM,
   FgMDM, Manifold Embedded Knowledge Transfer (MEKT-lite), Riemannian Procrustes Analysis.

## Evaluation protocol

Strict zero-shot LOSO throughout: no labeled target-subject data is ever used to fit any model or
classifier. Methods using Euclidean Alignment (EA) or MEKT's transductive pseudo-labeling access
the target subject's *unlabeled* trials collectively (never their true labels) — see the
manuscript's terminology note on "zero-shot" vs. "transductive" for the precise distinction.
`paper/utils.py::classification_metrics` computes accuracy and Cohen's kappa; true target labels
are used only at that final scoring step, never during fitting or model/method selection.

## Main experiments and results

| Result | Accuracy | Script | Status |
|---|---|---|---|
| Strongest classical baseline (EA+FgMDM) | 47.45% | `paper/eval_riemannian_advanced.py` | reproducible |
| MEKT-lite, default window/estimator | 48.48% | `paper/eval_mekt.py` | reproducible |
| **8-member MEKT ensemble (4 windows × 2 estimators), BCI-IV 2a** | **52.39% ± 14.92%** (κ=0.365) | `paper/eval_mekt_ensemble.py` | **reproducible, verified below** |
| 8-member ensemble + confidence-weighted fusion | 52.53% ± 14.95% | — | **not reproducible from this repo — see note below** |
| Cross-dataset replication, PhysioNet (109 subjects) | 51.95% | same ensemble logic, PhysioNet source | verified against `paper/results/physionet_full_pipeline_109.csv` (109 real per-fold rows) |
| Every other method tried (raw-EEG SSL, geometric DL, negative-result sweep) | see manuscript Tables | `paper/eval_zero_shot*.py`, `paper/eval_stacked_ensemble*.py`, `paper/eval_method_selection.py` | reproducible |

**On the 52.39% / 52.53% distinction:** the manuscript reports both. 52.39% is the base
equal-weight 8-member ensemble; 52.53% is a small additional refinement (per-trial max-softmax
confidence weighting instead of equal weighting) reported in the manuscript's negative-result
sweep section. **Only 52.39% is reproducible from this repository.** The original implementation
of the 52.53% refinement was not recovered when this repository was assembled; the manuscript's
claim is not altered or hidden, but this repository does not claim to reproduce it. If you are a
reviewer and this matters for your assessment, please raise it in review.

**Provenance of `paper/eval_mekt_ensemble.py`:** this script's configuration (4 analysis windows,
2 covariance estimators, MEKT hyperparameters) was recovered from a historical deployment run and
independently verified against that run's saved per-fold output
(`paper/results/mekt_ensemble_verified_reference.csv`) before being added here — see the script's
docstring for details. Nothing in it was tuned or reverse-engineered to hit the reported number.

## Repository structure

```
paper/
├── models/                # Architecture definitions (MAE, GRL, CLUB, SCIN, VIB, SPDNet,
│                           # Riemannian Transformer, EEGNet)
├── data.py, align.py, augment*.py, config.py, utils.py   # Shared data loading / preprocessing
├── train_*.py              # Training entry points for each method
├── eval_*.py                # Zero-shot LOSO evaluation entry points for each method
├── eval_mekt_ensemble.py    # Headline 8-member ensemble (52.39%) -- see table above
├── run_*.py                 # Kaggle/experiment orchestration scripts
├── representational_analysis.py, diagnose_in_distribution_probe.py, visualize_latent.py
│                           # Representation-quality diagnostics (t-SNE, linear probing, feature stats)
├── results/                 # Per-experiment CSV outputs (accuracy, std, kappa per method/fold)
└── writeup/
    ├── references.bib        # Bibliography (all DOIs verified live against Crossref/arXiv)
    ├── figures/               # All manuscript figures
    └── generate_manuscript_figures.py   # Regenerates every figure from results/
```

Trained model checkpoints and the raw PhysioNet / BCI-IV 2a datasets are not included in this
repository (checkpoints are large binaries re-derivable via the `train_*.py` scripts; the datasets
are public and MOABB-downloaded — see **Datasets** above).

## Setup

```bash
pip install torch numpy scipy scikit-learn moabb mne pyriemann matplotlib seaborn
```

(Exact pinned versions were not preserved from the original run; any reasonably recent release of
each package should reproduce the reported results within the reported variance. Pretraining ran
on a single GPU; all classical and domain-adaptation methods, including the headline ensemble, run
on CPU only.)

## Reproducing results

```bash
python paper/eval_riemannian_advanced.py   # classical baselines, incl. EA+FgMDM (47.45%)
python paper/eval_mekt.py                  # MEKT-lite, single default window/estimator (48.48%)
python paper/eval_mekt_ensemble.py         # headline 8-member ensemble (52.39%, see table above)
python paper/eval_ensemble.py              # separate experiment: soft-vote of FgMDM + Riemannian Transformer v2 + SPDNet (41.44%, not the headline result)
```

Methods requiring pretraining (Family 1/2) are trained first via the corresponding `train_*.py` /
`run_*.py` script, then evaluated zero-shot with the encoder frozen via the matching `eval_*.py`.
Every run is seeded at 42 unless a seed sweep is explicitly being performed
(`run_multiseed_dl_sweep.py`).

## Citation

Citation details are withheld during anonymous review and will be added once the review process
concludes.

```bibtex
@article{anonymous2026signalquality,
  title   = {Signal Quality, Not Model Complexity: A Systematic Study of Zero-Shot
             Cross-Subject Motor Imagery Decoding},
  author  = {Anonymous},
  note    = {Under double-blind review at Transactions on Machine Learning Research (TMLR)},
  year    = {2026}
}
```

## License

MIT (code). See `LICENSE`.

## Contact

Withheld during double-blind review.
