# Final reproducibility and anonymization audit

Date: 2026-10-28. Produced for the master audit directive's Phase 7.

## 1. Environment and dependencies

No `requirements.txt`/`environment.yml`/pinned versions exist in this repository. `README.md`
already disclosed this honestly before this session ("Exact pinned versions were not preserved
from the original run; any reasonably recent release of each package should reproduce the
reported results within the reported variance"). Not fixed by pinning retroactively (doing so
without having tested against those exact pins would itself be an unverified claim); disclosure
preserved as-is.

Core dependencies (from imports across `paper/*.py`): `torch`, `numpy`, `scipy`, `scikit-learn`,
`moabb`, `mne`, `pyriemann`, `einops`, `braindecode` (LaBraM only), `huggingface_hub` (LaBraM
only), `matplotlib`, `seaborn`.

## 2. Dataset source and license

- BCI Competition IV Dataset 2a: via MOABB's `BNCI2014_001`. Original dataset license/terms per
  the BCI Competition IV organizers (cited, `tangermann2012bci4`).
- PhysioNet EEG Motor Movement/Imagery Database: via MOABB's `PhysionetMI`. ODC-By licensed per
  PhysioNet's own terms (cited, `schalk2004physionet`, `goldberger2000physiobank`).
- This session's SPDNet `corr`-estimator run additionally used a third-party Kaggle mirror of the
  PhysioNet dataset (`gamalasran/physionet-eeg-motor-movement-imagery`, ODC-By) to avoid
  `physionet.org`'s observed unreliability -- see `reports/final_scientific_audit.md` Section 6
  for the integrity check and its one disclosed residual risk.

## 3. Commands to reproduce the main checks (consolidated; also added to `README.md` this session)

```bash
# Core BCI-IV 2a results
python paper/eval_riemannian_advanced.py      # classical baselines, EA+FgMDM (47.45%)
python paper/eval_mekt.py                     # MEKT-lite default (48.48%)
python paper/eval_mekt_ensemble.py            # headline 8-member ensemble (52.39%)

# PhysioNet comprehensive comparison (109 subjects)
python -m paper.eval_physionet_comprehensive  # 8 classical/domain-adaptation methods
python paper/run_physionet_deep_sweep.py      # 5 deep-learning architectures (GPU)
python -m paper.eval_physionet_loso_deep

# Foundation-model comparison
python -m paper.eval_foundation_model_labram  # LaBraM zero-shot, BCI-IV 2a

# Factorial preprocessing-vs-architecture experiment
python -m paper.eval_factorial                # 16-cell grid (CPU)
python paper/train_spdnet.py --cov-estimator corr   # SPDNet deep-learning-arm check (GPU)
python -m paper.eval_zero_shot_spdnet --cov-estimator corr
python -m paper.stats_factorial

# Statistics, robustness, and verification
python -m paper.stats_analysis
python -m paper.robustness_subject_influence
python -m paper.verify_all_claims             # every aggregate vs. raw per-subject CSVs
python -m paper.build_claim_evidence_audit    # -> reports/claim_evidence_audit.csv
```

All classical/domain-adaptation methods and the statistics/verification scripts are CPU-feasible
and deterministic (seed 42). Deep-learning pretraining requires a GPU; results were produced on a
single Kaggle Tesla T4 (free tier).

## 4. Claim-to-artifact mapping

Already machine-readable: `reports/claim_evidence_audit.csv` (46 rows, regenerated this session,
46/46 verified), produced by `paper/build_claim_evidence_audit.py`. This is the authoritative
mapping; this report does not duplicate it.

## 5. Expected output filenames

Documented inline in each script's docstring (e.g. `paper/eval_factorial.py` writes
`paper/results/factorial_<method>_t<tmin>_<tmax>_<estimator>.csv` per cell plus
`factorial_summary.csv`); not re-duplicated here to avoid a second, driftable source of truth.

## 6. Known limitations and non-rerunnable experiments

- The 4-seed ensemble bootstrap-resampling computation (51.98% ± 0.27%) has no standalone rerun
  script in this repository -- pre-dates this revision, disclosed in `reports/final_revision_audit.md`
  and `reports/claim_evidence_audit.csv`. Only its arithmetic has been independently checked.
- The confidence-weighted ensemble refinement (52.53%) has no recovered implementation --
  consistently disclosed in three places in the manuscript as not independently reproducible.
- The deep-learning methods' independent-retraining seed sweep (seeds 0/123/21/999, alongside 42)
  was not extended further; judged adequate given the existing 5-seed coverage.

## 7. Anonymity audit

| Check | Finding | Action |
|---|---|---|
| `.tex` source author block | Real names (gitignored, correct, deliberate) | None needed |
| Compiled PDF rendered content | Shows "Anonymous authors / Paper under double-blind review" (verified via `pdftotext` this session) -- `tmlr.sty`'s default mode hides the author block automatically | None needed |
| Compiled PDF metadata | No `/Author` field or other identifying metadata (checked via `pypdf` this session) | None needed |
| Compiled PDF git-tracking | **Was not gitignored** (only the `.tex` source was) -- an inconsistency, even though the current render is already anonymous | **Fixed this session**: added `/tmlr_2026.pdf` to `.gitignore` as defense-in-depth against a future compile in `[preprint]`/`[accepted]` mode being committed by mistake |
| Absolute personal file paths in tracked files | None found (`grep` for `C:\Users\hema0`, `hema0` across `paper/`, `reports/`, `README.md`, `tmlr_2026.tex`) | None needed |
| Institution/personal email in tracked non-`.tex` files | None found | None needed |
| `kaggle_kernel/` (contains the Kaggle account handle `sweety611` in run scripts) | Already fully gitignored (`.gitignore` line 3) -- never enters git history | None needed |
| `reports/*.md` (contains the Kaggle account handle `sweety611` in 8 files) | **Currently untracked** (the whole `reports/` directory has never been committed) -- no actual leak has occurred, but if these files are committed as-is, the handle would enter git history | **Flagged, not auto-fixed**: whether a Kaggle username is linkable to the authors' real identity is the authors' own judgment call (depends on that account's public profile), and scrubbing it would remove genuine provenance information (rule 9) without the authors having asked for it. Recommend the authors decide, before committing `reports/`, whether to redact the handle or accept the risk. |
| Git committer identity (`Hema6-ai` on the two existing commits) | Pre-existing, not something this audit can retroactively anonymize without rewriting commit history (a destructive operation this audit will not perform without explicit instruction) | Not acted on; flagged for the authors' awareness only |

## 8. Compilation status

Recompiled this session via `tectonic.exe --keep-logs tmlr_2026.tex` (build scaffolding -- `.sty`/
`.cls`/`.bst`/`references.bib`/figure `.png` files -- copied from `paper/writeup/` into the repo
root, then removed after compilation, consistent with this project's established build process).

- **Result**: success, 25 pages, ~8.7MB PDF.
- **Warnings present** (inspected in `tmlr_2026.log`, all pre-existing and unrelated to this
  session's edits): several `Underfull \vbox` warnings (cosmetic spacing, not content loss), one
  `Overfull \hbox (42.4pt too wide)` at the PhysioNet-comprehensive table (`tab:physionet-comprehensive`,
  pre-dates this session's edits -- not introduced by anything changed here), and one `Invalid
  UTF-8 byte or sequence` warning from `algorithm.sty` (a third-party style file, not this paper's
  content). **These warnings persist and are disclosed here rather than claimed fixed** -- the
  build is not literally "clean," but none of the warnings indicate missing or incorrect content
  (verified by text-extracting the final PDF and confirming every number/section this session
  touched is present and correct).
- No `[TBD]`, `TODO`, or other unresolved placeholder found in the final source (`\pending{}` macro
  is defined but never invoked).
- Text-extraction spot-check (`pdftotext` + `grep`) confirmed the new SPDNet-corr numbers (45.79%,
  +1.68pp, p=0.26), the corrected `fig:allmethods` caption, and the abstract's new clarifying
  sentence all appear in the compiled PDF.
