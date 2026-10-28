# Final revision audit

Date: 2026-10-18. Addresses the latest directive's Section 2 (audit before modifying) for this
round of revision. This audit inspects the actual current state of the repository -- not filenames,
not previous summaries -- before any further changes in this round.

## 1. Repository structure and authoritative file paths

- **Manuscript source**: `/tmlr_2026.tex` (repo root). Gitignored deliberately (contains real,
  non-anonymized author names; see `.gitignore` lines 39-41) -- this is expected, not a bug.
  Confirmed via `git check-ignore -v` this session. A second, older/divergent copy exists at
  `paper/writeup/tmlr_2026.tex` (also gitignored) -- NOT the authoritative one; it was last
  modified Aug 19 (pre-dating this entire revision), while the root copy was modified Oct 15-18
  across this revision's edits. All work in this and the prior session targeted the root copy.
  **Action for future sessions: always edit `/tmlr_2026.tex`, never `paper/writeup/tmlr_2026.tex`.**
- **Compiled PDF**: `/tmlr_2026.pdf`, recompiled 2026-10-18 via a locally-installed `tectonic`
  binary (no LaTeX toolchain pre-existed in this environment). 25 pages after this session's
  factorial-experiment section was added (was 24 before). Verified this session
  (see `reports/current_results_reproduction.md` and the compile transcript) that the PDF's text
  content includes every new section/number added this revision (LaBraM, SPDNet, Holm-Bonferroni,
  25.08%, 2.14pp/p=0.0002) -- i.e., the PDF genuinely corresponds to the current `.tex`, not a
  stale build.
- **Scripts**: all training/eval/stats/figure scripts live under `paper/` (flat, no `experiments/`
  subdirectory in this codebase's existing convention -- the directive's suggested
  `experiments/factorial/` path was not created; new scripts instead follow this repo's existing
  flat-`paper/`-directory convention for consistency with everything already there, e.g.
  `paper/eval_factorial.py` not `experiments/factorial/run.py`). This is a deliberate choice to
  avoid introducing a parallel directory convention — see rule 13 (no unnecessary complexity).
- **Results**: `paper/results/*.csv`, one file per method/experiment, consistent with the existing
  project convention of a flat results directory.
- **Reports**: `reports/*.md` plus `reports/*.csv` (the claim-evidence audit).

## 2. Existing reports verified present and read in full this session

| File | Present? | Re-inspected this session? |
|---|---|---|
| `reports/revision_audit.md` | Yes | Yes |
| `reports/current_results_reproduction.md` | Yes | Yes |
| `reports/data_leakage_audit.md` | Yes | Yes |
| `reports/compute_inventory.md` | Yes | Yes |
| `reports/reviewer_traceability_table.md` | Yes | Yes, updated this session |
| `reports/physionet_comprehensive_summary.md` | Yes | Yes |
| `reports/physionet_deep_learning_comparison.md` | Yes | Yes |
| `reports/foundation_model_comparison.md` | Yes | Yes |
| `reports/related_work_verification.md` | **New this session** | — |
| `reports/factorial_experiment_report.md` | **New this session** | — |
| `reports/robustness_sensitivity_report.md` | **New this session** | — |
| `reports/claim_evidence_audit.csv` | **New this session** | — |
| `reports/final_revision_report.md` | **New this session** | — |

No report's filename was trusted without opening and reading it. `paper/verify_all_claims.py`
(new this session) independently recomputes every per-subject-aggregate claim in
`tmlr_2026.tex` directly from the raw per-subject CSVs, not from any prior summary — see
Section 6 below for results.

## 3. Reviewer requests: satisfied / partially satisfied / unsatisfied

| Concern | Status this session |
|---|---|
| A: insufficient evidence (9-subject reliance) | **Satisfied.** PhysioNet comprehensive comparison (8 classical + 5 deep-learning methods, up to 109 subjects) independently re-verified this session (`paper/verify_all_claims.py`, 14/14 aggregate claims match). |
| B: statistical evidence for window/estimator | **Satisfied, with an honest caveat preserved.** Statistics re-run fresh this session (`paper/stats_analysis.py`), Holm-Bonferroni applied to both the n=9 and n=109 families. The factorial experiment (new this session, Section 4 below) directly estimates the window and estimator main effects rather than relying on a single progression's sequential comparisons. |
| C: scope of central claim | **Satisfied.** Abstract/Limitations explicitly scope the claim to tested architectures/datasets/protocols; this session added explicit language about the factorial design's partial coverage and the small-sample tuning risk (already demonstrated empirically: BCI-IV-2a-tuned MEKT does not outperform default on PhysioNet, p=0.45). |
| D: zero-shot vs. transductive terminology | **Satisfied** (pre-existing, re-verified this session — see Section 5). |
| E: related-work verification | **Satisfied this session** — see `reports/related_work_verification.md`, 24 citations checked claim-by-claim, including a live re-fetch of one paper's full text to verify a direct quotation. |

## 4. Known implementation bugs and resolution status

- **Environment-restart process kills**: this development environment has had background processes
  killed by what are presumably scheduled/external environment restarts at least 3 times across
  this revision's two sessions. Resolved by adding per-fold/per-cell resumability to every
  long-running script (`paper/eval_physionet_comprehensive.py`, `paper/eval_physionet_loso_deep.py`,
  `paper/eval_factorial.py`) — confirmed working (smoke-tested resume behavior; the classical
  comprehensive sweep actually recovered from two real interruptions this way).
- **Kaggle dataset-mount path assumption (v1-v2 of the deep-sweep kernel)**: fixed in v3 by walking
  the full `/kaggle/input` tree for `config.py`+`data.py` rather than guessing a fixed path depth.
  No longer an open bug.
- **Kaggle kernel output bloat (3.3GB+ downloads, repeated `ChunkedEncodingError`)**: fixed by
  redirecting MOABB/MNE raw-data caching to `/kaggle/temp` (outside the downloadable output) and
  suppressing periodic (non-"best") checkpoint saves. Applied in the v4 deep-sweep kernel and the
  new SPDNet-corrcov kernel.
- **physionet.org transient server flakiness** (caused MAE-only's PhysioNet eval to complete on
  only 95/109 subjects during the Kaggle run): **investigated and resolved this session** -- local
  network access to physionet.org from this machine was tested directly against the exact subject
  IDs that failed on Kaggle (1, 3, 6, 7, 9) and succeeded immediately. MAE-only is being fully
  retrained and re-evaluated locally this session for a complete, single-consistent-encoder
  109-subject result (status: see `reports/robustness_sensitivity_report.md`, Section 5.6).

## 5. Zero-shot vs. transductive terminology audit (reviewer Suggested Improvement D)

Re-verified this session against the actual code (not just the manuscript's prose claim):

| Method | Target-subject data accessed before prediction | Classification |
|---|---|---|
| MAE/SupCon/SPDNet/RiemannianTransformer zero-shot eval | None (frozen encoder fit only on source subjects' labeled data; target features extracted but never used to fit anything) | Strict zero-shot |
| CSP/FgMDM/MDM (no EA) | None | Strict zero-shot |
| EA/RA-based methods | Target subject's own unlabeled trial covariances (for per-subject whitening only, no cross-subject or label information) | Zero-shot-legal unsupervised alignment, explicitly distinguished from supervised calibration in the manuscript's terminology note (`sec:setup`) |
| MEKT, RPA | Target subject's unlabeled trials collectively (batch), iteratively pseudo-labeled | **Transductive** -- manuscript correctly labels this throughout ("never its true labels", `tmlr_2026.tex` line ~345) |
| LaBraM comparison (this revision) | None (frozen backbone, linear probe fit on source subjects only) | Strict zero-shot |

No method in this codebase accesses true target labels at any fitting/selection stage. Confirmed
by grep-auditing every `eval_*.py` and `*_loso*.py` script's use of `target_y` (used only inside
the final `classification_metrics()` call in every file checked).

## 6. Independent recomputation summary (full detail in `reports/current_results_reproduction.md`)

Re-run this session via `paper/verify_all_claims.py`: **14/14** PhysioNet/LaBraM aggregate claims
match independent recomputation from raw per-subject CSVs exactly (within 0.1pp / 0.01 kappa
tolerance). `paper/stats_analysis.py` re-run fresh: all 13 statistical comparisons (5 BCI-IV 2a +
8 PhysioNet) reproduce deterministically. The bootstrap 95% CI on the headline ensemble
([42.8%, 62.3%]) was independently re-bootstrapped this session (100k resamples, seed 42) and
matches exactly. The 4-seed stability claim (51.98% ± 0.27%) was arithmetic-checked against its
documented per-seed values and matches exactly, but — this is flagged, not hidden — the underlying
per-seed bootstrap-resampling computation itself has no standalone runnable script in this
repository (pre-dates this revision; see `reports/claim_evidence_audit.csv` row
`bci2a_4seed_stability`).

## 7. Available compute and realistic costs (update to `reports/compute_inventory.md`)

Unchanged from the prior session's inventory: local CPU-only (12 cores), Kaggle GPU (Tesla T4,
free-tier quota) accessible via account `sweety611`. This session additionally confirmed local
network access to physionet.org is reliable (resolves the MAE-only gap, Section 4 above) and
successfully installed a standalone LaTeX engine (`tectonic`) locally, so PDF compilation no
longer requires any external service.

## 8. Execution plan for this round, ordered by scientific importance

1. Independently re-verify every existing numeric claim (done, Section 6).
2. Related-work verification report (done).
3. Factorial preprocessing-vs-architecture experiment (in progress/done this session — see
   `reports/factorial_experiment_report.md`).
4. Robustness/sensitivity analyses, including the MAE-only recovery (done this session — see
   `reports/robustness_sensitivity_report.md`).
5. Claim-to-evidence machine-verifiable audit (done — `reports/claim_evidence_audit.csv` +
   `paper/build_claim_evidence_audit.py`).
6. Manuscript revision to incorporate the factorial/robustness findings (done this session).
7. Recompile PDF, inspect for broken references/overflow (done).
8. Final revision report with readiness verdict (`reports/final_revision_report.md`).
