# Final repository inventory

Date: 2026-10-28. Produced for the master audit directive's Phase 1. Establishes the actual
current state of the repository by direct inspection (`git status`, `git log`, `git check-ignore`,
file reads) -- not by trusting any prior report's filename or claimed status.

## 1. Authoritative manuscript source and PDF

- **Source**: `/tmlr_2026.tex` (repo root). Confirmed via `git check-ignore -v` to be intentionally
  gitignored (`.gitignore` line 39, under the "Real-author manuscript sources" block) because its
  `\author{}` block contains the real, non-anonymized names (P. Hema, Shridevi S, VIT). This is
  correct, deliberate practice, not an oversight.
- **A second, stale copy exists** at `paper/writeup/tmlr_2026.tex` (also gitignored). Confirmed via
  file modification time that it predates this revision and has not been touched by any session's
  edits. **It is not the authoritative file and must not be edited.**
- **Compiled PDF**: `/tmlr_2026.pdf`, recompiled this session via a locally-installed `tectonic`
  binary (`C:\Users\hema0\AppData\Local\tectonic\tectonic.exe`), 25 pages, ~8.7MB. Verified by
  direct text extraction (`pdftotext`) that the rendered title page shows **"Anonymous authors /
  Paper under double-blind review"**, not the real names -- the `tmlr.sty` package's default
  (non-`[accepted]`/`[preprint]`) mode hides the `\author{}` block automatically at compile time.
  PDF metadata (`/Creator`, `/Producer`, `/CreationDate`) contains no author-identifying fields.
  **Finding (fixed this session):** the compiled PDF itself was not gitignored (only the `.tex`
  source was), which was an inconsistency even though the current render is already anonymous --
  added `/tmlr_2026.pdf` to `.gitignore` as defense-in-depth against an accidental future compile
  in `[preprint]`/`[accepted]` mode being committed.

## 2. Reviewer reports

**No file in this repository contains the reviewer's original, verbatim comment text.** The
closest artifact is `reports/reviewer_traceability_table.md`, which organizes the concerns into
categories (A, B.1, B.2, C, D, D.1, Additional 1-3) attributed to "Reviewer ajnW" -- these are
**paraphrased summaries established in an earlier session from the user's own description of the
review**, not a pasted or quoted original review document, OpenReview export, or PDF. This
limitation is explicitly disclosed here per the directive's instruction not to invent or
reconstruct exact reviewer wording. All closure-matrix work in this audit (Section 2 of the master
directive) is therefore built on these paraphrased concern categories, with that caveat carried
forward into `reports/reviewer_comment_closure_matrix.md`.

## 3. Author-response / revision-note documents

No separate rebuttal draft or author-response document exists as a file. The equivalent is the
accumulated `reports/*.md` chain itself (each one a revision-session's working audit), culminating
in `reports/final_revision_report.md`.

## 4. Prior audit reports (all present, all re-opened and read this session)

| File | Session produced | Re-inspected this session |
|---|---|---|
| `reports/revision_audit.md` | 1 | Referenced |
| `reports/current_results_reproduction.md` | 1 | Referenced |
| `reports/data_leakage_audit.md` | 1 | Read in full -- used to explain the per-class-EA negative result (Section 6 below) |
| `reports/compute_inventory.md` | 1 | Referenced |
| `reports/reviewer_traceability_table.md` | 1 | Read in full |
| `reports/physionet_comprehensive_summary.md` | 1 | Referenced |
| `reports/physionet_deep_learning_comparison.md` | 1 | Referenced |
| `reports/foundation_model_comparison.md` | 1 | Referenced |
| `reports/related_work_verification.md` | 2 | Read in full -- reused as the base for the novelty audit |
| `reports/factorial_experiment_report.md` | 2 (updated session 3) | Read in full |
| `reports/robustness_sensitivity_report.md` | 2 | Referenced |
| `reports/final_revision_audit.md` | 2 | Read in full |
| `reports/final_revision_report.md` | 2 (updated session 3) | Read in full, updated again this session |
| `reports/claim_evidence_audit.csv` | 2 (regenerated session 3 and this session) | Regenerated and re-verified this session: **46/46 rows VERIFIED** |

No report's existence or filename was trusted without opening it; several numeric claims were
independently re-derived from raw CSVs rather than copied from these reports (Section 6 below and
`reports/final_scientific_audit.md`).

## 5. Experiment code, configs, results, statistics, figures

- Scripts: flat `paper/` directory (project's existing convention; no `experiments/` subdirectory
  was introduced). New files this revision: `eval_factorial.py`, `stats_factorial.py`,
  `eval_physionet_comprehensive.py`, `eval_physionet_loso_deep.py`, `run_physionet_deep_sweep.py`,
  `eval_foundation_model_labram.py`, `eval_mekt_progression.py`, `stats_analysis.py`,
  `verify_all_claims.py`, `build_claim_evidence_audit.py`, `robustness_subject_influence.py`.
- Results: `paper/results/*.csv`, one file per method/experiment (flat, consistent with the
  project's existing convention).
- Config: `paper/config.py` (seed 42 throughout for all deterministic/classical methods; seed
  documented per-run for deep-learning training).
- Dataset/caching code: `paper/data.py` (MOABB-based loaders for both BNCI2014_001 and
  PhysionetMI, plus this revision's `load_bnci_pretrain*` additions for the reversed-direction
  pretraining protocol).

## 6. Conflicts found and resolved this session (new finding, not previously reported)

**`paper/results/zero_shot_spdnet.csv` (the canonical `lwf`-estimator SPDNet baseline backing the
manuscript's existing 44.12% claim) held only 3 of 9 subjects** -- a stale, partial local run that
had been committed as-is since the original TMLR submission commit (`7db5875`, confirmed via
`git log --follow`). The actual complete 9-subject run was never copied back from its original
location (`kaggle_kernel/output_spdnet/results/zero_shot_spdnet.csv`, gitignored, local-only).
Recomputing that file's mean/std (44.12%/12.21%) matched `paper/results/final_results_table.csv`
row 23 and the manuscript's existing number exactly, confirming it as the correct source. Fixed by
copying it into the canonical path; the stale 3-subject file is preserved at
`paper/results/zero_shot_spdnet.csv.stale_3subj_bak` for provenance (per rule 9, nothing was
deleted). This conflict could not have been caught by `paper/verify_all_claims.py` alone before
the fix, because that script recomputes from whatever file is at the canonical path -- it verifies
internal consistency (does the file match the claimed number), not whether the file itself is the
complete/correct one. This is now disclosed explicitly so a future session does not re-trust the
script's "PASS" as proof the underlying file was always correct.

## 7. Missing/unresolved provenance (disclosed, not fabricated)

- The 4-seed ensemble bootstrap-resampling computation backing "51.98% ± 0.27%" has no standalone
  rerun script in this repository; only its arithmetic (mean/std of the 4 documented per-seed
  values) has been independently checked. Pre-dates this revision.
- The confidence-weighted ensemble refinement (52.53%, vs. the base 52.39% ensemble) has no
  recovered implementation; the manuscript explicitly and consistently discloses this in three
  places (lines ~224, ~232, ~488 of `tmlr_2026.tex`) as "reported; implementation not available for
  independent verification." Verified this session that this disclosure is present and consistent;
  not re-litigated further.

## 8. Reproducibility/environment files

- `README.md`: Setup/Reproducing-results sections present but did not list this revision's newer
  scripts (factorial, robustness, verification/audit scripts) until this session -- updated (see
  `reports/final_reproducibility_audit.md`).
- No `requirements.txt`/`environment.yml`/pinned versions exist; README already discloses this
  ("Exact pinned versions were not preserved from the original run").

## 9. Git state at start of this audit

- Branch: `main`. Last two commits: `09f96be` ("Fix two chart titles...") and `7db5875` (original
  TMLR submission). Everything from this revision (sessions 1-3) is **uncommitted** working-tree
  state: modifications to 11 tracked files, and ~60+ new untracked files (new scripts, new result
  CSVs, the entire `reports/` directory, the compiled PDF). No commits were made during this audit;
  this is recorded as a fact, not a recommendation to commit (committing was not requested).
