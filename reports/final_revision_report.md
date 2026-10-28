# Final revision report — Paper #11562

Date: 2026-10-28 (final update, third audit round). Synthesizes three revision sessions
(2026-10-13/15, 2026-10-18, 2026-10-20, and this session's full master-directive audit,
2026-10-28). Supersedes all prior versions of this file; superseded claims are not repeated
without re-verification.

## Overall verdict: READY

See Section K for the full justification. Short version: every reviewer concern has a verified
experiment or manuscript change behind it (Section A); every numeric claim in the manuscript
independently reproduces from raw data (46/46, Section C); this session's full audit pass found
and fixed three real prose-accuracy issues and one genuine (now-resolved) anonymity-hygiene gap,
and recovered one at-risk source asset (Section F) -- none of which indicate a deeper problem, and
all of which are now closed. No reviewer concern is left unresolved or unverifiable.

## A. Reviewer-response matrix

**Disclosure**: no file in this repository contains Reviewer ajnW's verbatim comment text; the
concern categories below are paraphrased summaries established from the user's own description of
the review, not a pasted original document. See `reports/reviewer_comment_closure_matrix.md` for
the full per-item closure table (9 items, all Resolved) and `reports/final_repository_inventory.md`
Section 2 for the full disclosure.

| Concern | Original claim | Experiment / manuscript change | Evidence artifact | Verification status | Remaining limitation |
|---|---|---|---|---|---|
| **A** (9-subject reliance) | Comparison relied almost entirely on BCI-IV 2a | Comprehensive 8-method classical/domain-adaptation + 5-method deep-learning comparison on PhysioNet (109 subjects) | `paper/results/physionet_comprehensive_*.csv`, `paper/results/physionet_loso_*.csv` | **Re-verified this session**: 16/16 aggregate claims match raw per-subject recomputation | Deep-learning arm uses reversed (BCI-IV-2a-source) pretraining — a disclosed population swap |
| **B** (statistical evidence for window/estimator) | Window/estimator effects observed sequentially, not independently tested | Full factorial design (`paper/eval_factorial.py`, `paper/stats_factorial.py`), including the deep-learning (SPDNet) estimator check | `paper/results/factorial_statistical_analysis.csv`, `reports/factorial_experiment_report.md` | Re-run fresh this session, deterministic, exact match | Deep-learning arm remains a reduced (default-window-only) design by deliberate scope choice |
| **C** (scope of central claim) | Risk of overgeneralizing beyond tested setting | Abstract/Limitations/Conclusion scoped; **this session**: added an explicit abstract clause distinguishing the window effect's generality from the estimator effect's configuration-specificity | `tmlr_2026.tex` abstract, `sec:factorial`, Limitations, Conclusion | Full manuscript re-scanned this session for absolute-quantifier overclaiming; 2 instances found and fixed (Section F) | None remaining found |
| **D** (zero-shot vs. transductive) | Needed explicit, consistent terminology | Verified every method's target-data access against the actual code | `reports/final_scientific_audit.md` Section 5 | Re-audited this session via direct code inspection | None found |
| **E** (related-work verification) | Novelty/citations not independently verified | Standalone verification report (24/24 citations) **plus this session's novelty comparison against 5 closest prior works, 2 newly found and verified from the current literature** | `reports/related_work_verification.md`, `reports/novelty_audit.md` | 24/24 citations verified; 2 new 2025-2026 papers fetched and verified directly from primary sources | One optional (not auto-applied) recommendation: cite the 2 newly-found papers if the authors, after reading them in full, judge they strengthen Related Work |

## B. Experiment execution summary

| Experiment | Status | Dataset / n | Main result |
|---|---|---|---|
| Classical/domain-adaptation PhysioNet comprehensive | Completed, re-verified this session | PhysioNet, 109 | MEKT default best: 51.54% |
| Deep-learning PhysioNet comprehensive | Completed, re-verified, 109/109 across all 5 methods | PhysioNet, 109 | SPDNet best: 49.40% |
| LaBraM zero-shot comparison | Completed, re-verified | BCI-IV 2a, 9 | 25.08% ± 1.74% (chance) |
| Factorial (classical/domain-adaptation arm) | Completed, re-verified this session | BCI-IV 2a, 9 | Window main effect significant (p=0.027); estimator/method main effects not significant |
| Factorial (SPDNet estimator arm) | Completed, re-verified this session | BCI-IV 2a, 9 | 45.79% ± 14.33% (corr) vs. 44.12% ± 12.21% (lwf), Δ=+1.68pp, p=0.26 — not significant |
| Subject-level jackknife robustness | Completed, re-verified this session | PhysioNet, 109 | All 4 key comparisons robust to single-subject removal |
| Statistical re-verification (both populations) | Re-run fresh this session | BCI-IV 2a (9) + PhysioNet (109) | Exact reproduction, deterministic |
| Claim-evidence audit | Rebuilt and extended this session | — | **46/46 rows VERIFIED**, 0 failures (grew from 35 after extending audit coverage to the factorial table, Section F) |
| Novelty audit | **New this session** | — | 5 closest prior works compared; no unsupported novelty language found |
| PDF recompilation | Re-run this session after manuscript edits | — | 25 pages; verified all new/changed text present via extraction |

No new experiment beyond the already-completed SPDNet check was judged necessary this session
(`reports/final_scientific_audit.md` Section 8) — this round's work was audit, verification, and
manuscript-accuracy correction, not new data collection.

## C. Statistical audit

Complete machine-readable record: `paper/results/statistical_tests_bci2a.csv`,
`paper/results/statistical_tests_physionet.csv`, `paper/results/factorial_statistical_analysis.csv`,
`paper/results/robustness_subject_influence.csv`. All re-run fresh this session; every value below
reproduced exactly.

| Comparison | Effect | 95% CI | p | Holm-sig. | Jackknife-robust? |
|---|---|---|---|---|---|
| FgMDM → 8-member ensemble (BCI-IV 2a) | +4.94pp | [+3.38,+6.58]pp | 0.0039 | Yes | n/a |
| SPDNet → MEKT default (PhysioNet) | +2.14pp | [+1.03,+3.24]pp | 0.0002 | Yes | **Yes** |
| MDM → FgMDM (PhysioNet) | +7.21pp | [+5.28,+9.15]pp | <0.0001 | Yes | **Yes** |
| FgMDM → MEKT default (PhysioNet) | +2.10pp | [+1.19,+3.03]pp | 0.0001 | Yes | **Yes** |
| MAE+GRL → SPDNet (PhysioNet) | +12.60pp | [+10.23,+14.99]pp | <0.0001 | Yes | **Yes** |
| MEKT default → MEKT tuned (PhysioNet) | -0.59pp | [-1.51,+0.31]pp | 0.4497 | No | — |
| Window: default → (0.3,2.5) (factorial, BCI-IV 2a) | +2.01pp | [+0.68,+3.18]pp | 0.0273 | No (7-contrast Holm family) | — |
| Estimator: lwf → corr, FgMDM/MEKT (factorial) | +0.04pp | [-0.80,+0.99]pp | 0.8203 | No | — |
| Estimator: lwf → corr, SPDNet only (factorial) | +1.68pp | [-0.35,+3.88]pp | 0.2626 | No | — |
| Method: FgMDM → MEKT (factorial) | -0.59pp | [-2.09,+0.70]pp | 0.6523 | No | — |
| Interaction: window × method (factorial) | -0.15pp | [-1.06,+0.79]pp | 0.8203 | No | — |

Methodology checks (new this session, `reports/final_scientific_audit.md` Section 3): units of
analysis (subject-paired throughout, verified by code read), mean/SD/CI not confused (explicit
in-text distinction verified), nonsignificance not presented as proof of equivalence (checked),
correction-family sizes internally consistent (7-contrast factorial family, 8-comparison PhysioNet
family, both verified against script output counts).

## D. Novelty assessment

Full detail in `reports/novelty_audit.md`. Summary: the paper's defensible contribution is a
controlled factorial separation of preprocessing effects from architecture effects, replicated
across two populations, with a disclosed negative-result catalog — not an algorithmic
contribution, and the manuscript does not claim one. Compared against 5 closest prior works (3
previously verified, 2 newly found and verified this session via direct primary-source fetch:
Manivannan & Lakshminarayan, arXiv:2511.18940, and Ramkumar & Lakshminarayanan, NeurIPS 2025
workshop poster). Neither new paper performs the same factorial separation or contradicts this
paper's zero-shot findings (one is explicitly transductive-framed, the other lacks a factorial
design). No unsupported "first"/"novel"/"state-of-the-art" language found anywhere in the
manuscript.

## E. Reproducibility and anonymity status

Full detail in `reports/final_reproducibility_audit.md`. Summary:
- **Reproducibility**: every result has a documented, re-runnable command (consolidated this
  session in `README.md` and `reports/final_reproducibility_audit.md`); known non-rerunnable items
  (4-seed bootstrap script, confidence-weighted-ensemble implementation) remain disclosed, not
  hidden. No pinned dependency versions (disclosed, pre-existing).
- **Anonymity**: `.tex` source correctly gitignored (real names, by design). Compiled PDF verified
  this session to render "Anonymous authors" with clean metadata — but was **not** gitignored
  itself (inconsistent with the `.tex` treatment); **fixed this session** by adding
  `/tmlr_2026.pdf` to `.gitignore`. One disclosed, non-auto-fixed item: `reports/*.md` (currently
  untracked, never committed) contains the Kaggle account handle `sweety611` in 8 files — flagged
  for the authors' own judgment before committing `reports/`, not scrubbed unilaterally.

## F. Issues found and fixed this session (new findings, full detail in `reports/final_scientific_audit.md`)

1. **`fig:allmethods` caption contradicted its own plotted data** ("every geometric or classical
   Riemannian method does substantially better" — false for the plotted "Per-class EA+TS+LDA"
   point at 25.41%, chance level). Fixed: caption now names the exception and its documented cause
   (a train/test alignment mismatch, already understood in `reports/data_leakage_audit.md`).
2. **A centroid-distance claim overgeneralized beyond what was measured** ("any learned
   representation" when only the MAE family and CSP were actually compared on that specific
   statistic). Fixed: reworded to name exactly what was measured.
3. **Abstract bundled two preprocessing effects of different generality** without noting the
   factorial check found only one to generalize. Fixed: added one clarifying sentence.
4. **Claim-evidence audit coverage gap**: the factorial table had zero corresponding audit rows
   despite backing manuscript numbers. Fixed: extended `paper/build_claim_evidence_audit.py` to
   cover it; audit grew from 35 to 46 rows, all verified.
5. **Anonymity hygiene gap**: compiled PDF not gitignored (Section E above). Fixed.
6. **A source figure asset (`paper/writeup/figures/pipeline_overview.png`) was found missing from
   disk during this session's recompilation** (not reproducible by any script — a hand-made
   diagram — and not under git version control, so its loss would have been unrecoverable through
   normal means). **Recovered** by extracting the embedded image directly from this session's
   still-intact, previously-compiled PDF (`pdftotext`/`PyMuPDF`, verified pixel content matches the
   expected 8-step pipeline diagram before restoring it to its source path). Root cause of the
   file's disappearance was not established (it was never git-tracked, so there is no history to
   diff against) — flagged as a residual risk, not fully explained. **Recommendation**: commit
   `paper/writeup/figures/*.png` to git now that this gap is closed, since 3 of 14 figure files
   (including this recovered one) currently exist only as untracked loose files with no backup
   beyond the compiled PDF.
7. **README reproducibility table was stale** (only listed session-1 scripts). Fixed: added rows
   for the factorial, robustness, and verification/audit scripts.

## G. Remaining gaps (not hidden, all previously disclosed and re-confirmed this session)

1. Deep-learning arm of the factorial is a reduced (default-window-only) design by deliberate
   scope choice, not incomplete work.
2. Deep-learning seed sweep not extended beyond 5 seeds (42/0/123/21/999) — judged adequate.
3. 4-seed ensemble bootstrap-resampling computation has no standalone rerun script (pre-dates this
   revision).
4. Jackknife robustness scoped to the 4 highest-stakes PhysioNet comparisons, not all 13.
5. Kaggle-dataset-mirror substitution for the SPDNet-corr run carries a residual, disclosed risk
   (a corrupted/mislabeled mirror file would not raise a download error) — see
   `reports/final_scientific_audit.md` Section 6.

## H. Compilation status

Recompiled this session via `tectonic.exe --keep-logs tmlr_2026.tex`. **Result: success, 25
pages, ~8.7MB.** Warnings present and disclosed (not claimed fixed): several cosmetic
`Underfull \vbox` spacing warnings; one pre-existing `Overfull \hbox (42.4pt too wide)` at the
PhysioNet-comprehensive table (predates this session); one `Invalid UTF-8` warning from a
third-party style file (`algorithm.sty`, not this paper's content). None indicate missing or
incorrect content — confirmed by text-extracting the final PDF and checking every
number/caption/sentence touched this session is present. No `[TBD]`/`TODO`/unresolved placeholder
found in the final source.

## I. Final artifact inventory

- Manuscript source: `/tmlr_2026.tex` (gitignored, contains real author names by design)
- Compiled PDF: `/tmlr_2026.pdf` (25 pages; now also gitignored, Section E)
- Claim-evidence audit: `reports/claim_evidence_audit.csv` (46/46 verified)
- New reports this session: `reports/final_repository_inventory.md`,
  `reports/reviewer_comment_closure_matrix.md`, `reports/novelty_audit.md`,
  `reports/final_scientific_audit.md`, `reports/final_reproducibility_audit.md`
- All prior sessions' artifacts unchanged in location (see prior version of this file / the new
  inventory report for the full list)

## J. Exact files changed this session (changelog)

| File | Change | Why |
|---|---|---|
| `tmlr_2026.tex` | 3 prose edits (Section F items 1-3) | Fix overclaiming/imprecision found by audit |
| `.gitignore` | Added `/tmlr_2026.pdf` | Anonymity defense-in-depth (Section E) |
| `README.md` | Extended reproducibility table | Cover this revision's newer scripts |
| `paper/build_claim_evidence_audit.py` | Added factorial-table coverage + SPDNet script mapping | Close audit coverage gap (Section F item 4) |
| `paper/verify_all_claims.py` | Added 2 BCI-IV-2a SPDNet claims | Same |
| `paper/writeup/figures/pipeline_overview.png` | Restored (was missing from disk) | Data-loss recovery (Section F item 6) |
| `reports/final_repository_inventory.md` | New | Directive Phase 1 deliverable |
| `reports/reviewer_comment_closure_matrix.md` | New | Directive Phase 2 deliverable |
| `reports/novelty_audit.md` | New | Directive Phase 3 deliverable |
| `reports/final_scientific_audit.md` | New | Directive Phases 4-6 deliverable |
| `reports/final_reproducibility_audit.md` | New | Directive Phase 7 deliverable |
| `reports/final_revision_report.md` | Rewritten (this file) | Directive Phase 8/final deliverable |
| `reports/claim_evidence_audit.csv` | Regenerated | 46/46, reflects the two fixes above |

No unrelated code, data, or configuration changes were introduced — confirmed via
`git diff --stat` on every modified tracked file this session before writing this report.

## K. Commands to reproduce the main checks

```bash
python -m paper.verify_all_claims            # 16/16 aggregate claims
python -m paper.build_claim_evidence_audit   # 46/46 audit rows -> reports/claim_evidence_audit.csv
python -m paper.stats_analysis               # 13 paired statistical tests
python -m paper.stats_factorial              # 9 factorial contrasts
python -m paper.robustness_subject_influence # 4 jackknife robustness checks
```

Full command list (every experiment, not just checks): `reports/final_reproducibility_audit.md`
Section 3.

## L. Three highest-priority remaining actions (if any)

1. **Commit `paper/writeup/figures/*.png` to git**, especially `pipeline_overview.png` (just
   recovered, currently has zero backup beyond the compiled PDF).
2. **Decide on the `sweety611` Kaggle-handle disclosure** in `reports/*.md` before committing
   `reports/` (Section E) — an authors'-judgment call, not something this audit can decide.
3. **Optional**: read the two newly-found 2025-2026 papers in full
   (`reports/novelty_audit.md`) and decide whether to cite them in Related Work.

None of these three block the READY verdict; all are disclosed, bounded, and do not affect any
numeric claim, statistical conclusion, or the manuscript's central findings.
