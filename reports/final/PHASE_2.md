# Phase 2: Automated Consistency Checks

## 2.1 Numbers (abstract, introduction, conclusion, Limitations, captions, both letters)

Every percentage/p-value/pp-difference token extracted from the abstract, introduction,
conclusion, and Limitations sections maps to an existing, previously-verified result:
49.94%/52.39% (nested vs. in-sample ensemble), +2.01pp/p=0.0273 and +0.88pp/p=0.0383 (time-window
effect at n=9/n=109), 42.8%/62.3% (bootstrap CI). All 36 table/figure captions were extracted with
a brace-balanced parser (a naive regex truncates at the first internal `}`, e.g. inside `p{=}...`,
and was replaced after it produced false "no numbers" results) and checked; every numeric claim in
a caption (52.39%, 49.94%, +4.94pp/p=0.0039, +2.49pp/p=0.0742, 2.4--3.1pp, +2.01pp, 50%/33% chance
levels, kappa=0.333) is a pre-existing, previously-audited number, not new to this pass.

**Status: PASS.** No unmapped numeric claim found.

## 2.2 Counts

Checked against rendered PDF text: "four method families" (4 occurrences, consistent), "three
populations" (6, consistent), "two exceptions" (2, consistent), "three observations" (1, matches
the First/Second/Third list), "two further controls" (1, matches the EEGNetv4-population-swap +
RT-matched items), "ten further extensions" (2 occurrences, consistent with each other and with
Algorithm 1's step), "7 of 9" / "7 of the 9" (4 occurrences, all referring to the same
alignment-success count, consistent), "three supervised architectures" (uses the phrase "three
supervised deep architectures", consistent), LaBraM's "three adaptation regimes" (2 occurrences,
consistent; a separate, unrelated "two regimes" refers to MAE-only's backbone-fine-tuning
ablation, not LaBraM -- verified by reading full context, not a conflict).

**Status: PASS.**

## 2.3 Cross-references

PDF rendered text: 0 occurrences of "??" (no undefined references render visibly; also 0
undefined-reference warnings in the compile log). Both response letters: 0 raw LaTeX
(`\ref`, `Table~`, etc.).

`RESPONSE_TO_REVIEWER_c1um.md` (written after this round's table insertion) already used the
correct current numbers (Table 6, Table 8, Table 17).

`RESPONSE_TO_REVIEWER.md` (the ajnW letter, written across earlier rounds) had **gone stale a
second time**: this round's new Table 8 (matched-protocol SPDNet results) shifted every table
number from 8 onward by one again. Found and fixed, by content meaning checked against the
current `tmlr_2026.aux`, not by blind renumbering:
- "Table 7" -> **Table 15** (physionet-comprehensive) in three places where the surrounding
  sentence described the full method-family comparison at PhysioNet scale -- these were
  pointing at the wrong table (physionet-stats, a narrower FgMDM/MEKT-only table) even before
  this round's shift, not purely a renumbering artifact.
- "Table 8" -> **Table 9** (cho2017), three places.
- "Table 14" -> **Table 15** (physionet-comprehensive), one place.
- "Table 15" -> **Table 16** (e1-architectures), three places, including two rows of the
  summary-of-changes table.
- Section numbers (1, 3.1, 5.1, 10.1, 13.1-13.5, 14, 15) were all re-checked against the aux file
  and are unaffected by the table insertion -- no section-number fixes were needed.

**Status: FIXED, now PASS.**

## 2.4 Terminology

- "zero-label": 33 occurrences, used as the primary term throughout (title, abstract, body).
- "zero-shot": 6 occurrences, every one scoped to the LaBraM comparison (its definition, its two
  uses in Section 13.4, the results-table row, and the anonymized repository's own name) or the
  one explicit definition sentence reserving the term for it. No stray use elsewhere.
- "transductive": defined once (Section 3.1), used consistently (5 occurrences).
- "MEKT-lite": 20 occurrences, capitalization consistent (0 "MEKT-Lite" variants found).
- "EA": 88 occurrences, no inconsistent expansion found.
- "pp" (percentage points) used consistently for differences; "%" for absolute accuracies; no
  mixing found in the sections checked.

**Status: PASS.**

## 2.5 Revision and process language

Grepped both `tmlr_2026.tex` and the rendered PDF text for: previously, originally, earlier,
newly, surfaced, revisit, this round, reviewer, revision, honest, genuine, notably, session,
flaky, retried, Kaggle account names, file paths, script names in the manuscript body.

Zero hits for: previously, originally, earlier, newly, surfaced, revisit, this round, reviewer,
revision, honest, genuine, notably.

Three hits required justification (logged below and in `FINAL_QA.md`):
- "session" (1): "a lengthy calibration **session** is itself burdensome" -- ordinary English
  about a BCI calibration session, unrelated to this paper's writing process. **Justified, kept.**
- "retried" (2): both describe real experimental infrastructure (`physionet.org` download
  timeouts during data collection, and a subject-level retry during MAE-only training), reported
  in the paper's own Reproducibility/Compute disclosures. Pre-existing methodological detail, not
  revision-process narration. **Justified, kept.**
- "Kaggle" (1): "Deep-learning pretraining ran on a single Kaggle GPU (Tesla P100/T4, free
  tier)" -- a standard compute-resources disclosure, no account name or username attached.
  **Justified, kept.**

File paths / script names in the manuscript body: "paper/" = 0 occurrences; "reports/" = 0;
`tmlr_2026` = 0; ".aux" = 0. Three bare `.py` filenames remain, all pre-existing reproducibility
citations (`eval_mekt_ensemble.py`, `eval_e4_nested_selection.py`,
`eval_e7_confidence_weighted_fusion.py`) naming the exact script that reproduces a specific
number, with no directory path. **Justified, kept** -- consistent with this round's own choice
(Section 13.3's `stats_eegnet_comprehensive.csv`/`stats_e1_all_architectures.csv` mentions) to
allow a bare, real filename as a reproducibility pointer while removing anything that reveals
directory structure.

Both response letters: "reviewer" appears throughout `RESPONSE_TO_REVIEWER_c1um.md` and
`RESPONSE_TO_REVIEWER.md` by necessity -- each is a response addressed to a named reviewer ID,
quoting their comments. **Justified, kept** (removing it would make the letters unusable).

**Status: PASS (3 justified exceptions logged in FINAL_QA.md).**

## 2.6 Anonymity

- PDF metadata: Title/Author/Subject/Keywords/Creator all empty or generic
  ("LaTeX with hyperref"). No encryption, no trapped flag.
- PDF rendered text: no hits for any author name, institution, personal email, or GitHub
  username. The only repository link in the manuscript is the anonymized one
  (`https://anonymous.4open.science/r/eeg-zero-shot-decoding-BA8B/`); no `github.com` link is
  exposed.
- Every figure PNG's embedded metadata contains only a generic Matplotlib software tag and DPI --
  no author/path/username fields.
- `references.bib` and `.tex` comments: no author name, institution, or identifying string found
  outside the (correctly hidden-by-`tmlr.sty`) `\author{}` block itself.
- Both response letters: no name, email, institution, username, or local path found.

**Status: PASS.**

## 2.7 Statistics: Holm-adjusted p recomputation

Recomputed Holm-adjusted p-values directly from each stats CSV's raw `wilcoxon_p` column
(independent re-implementation of the step-down procedure, not reusing `stats_analysis.py`'s own
code), grouped by the bracketed family prefix in each `comparison` string (the convention this
project's scripts already use to mark family membership) and cross-checked against the stored
`holm_significant` boolean.

- `stats_spdnet_matched_protocol.csv` (this round's new table): 3 families of m=4 (bci2a,
  physionet30, physionet_full) -- 0 mismatches. Matches the manuscript's Table 8 exactly.
- `stats_eegnet_comprehensive.csv`: 6 families (m=3 or m=1 each) -- 0 mismatches.
- `e5_cho2017_eegnet_vs_classical_stats.csv`, `stats_e9_alignment_ablation.csv`: 0 mismatches.
- `stats_e1_all_architectures.csv`: 4 within-dataset families of m=4 found 1 apparent mismatch on
  first pass (ShallowFBCSPNet vs. nested-ensemble, PhysioNet30: raw p=0.0267, naive m=4
  Holm-adjusted p=0.0267, which would be significant, vs. the stored flag of
  not-significant). Investigating resolved this as **not a bug**: the manuscript text itself
  states this specific comparison is Holm-corrected at **m=2, combined with its BCI-IV 2a
  counterpart** ("Holm-adj. 0.0533 at m=2 with its BCI-2a counterpart"), not within the
  four-comparison per-dataset family. Recomputing with that m=2 cross-dataset family
  (p=0.0267 for PhysioNet30, p=0.1289 for BCI-IV 2a) gives Holm-adjusted p=0.0533 for the
  PhysioNet30 row -- an exact match to the manuscript's stated value. Confirmed by checking the
  ATCNet counterpart rows too (same m=2 cross-dataset convention, consistent). The stored
  `holm_significant=False` flag is correct; only my first-pass grouping heuristic was wrong.
- Four single-row CSVs (`stats_eegnetv4_vs_ensemble_full_physionet.csv`,
  `stats_fgmdm_vs_nested_ensemble.csv`, `stats_alignment_baseline_tslr.csv`,
  `stats_eegnet_cho2017_alignment.csv`) carry no `holm_significant` column by design: each is one
  input to the pooled 9-comparison alignment-success family ("7 of 9") reported directly in prose
  rather than re-stored per file.

**Status: PASS.** One apparent discrepancy investigated and explained (correct, documented
cross-dataset family); zero real discrepancies found.

## 2.8 Compile

- 0 undefined references (confirmed via log grep and 0 "??" in rendered text).
- 4 overfull `\hbox` warnings, all negligible and pre-existing in scale (3.76pt, 2.80pt, 1.09pt,
  8.87pt) -- none visibly affects output at normal zoom; no overfull `\vbox` beyond standard
  underfull-only float-placement warnings.
- All 24 font subsets embedded (every font name carries the standard LaTeX subsetting prefix,
  e.g. `AKHCZK+LMRoman10-Italic`; none referenced externally).
- All figures present (compile would have failed on a missing `\includegraphics` target; it did
  not).
- Spell check: three independent passes (duplicate-word scan, 90-term common-typo list with word
  boundaries, domain-aware `pyspellchecker` over the LaTeX source with math/commands/tables/
  citations stripped) found no genuine misspelling.
- **Page count: 34 total, 24 main-text pages** (Appendix A begins on page 25; References begin
  on page 27, inside the appendix span).

**Status: PASS.**

## 2.9 Rerun verify_all_claims and claim-evidence audits

- `paper/verify_all_claims.py`: **20/20 claims independently verified** (16 pre-existing +
  4 this-round additions), all new numeric strings found in the manuscript text.
- `paper/build_claim_evidence_audit.py`: 49/49 verified.
- `paper/build_claim_evidence_audit_r2.py`: 52/52 verified.
- `paper/build_claim_evidence_audit_r3.py`: 16/16 verified.
- **117/117 total audit rows verified, 0 failures.**

**Status: PASS.**
