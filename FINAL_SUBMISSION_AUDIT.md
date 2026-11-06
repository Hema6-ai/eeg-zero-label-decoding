# Final Submission Audit — Paper #11562

**Scope.** Full scientific, statistical, editorial, and anonymity audit of `tmlr_2026.tex` /
`tmlr_2026.pdf` and `RESPONSE_TO_REVIEWER.md`, against the actual repository state (not against
any prior audit's claims, which were independently re-verified here). Local-only document; not
pushed or committed (see Submission Package note at the end).

## 1. Repository state at start of this pass

- Branch `main`, clean except two intentionally-untracked files (`RESPONSE_TO_REVIEWER.md`,
  `neuro.code-workspace`) and the gitignored `tmlr_2026.tex`/`.pdf` (real-author source kept local
  during review by design — `.gitignore` documents why).
- Latest commit `bca7ee3`: the PhysioNet factorial replication, Cho2017 evaluation, supervised
  architecture comparisons, and foundation-model fine-tuning experiments, with their result CSVs
  and statistical-analysis scripts.
- `reports/` (historical audit/process markdown) was removed from the repository per explicit
  instruction in a prior turn; this pass regenerates only `reports/claim_evidence_audit.csv` as a
  machine-readable verification artifact (see Section 4), kept local.

## 2. Scientific review findings (Phase 2) and corrections made

| # | Finding | Correction |
|---|---|---|
| 1 | Introduction described the protocol as "all models trained or adapted on source subjects only," which is inaccurate for EA and MEKT (both access the held-out subject's own unlabeled trials transductively) | Rewrote the paragraph: source-only *fitting*, with transductive target-trial access named explicitly and pointed to Section 3.1 |
| 2 | Introduction and Contributions implied "one fixed evaluation protocol" without acknowledging dataset/method-specific scope differences (seed counts, population sizes, which architectures ran at which scale) | Replaced with "one shared zero-label evaluation framework" and an explicit sentence pointing to where deviations are disclosed |
| 3 | Section 11's opening sentence still said "per Section 10.1's controlled check, only the window effect was confirmed as a general main effect" — this directly contradicts the Holm-correction fix made earlier this revision (neither effect survives correction) | Rewrote to state plainly that neither effect survives multiple-comparison correction as a general main effect |
| 4 | Section 5.1 (nested cross-validation) explains the restricted 9-candidate check and the broader per-fold search in two separate paragraphs, but never states in one unmistakable sentence, immediately after the restricted check's result, that it is *not* the primary estimate | Added that sentence directly after the restricted-check paragraph |
| 5 | Figure 4 (progression chart) and Figure 5 (all-methods bar chart) captions did not flag that the ensemble value they display (52.39%) is the in-sample, optimistic figure | Both captions now state this explicitly and name the nested 49.94% primary estimate |
| 6 | Table 4 (cross-dataset generalization) reported PhysioNet's ensemble result only relative to the in-sample 52.39% (−0.44pp); the comparison to the primary 49.94% estimate (+2.01pp) was absent | Added to the caption |
| 7 | Table 17 (appendix, full method inventory) lists only 52.39% for the ensemble, with no pointer to the 49.94% primary estimate | Added to the caption |
| 8 | The covariance-estimator/EA residual-scale explanation (Section 10) was stated as a finding rather than an interpretation | Rewrote to lead with "one plausible interpretation... rather than a mechanism established by the factorial results" |
| 9 | Genuine duplicate-word typo: "To obtain a a multi-seed stability check" (Section 12) | Fixed |
| 10 | Table 3 (BCI-2a factorial) had three row labels verbose enough to cause a 17pt table overflow (fixed in Phase 6 below) | Shortened row labels; the averaging convention they described is already stated once in the paragraph immediately preceding the table |

All ten were verified against the actual `.tex` source before editing (not assumed from the
directive's own description) and against the compiled PDF after editing.

## 3. Items checked and found already correct (no action needed)

- Central claim consistency across abstract/intro/results/conclusion (alignment generalizes, with
  the two disclosed exceptions: MEKT on Cho2017, raw-EEG self-supervised encoder on BCI-IV 2a).
- PhysioNet/Cho2017 per-method-family scope (classical/domain-adaptation at n=109; SPDNet and the
  self-supervised family via population-swapped pretraining; EEGNetv4 at n=30 and n=109;
  ShallowFBCSPNet/ATCNet at n=30 only; Cho2017 for classical methods, the ensemble, and EEGNetv4)
  — already stated identically in the Introduction's architecture subsection, Section 13.3,
  Section 14 (Limitations), and the Table 14 caption, from a prior pass this revision.
- Zero-label / transductive / zero-shot terminology — defined once in Section 3.1, used
  consistently everywhere checked.
- LaBraM scope (one checkpoint, 20-of-22 channels, frozen + positive control + two fine-tuning
  regimes, all at chance) is not generalized to "every EEG foundation model"; CBraMod/BIOT
  exclusions are each given a specific, verified reason in Section 14, not a blanket unavailability
  claim.
- MIRepNet and the Euclidean-Alignment-for-deep-decoders citation are used to scope this paper's
  own findings as consistent with, not novel beyond, that prior work (Section 13.3, 13.4).
- Citation necessity and attribution: all 30 bibliography entries are cited at the exact point the
  method/dataset they belong to is introduced; none are padding; every borrowed method (MAE, GRL,
  CLUB, VIB, EA, RA, RPA, MEKT/JDA, CSP, MDM, FgMDM, SPDNet, ReZero, EEGNetv4, BENDR, BIOT, LaBraM)
  is attributed, and the paper's own variants (MEKT-lite, the tangent-space RPA implementation) are
  explicitly labeled as the authors' implementation of a cited method, never as a new algorithm.

## 4. Numerical audit (Phase 4)

See `FINAL_NUMERICAL_CLAIM_AUDIT.md` for the full breakdown. Summary: `verify_all_claims.py`
independently recomputes 16 aggregate claims from raw per-subject CSVs — 16/16 match.
`build_claim_evidence_audit.py` + `build_claim_evidence_audit_r2.py` map 98 manuscript numbers to
their producing script and result file — 98/98 verified, 0 failed. No numerical claim was changed
silently; the two corrections made this revision (the BCI-2a window-effect Holm label, and one
stale ShallowFBCSPNet significance verdict from a since-corrected CSV) were presentation/labeling
fixes, documented in the manuscript's own text, not value changes.

## 5. Response-letter audit (Phase 5)

See `FINAL_REVIEWER_CLOSURE_MATRIX.md`. Two inaccuracies were found and fixed this pass:

1. Critical Change 1 overstated PhysioNet scope ("every deep-learning method... now also runs on
   PhysioNet's full 109-subject population") without the per-family qualification; corrected to
   state the exact scope per method family with section/table citations.
2. Critical Change 2's nested-estimate paragraph did not distinguish the two nested checks (the
   fixed 9-candidate check that reproduces 52.39% vs. the from-scratch 32-config per-fold search
   that gives 49.94%); corrected to describe both and state which is primary, matching the
   corresponding Section 5.1 fix above.

## 6. LaTeX / PDF audit (Phase 6)

- Recompiled with `tectonic --keep-logs --keep-intermediates`. Final compile: **0 undefined
  references, 0 multiply-defined labels, 0 missing citations.**
- Page count: **32 pages total** (main text ends page 22, appendix follows), up from 31 before
  this pass — the increase is the added clarifying sentences/captions in Section 5, Section 5.1,
  Section 11, Figure 4/5 captions, Table 4/17 captions; no content was removed to compensate, and
  this trade-off (one extra page for internal consistency) is intentional.
- Overfull-hbox check: two genuine overflows found and fixed (Table 3, up to 17pt past the margin
  from verbose row labels; Table 4, 31pt past the margin from an added row-label suffix later
  reverted). One pre-existing, sub-visible overflow remains (Table 1's bootstrap-CI column, 3.76pt
  ≈ 0.05mm — not a visible defect; present before this entire revision began).
- PDF metadata: title/author/subject/keywords all empty, as required for anonymous review.
- Rendered PDF text: 0 occurrences of any author name, email, institution, local file path, or
  GitHub/Kaggle username; 0 standalone `?` characters (would indicate an unresolved ref/citation).

## 7. Anonymity and repository hygiene (Phase 7)

- `sweety611` (flagged by a prior audit) is found only inside `kaggle_kernel/` (a Kaggle dataset
  slug used for private experiment orchestration), which is listed in `.gitignore` and was never
  committed or pushed. It is not part of the manuscript, the response letter, or the public
  repository. No action needed beyond confirming this.
- No `sweety611` or any other identifying string appears in the `.tex` source, the compiled PDF
  text, PDF metadata, figure file metadata (checked: all 15 figures carry only
  `matplotlib`/dpi metadata), or `RESPONSE_TO_REVIEWER.md`.
- The anonymized code-availability link in the manuscript
  (`https://anonymous.4open.science/r/eeg-zero-shot-decoding-BA8B/`) points to an
  anonymous.4open.science mirror, not the real GitHub account. **This tool has no browser/HTTP
  access and cannot verify that this mirror currently reflects the latest commit
  (`bca7ee3`).** The author must re-sync or re-generate this anonymized link against the current
  `main` branch before submission; this is flagged as an open item in
  `FINAL_SUBMISSION_CHECKLIST.md`.

## 8. Reproducibility (Phase 8)

- `README.md` was found to be substantially stale: old title, old central claim ("signal quality,
  not model complexity" — the exact claim this revision retracts), two datasets instead of three,
  three method families instead of four, "zero-shot" terminology throughout instead of
  "zero-label," and pointers into the now-deleted `reports/` directory. **Rewritten in full** to
  match the final manuscript's title, claim, terminology, datasets, method families, and headline
  numbers (49.94% primary, 52.39% in-sample, both explained).
- Every script path referenced in the new README was checked to exist in the current repository
  (26/26 confirmed present).
- No training or evaluation implementation was modified. No experiment was rerun from scratch;
  existing result CSVs and statistics scripts were re-executed to confirm they still reproduce the
  numbers already in the manuscript (Section 4 above).

## 9. Submission package note

This audit, `FINAL_REVIEWER_CLOSURE_MATRIX.md`, `FINAL_NUMERICAL_CLAIM_AUDIT.md`, and
`FINAL_SUBMISSION_CHECKLIST.md` are written to the repository root as local working documents.
Per an explicit instruction earlier in this project ("only tex, pdf, reviewer letter... remove the
reports too" — for the public GitHub repository), **none of these four files, nor the
regenerated `reports/claim_evidence_audit.csv`, have been committed or pushed.** They exist only
on this machine for your review. Tell me if you want any of them added to the repository or
submission package; by default I have left the public repo exactly as you last set it.

## 10. Submission action (Phase 10)

**Not performed.** This environment has no browser, no OpenReview/TMLR API access, and no
credentials for Paper #11562's submission portal — only local filesystem, git, and shell tools.
I cannot open the submission interface, upload files, or click any confirmation. See
`FINAL_SUBMISSION_CHECKLIST.md` for exactly what remains and must be done manually.
