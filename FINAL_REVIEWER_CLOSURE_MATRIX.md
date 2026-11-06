# Reviewer Closure Matrix — Paper #11562, Reviewer ajnW

Each row maps one point from the actual verbatim review (reproduced in full in
`RESPONSE_TO_REVIEWER.md`) to the manuscript section/table that addresses it and the verification
status as of the final compiled PDF (32 pages; see `FINAL_SUBMISSION_AUDIT.md` Section 6).

| Reviewer point | Manuscript location | Verified against final PDF | Notes |
|---|---|---|---|
| Central claim evidence insufficient ("does not convincingly establish preprocessing choices are generally more important than model complexity") | Abstract; Section 15 (Conclusion) | Yes | Central claim rewritten around alignment, not preprocessing-vs-architecture |
| Most comparisons rely on 9-subject BCI-IV 2a | Section 13.2 (Table 6), Section 13.5 (Table 7) | Yes | Full method-family comparison now also run on PhysioNet (n=109) and Cho2017 (n=52) |
| Comprehensive comparisons not reproduced on PhysioNet | Section 13.2 (Table 6, Table 13 appendix) | Yes | Every classical/domain-adaptation method and the self-supervised/SPDNet family (population-swap protocol) now evaluated at full PhysioNet scale |
| No individual preprocessing improvement reaches significance; cumulative improvement is significant | Section 10.1 (Table 3), Section 13.1 (Table 5), Section 5.1 (Table 1) | Yes | Factorial check shows neither window nor estimator effect survives Holm correction on either population; the cumulative FgMDM-to-ensemble result is significant only in-sample (p=0.0039) and not significant under the nested re-search (p=0.0742) |
| Limited deep-learning architecture coverage; no foundation-model comparison | Section 13.3 (Table 14), Section 13.4 | Yes | Three supervised architectures added (EEGNetv4, ShallowFBCSPNet, ATCNet); LaBraM tested frozen, with a positive control, and under two fine-tuning regimes; CBraMod/BIOT exclusions documented with specific reasons |
| Clarify zero-shot vs. transductive terminology | Section 3.1 | Yes | "Zero-label," "transductive," and "zero-shot" each defined once and used consistently; Introduction's protocol description corrected this pass to not imply EA/MEKT are strictly source-only |
| Methodological novelty relative to existing Riemannian/domain-adaptation work | Section 2 (Related Work), Section 13.3, Section 13.4 | Yes | MIRepNet and the EA-for-deep-decoders paper added, used to scope (not inflate) this paper's own findings |

## Items corrected in this closure pass specifically

1. **Critical Change 1** in `RESPONSE_TO_REVIEWER.md` previously said the full comparison "now
   also runs on PhysioNet's full 109-subject population" without qualifying that
   ShallowFBCSPNet/ATCNet ran only at n=30 and the self-supervised family used a population-swap
   protocol. Corrected to state the exact scope per method family, with section/table numbers.
2. **Critical Change 2** previously described the nested re-search as a single check. Corrected to
   describe both nested checks (the fixed 9-candidate check, which reproduces 52.39% and is not
   the primary estimate; the from-scratch 32-config per-fold search, which gives 49.94% and is the
   primary estimate), matching the equivalent fix made in Section 5.1 of the manuscript itself.
3. Every claim in the response letter was checked against the actual final-PDF section/table
   numbers extracted from the compiled `.aux` file (not assumed or hand-counted), so every
   "Section X, Table Y" citation in the letter resolves to the content described.

## What is not claimed

The review quoted in `RESPONSE_TO_REVIEWER.md` is presented as the actual verbatim text supplied
by the author for this revision (a full review-form export: Summary of Contributions, the
evidence/audience-interest questions and explanations, Critical changes 1–3, Suggested
improvements 1–2). It is not a reconstruction. No sentence in the response letter claims a result,
experiment, or scope broader than what Sections 4 (claim audit) and 5 (reviewer audit) above
verified against the manuscript.
