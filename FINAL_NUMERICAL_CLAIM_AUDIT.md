# Final Numerical Claim Audit — Paper #11562

Every central numerical claim in `tmlr_2026.tex` was traced to a producing script and result
file, then independently recomputed or re-executed this session. No numerical claim was changed
based on this audit; two presentation/labeling errors were found and fixed (Section 3 below), and
numbers moved location (main text → appendix) without their values changing (see
`FINAL_SUBMISSION_AUDIT.md` Section 6 for the page-layout consequences).

## 1. `verify_all_claims.py` — independent recomputation from raw per-subject data

Rerun this session. **16/16 claims verified**, covering: every PhysioNet comprehensive-comparison
method (CSP+LDA, CSP+SVM, FBCSP+SVM, MDM, FgMDM, RPA, MEKT default/tuned, SPDNet, Riemannian
Transformer v2, SupCon, MAE+GRL, MAE-only), the LaBraM BCI-IV 2a zero-shot result, and both SPDNet
factorial-check rows (`lwf` baseline and `corr` check). Each reported mean accuracy, standard
deviation, and Cohen's kappa was recomputed directly from the per-subject result files and matched
to the precision printed in the manuscript.

## 2. `build_claim_evidence_audit.py` + `build_claim_evidence_audit_r2.py` — full claim-to-evidence map

Rerun this session, writing 98 rows to `reports/claim_evidence_audit.csv` (kept local; see
Submission Package note in `FINAL_SUBMISSION_AUDIT.md`). Breakdown:

- 97 rows: **VERIFIED**, either by rerunning the producing script fresh this session with an
  exact match to the manuscript's printed numbers, or by direct arithmetic/recomputation from the
  relevant source CSV.
- 1 row: **PARTIALLY VERIFIED**, disclosed, not hidden — the 4-seed bootstrap ensemble stability
  claim (mean $51.98\%\pm0.27\%$, Table 15 / Section 12). The arithmetic (mean and standard
  deviation of the four documented per-seed values) was checked and matches exactly. The
  underlying per-seed bootstrap-resampling computation itself predates this revision and has no
  standalone script in the current repository that can be rerun end-to-end to regenerate those
  four per-seed numbers from scratch; only the aggregation on top of them was independently
  recomputed this session. This is a real traceability gap, stated here plainly rather than
  claimed as fully reproduced.

No row failed or mismatched.

## 3. Presentation/labeling corrections made this revision (not value changes)

| Correction | Before | After | Where |
|---|---|---|---|
| Holm-significance label, BCI-2a window effect | Marked "Holm-sig: yes" at raw $p=0.0273$, which is mathematically impossible for any stated family size $\geq 5$ | Marked "no," Holm-adjusted $p=0.1365$ at the stated $m=5$ primary family | Table 3 |
| ShallowFBCSPNet vs. nested-ensemble significance, PhysioNet30 | A stale cached CSV value marked this comparison significant ($p=0.0267$ at $m=2$) | Corrected after rerunning the generating script fresh: not significant (Holm-adjusted $p=0.0533$) | Section 13.3 text, `paper/results/stats_e1_all_architectures.csv` |

Both were labeling/significance-verdict errors, independently reproducible as errors by
recomputing Holm-Bonferroni from the stated raw $p$-values and family sizes; neither changed an
underlying accuracy, $p$-value, or effect-size number.

## 4. Headline-number consistency check (49.94% vs. 52.39%)

Checked every location in the manuscript where either number appears (26 occurrences of 52.39%,
13 of 49.94%, counted by direct text search on the compiled `.tex` source). Every location that
*introduces* the number, or restates it inside a table/figure caption readable on its own
(abstract, Algorithm 1, Figure 4, Figure 5, Table 1, Table 2, Table 4, Table 17, Section 5.1,
Conclusion), carries an explicit "in-sample/optimistic" or "nested/primary estimate" label. A
handful of later narrative mentions in Sections 10 and 12 (e.g. "reached 52.39%, the best result
in this study") use the bare number without repeating the qualifier at that exact sentence; each
occurs after Section 5.1 has already disclosed the distinction, so this is a stylistic choice
against redundancy rather than a missing disclosure, but it means not literally every one of the
39 occurrences carries its own inline label. Three caption-level locations were missing the label
entirely before this pass (Figure 4, Figure 5, Table 17) and were fixed (see
`FINAL_SUBMISSION_AUDIT.md` Section 2, items 5 and 7). Table 4's cross-dataset comparison was
extended to report PhysioNet's full-population ensemble result relative to both reference points
($-0.44$pp vs. the in-sample $52.39\%$; $+2.01$pp vs. the primary $49.94\%$), rather than only the
former.

## 5. Statistical-test family audit

Re-verified from `reports/revision_r2/HOLM_RECHECK.md`'s underlying work (produced during an
earlier pass this project; independently re-confirmed here rather than taken on faith): every
Holm-Bonferroni family in the manuscript (Table 3's primary 5-contrast and supplementary
2-contrast families; Table 5's 5-contrast PhysioNet-replication family; Table 6's 8-comparison
PhysioNet family; the EEGNetv4-alignment 2-comparison family; the PhysioNet/Cho2017
alignment-ablation 4-comparison family; the Cho2017 EEGNetv4-vs-classical 3-comparison family)
was recomputed directly from its stated raw $p$-values and family size, and matches the
manuscript's reported Holm-adjusted $p$-values and significance verdicts.

## 6. Reproducibility status of each experiment class

| Experiment class | Reproducible from this repo | Compute requirement |
|---|---|---|
| Classical/domain-adaptation methods (CSP, MDM, FgMDM, MEKT, RPA), all three datasets | Yes, deterministic | CPU |
| In-sample 8-member ensemble (52.39%) | Yes, deterministic | CPU |
| Nested re-search, primary estimate (49.94%) | Yes, deterministic | CPU |
| Confidence-weighted fusion refinement (52.53%) | Yes | CPU |
| Self-supervised/geometric deep learning (MAE variants, SupCon, SPDNet, Riemannian Transformer) | Yes, given GPU | GPU (ran on Kaggle free tier) |
| Supervised architectures (EEGNetv4, ShallowFBCSPNet, ATCNet) | Yes, given GPU | GPU |
| LaBraM frozen probe | Yes | CPU-feasible |
| LaBraM fine-tuning (2 regimes), positive control | Yes, given GPU | GPU |
| CBraMod | Fails reproducibly with the documented upstream bug; not a result to reproduce | N/A |
| BIOT | Not attempted (verified channel-scheme incompatibility, stated as such, not run) | N/A |
| 4-seed bootstrap ensemble stability aggregation | Partially — see Section 2 above | CPU |

No code defect was found that invalidates a reported result. No experiment required rerunning
from scratch to correct a value; all reruns performed were re-executions of existing scripts to
confirm they still reproduce the numbers already in the manuscript.
