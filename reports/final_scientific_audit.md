# Final scientific audit

Date: 2026-10-28. Produced for the master audit directive's Phases 4-6 (independent numerical
audit, methodological-weakness check, manuscript-accuracy revision). Everything below was either
re-run from scratch this session or read directly from the current source; nothing is copied from
a prior report without independent re-verification.

## 1. Issues found and fixed this session (new findings)

Three real, previously-undetected manuscript-accuracy issues were found by direct inspection and
fixed with minimal, targeted edits:

### 1.1 `fig:allmethods` caption contradicted its own plotted data
Caption claimed "every geometric or classical Riemannian method... does substantially better
[than raw-EEG methods], regardless of whether it is learned or hand-crafted." The figure's own
data (`paper/writeup/generate_manuscript_figures.py`) plots "Per-class EA+TS+shrinkage LDA" at
**25.41% ± 3.37%** -- at 4-class chance (25%), not "substantially better." Root cause (verified by
reading `paper/eval_riemannian_advanced.py::run_tslda_per_class_ea` directly): per-class EA
requires labels, so only source subjects get it; the held-out subject gets a mismatched,
label-free global EA, which is a real, already-understood (`reports/data_leakage_audit.md`)
negative result about that one alignment choice, not a failure of the Riemannian/classical family
in general. **Fix**: caption now names the exception and its documented cause instead of making an
absolute claim.

### 1.2 Centroid-distance claim overgeneralized beyond what was measured
Text claimed "every pair of classes is further apart under CSP than under any learned
representation." The underlying table (`tab:feature-stats`) only measured MAE-only, MAE+GRL,
MAE+CLUB+SCIN+VIB, and CSP -- SPDNet, the Riemannian Transformer, and SupCon were never put
through this specific diagnostic. **Fix**: reworded to "any of the raw-EEG representations
examined in this diagnostic," with an explicit parenthetical naming what was not measured.

### 1.3 Abstract bundled two preprocessing effects of different generality
The abstract's closing "central finding" sentence described the time-window and
covariance-estimator effects together, without noting that `sec:factorial`'s dedicated factorial
check (already in the body) found only the window effect to survive as an independent main effect
(p=0.027) while the estimator effect did not (p=0.82 for FgMDM/MEKT; p=0.26 for the SPDNet check
added this session). This risked a reader walking away thinking both effects were equally general.
**Fix**: added one sentence to the abstract making this distinction explicit, pointing to
`sec:factorial`.

No other instance of an absolute quantifier ("every"/"all") contradicted by its own data was found
in a full re-scan of Results, Discussion, Limitations, and every figure/table caption (grep for
`\bevery\b|\ball\b`, ~45 matches individually checked against the cited table/figure/computation).

## 2. Independent numerical re-verification (re-run fresh this session, not copied from a prior report)

| Check | Command | Result |
|---|---|---|
| Every aggregate claim vs. raw per-subject CSVs | `python -m paper.verify_all_claims` | **16/16 verified** (includes the two BCI-IV-2a SPDNet rows added this session) |
| Machine-readable claim-to-evidence audit | `python -m paper.build_claim_evidence_audit` | **46/46 rows VERIFIED**, 0 failures (extended this session to cover the factorial table, which previously had zero audit rows at all -- see Section 4 below) |
| BCI-IV 2a + PhysioNet statistical tests | `python -m paper.stats_analysis` | All 13 comparisons (5 + 8) reproduce deterministically, exact match to manuscript text |
| Factorial design (16 cells + SPDNet check) | `python -m paper.stats_factorial` | All 9 contrasts reproduce deterministically, exact match to `tab:factorial` |
| Jackknife robustness (4 key PhysioNet comparisons) | `python -m paper.robustness_subject_influence` | All 4 robust: significance conclusion unchanged under removal of any single subject |
| 52.39% headline ensemble | independent recomputation from `paper/results/zero_shot_mekt_ensemble.csv` | n=9, mean=52.392%, std=14.917% -- exact match |
| Subject 2's "never exceeds ~32%" claim (Ethics section) | independent scan of every BCI-IV-2a per-subject result file found in the repo (SPDNet lwf/corr, LaBraM, MEKT window variants, ensemble) | Max observed: 32.12% (MEKT window, lwf). Claim holds across every file checked, including the two result files added this session |

**52.53% confidence-weighted refinement**: confirmed this is consistently and correctly disclosed
in three places in the manuscript as "reported; implementation not available for independent
verification," and the base `paper/eval_mekt_ensemble.py` script's own docstring states the same.
Not presented anywhere as independently reproducible. No action needed.

## 3. Statistical-methodology check

- **Units of analysis**: every comparison treats the 9 (or 109) subjects as the paired,
  repeated-measures unit -- confirmed by reading `paper/stats_analysis.py`/`paper/stats_factorial.py`
  directly; no comparison collapses folds into independent observations.
- **Mean/SD/CI are not confused**: `tmlr_2026.tex` explicitly states "The ± in Table~\ref{tab:main-results}
  is the standard deviation across the 9 LOSO folds... not a confidence interval on the mean," and
  separately reports the bootstrap 95% CI. Verified this distinction is maintained everywhere a ±
  appears alongside a CI claim.
- **Nonsignificance is not presented as proof of equivalence**: checked every use of "indistinguishable"/
  "no difference"/"equivalent" in the manuscript (1 instance: "statistically indistinguishable from
  no effect at all," describing a failed-to-reject-null result, correctly hedged, not a claim of
  true equivalence).
- **Correction family sizes are internally consistent**: the factorial table's caption now says "7
  contrasts" (corrected from "6" when the SPDNet row was added this session) and `paper/stats_factorial.py`'s
  Holm-Bonferroni loop operates over exactly 7 valid-p rows (verified by counting output rows). The
  PhysioNet family's "8 comparisons" claim matches `paper/stats_analysis.py`'s 8 saved rows.
- **Power**: the manuscript already discloses, repeatedly, that the n=9 BCI-IV-2a estimates carry
  wide uncertainty (explicit 95% CI [42.8%, 62.3%] on the headline result) and frames the n=109
  PhysioNet replication as the better-powered confirmation -- this framing is accurate given the
  jackknife-robustness results in Section 2 above.
- **Not called a "complete" factorial when it isn't**: `sec:factorial`'s own text and
  `reports/factorial_experiment_report.md` both explicitly describe the deep-learning (SPDNet) arm
  as a reduced, default-window-only design, not a full 4-window grid -- re-confirmed this session,
  language unchanged from the already-correct prior session's disclosure.

## 4. Claim-evidence audit scope gap found and fixed

`paper/build_claim_evidence_audit.py` previously only pulled from `paper/verify_all_claims.CLAIMS`,
`statistical_tests_bci2a.csv`, `statistical_tests_physionet.csv`, and the robustness CSV --
**the entire factorial table (`factorial_statistical_analysis.csv`) had zero corresponding rows in
the audit**, despite backing manuscript numbers in `tab:factorial`. Fixed by adding a loop over
that file (mirroring the existing statistical-tests loop) and by adding the two new BCI-IV-2a
SPDNet claims to `verify_all_claims.CLAIMS`. Audit grew from 35 to 46 rows as a direct result; all
46 verify.

## 5. Zero-shot vs. transductive re-audit (direct code inspection, not prose-trusting)

Re-confirmed this session by re-reading the relevant `eval_*.py` files' `target_y` usage:

| Method class | Target data accessed before prediction | Classification |
|---|---|---|
| MAE/SupCon/SPDNet/RiemannianTransformer/LaBraM zero-shot eval | None (frozen encoder, fit only on source subjects) | Strict zero-shot |
| CSP/FgMDM/MDM (no EA) | None | Strict zero-shot |
| EA/RA-based methods | Target's own unlabeled trial covariances only (per-subject whitening, no cross-subject or label info) | Zero-shot-legal unsupervised alignment |
| MEKT, RPA | Target's unlabeled trials collectively (batch), iteratively pseudo-labeled | Transductive -- labeled as such throughout |
| Per-class EA+TS+LDA (Section 1.1 above) | Target gets global EA only; source gets per-class EA | Zero-shot-legal, but a train/test distribution mismatch, which is the method's own documented failure mode |

No label leakage, no cross-subject leakage into the target fold, no use of target outcomes beyond
the final scoring call, in any method. Consistent with the prior session's audit; re-verified by
direct code read, not assumed from that prior audit's conclusion.

## 6. Dataset/caching integrity (new check this session, re: the SPDNet-corr Kaggle run)

The final SPDNet `corr`-estimator training run pre-populated MOABB's local PhysioNet cache from a
third-party Kaggle dataset mirror (`gamalasran/physionet-eeg-motor-movement-imagery`) rather than
downloading from `physionet.org` directly, to work around that server's observed unreliability.
Checked this session:
- The mirror's directory layout (`files/SXXX/SXXXRXX.edf`) and subject count (109) were confirmed
  against `RECORDS` and the `S0*`/`S1*` subdirectory structure before use (done in a prior turn
  this session, via `kaggle datasets files`/`metadata`).
- License: ODC-By, openly stated on the Kaggle dataset listing.
- MOABB/`pooch` only skip-downloads a file if one already exists at the destination path and
  treats its hash as current -- meaning if the mirror's files were corrupted or mislabeled, the
  pipeline would silently train on bad data without a download error. **This is a residual,
  disclosed risk of the mirror-substitution approach** (not fully eliminable without
  re-downloading from the original source for comparison, which was the very problem being worked
  around) -- it is called out explicitly here rather than treated as watertight.
- The resulting SPDNet `corr` result (45.79% ± 14.33%) is consistent in magnitude and variance
  with every other BCI-IV-2a SPDNet-family number in this study, which is circumstantial but not
  conclusive evidence against silent data corruption.

## 7. Foundation-model (LaBraM) scope re-check

- Chance level for BCI-IV 2a (4-class) is 25%; LaBraM's 25.08% ± 1.74% is correctly described as
  "indistinguishable from 4-class chance."
- Checkpoint, channel-index mapping, and frozen-backbone/linear-probe setting were verified
  against LaBraM's original released source in a prior session (`get_input_chans`/
  `standard_1020`, reproduced verbatim from the official repository) -- re-confirmed present and
  unchanged in `paper/eval_foundation_model_labram.py` this session.
- Manuscript explicitly scopes the conclusion to "one checkpoint, one frozen linear probe, no
  fine-tuning... should not be read as a claim about LaBraM generally" -- present in both
  `sec:labram` and Limitations, re-verified this session.

## 8. No new experiments were judged necessary

Per directive Phase 5's instruction to run only experiments that resolve a real, otherwise-unresolved
scientific ambiguity: the one previously-open experimental gap (the SPDNet `corr`-estimator
factorial cell) was already completed earlier this session (see conversation history) and is now
fully integrated into the statistics, manuscript, and audit. No other item in Phase 5's checklist
(matched-data comparison, additional foundation model, hyperparameter-leakage audit, dataset
harmonization) surfaced a gap not already disclosed in the existing Limitations section -- see
Section 3's power/disclosure notes above and `reports/final_repository_inventory.md` Section 7.
