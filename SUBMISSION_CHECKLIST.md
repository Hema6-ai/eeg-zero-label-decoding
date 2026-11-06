# Submission Checklist (Paper #11562)

One row per reviewer comment: where it is addressed, the table/figure, the key number, and the
remaining limitation.

| Reviewer | Comment | Manuscript location | Table/Figure | Key number | Remaining limitation |
|---|---|---|---|---|---|
| ajnW | Summary of contributions (methodological novelty) | Section 1 | -- | -- | Contribution is a controlled comparison, not a new algorithm; stated directly |
| ajnW | Claims not convincingly supported (limited evidence on n=9) | Section 13.2 | Table 15 | MEKT default 51.54%±14.35% (PhysioNet, n=109) | Comprehensive comparison still uses population-swap pretraining for self-supervised/SPDNet families, not within-dataset training |
| ajnW | Critical change 1: extend to a larger/third dataset | Sections 13.2, 13.5 | Tables 15, 9 | Cho2017 EEGNetv4+EA 68.68%±12.99% vs. ensemble 66.12%±10.22%, p<0.0001 | ShallowFBCSPNet/ATCNet evaluated only at PhysioNet's n=30 subset, not n=109, for compute-budget reasons |
| ajnW | Critical change 2: strengthen preprocessing statistics | Sections 10.1, 13.1 | Tables 3, 5 | Time-window +2.01pp p=0.0273 (n=9), +0.88pp p=0.0383 (n=109), neither Holm-significant | Covariance-estimator and time-window effects are disclosed as not general; no omnibus correction across every family in the paper |
| ajnW | Critical change 3: scope vs. recent foundation models | Sections 13.3, 13.4, 14 | Table 16, Table "allmethods" (zero-shot LaBraM) | LaBraM frozen zero-shot 25.08%±1.74% (chance); matched SPDNet at n=109: 47.18%±12.16% untuned, 49.17%±12.80% tuned | CBraMod blocked by a verified upstream bug; BIOT excluded for channel-scheme incompatibility; RT-matched and EEGNetv4-under-population-swap controls not run |
| ajnW | Suggested improvement 1: zero-shot vs. transductive terminology | Section 3 (terminology note) | -- | -- | Practical calibration-time cost of the transductive batch is now disclosed but not quantified |
| ajnW | Suggested improvement 2: novelty vs. related Riemannian/domain-adaptation work | Section 13.4, Related Work | -- | -- | -- |
| c1um | Fairness of comparisons across model families (matched-protocol SPDNet) | Section 13.3 | Table 8 | Tuned matched SPDNet at n=109: 49.17%±12.80%, significantly below EEGNetv4 (-3.82pp, p<0.0001), MEKT (-2.37pp, p<0.0001), and the ensemble (-2.78pp, p<0.0001); ties EA+FgMDM (-0.26pp, p=0.4665) | Matched protocol removes the population-size asymmetry but not the self-supervised-objective difference; EEGNetv4-under-population-swap and RT-matched controls still not run; EEGNetv4 did not receive an equivalent hyperparameter search |
| c1um | Protocol summary table requested | Sections 13.2, Appendix A.6 | Tables 6, 17 | -- | Seeds and exact per-dataset source counts stated for every method |
| c1um | Typo: "properly- implemented" | Section 1 (Introduction) | -- | -- | Fixed |
| c1um | Typo: "Two observations" should read "Three" | Section 11 | -- | -- | Fixed |
| Internal QA | No omnibus multiple-comparisons control across the paper | Limitations (Section 14) | -- | -- | Disclosed; not fixable without a single paper-wide test design, which was judged out of scope |
| Internal QA | Ensemble-member diversity never examined | Limitations (Section 14) | -- | -- | Disclosed; full fix needs per-trial vote logging (not run) |
| Internal QA | No published state-of-the-art comparison | Limitations (Section 14) | -- | -- | Not added this pass -- needs individually verified citations, out of scope for this pass |
| Internal QA | No training-time/compute-cost comparison | Limitations (Section 14) | -- | -- | Disclosed; not quantified |
| Internal QA | Cho2017's cross-hardware role not stated explicitly | Limitations (Section 14) | Table 9 | -- | Disclosed; not analyzed specifically as a hardware-transfer test |
