# Phase 4.4: Reproducibility Inventory

Every `script`, `stats_script`, `result_file`, and `prediction_artifact` field in
`reports/claim_evidence_audit.csv` (126 rows, covering every quantitative claim verified by
`paper/verify_all_claims.py` and the four `build_claim_evidence_audit*.py` scripts, including the
Phase 1 robustness-check extension) was parsed out into a flat list of script and result-file
names, then checked for existence against the repository.

- **31 generating scripts** referenced (e.g. `eval_mekt_ensemble.py`, `eval_factorial.py`,
  `eval_spdnet_matched_protocol.py`, `eval_spdnet_matched_protocol_tuned.py`,
  `tune_spdnet_matched_protocol.py`, `train_spdnet.py`).
- **54 result/prediction CSV files** referenced under `paper/results/`, including this check's
  own new files (`spdnet_hparam_search_physionet30.csv`,
  `e1_dlloso_spdnet_tuned_ea_default_seed{42,0,123}_*.csv`,
  `stats_spdnet_tuned_protocol.csv`).

**85 total referenced items checked; 84 present, 1 apparent gap.** The one apparent gap
(`_full.csv`) is a parsing artifact of this check's own comma-splitting regex choking on a
filename containing a brace-expansion-style seed list
(`e1_dlloso_spdnet_matched_ea_default_seed{42,0,123}_physionet_full.csv`) inside one audit row's
free-text field -- the three real per-seed files it refers to are already separately listed and
confirmed present (both the untuned and tuned variants).

**Status: every script and result file needed to reproduce every number in the paper, including
the Phase 1 hyperparameter robustness check, is present in the repository.**

This check is a repository-presence check, not a re-run of the pipeline end to end; the separate
`verify_all_claims.py`/`build_claim_evidence_audit*.py` scripts (Phase 2.9) independently confirm
that the numbers these files contain recompute to the numbers printed in the manuscript.
