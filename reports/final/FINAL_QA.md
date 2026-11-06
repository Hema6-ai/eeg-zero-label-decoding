# Final QA: justified exceptions to the zero-hits process-language/path rule (Phase 2.5)

Each item below is a literal hit for one of the flagged terms (previously, originally, earlier,
newly, surfaced, revisit, this round, reviewer, revision, honest, genuine, notably, session,
flaky, retried, Kaggle, file paths, script names) that was reviewed and kept because it is
legitimate paper content, not revision-process narration.

1. **"session"** (`tmlr_2026.tex`, Ethical Considerations): "a lengthy calibration *session* is
   itself burdensome." Ordinary English describing a BCI calibration session for end users;
   unrelated to the paper's own writing/editing process.

2. **"retried"** x2 (`tmlr_2026.tex`, reproducibility/compute disclosures): transient
   `physionet.org` download timeouts during data collection, and a subject-level retry during
   MAE-only's own training run. Both describe real experimental infrastructure reported as a
   methodological disclosure, not this paper's revision history.

3. **"Kaggle"** (`tmlr_2026.tex`, Compute disclosure): "Deep-learning pretraining ran on a single
   Kaggle GPU (Tesla P100/T4, free tier)." A standard compute-resources statement with no account
   name, username, or other identifying string attached.

4. **Bare script filenames** (`.py`/`.csv`, no directory path) in the manuscript body:
   `eval_mekt_ensemble.py`, `eval_e4_nested_selection.py`, `eval_e7_confidence_weighted_fusion.py`
   (pre-existing reproducibility citations), `stats_eegnet_comprehensive.csv`,
   `stats_e1_all_architectures.csv` (this round's fairness-check discussion). Each names the exact
   script/file that reproduces one specific number, with the `paper/`/`reports/` directory prefix
   removed. Kept as a reproducibility pointer; removing the bare filename entirely was judged to
   lose real reproducibility value for no anonymity gain (no author, account, or path is
   revealed by a bare filename alone).

5. **"reviewer"** throughout both response letters. Each letter is a response addressed to a
   named reviewer ID (ajnW, c1um), quoting their review comments verbatim. The word is
   structurally required; removing it would make the letters unreadable as responses.

No other hit was found for any flagged term in the compiled PDF text, the `.tex` source, or
either response letter.

## Phase 4.2 letter-specific exceptions

6. **"revision"** x2 in `RESPONSE_TO_REVIEWER.md`'s opening ("This revision changes the paper's
   central claim"; "This is the central finding of the revision"). A response-to-reviewers
   letter's subject matter *is* the revision being described; this is standard review-response
   language, not internal workflow narration, and removing it would make the letter read
   strangely. **Justified, kept.**

7. **One bare `.py` filename** in `RESPONSE_TO_REVIEWER_c1um.md`
   (`` `eval_spdnet_matched_protocol.py` ``), no directory path. Kept per the explicit standing
   instruction that bare, real script names naming the exact reproducing script are allowed in
   both the manuscript and the letters, only directory-revealing paths (`paper/...`,
   `reports/...`) are removed.

## Phase 1 (hyperparameter robustness check) exception

8. **"genuinely"** (`tmlr_2026.tex`, new Limitations sentence on ensemble-member diversity):
   "...whether it helps mainly by averaging over correlated, similar errors or by combining
   genuinely different error patterns is not established here." Describes the ensemble's error
   statistics, not a claim about this paper's own process. **Justified, kept.**

## Final cleanup pass: re-grep for judged/out of scope/retried/timeouts/historical/independently
verified/re-run/".py"/field convention/honest/reviewer/round/pass

Zero hits for: judged, out of scope, timeouts, historical, independently verified, ".py", field
convention, honest, reviewer. Three terms had hits, all legitimate methodology, not revision
language:

9. **"retried"** (1, pre-existing, already logged as item 2 above): MAE-only's own training
   methodology (a subject-level retry during that experiment).

10. **"re-run"/"re-running"** (5): all describe the paper's own nested cross-validation
    *method* -- "re-running the entire window/estimator/method search per outer fold" is the
    literal definition of the nested check this paper reports, not a statement about editing the
    manuscript. **Justified, kept.**

11. **"round"** (3): "a second round of test-set tuning" (methodology, avoiding double-dipping)
    and two false-positive substring matches inside "worked around." **Justified, kept.**

12. **"pass"** (4): all four are "forward pass" (standard neural-network terminology), unrelated
    to a "revision pass." **Justified, kept** (coincidental substring match, not a real hit).
