# Phase 1: One Robustness Check (SPDNet Tuning)

## Search

36-configuration grid (BiMap output dimension {8, 12, 16, 22} x depth {1, 2, 3} x learning rate
{1e-3, 3e-4, 1e-4}) for matched-protocol SPDNet, selected purely on PhysioNet30
source-validation accuracy (never target/held-out-subject accuracy), on a fixed 10-subject fold
subset (every 3rd subject of PhysioNet30's 30-subject list, sorted) to keep the search
tractable. Full results: `paper/results/spdnet_hparam_search_physionet30.csv`.

**Selected configuration: BiMap dims (22, 22) (depth 1, no reduction), lr = 1e-3** (mean
source-validation accuracy 50.15% on the search subset).

## Full reruns with the selected configuration

Primary comparison cell only per dataset (seed 42, EA, default window for BCI-IV 2a/PhysioNet30;
seeds {42, 0, 123} average for PhysioNet-full) -- matching exactly what Table 8 reports, not the
full 5-seed x 2EA x 2window grid, which would cost as much compute as the search itself.

- BCI-IV 2a: local download of `lampx.tugraz.at` failed repeatedly (read timeouts); moved to a
  Kaggle kernel (`sweety611/eeg-spdnet-tuned-protocol-bci2a`), which completed cleanly: 9/9 folds.
- PhysioNet30: ran locally without issue (data already cached from the search): 30/30 folds.
- PhysioNet-full: local run hit intermittent `physionet.org` timeouts on ~20 of 109 subjects
  (different subjects failing across attempts); the resulting per-seed files were incomplete
  (88/109) and were discarded rather than used. Moved to a Kaggle kernel
  (`sweety611/eeg-spdnet-tuned-protocol-physionet-full`), which completed cleanly: 109/109 folds
  for all three seeds (~5 hours total runtime). Both kernels' outputs were pulled by paginating
  Kaggle's `list_kernel_session_output` API directly (the default single-page CLI/API call missed
  the genuinely new `results/` files, sorted alphabetically onto a later page behind hundreds of
  copied input-dataset files -- this was the same failure mode diagnosed in an earlier round, now
  fixed by paginating explicitly).

## EEGNetv4 did not get an equivalent search

Per the directive's "if not, say so": not run. EEGNetv4's hyperparameter space (kernel sizes,
temporal/spatial filter counts, dropout) is not directly comparable to SPDNet's (BiMap
dimension/depth/learning rate), and designing a fair equivalent 36-point search was judged out of
scope alongside the SPDNet search and its Kaggle fallback runs. Stated in Limitations (Section 14).

## Results

| Scale | Tuned SPDNet acc. | vs. EEGNetv4 | vs. EA+FgMDM | vs. MEKT | vs. ensemble | vs. untuned matched | vs. original-protocol |
|---|---|---|---|---|---|---|---|
| BCI-IV 2a (n=9) | 47.76%±12.06% | +0.98pp, p=0.6523 | +0.31pp, p=0.9102 | -0.62pp, p=0.7344 | -2.18pp, p=0.3594 (nested) | +2.12pp, p=0.0742 | +3.65pp, p=0.0195 |
| PhysioNet30 (n=30) | 43.86%±9.94% | -1.55pp, p=0.3676 | **-4.17pp, p=0.0010** | **-5.06pp, p=0.0016** | **-6.40pp, p=0.0035** (full) | -1.44pp, p=0.4236 | -3.63pp, p=0.0101 |
| PhysioNet-full (n=109) | 49.17%±12.80% | **-3.82pp, p<0.0001** | -0.26pp, p=0.4665 | **-2.37pp, p<0.0001** | **-2.78pp, p<0.0001** | **+1.99pp, p=0.0002** | -0.23pp, p=0.8814 |

Bold = Holm-significant within its 4-comparison family (EEGNetv4/EA+FgMDM/MEKT/ensemble, per
dataset). The "vs. untuned" and "vs. original-protocol" columns are single comparisons, not
Holm-corrected. Full table: `paper/results/stats_spdnet_tuned_protocol.csv` (18 rows).

## What changed and what did not

**The core finding at n=109 holds, with one real nuance.** Tuning significantly improves
matched-protocol SPDNet at the full-population scale (+1.99pp over the untuned configuration,
p=0.0002) and it remains significantly below EEGNetv4, MEKT, and the ensemble -- the same
qualitative story as the untuned check. The nuance: under the untuned configuration, matched
SPDNet was also significantly below EA+FgMDM (-2.26pp, p=0.0006); under the tuned configuration
it is not (-0.26pp, p=0.4665). This is reported honestly in the manuscript (Section 13.3,
"Robustness to hyperparameter tuning") rather than rounded off in either direction.

**Tuning did not transfer uniformly across scales.** At BCI-IV 2a (n=9), tuning helped (nominally
ties or beats every baseline, though none of these were significant even before tuning given the
low power at n=9). At PhysioNet30 (n=30) -- the scale the search was actually tuned on -- tuning
made things nominally *worse* against every baseline (not significant, p=0.4236 vs. the untuned
configuration), and EA+FgMDM/MEKT/the full ensemble became Holm-significant there for the first
time (they were only marginal under the untuned configuration). This is reported as a limitation
of small validation-accuracy-selected searches in general, not specific to SPDNet.

**Also discovered and fixed in the course of this check**: the abstract and conclusion's claim
that "SPDNet is significantly below EA+FgMDM, MEKT, and the ensemble... under both its original
and a matched... protocol" was imprecise even before considering the tuned configuration --
recomputing directly, **original-protocol SPDNet ties EA+FgMDM** (-0.03pp, p=0.6192, not
significant) and is significantly below only MEKT (-2.14pp, p=0.0002) and the ensemble (-2.55pp,
p=0.0001). This was an error in an earlier pass of this same final cleanup (added without
verifying against the actual original-protocol-vs-FgMDM numbers), caught and fixed while
preparing this check's numbers. The abstract and conclusion now state the accurate, protocol-by-
protocol picture.

## Manuscript changes

- Section 13.3: new "Robustness to hyperparameter tuning" paragraph (full search description,
  tuned results at all three scales, the honest non-uniform-transfer finding).
- Limitations (Section 14): "may be under-capacity" caveat deleted (the robustness check now
  directly addresses it); replaced with a factual statement of what was searched, what was
  selected, and that EEGNetv4 did not get an equivalent search.
- Abstract and Conclusion: corrected to the precise, protocol-by-protocol claim (see above).
- `paper/verify_all_claims.py`: extended to 23/23 verified claims (4 new: BCI-2a tuned, PhysioNet30
  tuned, PhysioNet-full tuned seed-average, plus the pre-existing untuned entries).
- `paper/build_claim_evidence_audit_phase1.py` (new): 7/7 verified rows appended to
  `reports/claim_evidence_audit.csv`, covering every paired comparison quoted in the new
  paragraph.
- Total audit: 126/126 rows verified across all four audit scripts (51 + 52 + 16 + 7).
