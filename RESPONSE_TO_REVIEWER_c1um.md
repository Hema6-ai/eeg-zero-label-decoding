# Response to Reviewer c1um (Paper #11562)

We thank the reviewer for the careful, specific suggestion and for catching two typos we had
missed.

## Requested change: fairness of comparisons across model families

*"Improve the fairness of comparisons across model families (Sections 13.2-13.3). In my
understanding, some deep learning models are pretrained on only 9 BCI-IV 2a subjects when
evaluated on PhysioNet, while classical methods and EEGNetv4 can use up to 108 source subjects.
This makes it difficult to attribute performance differences to model architectures. I suggest
adding a controlled comparison where at least one geometric deep learning model (e.g., SPDNet) is
trained using the same source subjects and supervision as EEGNetv4. At minimum, the authors should
provide a table summarizing the training data, supervision, and adaptation protocols for each
method."*

We trained SPDNet end-to-end, supervised, from scratch, on source subjects only, under the exact
LOSO fold structure, supervision, and EA/window sweep as EEGNetv4 (`eval_spdnet_matched_protocol.py`),
at all three scales. On BCI-IV 2a (n=9) and PhysioNet30 (n=30), both models are compared at seed
42; on PhysioNet-full (n=109), both are compared as an average across seeds 42/0/123, matching
EEGNetv4's reduced 3-seed scope at that scale. Matched SPDNet and EEGNetv4 use the same seed set
in every comparison below.

| Dataset (n), matched SPDNet acc. | Comparison | Diff (pp) | Raw p | Holm-adj. p |
|---|---|---|---|---|
| BCI-IV 2a (n=9), seed 42, 45.64%±12.96% | vs. EEGNetv4 | -1.14 | 0.5703 | 0.5703 |
| | vs. EA+FgMDM | -1.81 | 0.2500 | 0.5000 |
| | vs. MEKT | -2.74 | 0.0742 | 0.2969 |
| | vs. nested ensemble | -4.30 | 0.0977 | 0.2969 |
| | vs. original-protocol SPDNet | +1.52 | 0.2626 | -- (outside family) |
| PhysioNet30 (n=30), seed 42, 45.30%±11.76% | vs. EEGNetv4 | -0.11 | 0.9053 | 0.9053 |
| | vs. EA+FgMDM | -2.73 | 0.0478 | 0.0956 |
| | vs. MEKT | -3.62 | 0.0157 | 0.0629 |
| | vs. full ensemble | -4.96 | 0.0293 | 0.0880 |
| | vs. original-protocol SPDNet | -2.19 | 0.1512 | -- (outside family) |
| PhysioNet-full (n=109), seeds 42/0/123 avg., 47.18%±12.16% | vs. EEGNetv4 | -5.81 | <0.0001 | <0.0001 |
| | vs. EA+FgMDM | -2.26 | 0.0006 | 0.0006 |
| | vs. MEKT | -4.36 | <0.0001 | <0.0001 |
| | vs. full ensemble | -4.77 | <0.0001 | <0.0001 |
| | vs. original-protocol SPDNet | -2.22 | 0.0010 | -- (outside family) |

Within each dataset, the four comparisons against EEGNetv4, EA+FgMDM, MEKT, and the ensemble
(nested for BCI-IV 2a, full for PhysioNet) form one Holm-corrected family (m=4); both raw and
Holm-adjusted p are shown. The comparison against original-protocol SPDNet is a single
comparison, outside that family, reported uncorrected.

At the two smaller scales, none of the four within-family comparisons survive Holm correction. At
the full 109-subject scale, all four are Holm-significant, and matched-protocol SPDNet is below
every one of them. Against original-protocol SPDNet, the difference is not significant at the two
smaller scales but is significant at the full scale (-2.22pp, p=0.0010): the protocol change
itself produces a small but real effect once enough source subjects are available.

This does not change the paper's claim. "Geometric deep learning does not beat classical
geometry" (Section 9) continues to hold under the matched protocol at every scale tested. Under
the matched protocol, SPDNet's point estimate is below EA+FgMDM, MEKT and the ensemble at every
scale; the difference is significant after Holm correction only at n=109. The EEGNetv4 tie
reported at n=9 and n=30 source subjects per fold does not hold at 108 source subjects per fold:
the apparent SPDNet/EEGNetv4 parity at the two smaller scales is a small-source-population effect,
not a general property of the matched protocol, as stated directly in the manuscript (Section
13.3).

**Follow-up robustness check: does hyperparameter tuning change this?** The configuration above
was fixed in advance, not tuned, so we ran a small search for matched-protocol SPDNet (BiMap
output dimension {8, 12, 16, 22}, depth {1, 2, 3}, learning rate {1e-3, 3e-4, 1e-4}; 36
configurations), selected purely on PhysioNet30 source-validation accuracy (never target
accuracy, on a fixed 10-subject fold subset), and reran all three scales with the selected
configuration (dims (22,22), lr=1e-3). Matched SPDNet and EEGNetv4 use the same seeds in every
comparison here too: seed 42 at n=9 and n=30, seeds 42/0/123 averaged at n=109.

| Scale | Tuned SPDNet acc. | Comparison | Diff (pp) | Raw p | Holm-adj. p |
|---|---|---|---|---|---|
| BCI-IV 2a (n=9) | 47.76%±12.06% | vs. EEGNetv4 | +0.98 | 0.6523 | 1.0000 |
| | | vs. EA+FgMDM | +0.31 | 0.9102 | 1.0000 |
| | | vs. MEKT | -0.62 | 0.7344 | 1.0000 |
| | | vs. nested ensemble | -2.18 | 0.3594 | 1.0000 |
| | | vs. untuned matched SPDNet | +2.12 | 0.0742 | -- (outside family) |
| PhysioNet30 (n=30) | 43.86%±9.94% | vs. EEGNetv4 | -1.55 | 0.3676 | 0.3676 |
| | | vs. EA+FgMDM | -4.17 | 0.0010 | 0.0042 |
| | | vs. MEKT | -5.06 | 0.0016 | 0.0048 |
| | | vs. full ensemble | -6.40 | 0.0035 | 0.0070 |
| | | vs. untuned matched SPDNet | -1.44 | 0.4236 | -- (outside family) |
| PhysioNet-full (n=109) | 49.17%±12.80% | vs. EEGNetv4 | -3.82 | <0.0001 | <0.0001 |
| | | vs. EA+FgMDM | -0.26 | 0.4665 | 0.4665 |
| | | vs. MEKT | -2.37 | <0.0001 | <0.0001 |
| | | vs. full ensemble | -2.78 | <0.0001 | <0.0001 |
| | | vs. untuned matched SPDNet | +1.99 | 0.0002 | -- (outside family) |

Within each dataset, the four comparisons against EEGNetv4, EA+FgMDM, MEKT, and the ensemble form
one Holm-corrected family (m=4), consistent with the first table; the comparison against the
untuned configuration is a single comparison, outside that family, reported uncorrected.

At n=109, tuning is a significant improvement over the untuned configuration (+1.99pp,
p=0.0002), and tuned matched SPDNet remains significantly below EEGNetv4, MEKT, and the ensemble;
it is no longer significantly below EA+FgMDM specifically. The core finding holds: tuning narrows
but does not close the gap to EEGNetv4, MEKT, or the ensemble at this scale. Tuning did not
transfer uniformly, however: at n=30 -- the scale the search itself was tuned on -- it is
nominally worse than the untuned configuration against every baseline (not significant), and
EA+FgMDM/MEKT/the ensemble become Holm-significant there for the first time. We report this as a
limitation of this kind of small validation-accuracy-selected search in general, not only of
SPDNet.

We did not run an equivalent search for EEGNetv4; its hyperparameter space (kernel sizes, filter
counts, dropout) is not directly comparable to SPDNet's. This is stated in Limitations
(Section 14).

We have added Table 6 (Section 13.2) and Table 17 (Appendix Section A.6, "Full training protocol
table") giving the exact pretraining/downstream data, supervision, alignment, target-data access,
and model-selection procedure for every method in the paper, as requested.

We did not run EEGNetv4 under SPDNet's population-swap protocol or a matched-protocol Riemannian
Transformer. Each requires a training campaign comparable to the SPDNet check. We list both in
Limitations (Section 14).

**Remaining limitation.** The self-supervised methods (MAE variants, SupCon) still pretrain on an
upstream corpus with a self-supervised or contrastive objective, not the supervised, source-only
training EEGNetv4 and the matched-protocol architectures use; this difference in *objective*, not
just population size, is not resolved by the matched-protocol check and is stated plainly in
Limitations (Section 14).

## Typos

*"Page 2 'properly- implemented' -> remove blank. Page 15 'Two observations generalize across
this sweep' but there are actually three."*

Both fixed:
- The stray space after "properly-" (a line-wrap artifact in the LaTeX source that rendered as a
  visible gap in the compiled PDF) is fixed by joining the hyphenated compound onto one source
  line so it no longer breaks at that point.
- "Two observations" is corrected to "Three observations" (Section 11), matching the "First,
  Second, Third" list that follows it.

We additionally checked every other count claim in the manuscript against the list it refers to
(four method families, three populations, two exceptions, two LaBraM fine-tuning regimes, three
supervised architectures, ten follow-up extensions), ran a duplicated-word check, a common-typo
scan, and a domain-aware spell-check pass over the full manuscript; no further mismatches or
misspellings were found.

## Summary of changes by section

| Section | What changed | Table(s) |
|---|---|---|
| Section 13.2, opening | States the protocol difference (classical/EEGNetv4 trained on source subjects directly vs. self-supervised/geometric methods via population-swap pretraining) before the results | Table 6 |
| Section 13.2 | Added a compact protocol table (every method's pretraining data, frozen/trained status, target-data access, model selection) | Table 6 |
| Appendix, Section A.6 | Added the full per-dataset protocol table with exact source-subject counts and seeds | Table 17 |
| Section 9 (text), Contributions | Added a protocol qualifier to the "geometric deep learning does not beat classical geometry" claim. Title unchanged: matched-protocol SPDNet is never significantly above MEKT, EA+FgMDM, or the ensemble at any scale tested (point estimates vary and are nominally above EA+FgMDM and EEGNetv4 for the tuned configuration at n=9, but not significantly), and is significantly below EEGNetv4 at the full 109-subject scale | -- |
| Section 13.3 | Split into three parts: what was trained and how, the results table, and the interpretation | Table 8 |
| Section 13.3 | Added a hyperparameter-tuning robustness check for matched-protocol SPDNet (36-configuration search, selected on PhysioNet30 validation accuracy only) at all three scales | -- |
| Section 11, Section 1 | Typo fixes | -- |
| Limitations (Section 14) | Added sentences on the remaining cross-family protocol difference, the two controls not run, and that EEGNetv4 did not receive an equivalent hyperparameter search; removed the earlier "may be under-capacity" caveat now that tuning has been checked directly | -- |
| Abstract, Conclusion | Updated to state the SPDNet results by protocol (original, matched, tuned) | -- |

All reported numbers were recomputed from the per-subject result files by scripts in the
anonymized repository.
