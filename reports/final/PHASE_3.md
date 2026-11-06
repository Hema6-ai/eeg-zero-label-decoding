# Phase 3: Hostile Reviewer Pass

Read the final PDF as two reviewers who disagree with the paper's claims. Five objections each,
quoted against the manuscript text that would draw them, classified, and (a)/(b) fixes applied
directly to `tmlr_2026.tex`.

## Reviewer 1: "The Methodological Rigor Skeptic"

**1. No omnibus/paper-wide multiple-comparisons control.**
Quote: every Holm-correction statement in the paper ("Holm-corrected across this 4-comparison
family", "Holm-Bonferroni corrected across all 8 comparisons tested", etc.) states only a
within-family correction; nowhere does the paper discuss the many separate small families
(m=2 to m=9) scattered across dozens of total comparisons, or note that family-wise error control
is not the same guarantee as a paper-wide one. Verified zero hits for "omnibus", "family-wise",
or "paper-wide" anywhere in the text.
**Classification: (b) needs an added disclosure. Applied** -- new Limitations paragraph states
this explicitly.

**2. Ensemble-member diversity is never examined.**
Quote: "An 8-member ensemble reaches 52.39%..." (Abstract, Table 1, Table 2) -- the paper reports
the ensemble's accuracy gain repeatedly but never reports pairwise agreement, error correlation,
or any other diversity measure between the 8 members' own predictions, so a reader cannot tell
whether the gain is genuine complementary information or mostly variance reduction among
similar, correlated classifiers.
**Classification: (b) for the disclosure (applied, new Limitations sentence); the full fix is (c)**
-- computing real pairwise prediction correlation requires re-running inference to save each
member's per-trial votes, which are not currently persisted anywhere in `paper/results/`. Not run
this pass. *Recommendation: add per-trial vote logging to `eval_mekt_ensemble.py` and report
pairwise agreement/correlation in a follow-up table; this is cheap (no retraining) once votes are
persisted.*

**3. The matched-protocol SPDNet hyperparameter search (Phase 1 of this pass) tunes on one
dataset and transfers the choice untested.**
Quote (pre-Phase-1): "SPDNet's configuration was fixed from the 9-subject setting and may be
under-capacity at 108 source subjects" (Limitations) -- the proposed fix substitutes one
untested, fixed configuration (9-subject-derived) for another (PhysioNet30-derived), without
separately validating the newly-selected configuration's fit to BCI-IV 2a's very different,
much smaller setting.
**Classification: (b) needs an added disclosure. Applied** -- see Phase 1 report; a caveat
sentence is added wherever the tuned configuration's results are reported.

**4. BCI-IV 2a (the least-powered population, n=9) anchors the paper's most-quoted numbers.**
Quote: the abstract's closing numbers, "An 8-member ensemble reaches 52.39%... a fully nested
re-search gives 49.94% instead, the estimate this paper reports throughout," are BCI-IV 2a
figures, reported ahead of the better-powered PhysioNet (n=109) results throughout the abstract
and title framing.
**Classification: (a) fixable by rewording. Applied** -- added a clause to the abstract's closing
sentence stating explicitly that the BCI-IV 2a figures carry the least statistical weight of the
three populations despite being reported first.

**5. "Ties with" claims at n=9 are not uniformly caveated as low-power null results.**
Quote: "EEGNetv4 ties the classical ensemble on two populations, BCI-IV 2a and PhysioNet" and
similar "ties"/"tied" statements (5 occurrences) are not every one individually flagged with the
power caveat given elsewhere ("unsurprising given how little power n=9 has," Section 13.3) --
a reader could read some ties as stronger evidence of true equivalence than the sample size
supports.
**Classification: (a)/(b). Applied** -- added one general Limitations sentence stating this
applies to every "ties"/"not significantly different" claim in the paper, rather than editing
each of the 5 occurrences individually (a more surgical per-sentence rewording was judged
unnecessary once the general caveat is stated plainly).

## Reviewer 2: "The External Validity / Deployment Skeptic"

**1. No training-time or compute-cost comparison across methods.**
Quote: "Deep-learning pretraining ran on a single Kaggle GPU (Tesla P100/T4, free tier); all
classical/domain-adaptation methods and ev[aluation]..." (Compute paragraph) states what hardware
was used but never compares relative training cost, so a reader cannot judge whether EEGNetv4's
accuracy edge at PhysioNet's full population (Section 13.3) is worth its cost relative to the
near-instant classical baselines.
**Classification: (b) needs an added disclosure. Applied** -- new Limitations sentence.

**2. The practical time/calibration burden of the transductive batch is defined but not costed.**
Quote: "We reserve 'zero-shot' specifically for describing the LaBraM comparison... and for
quoting other papers' own terminology" (end of the terminology note, Section 3) -- the zero-label
vs. transductive distinction is precisely defined, but the paper never states that collecting the
target's unlabeled batch before EA/MEKT can run is itself a real deployment cost, separate from
needing no label.
**Classification: (b) needs an added disclosure. Applied** -- new sentence added immediately
after the terminology note.

**3. No comparison to published state-of-the-art results.**
Quote: nowhere in the paper (0 hits for "state of the art" or "state-of-the-art"/"published
benchmark") is any external, previously-published accuracy number cited for BCI-IV 2a or
PhysioNet motor-imagery decoding, so a reader cannot calibrate whether this paper's ~45-53%
(3-4-class) numbers are competitive with or far behind the field.
**Classification: (c) needs a new experiment/literature verification. Not applied** -- adding
specific competing numbers risks citing unverified or stale figures without checking each one's
exact protocol match (class count, subject set, calibration data use), which this pass's time
budget does not allow to do carefully. *Recommendation: add a small table of 3-5 published
zero-/few-shot cross-subject results on BCI-IV 2a and PhysioNet in a follow-up pass, each
individually verified (DOI/arXiv) and protocol-matched before citing.*

**4. The "cross-subject generalization" framing is tested mostly within one hardware/lab per
dataset; cross-hardware transfer is only implicitly touched.**
Quote: "A single replication, however well-powered, leaves open whether the pattern is a property
of this paper's pipeline in general or an artifact shared by exactly two datasets. We evaluate a
third, independent population -- Cho2017..." (Section 13.5 opening) -- this motivates Cho2017 as
a third population, but does not state that Cho2017 also uses different recording hardware and
protocol from the other two, making it a (limited) cross-hardware check as well as a third
same-type replication.
**Classification: (a) fixable by rewording. Applied** -- new Limitations sentence states this
explicitly.

**5. EEGNetv4 was not given an equivalent hyperparameter search for Phase 1's fairness check.**
Quote: see Phase 1 report -- the directive for this pass explicitly asks for an equally-sized
small search for EEGNetv4 "if you can."
**Classification: (b), addressed in the Phase 1 writeup** -- an equal search was not run this
pass (compute budget already committed to the 36-configuration SPDNet grid); said so explicitly
in the Phase 1 report and in the manuscript (Section 13.3 addition, see `PHASE_1.md`).
