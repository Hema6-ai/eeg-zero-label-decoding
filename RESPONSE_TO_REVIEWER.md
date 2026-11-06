# Response to Reviewer ajnW (Paper #11562)

We thank the reviewer for a careful reading. The paper's central claim has changed.
The submitted version argued that two preprocessing choices, time-window selection and
covariance estimation, matter more than architecture. We no longer make that claim. Testing the
same preprocessing choices on a second, 109-subject population, PhysioNet, shows that the
time-window effect is nominally positive on both datasets but does not survive Holm correction on
either; the covariance-estimator effect is null on both (Section 10.1, Table 3; Section 13.1,
Table 5). The claim that does hold up is narrower: alignment (Euclidean Alignment / domain
adaptation) is the lever that generalizes across populations and method families, with two
disclosed exceptions (Section 13.3). We also found that our original headline accuracy was
optimistic: it was selected on the same subjects it was evaluated on, and a fully nested
re-search gives a number 2.4-3.1 points lower, which we now report throughout (Section 5.1, page
9, Table 1). We address each point below, quoting the review in italics.

TMLR's review form asks reviewers to judge whether the claims are correct and whether the
findings would interest the journal's audience; both are addressed directly in what follows.

## Summary of contributions

*"This paper investigates zero-shot cross-subject motor imagery decoding by comparing
self-supervised representation learning, geometric deep learning, and classical Riemannian
approaches. The authors report that preprocessing choices, particularly time-window selection and
covariance estimation, provide greater performance improvements than the architectural
modifications examined. The study includes extensive negative results and additional validation
on PhysioNet. The topic is relevant, and the systematic comparison is valuable. However, the
methodological novelty is limited, and the experimental evidence does not fully support the
central conclusions."*

We agree with the evidence concern and have changed the central claim accordingly (see above). On
methodological novelty: the paper's contribution is the controlled comparison itself (the same
protocol applied across method families, preprocessing choices, and three independent
populations), not a new algorithm; we state this directly in the introduction (Section 1) rather
than implying otherwise.

## Are the claims supported by accurate, convincing, and clear evidence? (reviewer answered "No")

*"The main concern is the limited experimental evidence supporting the central claims. Most
comparisons rely on BCI Competition IV Dataset 2a, involving only nine subjects, which limits the
reliability of conclusions given substantial inter-subject variability. Although the final
pipeline is evaluated on PhysioNet with 109 subjects, the comprehensive comparisons across method
families are not reproduced on this larger dataset. Furthermore, none of the individual
preprocessing improvements achieves statistical significance in the reported paired tests,
although the cumulative improvement is significant. Consequently, the results support the
effectiveness of the proposed pipeline but do not convincingly establish that preprocessing
choices are generally more important than model complexity."*

This is the paper's central finding now. We extended the full method-family comparison, not
just the final pipeline, to PhysioNet's 109-subject population (Section 13.2, Table 15), and
separately re-ran the exact time-window/covariance-estimator factorial design used on BCI-IV 2a
at PhysioNet scale (Section 13.1, Table 5). The covariance-estimator effect stays null on both
populations (Table 3, Table 5). The time-window effect is nominally significant on both (raw
p=0.0273 at n=9, Table 3; raw p=0.0383 at n=109, Table 5) but, once corrected for multiple
comparisons on the same 5-contrast family on both datasets, is not significant on either. We also
added a third, independent population (Cho2017, 52 subjects, different hardware and protocol;
Section 13.5, Table 9) and trained three supervised deep architectures directly on each dataset's
own subjects rather than only through a population-swapped pretraining step (Section 13.3, Table
16). The paper's claim is now that alignment is the lever that generalizes, not that preprocessing
in general outweighs model complexity: on one of the three populations (Cho2017), a supervised CNN
(EEGNetv4) outperforms the classical ensemble outright (Section 13.5), which is itself evidence
against a blanket "preprocessing beats architecture" reading. We also corrected the
cumulative-improvement number the reviewer references: it was significant only when the ensemble
was selected on the same subjects used to test it; a fully nested re-search gives a smaller,
non-significant improvement, and we report that nested estimate throughout (Section 5.1, page 9,
Table 1).

## Would TMLR's audience be interested? (reviewer answered "Yes")

*"The findings are relevant to researchers working on EEG decoding, brain-computer interfaces,
domain adaptation, and cross-subject generalization. The systematic reporting of negative results
and the emphasis on preprocessing provide useful practical insights, even though the conclusions
require stronger experimental validation."*

No response needed; the stronger validation requested is addressed in the critical changes below.

## Critical change 1

*"Extend the comparisons across all major method families to PhysioNet or another sufficiently
large dataset to verify whether the main findings generalize beyond nine subjects."*

We extended the comparison to PhysioNet's full 109-subject population and to a third population,
Cho2017 (n=52). The scope differs by method family and we state it exactly. Every classical and
domain-adaptation method (FgMDM, MEKT, MDM, CSP/FBCSP variants) runs at PhysioNet's full n=109
(Section 13.2, Table 15). SPDNet and the raw-EEG self-supervised family (MAE-only, MAE+GRL, SupCon,
Riemannian Transformer v2) also run at n=109, under a population-swapped protocol: re-pretrained
on BCI-IV 2a's 9 subjects and evaluated zero-label across PhysioNet, since their original
PhysioNet-pretrained checkpoints cannot be evaluated within PhysioNet without leakage (Section
13.2, Table 15). EEGNetv4 is evaluated at both n=30 and the full n=109; ShallowFBCSPNet and ATCNet
are evaluated only at n=30, since training three architectures end-to-end at 5 seeds across the
full population was not tractable within this study's compute budget (Section 13.3, Table 16).
Cho2017 (n=52) covers the classical methods, the 8-member ensemble, and EEGNetv4 (Section 13.5,
Table 9). The classical pattern, an ensemble of aligned methods beating any single classical
method, replicates on all three populations. The architecture pattern does not: EEGNetv4 ties the
classical ensemble on two populations, BCI-IV 2a and PhysioNet (Section 13.3), and significantly
beats it on the third, Cho2017 (Section 13.5, Table 9). We report this directly rather than
averaging it into a single number.

## Critical change 2

*"Strengthen the statistical evidence supporting the contributions of time-window selection and
covariance estimation, and distinguish statistically supported findings from empirical trends."*

We repeated the exact 4x2x2 time-window/estimator/method factorial design from BCI-IV 2a (Section
10.1, Table 3) on PhysioNet, first at a disclosed n=30 subset, then at the full n=109 population
(Section 13.1, Table 5). Neither effect survives Holm correction at either scale. The
covariance-estimator effect remains null, consistent with the original finding. The time-window
effect is positive and in the same direction at both sample sizes but shrinks as n grows and is
not significant after correction on either dataset; we no longer describe it as a general property
of the pipeline. A raw p-value of 0.0273 cannot be significant after Holm correction across five or
more contrasts; the table is corrected (Table 3).

Separately, the paper's one previously Holm-significant paired-statistics result (FgMDM to the
final ensemble, p=0.0039, Table 1) used an ensemble selected on the same 9 subjects it was tested
on. We built two nested checks (Section 5.1, page 9). The first fixes the candidate set to the 8
individual configurations plus the pre-built ensemble and nests only the choice among those 9;
it reproduces 52.39% exactly, which shows that choice was not leaky, but is not the primary
generalization estimate, since its candidate set was itself built from the original, non-nested
study. The second rebuilds the entire window/estimator/method search from scratch per outer fold,
with no reference to that fixed candidate set, giving +2.49pp, p=0.0742 (not significant) over
FgMDM and 49.94% for the top-8 ensemble. The fully nested re-search gives 49.94%, which we now
use as the primary generalization estimate throughout the paper (Table 1, Table 2). We retain
52.39% only where explicitly identified as the in-sample, optimistically selected result, for
transparency and reproducibility.

## Critical change 3

*"Further clarify the scope of the central claim, particularly given the limited selection of
deep-learning architectures and the absence of direct comparisons with recent EEG foundation
models."*

We closed this in three parts. First, we tested LaBraM, a pretrained EEG foundation model, with a
positive-control check (10 variants in the easiest possible same-subject setting) and with
source-only fine-tuning of the backbone under two regimes, not only a frozen linear probe; it
stays at chance under every condition (Section 13.4). Second, we attempted CBraMod and document
the specific upstream checkpoint-loading bug that prevented it from running, rather than stating
only that it was unavailable; BIOT was excluded for a verified channel-scheme incompatibility
(Section 14). Third, and most substantively, we trained three supervised architectures (EEGNetv4,
ShallowFBCSPNet, ATCNet) directly on each dataset's own subjects, not only via the
population-swapped pretraining the self-supervised methods use (Section 13.3, Table 16). On
PhysioNet's 30-subject subset, all three tie or lose to the classical baselines. At the full
109-subject population, which only EEGNetv4 was rerun at (ShallowFBCSPNet and ATCNet were
evaluated only at n=30), EEGNetv4 significantly outperforms two individual classical baselines it
had only tied at the smaller scale, though it still does not outperform the full ensemble (Section
13.3). Training-population size interacting with architecture this way is a result this comparison
surfaced; we state it as a finding, not only as a methodological correction, and the central claim
is scoped accordingly: alignment generalizes across the method families and populations we tested
it on (Section 13.3); which specific method or architecture wins outright does not.

## Suggested improvement 1

*"Clarify the distinction between strict zero-shot decoding and transductive adaptation using
unlabeled target-subject data."*

We introduce "zero-label" as the paper's primary term (no labeled target-subject data is used to
fit any model) and use it in the title, abstract, and throughout; it is defined in Section 3.1.
"Transductive" is used explicitly wherever a method uses the target subject's complete unlabeled
batch at once (Euclidean/Riemannian Alignment, MEKT's pseudo-labeling; Section 3.1). "Zero-shot"
is reserved for the one setting that is strictly inductive: the LaBraM comparison (Section 13.4),
which uses neither alignment nor pseudo-labeling, and for quoting other papers' own terminology.

## Suggested improvement 2

*"Provide a more focused discussion of the methodological novelty relative to existing Riemannian
geometry and domain-adaptation approaches."*

We added two directly relevant papers, verified by reading their full text and confirming arXiv
identifiers: MIRepNet (arXiv:2507.20254), a motor-imagery-specific foundation model that reports
beating LaBraM after fine-tuning, which scopes our LaBraM null result to frozen or minimally
adapted general-purpose checkpoints rather than every learned representation under every protocol
(Section 13.4); and an existing systematic evaluation of Euclidean Alignment for deep EEG decoders
(arXiv:2401.10746), which scopes our own EA-helps-EEGNetv4 finding as a replication of an
established phenomenon on this paper's own populations, not a new mechanism (Section 13.3).

## Summary of changes by section

| Section | What changed | Why |
|---|---|---|
| Abstract, Title | Rewritten around alignment as the central claim; "zero-label" introduced as the primary term; time-window effect stated symmetrically for both datasets | Central claim changed; terminology precision (Suggested 1) |
| Section 1, Introduction | Added a sentence stating the contribution is a controlled comparison under one protocol, not a new algorithm; removed an inaccurate blanket description of the evaluation protocol | Methodological-novelty concern |
| Section 3.1, Data canonicalization | "zero-label" and "transductive" defined precisely | Suggested 1 |
| Section 5.1, Nested cross-validation (page 9) | Added an explicit statement that the restricted-candidate nested check (reproduces 52.39%) is not the primary generalization estimate; the broader per-fold search (49.94%) is used throughout as the primary estimate | Critical 2 |
| Section 10.1, Factorial check on BCI-IV 2a (Table 3) | Table restructured around a 5-contrast primary family and a 2-contrast supplementary family, both raw and Holm-adjusted p shown; mislabeled window-effect significance corrected | Critical 2; labeling-error correction |
| Section 13.1, Factorial replication on PhysioNet (Table 5) | New section; identical design re-run at n=30 and n=109 | Critical 1, Critical 2 |
| Section 13.2, Comprehensive method-family comparison on PhysioNet (Tables 7, 15) | New section; every method family re-evaluated at full PhysioNet scale; PhysioNet alignment-ablation numbers (FgMDM, MEKT) added | Critical 1 |
| Section 13.3, Supervised deep architectures (Table 16) | New section; EEGNetv4 at n=30 and n=109, ShallowFBCSPNet/ATCNet at n=30 only; one comparison (ShallowFBCSPNet vs. nested ensemble) corrected from a stale significant result to the correct non-significant one | Critical 3 |
| Section 13.4, Foundation model (LaBraM) | Positive-control check and source-only backbone fine-tuning added | Critical 3 |
| Section 13.5, Third population, Cho2017 (Table 9) | New section; classical ensemble replicates, but EEGNetv4+EA beats it, the opposite direction from the first two datasets | Critical 1, Critical 3 |
| Section 14, Limitations | CBraMod/BIOT exclusion documented; PhysioNet scope restated exactly for all three supervised architectures; BCI-IV 2a hyperparameters tuned on 9 subjects do not transfer to PhysioNet | Critical 2, Critical 3 |
| Section 15, Conclusion | Shortened; restated around alignment as the generalizing lever, with the two stated exceptions, and the dataset-dependence of which method wins outright | Central claim change |
