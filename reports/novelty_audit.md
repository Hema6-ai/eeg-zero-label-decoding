# Novelty audit

Date: 2026-10-28. Produced for the master audit directive's Phase 3.

## Step 1: What does this paper actually contribute?

Direct read of the Introduction, contribution list, and Conclusion (`tmlr_2026.tex`). The paper
does **not** propose a new algorithm, architecture, or theoretical result. Every method used
(CSP, MDM, FgMDM, EA/RA, JDA/MEKT, RPA, SPDNet, MAE, SupCon, CLUB, VIB, GRL) is attributed to its
original authors with a citation (`reports/related_work_verification.md`, 24/24 citations
checked, 0 required correction). The actual, defensible contribution is a combination of:

1. A **controlled empirical analysis** that separates preprocessing effects (time window,
   covariance estimator) from architecture effects, via a full factorial design
   (`sec:factorial`) rather than a sequential tuning narrative -- this is the paper's
   methodologically strongest piece, because it is the one place the paper does something a
   simple comparison table could not: it isolates *which* factor is responsible for an observed
   gain.
2. A **larger-population replication** (PhysioNet, 109 subjects) of a finding originally made at
   n=9, with real statistical power (Holm-corrected, p=0.0002 for the central comparison).
3. A **negative-result catalog**: nine learned/geometric methods and seven further refinements
   that did not beat a decades-old classical baseline, reported rather than suppressed.
4. A **reproducibility/leakage audit** of the evaluation protocol itself (zero-shot vs.
   transductive, training-data exposure, hyperparameter-tuning-on-test-set), disclosed explicitly
   in Limitations rather than assumed away.

This is accurately self-described in the Introduction/Conclusion as a systematic empirical
comparison, not an algorithmic contribution. No correction needed on this point (re-confirmed by
scanning for "first," "novel," "state-of-the-art," "unprecedented" -- none found in a fresh grep
this session, consistent with the prior session's finding).

## Step 2: Closest prior work

Five papers compared, three carried over (and re-verified) from
`reports/related_work_verification.md`, two newly found and verified this session via direct
primary-source fetches (not search-snippet summaries).

| Paper | Research question | Dataset / n | Protocol | Preprocessing factorial? | Stats correction | Code/reproducibility | Main conclusion |
|---|---|---|---|---|---|---|---|
| **This paper** | Does preprocessing or architecture drive cross-subject MI gains? | BCI-IV 2a (9) + PhysioNet (109) | Strict zero-shot LOSO, inductive; transductive methods explicitly labeled as such | **Yes** -- 4 windows x 2 estimators x {FgMDM, MEKT}, plus a default-window SPDNet estimator check | Holm-Bonferroni across both families; bootstrap CIs; jackknife robustness | Full per-subject CSVs + regenerating scripts for every number (46/46 claim-audit rows verified this session) | Time-window effect generalizes; covariance-estimator and raw method-family effects do not, once controlled for each other |
| `zhang2020mekt` (Zhang & Wu, IEEE TNSRE 2020) | Propose MEKT itself (JDA specialized to BCI tangent-space features) | BCI-IV 2a and others | Transductive (batch access to unlabeled target trials) | No | Not reported in the original paper | Not independently assessed here (predates this study) | MEKT improves cross-subject transfer -- this paper's central domain-adaptation method is a direct, attributed reimplementation, not a reinvention |
| `zhou2026eegfmaudit` (Zhou & Roy, arXiv:2609.23924, 2026) | Audit whether EEG foundation models beat supervised baselines under matched input/preprocessing | BCI Competition IV-2a (same dataset as this paper) | Frozen-backbone, matched preprocessing | No | Not a factorial design, but a controlled matched-input comparison | Preprint, code status not verified by us | Every supervised baseline tested beats every LaBraM/CBraMod configuration on BCI-IV-2a -- **the single most directly corroborating precedent for this paper's own LaBraM finding, same dataset** |
| `xiong2025eegfmbench` (Xiong et al., arXiv:2508.17742 / ICML 2026 poster) | Broad frozen-backbone benchmark of EEG foundation models across many datasets/tasks | Many datasets (not just BCI-IV 2a) | Frozen-backbone fine-tuning-free protocol, directly comparable in spirit | No | Not reported | Benchmark paper, broader scope | "Catastrophic performance collapse... across all models and nearly all tasks" -- broader-scope corroboration that frozen EEG-FM evaluation often yields chance-level results, consistent with but not specific to this paper's LaBraM result |
| **Manivannan & Lakshminarayan, "Geometry-Aware Deep Congruence Networks for Manifold Learning in Cross-Subject Motor Imagery," arXiv:2511.18940** (v1: 24 Nov 2025; v3: 9 Jul 2026, the version verified) -- **not currently cited** | Propose new manifold-alignment architectures (DCT/DLDCT/DDCT-UNet) to improve cross-subject MI decoding | Not stated on the abstract page (would require the full PDF) | **Explicitly transductive** in the current (v3) abstract ("consistent improvements in transductive decoding performance") -- confirmed by direct fetch of the arXiv abstract page, not inferred | Not mentioned on the abstract page | Not stated on the abstract page | Preprint (v3 as of 2026-07-09); code status not checked | Proposes a new architecture claiming ~2-3pp gains over strong baselines -- but in the **transductive**, not zero-shot, setting, so it is not a direct counter-example to this paper's strict zero-shot claim |
| **Ramkumar & Lakshminarayanan, "Riemannian Transfer Learning in Motor Imagery decoding: Reproducibility and Standardized Benchmarks," NeurIPS 2025 workshop poster ("Learning from Time-Series for Health")** -- **not currently cited** | Standardize evaluation protocol for transfer-learning techniques across cross-session/cross-subject MI, with ablations on signal length and preprocessing parameters | Not stated on the listing page | Not stated explicitly | Ablations exist on "signal length and preprocessing parameters" but **not described as a factorial design isolating preprocessing from architecture** (confirmed by direct fetch; the page does not make this claim) | Not stated | Explicitly provides Python implementations for reproducibility | Addresses a benchmarking/standardization gap similar in spirit to this paper's reproducibility concerns, from the same research group as the entry above |

## Step 3: Testing each novelty claim

1. **Is the controlled factorial separation genuinely distinct from prior work?** Based on the
   five comparators above (the three previously verified plus the two newly found this session),
   yes with moderate confidence: none of the five papers describe a factorial design isolating
   time-window and covariance-estimator effects from architecture choice. The two newest,
   most topically adjacent preprints (same narrow sub-field, overlapping authorship across the
   two) do not claim this either -- one is an architecture paper in the transductive setting, the
   other is a reproducibility/benchmarking paper whose ablations are not described as factorial.
   This is not proof no such paper exists anywhere in the literature (a negative claim cannot be
   established by five comparisons), so the manuscript should continue to avoid "first" language,
   which it already does.
2. **Is it demonstrated by the experiments?** Yes -- `paper/eval_factorial.py` ran all 16 cells
   plus the SPDNet check; `paper/stats_factorial.py` reproduces the main-effect estimates exactly
   this session.
3. **Is it useful to the research community?** Plausibly yes: the factorial result is a genuine,
   actionable methodological caution (a sequentially-discovered preprocessing gain may not
   generalize as an independent effect) that applies beyond this specific paper's methods.
4. **Supported by the evaluation protocol?** Yes -- strict zero-shot LOSO throughout, with
   transductive methods (MEKT, RPA) explicitly and consistently labeled as such.
5. **Does the manuscript explain why it matters?** Yes, in `sec:factorial`'s interpretation
   paragraph and the Conclusion's closing sentence about "small-sample tuning decisions deserve
   the same scrutiny we apply to architectural choices."
6. **Still valid if the strongest positive result (window effect) became nonsignificant?** The
   paper's claim would weaken to "no preprocessing or architecture effect survives a factorial
   control at n=9" -- which the paper would still be able to report honestly, since it already
   reports the estimator and method-family null results alongside the window's significant one.
   The contribution (the factorial methodology itself, and the discipline of reporting nulls) does
   not depend on any one cell being significant.
7. **Stated more broadly than the data allow?** This audit found and fixed three instances where
   wording in captions/abstract risked this (Section 1 below); after the fixes, no remaining
   instance was found in a full re-scan of absolute quantifiers ("every," "all") in the Results,
   Discussion, and figure captions.

## Step 4: Strengthening the defensible contribution

No new experiments, terminology, or theoretical claims were added for the purpose of manufacturing
novelty (per directive rule 6). The existing framing in Introduction/Conclusion already states the
contribution as the controlled empirical separation plus the two-population replication; this
audit's only novelty-related *content* changes were (a) the abstract clarification distinguishing
the window effect's generality from the estimator effect's configuration-specificity (Section 1,
`reports/final_scientific_audit.md`), which makes the strongest contribution -- the factorial
separation itself -- more visible and precise, and (b) flagging two citable, very recent papers as
an optional strengthening opportunity for Related Work (not auto-added; see table above and the
rationale in the closure matrix's "Additional 2" row).

## Summary

- Novelty claim: a controlled factorial separation of preprocessing from architecture effects,
  replicated across two populations, with a disclosed negative-result catalog. Not an algorithmic
  contribution; the manuscript does not claim one.
- 5 closest works compared; 2 of the 5 found and verified this session from the current (2025-2026)
  literature, directly in this paper's sub-field.
- No unsupported "first"/"novel"/"SOTA" language found anywhere in the manuscript.
- One optional, disclosed (not auto-applied) recommendation: cite the two newly-found papers in
  Related Work if the authors judge, after reading the full papers (not just abstracts), that they
  support the claims this audit infers from the abstract pages alone.
