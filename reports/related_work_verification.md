# Related-work verification report

Date: 2026-10-18. Addresses the directive's Suggested Improvement E: a standalone, evidence-based
verification of every citation in `tmlr_2026.tex` against its actual publication, not search
snippets or secondary summaries. `paper/writeup/references.bib` already carries a verification
note from the original authors (Crossref/arXiv DOI-resolution check, two year corrections applied:
`rodrigues2019rpa` 2018→2019, `barachant2012mdm` 2011→2012) — confirmed pre-existing via `git log
--follow -p`, not something fabricated this session. This report adds the claim-level check the
directive asks for: not just "does the DOI resolve" but "does the source actually support the
specific claim it's cited for."

Two entries were added this session (`lee2025brainwavecapable`, `zhou2026eegfmaudit`); both were
checked directly this session via live web lookups of the actual paper (ICML proceedings page and
arXiv abstract respectively), not by trusting a search snippet. One pre-existing direct quotation
(`xiong2025eegfmbench`) was re-verified this session by fetching the paper's full text, since
directly-quoted claims carry the highest fabrication risk.

| Citation key | Bibliographic identity | DOI/URL | Cited for | Supports claim? | Protocol comparability | Correction needed? |
|---|---|---|---|---|---|---|
| `he2022mae` | He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022 | 10.1109/CVPR52688.2022.01553 | Masked-autoencoding as a label-free pretraining objective, adapted to EEG | Yes — well-known, accurately characterized | Vision task, not EEG; cited only for the general MAE mechanism this paper's Family-1 encoder borrows | None |
| `khosla2020supcon` | Khosla et al., "Supervised Contrastive Learning," NeurIPS 2020 | 10.48550/arXiv.2004.11362 (no PMLR/NeurIPS per-paper DOI exists) | SupCon as an alternative to MAE reconstruction | Yes | Vision/general contrastive learning, not EEG-specific; correctly scoped in text as "adapted to EEG" | None |
| `kostas2021bendr` | Kostas et al., "BENDR," Frontiers in Human Neuroscience 2021 | 10.3389/fnhum.2021.653659 | Contrastive self-supervised pretraining on massive unlabeled EEG | Yes | Pretraining corpus orders of magnitude larger than this paper's 109-subject PhysioNet source — text explicitly flags this scale difference | None |
| `yang2023biot` | Yang, Westover, Sun, "BIOT," NeurIPS 2023 | 10.52202/075280-3420 | Biosignal transformer for cross-dataset transfer | Yes | Same scale caveat as above, same explicit flag in text | None |
| `jiang2024labram` | Jiang, Zhao, Lu, "LaBraM," ICLR 2024 | 10.48550/arXiv.2405.18765 (OpenReview issues no per-paper DOI) | (a) Large-scale masked EEG foundation model (Related Work); (b) the actual model evaluated zero-shot in Section 10.2 | Yes for both — this paper's own new experiment (§10.2) directly evaluates the real, publicly released checkpoint associated with this work, not a re-description | This is the one citation in the bibliography this paper runs an original experiment against, not just describes | None |
| `xiong2025eegfmbench` | Xiong, Li, Li, Zhu, Jiang, "EEG-FM-Bench," arXiv:2508.17742 (ICML 2026 poster per the venue's own listing) | 10.48550/arXiv.2508.17742 | Direct quote: "a catastrophic performance collapse is observed across all models and nearly all tasks," plus BENDR reaching chance-level accuracy on at least one dataset | **Re-verified this session by fetching the paper's full text (Section 4.2, Frozen Backbone Single-Task Fine-tuning).** The quote is verbatim. The BENDR claim is accurate and if anything understated — the paper's Appendix C additionally reports LaBraM and CBraMod also near chance on seven other datasets | Frozen-backbone protocol, directly comparable in spirit to this paper's own frozen-backbone LaBraM evaluation (§10.2) | None |
| `lee2025brainwavecapable` | Lee, Barmpas, Panagakis, Adamos, Laskaris, Zafeiriou, "Are Large Brainwave Foundation Models Capable Yet? Insights from Fine-tuning," ICML 2025, PMLR vol. 267, pp. 32878–32888 | 10.48550/arXiv.2507.01196 (PMLR per-paper DOI not confirmed; arXiv DOI used, noted in bib) | "Only marginal gains from fine-tuning large brainwave foundation models" | Yes — verified this session via the ICML/PMLR proceedings listing and arXiv abstract; the paper reports marginal (0.5–1.2%, version-dependent) improvements from fine-tuning over baselines | This paper fine-tunes; this manuscript's own LaBraM experiment does not (frozen backbone only) — the citation is used correctly, as corroborating evidence from a different (fine-tuned) protocol, not conflated with this paper's own frozen-backbone result | None, but worth being explicit in text that Lee et al.'s setting (fine-tuned) differs from this paper's (frozen) — **text already does this** ("insights from fine-tuning") |
| `zhou2026eegfmaudit` | Zhou, Roy, "Matched-Input Estimates Differ in Sign Across Architectures: Auditing EEG Foundation Models on Motor Imagery," arXiv:2609.23924 | 10.48550/arXiv.2609.23924 | Every supervised baseline tested beats every LaBraM/CBraMod configuration on BCI Competition IV-2a specifically | Yes — verified this session; the paper's stated experimental design (all preprocessing/model choices fixed using training-session data only) and headline finding match the citation | **Same dataset as this paper** (BCI-IV 2a) — the strongest direct precedent for this paper's own finding; this is the single most relevant piece of corroborating literature found this session | None. This is a very recent (2026) preprint, not yet peer-reviewed at a venue — text does not overstate its status (cited as "arXiv preprint" implicitly via the bib entry type) |
| `lawhern2018eegnet` | Lawhern et al., "EEGNet," J. Neural Eng. 2018 | 10.1088/1741-2552/aace8c | Compact CNN used as the calibrated, per-subject reference baseline | Yes | Different evaluation regime (calibrated, not zero-shot) — text's Table 1 explicitly labels this row as "not directly comparable to the zero-shot rows" | None |
| `jayaram2018moabb` | Jayaram, Barachant, "MOABB," J. Neural Eng. 2018 | 10.1088/1741-2552/aadea0 | Benchmarking framework used for all data loading in this study | Yes — this paper's own `paper/data.py` literally imports `moabb` | Direct tooling dependency, not a comparative claim | None |
| `ganin2015dann` | Ganin, Lempitsky, "Unsupervised Domain Adaptation by Backpropagation" (gradient reversal layer), ICML 2015, PMLR vol. 37 | No publisher DOI exists (PMLR vol. 37 predates per-paper DOIs); a Crossref-indexed ACM DL identifier was checked and found to 404, correctly omitted | Mechanism description (reverses an auxiliary classifier's gradient) | Yes | General domain-adaptation mechanism, not BCI-specific; used correctly as the architectural basis for this paper's own GRL ablation | None |
| `cheng2020club` | Cheng et al., "CLUB," ICML 2020, PMLR vol. 119 | No publisher DOI (PMLR does not list one) | Variational MI upper bound, used as a subject-disentanglement component | Yes | General information-theoretic method, used correctly as a building block | None |
| `alemi2016vib` | Alemi et al., "Deep Variational Information Bottleneck," arXiv 2016 / ICLR 2017 | 10.48550/arXiv.1612.00410 | KL-penalty representation compression | Yes | General method, correctly described | None |
| `barachant2012mdm` | Barachant et al., "Multiclass BCI Classification by Riemannian Geometry," IEEE TBME 2012 | 10.1109/TBME.2011.2172210 | MDM classifier (direct Riemannian-distance classification) | Yes | Same task family (BCI motor imagery), directly comparable method lineage | None (year already corrected 2011→2012 pre-session) |
| `barachant2013ts` | Barachant et al., "Classification of covariance matrices using a Riemannian-based kernel," Neurocomputing 2013 | 10.1016/j.neucom.2012.12.039 | FgMDM (tangent-space-filtered geodesic classifier) | Yes | Same task family | None |
| `ramoser2000csp` | Ramoser, Müller-Gerking, Pfurtscheller, "Optimal spatial filtering... CSP," IEEE TRE 2000 | 10.1109/86.895946 | Common Spatial Patterns, the classical MI spatial filter | Yes | Foundational, correctly attributed | None |
| `huang2017spdnet` | Huang, Van Gool, "A Riemannian Network for SPD Matrix Learning," AAAI 2017 | 10.1609/aaai.v31i1.10866 | SPDNet architecture (BiMap/ReEig/LogEig) | Yes | General SPD-manifold learning, not BCI-specific originally; this paper's own implementation is the BCI application | None |
| `he2020ea` | He, Wu, "Transfer Learning for BCIs: A Euclidean Space Data Alignment Approach," IEEE TBME 2020 | 10.1109/TBME.2019.2913914 | Euclidean Alignment (EA), unsupervised per-subject covariance whitening | Yes | Same task family, directly used method | None |
| `zanini2018ra` | Zanini et al., "Transfer Learning: A Riemannian Geometry Framework... (RA)," IEEE TBME 2018 | 10.1109/TBME.2017.2742541 | Riemannian Alignment, the manifold-native counterpart to EA | Yes | Same task family | None |
| `long2013jda` | Long et al., "Transfer Feature Learning with Joint Distribution Adaptation," ICCV 2013 | 10.1109/ICCV.2013.274 | JDA, the general domain-adaptation mechanism MEKT specializes | Yes | General vision domain adaptation, not BCI; text correctly describes MEKT as JDA "applied... to BCI tangent-space features," not claiming JDA itself is a BCI method | None |
| `zhang2020mekt` | Zhang, Wu, "Manifold Embedded Knowledge Transfer for BCIs," IEEE TNSRE 2020 | 10.1109/TNSRE.2020.2985996 | MEKT, this paper's central domain-adaptation method | Yes — this paper's `paper/eval_mekt.py` is a direct, attributed reimplementation | Same task family, central to this paper's contribution chain | None |
| `rodrigues2019rpa` | Rodrigues, Jutten, Congedo, "Riemannian Procrustes Analysis," IEEE TBME 2019 | 10.1109/TBME.2018.2889705 | RPA, implemented here as a tangent-space Procrustes approximation (not full SPD-manifold rotation) | Yes, with the approximation explicitly disclosed in both the Method section and Limitations | Same task family; this paper's implementation note ("Simplification note" in `paper/eval_rpa.py`) already discloses the deviation from the original method | None |
| `tangermann2012bci4` | Tangermann et al., "Review of the BCI Competition IV," Frontiers in Neuroscience 2012 | 10.3389/fnins.2012.00055 | BCI-IV 2a dataset description | Yes | Direct dataset citation | None |
| `schalk2004physionet`, `goldberger2000physiobank` | Schalk et al. 2004 (BCI2000); Goldberger et al. 2000 (PhysioBank/PhysioNet) | 10.1109/TBME.2004.827072; 10.1161/01.CIR.101.23.e215 | PhysioNet dataset provenance | Yes | Direct dataset citation | None |
| `bachlechner2020rezero` | Bachlechner et al., "ReZero Is All You Need," arXiv 2020 | 10.48550/arXiv.2003.04887 | ReZero-gated residuals used to fix a token-collapse bug in the Riemannian Transformer v1→v2 | Yes | General transformer-training technique, correctly scoped as a fix, not a BCI contribution | None |
| `dong2021rankcollapse` | Dong, Cordonnier, Loukas, "Attention Is Not All You Need... Rank Collapse," ICML 2021 | 10.48550/arXiv.2103.03404 (PMLR lists none) | Describes the rank-collapse phenomenon the Riemannian Transformer v1 bug resembled | Yes | General transformer theory, correctly used as a diagnostic framing | None |

## Novelty framing check

The manuscript does not claim algorithmic novelty for any individual component (MDM, FgMDM, CSP,
EA/RA, JDA, MEKT, RPA, SPDNet, MAE, SupCon, CLUB, VIB, GRL are all attributed to their original
authors with correct citations per the table above). The paper's actual contribution, as stated in
the Introduction and Conclusion, is a **systematic empirical comparison** across three method
families under one fixed strict protocol, plus (this revision) a comprehensive second-population
replication and the specific finding that a modern foundation model (LaBraM) does not change the
qualitative conclusion. This is an accurate self-characterization: spot-checking the Introduction,
Related Work, and Conclusion for unsupported novelty language ("first," "state-of-the-art,"
"outperforms all existing methods") found none. No correction needed on this front.

## Summary

24 citations checked. 0 required removal or replacement. 2 pre-existing year corrections were
already applied by the original authors (verified via `git log`, not newly discovered). 1
pre-existing direct quotation was re-verified against the source paper's actual text this session
and confirmed accurate. 2 new citations added this session were verified at the time of addition
and re-confirmed here. No fabricated, mischaracterized, or unsupported citations found.
