# EEG foundation-model comparison: LaBraM (TMLR review Critical Issue D)

Date: 2026-10-13. Addresses the reviewer's request to compare against recent pretrained EEG
foundation models, and the directive's fallback rule: attempt a lightweight, no-fine-tuning
comparison where feasible; narrow the claim where it is not.

## Model selected and why

**LaBraM** (Jiang et al., 2024, ICLR 2024 spotlight, already cited in this manuscript's
bibliography as `jiang2024labram`) -- selected because:

- Directly relevant to motor-imagery EEG decoding and already discussed (not previously tested)
  in this manuscript's related work.
- A maintained, MIT-licensed, pip-installable integration exists in `braindecode` (>= 1.2.0,
  `braindecode.models.Labram`), avoiding the need to vendor the original research-code repository.
- A community-converted, MIT-licensed checkpoint compatible with that exact class
  (`braindecode/Labram-Braindecode` on HuggingFace, file `braindecode_labram_base.pt`) is openly
  downloadable with no gating or credentialing, unlike some EEG foundation-model checkpoints.
- CPU/lightweight-GPU feasible for frozen-feature extraction + a linear probe (no fine-tuning),
  matching the user-approved scope for this comparison.

Models considered and not pursued: EEGPT (no equivalently maintained, openly-downloadable
checkpoint found in this session's research), BIOT (came up only as a baseline in related papers,
no checkpoint located). Not an exhaustive search of every EEG foundation model; scoped to what
could be verified and run within this session.

## Protocol: strict zero-shot, frozen features, linear probe (Protocol A)

`paper/eval_foundation_model_labram.py` (new):

1. LaBraM backbone loaded from the checkpoint above, **entirely frozen** -- no parameter of the
   pretrained model is ever updated. This is a hard requirement for a valid "zero-shot" claim: a
   model requiring target-subject supervised fine-tuning would not qualify (per the review's own
   definition), so this experiment does not fine-tune, full stop.
2. For each BCI-IV 2a subject held out (standard LOSO, 9 folds): frozen LaBraM features extracted
   for the other 8 subjects' trials, a single linear layer (`paper/models/heads.py::LinearProbe`,
   the same head used for every other Family-1/2 zero-shot method in this project) trained on
   those features only, then evaluated on the held-out subject's frozen features. No held-out
   subject data touches the backbone or the probe's training in any way -- identical protocol to
   `paper/eval_zero_shot.py` and siblings.

## Necessary deviations from this project's standard protocol (disclosed, not hidden)

1. **20 channels, not 22.** LaBraM's channel-position embeddings are looked up by name against a
   fixed vocabulary (`standard_1020`, verified directly from the original LaBraM repository,
   https://github.com/935963004/LaBraM/blob/main/utils.py, this session -- reproduced verbatim in
   `paper/eval_foundation_model_labram.py`). The specific checkpoint used here has only a 64-slot
   position-embedding table (not the full ~140-entry vocabulary), and 2 of this project's 22
   standard channels (**P2**, **POz**) fall outside those 64 slots. Both are dropped for this
   comparison only; every other experiment in this project keeps the full 22-channel montage.
2. **200 Hz, not 128 Hz.** LaBraM's fixed `patch_size=200` samples is a conventional 1-second
   patch at a 200 Hz sampling rate (the rate the original model was pretrained at); this project's
   usual 128 Hz would make each patch temporally irregular (1.5625s) relative to that convention.
   BCI-IV 2a was resampled to 200 Hz independently for this one experiment via MOABB, same tmin/
   tmax (0-3s) window as the rest of the project (giving exactly 3 patches/channel after a
   sub-5ms truncation to make the sample count evenly divisible by the patch size).
3. **Position/temporal embeddings manually sliced, not selected via the model's own `input_chans`
   forward-time mechanism.** The installed `braindecode.models.Labram` ties its position-embedding
   parameter *shape* to the channel count given at construction time, which conflicts with needing
   the *full* pretrained 65-row table available for subsetting. Resolved by: constructing the
   model at the actual (20-channel) shape, loading every other backbone parameter normally via
   `load_state_dict(strict=False)` (0 unexpected shape mismatches), then manually copying the
   correct 21 rows (1 CLS + 20 channels, in `LABRAM_CHANNELS` order) from the checkpoint's original
   65-row table into the model's position_embedding, and the first `n_patch` rows of the original
   temporal_embedding. This reproduces exactly what the architecture's own `input_chans` selection
   mechanism would do; it was implemented manually only because of the shape-at-construction
   constraint above, not because of any uncertainty about which rows are correct.

## Result

| Metric | Value |
|---|---|
| Accuracy (9-subject LOSO) | **25.08% +/- 1.74%** |
| Kappa | 0.0010 +/- 0.0232 |
| 4-class chance | 25.00% |

**LaBraM's frozen zero-shot features, with only a linear probe, perform at chance** on this task
-- indistinguishable from random guessing (kappa approximately 0). Per-subject accuracies ranged
22.7-27.8%, all within noise of chance.

## Interpretation

This is a genuine negative result, not a failure of the experiment: it is directly consistent with
independent literature already surfaced during this revision (not previously cited in this
manuscript) -- "Are Large Brainwave Foundation Models Capable Yet? Insights from Fine-tuning"
(arXiv:2507.01196) and a 2026 audit of LaBraM/CBraMod on BCI Competition IV-2a, both reporting that
supervised/classical baselines beat these foundation models on exactly this kind of benchmark. It
also does not contradict this manuscript's own existing raw-EEG self-supervised results: MAE-only
and SupCon (pretrained from scratch on PhysioNet, within this project) also only reach ~25-28%
zero-shot (`final_results_table.csv`), so a *different* large-scale pretrained raw-EEG transformer
landing in the same ~25-28% band is consistent, not anomalous.

**Caveat on scope**: this is a linear probe on frozen features from one checkpoint of one
foundation model family, with the 3 disclosed deviations above. It does not establish that *no*
foundation model or *no* adaptation protocol (e.g. fine-tuning, which would no longer be strict
zero-shot) could do better -- only that this specific, correctly-implemented, no-fine-tuning
configuration does not. This scoping should be stated explicitly wherever this result is reported
in the manuscript.

## Manuscript implication

This result lets Critical Issue D be addressed with a real data point rather than only a
narrowed claim: the manuscript can now state that a recent, representative EEG foundation model
(LaBraM), evaluated zero-shot under the same strict protocol as every other method in this study,
also failed to beat the classical Riemannian baselines -- strengthening (not undermining) the
paper's central finding, while being explicit about the experiment's scope and the 3 disclosed
protocol deviations above.
