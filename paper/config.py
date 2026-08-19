"""Central configuration for EEG-MAE pretraining, ablation, and zero-shot eval."""
import random

import numpy as np
import torch

SEED = 42

# --- Common preprocessing (shared across PhysioNet pretraining and BNCI-IV-2a eval) ---
# The full 22-channel BNCI2014_001 (BCI-IV 2a) montage -- verified as a subset of
# PhysionetMI's 64 channels (both use MNE's standardized 10-20 names, e.g. "FC3", "Cz", "POz").
COMMON_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
RESAMPLE_HZ = 128
TMIN, TMAX = 0.0, 3.0  # seconds relative to cue onset

# --- Patching / architecture ---
PATCH_SIZE = 50
EMBED_DIM = 256
ENCODER_DEPTH = 6
ENCODER_HEADS = 8
DECODER_DIM = 128
DECODER_DEPTH = 4
DECODER_HEADS = 8
MASK_RATIO = 0.75

# --- Pretraining ---
PRETRAIN_SUBJECTS = list(range(1, 110))  # PhysioNet: 109 subjects
PRETRAIN_EPOCHS = 100
PRETRAIN_BATCH_SIZE = 128
PRETRAIN_LR = 1.5e-4
GRL_LAMBDA_MAX = 1.0
GRL_WARMUP_EPOCHS = 20
CHECKPOINT_EVERY = 10

# --- Zero-shot eval (BCI-IV 2a) ---
EVAL_SUBJECTS = list(range(1, 10))  # 9 subjects
N_MI_CLASSES = 4
PROBE_EPOCHS = 50
PROBE_LR = 1e-3

# --- MAE+CLUB+SCIN (proposed method) ---
CLUB_LAMBDA = 0.01
CLUB_LR = 1e-3
VIB_BETA = 1e-3
CLUB_SCIN_EPOCHS = 300
CLUB_SCIN_WARMUP_EPOCHS = 10
CLUB_SCIN_CHECKPOINT_EVERY = 20
AUGMENT_TIME_WARP_PROB = 0.5
AUGMENT_MAX_WARP = 0.1
AUGMENT_CHANNEL_DROPOUT_PROB = 0.1

# Zero-shot head for the proposed method: 2-layer MLP instead of linear, more epochs, ensembled.
ZEROSHOT_MLP_HIDDEN = 128
ZEROSHOT_MLP_DROPOUT = 0.3
ZEROSHOT_MLP_EPOCHS = 100
ZEROSHOT_ENSEMBLE_SIZE = 5

# --- SupCon pretraining (supervised contrastive, replaces MAE reconstruction) ---
SUPCON_CLASSES = ("feet", "left_hand", "right_hand")  # PhysioNet has no 'tongue' class
SUPCON_EPOCHS = 200
SUPCON_BATCH_SIZE = 256
SUPCON_LR = 1e-4
SUPCON_WEIGHT_DECAY = 0.05
SUPCON_WARMUP_EPOCHS = 10
SUPCON_TEMPERATURE = 0.07
SUPCON_CHECKPOINT_EVERY = 20
SUPCON_MLP_HIDDEN = 128
SUPCON_MLP_DROPOUT = 0.3
SUPCON_HEAD_EPOCHS = 100

# --- Riemannian Transformer (covariance -> tangent space -> transformer) ---
RT_TANGENT_DIM = 253  # 22*23/2 upper-triangle elements for 22 channels
RT_EMBED_DIM = 128
RT_DEPTH = 4
RT_HEADS = 4
RT_EPOCHS = 100
RT_BATCH_SIZE = 128
RT_LR = 1e-4
RT_WEIGHT_DECAY = 0.05
RT_CHECKPOINT_EVERY = 20
RT_HEAD_HIDDEN = 64
RT_HEAD_DROPOUT = 0.3
RT_HEAD_EPOCHS = 100

# --- v2: trial-level attention (N trials attend to each other, not 1 trial per forward pass) ---
RTV2_DEPTH = 6
RTV2_N_TRIALS = 32
RTV2_EPOCHS = 150
RTV2_LR = 1e-4
RTV2_WEIGHT_DECAY = 0.05
RTV2_WARMUP_EPOCHS = 10
RTV2_CHECKPOINT_EVERY = 20

# --- SPDNet (BiMap/ReEig/LogEig -- stays on the SPD manifold, no early tangent-space flattening) ---
SPDNET_DIMS = (22, 16, 10)
SPDNET_EPOCHS = 150
SPDNET_BATCH_SIZE = 64
SPDNET_LR = 1e-3
SPDNET_WEIGHT_DECAY = 1e-4
SPDNET_WARMUP_EPOCHS = 10
SPDNET_CHECKPOINT_EVERY = 20
SPDNET_HEAD_HIDDEN = 64
SPDNET_HEAD_DROPOUT = 0.3
SPDNET_HEAD_EPOCHS = 100

CHECKPOINT_DIR = "paper/checkpoints"
RESULTS_DIR = "paper/results"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
