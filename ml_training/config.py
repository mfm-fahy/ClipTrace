"""
ClipTrace ML Training Configuration
All hyperparameters and paths in one place.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"          # original + edited videos go here
SEGMENTS_DIR   = DATA_DIR / "segments"     # extracted 1-sec frame folders
PAIRS_DIR      = DATA_DIR / "pairs"        # saved pair index (CSV)
CHECKPOINTS    = BASE_DIR / "checkpoints"
LOGS_DIR       = BASE_DIR / "logs"
EXPORTS_DIR    = BASE_DIR / "exports"      # final .pt model exports

for d in [RAW_DIR, SEGMENTS_DIR, PAIRS_DIR, CHECKPOINTS, LOGS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Segmentation ──────────────────────────────────────────────────────────────
SEGMENT_DURATION   = 1.0      # seconds
FRAMES_PER_SEGMENT = 8        # frames sampled per segment for training
FRAME_SIZE         = (224, 224)

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_DIM  = 256          # final embedding size
BACKBONE       = "mobilenet_v3_small"   # or "resnet18", "efficientnet_b0"

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
NUM_EPOCHS     = 30
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
LR_STEP_SIZE   = 10
LR_GAMMA       = 0.5
NUM_WORKERS    = 0            # set >0 on Linux; keep 0 on Windows to avoid spawn issues

# ── Loss ──────────────────────────────────────────────────────────────────────
LOSS_TYPE      = "triplet"    # "triplet" | "contrastive"
TRIPLET_MARGIN = 0.5
CONTRASTIVE_MARGIN = 1.0

# ── Data split ────────────────────────────────────────────────────────────────
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
TEST_RATIO     = 0.15

# ── Augmentation ──────────────────────────────────────────────────────────────
# Applied to edited/query frames to simulate real-world transformations
AUG_PROB       = 0.8

# ── Evaluation ────────────────────────────────────────────────────────────────
TOP_K_VALUES   = [1, 5, 10]   # Recall@K metrics
SIMILARITY_THRESHOLD = 0.75
