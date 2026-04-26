import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite+aiosqlite:///{BASE_DIR}/cliptrace.db"

SEGMENT_DURATION = 1.0          # seconds per segment
FINGERPRINT_DIM = 832           # 256 deterministic + 576 MobileNetV3 ML embedding
SIMILARITY_THRESHOLD = 0.75     # minimum cosine similarity for a match
MIN_MATCHING_SEGMENTS = 2       # minimum segments to confirm a match
CONFIDENCE_CONTINUITY_BONUS = 0.05  # bonus per consecutive matching segment

SECRET_KEY = os.getenv("CLIPTRACE_SECRET", "cliptrace-dev-secret-change-in-prod")
ALGORITHM = "HS256"
