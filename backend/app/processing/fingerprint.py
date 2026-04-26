"""
Fingerprint Extractor
Combines three modalities into one deterministic fingerprint per segment:
  1. Visual  – perceptual hash (pHash) of the median frame
  2. Motion  – mean optical-flow magnitude histogram
  3. Audio   – MFCC mean coefficients (if audio available)

Output:
  fingerprint_hex  – 64-char hex string (robust perceptual identity)
  embedding        – float32 numpy array of shape (FINGERPRINT_DIM,)
"""
import hashlib
import numpy as np
import cv2
from PIL import Image
import imagehash
from app.core.config import FINGERPRINT_DIM

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False


# ── Visual ────────────────────────────────────────────────────────────────────

def _phash_frame(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h = imagehash.phash(img, hash_size=8)   # 64-bit hash
    bits = np.array(h.hash, dtype=np.float32).flatten()  # shape (64,)
    return bits


def _visual_features(frames: list[np.ndarray]) -> np.ndarray:
    mid = frames[len(frames) // 2]
    phash_bits = _phash_frame(mid)                        # (64,)

    # colour histogram (3 channels × 16 bins = 48 values)
    hist_parts = []
    for ch in range(3):
        h, _ = np.histogram(mid[:, :, ch], bins=16, range=(0, 256))
        hist_parts.append(h.astype(np.float32) / (mid.shape[0] * mid.shape[1]))
    colour_hist = np.concatenate(hist_parts)              # (48,)

    return np.concatenate([phash_bits, colour_hist])      # (112,)


# ── Motion ────────────────────────────────────────────────────────────────────

def _motion_features(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) < 2:
        return np.zeros(32, dtype=np.float32)

    magnitudes = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(mag.mean())
        prev_gray = gray

    mag_arr = np.array(magnitudes, dtype=np.float32)
    hist, _ = np.histogram(mag_arr, bins=32, range=(0, mag_arr.max() + 1e-6))
    return hist.astype(np.float32) / (len(magnitudes) + 1e-6)   # (32,)


# ── Audio ─────────────────────────────────────────────────────────────────────

def _audio_features(audio_path: str | None) -> np.ndarray:
    if not LIBROSA_OK or audio_path is None:
        return np.zeros(40, dtype=np.float32)
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=1.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=20)
        return np.concatenate([mfcc.mean(axis=1), chroma.mean(axis=1)]).astype(np.float32)  # (40,)
    except Exception:
        return np.zeros(40, dtype=np.float32)


# ── Combined ──────────────────────────────────────────────────────────────────

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def extract_fingerprint(segment: dict) -> tuple[str, np.ndarray]:
    """
    Returns (fingerprint_hex, embedding_vector).
    fingerprint_hex  – deterministic 64-char hex from pHash of median frame
    embedding_vector – L2-normalised float32 array of shape (FINGERPRINT_DIM,)
    """
    frames = segment["frames"]
    audio_path = segment.get("audio_path")

    visual = _visual_features(frames)    # (112,)
    motion = _motion_features(frames)    # (32,)
    audio  = _audio_features(audio_path) # (40,)

    raw = np.concatenate([visual, motion, audio])  # (184,)

    # Pad / truncate to FINGERPRINT_DIM
    if len(raw) < FINGERPRINT_DIM:
        raw = np.pad(raw, (0, FINGERPRINT_DIM - len(raw)))
    else:
        raw = raw[:FINGERPRINT_DIM]

    embedding = _l2_normalize(raw.astype(np.float32))

    # Deterministic hex fingerprint from pHash bits of median frame
    mid_frame = frames[len(frames) // 2]
    img = Image.fromarray(cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB))
    phash_val = str(imagehash.phash(img, hash_size=8))
    fingerprint_hex = hashlib.sha256(phash_val.encode()).hexdigest()

    return fingerprint_hex, embedding
