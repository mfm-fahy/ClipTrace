"""
Shared video processing helper.
All CPU-heavy work (segmentation, fingerprinting, ML inference) runs in
a thread-pool executor so it never blocks the async event loop.
"""
import asyncio
import numpy as np
from pathlib import Path
from functools import partial

from app.processing.segmenter import extract_segments, cleanup_audio_files
from app.processing.fingerprint import extract_fingerprint
from app.ml.embedder import get_ml_embedding


def _process_video_sync(video_path: str) -> tuple[list[dict], list[np.ndarray], list[str]]:
    """
    Runs synchronously in a thread pool.
    Returns (segments, embeddings, fingerprints_hex).
    """
    segments = extract_segments(video_path)
    embeddings = []
    fingerprints_hex = []

    for seg in segments:
        fp_hex, det_emb = extract_fingerprint(seg)
        ml_emb = get_ml_embedding(seg["frames"])

        if ml_emb is not None:
            combined = np.concatenate([det_emb, ml_emb])
            norm = np.linalg.norm(combined)
            final_emb = (combined / (norm + 1e-8)).astype(np.float32)
        else:
            final_emb = det_emb

        embeddings.append(final_emb)
        fingerprints_hex.append(fp_hex)

    return segments, embeddings, fingerprints_hex


async def process_video(video_path: str) -> tuple[list[dict], list[np.ndarray], list[str]]:
    """Async wrapper — offloads CPU work to thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _process_video_sync, video_path)
