"""
Video Segmentation Engine
Splits a video into SEGMENT_DURATION-second chunks and yields frames + audio per segment.
"""
import cv2
import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path
from app.core.config import SEGMENT_DURATION


def extract_segments(video_path: str) -> list[dict]:
    """
    Returns a list of segment dicts:
      { index, timestamp_start, timestamp_end, frames: [np.ndarray], audio_path: str|None }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames_per_seg = max(1, int(fps * SEGMENT_DURATION))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    segments = []
    seg_index = 0
    frame_buffer = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)
        frame_count += 1

        if len(frame_buffer) == frames_per_seg:
            t_start = seg_index * SEGMENT_DURATION
            t_end = min(t_start + SEGMENT_DURATION, duration)
            audio_path = _extract_audio_segment(video_path, t_start, SEGMENT_DURATION)
            segments.append({
                "index": seg_index,
                "timestamp_start": t_start,
                "timestamp_end": t_end,
                "frames": frame_buffer.copy(),
                "audio_path": audio_path,
            })
            frame_buffer = []
            seg_index += 1

    # flush remaining frames as a partial segment
    if frame_buffer:
        t_start = seg_index * SEGMENT_DURATION
        t_end = duration
        audio_path = _extract_audio_segment(video_path, t_start, t_end - t_start)
        segments.append({
            "index": seg_index,
            "timestamp_start": t_start,
            "timestamp_end": t_end,
            "frames": frame_buffer,
            "audio_path": audio_path,
        })

    cap.release()
    return segments


def _extract_audio_segment(video_path: str, start: float, duration: float) -> str | None:
    """Extract a short audio clip using ffmpeg; returns temp wav path or None."""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start), "-t", str(max(duration, 0.1)),
            "-i", video_path,
            "-ac", "1", "-ar", "22050",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and os.path.getsize(tmp.name) > 0:
            return tmp.name
        os.unlink(tmp.name)
        return None
    except Exception:
        return None


def cleanup_audio_files(segments: list[dict]):
    for seg in segments:
        if seg.get("audio_path") and os.path.exists(seg["audio_path"]):
            os.unlink(seg["audio_path"])
