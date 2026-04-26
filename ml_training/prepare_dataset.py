"""
Dataset Preparation
1. Scans RAW_DIR for videos (expects subfolders: original/, edited/)
2. Extracts FRAMES_PER_SEGMENT evenly-spaced frames per 1-second segment
3. Saves frames as JPEGs under SEGMENTS_DIR/{video_id}/{seg_idx}/frame_{n}.jpg
4. Builds a pairs CSV: (anchor_path, positive_path, negative_path, video_id)

Directory layout expected:
  data/raw/
    original/
      video_001.mp4
      video_002.mp4
    edited/
      video_001_crop.mp4      ← edited version of video_001
      video_001_blur.mp4
      video_002_filter.mp4
"""
import cv2
import csv
import random
import numpy as np
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
import config

# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_segment_frames(video_path: Path, out_dir: Path) -> int:
    """
    Extract FRAMES_PER_SEGMENT frames per 1-second segment.
    Returns number of segments extracted.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path.name}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_seg = max(1, int(fps * config.SEGMENT_DURATION))
    n_segs = total_frames // frames_per_seg

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    seg_count = 0
    for seg_idx in range(n_segs):
        seg_frames = all_frames[seg_idx * frames_per_seg: (seg_idx + 1) * frames_per_seg]
        if not seg_frames:
            continue

        # Sample FRAMES_PER_SEGMENT evenly
        indices = np.linspace(0, len(seg_frames) - 1, config.FRAMES_PER_SEGMENT, dtype=int)
        sampled = [seg_frames[i] for i in indices]

        seg_dir = out_dir / str(seg_idx)
        seg_dir.mkdir(parents=True, exist_ok=True)

        for f_idx, frame in enumerate(sampled):
            resized = cv2.resize(frame, config.FRAME_SIZE)
            cv2.imwrite(str(seg_dir / f"frame_{f_idx:02d}.jpg"), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

        seg_count += 1

    return seg_count


def prepare_segments():
    """Extract frames from all videos in RAW_DIR."""
    original_dir = config.RAW_DIR / "original"
    edited_dir   = config.RAW_DIR / "edited"

    if not original_dir.exists():
        original_dir.mkdir(parents=True)
        print(f"[INFO] Created {original_dir} — add original videos here")

    if not edited_dir.exists():
        edited_dir.mkdir(parents=True)
        print(f"[INFO] Created {edited_dir} — add edited videos here")

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    all_videos = list(original_dir.glob("*")) + list(edited_dir.glob("*"))
    all_videos = [v for v in all_videos if v.suffix.lower() in video_exts]

    if not all_videos:
        print("[WARN] No videos found. Generating synthetic dataset for demo...")
        _generate_synthetic_dataset()
        return

    print(f"[INFO] Processing {len(all_videos)} videos...")
    for video_path in tqdm(all_videos, desc="Extracting segments"):
        # video_id = stem without edit suffix (e.g. video_001_blur → video_001)
        stem = video_path.stem
        video_id = stem.split("_crop")[0].split("_blur")[0].split("_filter")[0].split("_edit")[0]
        variant  = stem  # full name as variant key

        out_dir = config.SEGMENTS_DIR / video_id / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        n = extract_segment_frames(video_path, out_dir)
        print(f"  {video_path.name}: {n} segments")


# ── Synthetic dataset (demo / CI) ─────────────────────────────────────────────

def _generate_synthetic_dataset():
    """
    Creates synthetic segment folders with random coloured frames
    so the full pipeline can be tested without real videos.
    10 video IDs × 2 variants × 20 segments × 8 frames.
    """
    print("[INFO] Generating synthetic dataset (10 videos × 2 variants × 20 segments)...")
    random.seed(42)
    np.random.seed(42)

    for vid_id in range(10):
        video_id = f"video_{vid_id:03d}"
        # Base colour for this video (so same-video segments share colour family)
        base_color = np.array([random.randint(50, 200) for _ in range(3)], dtype=np.uint8)

        for variant_idx, variant_name in enumerate(["original", f"edited_v1"]):
            for seg_idx in range(20):
                seg_dir = config.SEGMENTS_DIR / video_id / variant_name / str(seg_idx)
                seg_dir.mkdir(parents=True, exist_ok=True)

                for f_idx in range(config.FRAMES_PER_SEGMENT):
                    # Same-video: small noise around base colour
                    noise = np.random.randint(-30, 30, 3)
                    color = np.clip(base_color + noise, 0, 255).astype(np.uint8)

                    if variant_idx == 1:
                        # Edited variant: add stronger perturbation
                        edit_noise = np.random.randint(-60, 60, 3)
                        color = np.clip(color + edit_noise, 0, 255).astype(np.uint8)

                    frame = np.ones((*config.FRAME_SIZE, 3), dtype=np.uint8) * color
                    # Add some structure (gradient + random shapes)
                    frame[:, :, 0] = np.linspace(int(color[0]), min(int(color[0]) + 40, 255), config.FRAME_SIZE[1])
                    cv2.circle(frame, (112 + seg_idx * 3, 112), 30, color.tolist(), -1)

                    cv2.imwrite(str(seg_dir / f"frame_{f_idx:02d}.jpg"), frame)

    print("[INFO] Synthetic dataset ready.")


# ── Pair building ─────────────────────────────────────────────────────────────

def build_pairs():
    """
    Build triplet pairs CSV: anchor, positive, negative, video_id
    - anchor   = segment from original variant
    - positive = same segment index from edited variant (same video)
    - negative = any segment from a different video
    """
    # Collect all segment paths grouped by video_id
    video_segments: dict[str, list[Path]] = {}

    for video_dir in sorted(config.SEGMENTS_DIR.iterdir()):
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name
        segs = []
        for variant_dir in video_dir.iterdir():
            for seg_dir in sorted(variant_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 0):
                if seg_dir.is_dir() and any(seg_dir.glob("*.jpg")):
                    segs.append(seg_dir)
        if segs:
            video_segments[video_id] = segs

    if len(video_segments) < 2:
        print("[ERROR] Need at least 2 videos to build pairs.")
        return

    video_ids = list(video_segments.keys())
    triplets = []

    for video_id, segs in video_segments.items():
        # Group by segment index across variants
        by_index: dict[int, list[Path]] = {}
        for seg_path in segs:
            idx = int(seg_path.name) if seg_path.name.isdigit() else 0
            by_index.setdefault(idx, []).append(seg_path)

        for seg_idx, variants in by_index.items():
            if len(variants) < 2:
                continue  # need at least 2 variants for positive pair

            for anchor, positive in combinations(variants, 2):
                # Pick a negative from a different video
                neg_video = random.choice([v for v in video_ids if v != video_id])
                negative = random.choice(video_segments[neg_video])
                triplets.append((str(anchor), str(positive), str(negative), video_id))

    # Shuffle and save
    random.shuffle(triplets)
    out_csv = config.PAIRS_DIR / "triplets.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["anchor", "positive", "negative", "video_id"])
        writer.writerows(triplets)

    print(f"[INFO] Built {len(triplets)} triplets -> {out_csv}")
    return len(triplets)


if __name__ == "__main__":
    print("=== Step 1: Extracting segments ===")
    prepare_segments()
    print("\n=== Step 2: Building triplet pairs ===")
    build_pairs()
    print("\nDone. Run train.py next.")
