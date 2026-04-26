"""
Dataset Classes
- TripletVideoDataset  : yields (anchor, positive, negative) frame tensors
- ContrastiveVideoDataset : yields (frame1, frame2, label) for contrastive loss
- EmbeddingDataset     : yields (frames, video_id, seg_path) for evaluation
"""
import csv
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import config
from augmentations import apply_random_augmentations, get_train_transform, get_eval_transform


def _load_segment_frames(seg_dir: Path, augment: bool = False) -> torch.Tensor:
    """
    Load FRAMES_PER_SEGMENT frames from a segment directory.
    Returns tensor of shape (FRAMES_PER_SEGMENT, 3, H, W).
    """
    frame_paths = sorted(seg_dir.glob("frame_*.jpg"))
    if not frame_paths:
        # Return blank tensor if segment is empty
        return torch.zeros(config.FRAMES_PER_SEGMENT, 3, *config.FRAME_SIZE)

    transform = get_train_transform() if augment else get_eval_transform()
    frames = []

    # Sample or pad to exactly FRAMES_PER_SEGMENT
    indices = np.linspace(0, len(frame_paths) - 1, config.FRAMES_PER_SEGMENT, dtype=int)
    selected = [frame_paths[i] for i in indices]

    for fp in selected:
        img = cv2.imread(str(fp))
        if img is None:
            img = np.zeros((*config.FRAME_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if augment and random.random() < config.AUG_PROB:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bgr = apply_random_augmentations(bgr, n_augs=random.randint(1, 3))
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        frames.append(transform(img))

    return torch.stack(frames)   # (F, 3, H, W)


def _mean_pool_frames(frame_tensor: torch.Tensor) -> torch.Tensor:
    """Average frames into a single (3, H, W) tensor for CNN input."""
    return frame_tensor.mean(dim=0)


# ── Triplet Dataset ───────────────────────────────────────────────────────────

class TripletVideoDataset(Dataset):
    """
    Yields (anchor, positive, negative) where each is a (3, H, W) tensor
    representing the mean-pooled frames of a 1-second segment.
    """
    def __init__(self, csv_path: Path, augment: bool = True):
        self.augment = augment
        self.triplets = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.triplets.append((
                    Path(row["anchor"]),
                    Path(row["positive"]),
                    Path(row["negative"]),
                    row["video_id"],
                ))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, pos_path, neg_path, video_id = self.triplets[idx]

        anchor   = _mean_pool_frames(_load_segment_frames(anchor_path, augment=False))
        positive = _mean_pool_frames(_load_segment_frames(pos_path,    augment=self.augment))
        negative = _mean_pool_frames(_load_segment_frames(neg_path,    augment=False))

        return anchor, positive, negative


# ── Contrastive Dataset ───────────────────────────────────────────────────────

class ContrastiveVideoDataset(Dataset):
    """
    Yields (frame1, frame2, label) where label=1 means same video, 0 means different.
    Built dynamically from the triplets CSV.
    """
    def __init__(self, csv_path: Path, augment: bool = True):
        self.augment = augment
        self.pairs = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Positive pair
                self.pairs.append((Path(row["anchor"]), Path(row["positive"]), 1))
                # Negative pair
                self.pairs.append((Path(row["anchor"]), Path(row["negative"]), 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        f1 = _mean_pool_frames(_load_segment_frames(p1, augment=False))
        f2 = _mean_pool_frames(_load_segment_frames(p2, augment=self.augment and label == 1))
        return f1, f2, torch.tensor(float(label))


# ── Evaluation Dataset ────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """
    Iterates all segments in SEGMENTS_DIR for embedding extraction / evaluation.
    Yields (frame_tensor, video_id, seg_path_str).
    """
    def __init__(self, segments_dir: Path = config.SEGMENTS_DIR):
        self.items = []
        for video_dir in sorted(segments_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            for variant_dir in video_dir.iterdir():
                for seg_dir in sorted(variant_dir.iterdir(),
                                      key=lambda x: int(x.name) if x.name.isdigit() else 0):
                    if seg_dir.is_dir() and any(seg_dir.glob("*.jpg")):
                        self.items.append((seg_dir, video_id))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seg_dir, video_id = self.items[idx]
        frames = _mean_pool_frames(_load_segment_frames(seg_dir, augment=False))
        return frames, video_id, str(seg_dir)


# ── Train / Val / Test split ──────────────────────────────────────────────────

def split_csv(csv_path: Path) -> tuple[Path, Path, Path]:
    """Split triplets.csv into train/val/test CSVs."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)

    splits = {
        "train": rows[:n_train],
        "val":   rows[n_train:n_train + n_val],
        "test":  rows[n_train + n_val:],
    }

    paths = {}
    for split_name, split_rows in splits.items():
        out = config.PAIRS_DIR / f"{split_name}.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"  {split_name}: {len(split_rows)} triplets -> {out}")
        paths[split_name] = out

    return paths["train"], paths["val"], paths["test"]
