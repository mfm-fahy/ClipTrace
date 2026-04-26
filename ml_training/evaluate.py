"""
Evaluation Script
Metrics:
  - Recall@K (K=1,5,10)  — retrieval accuracy
  - Precision / Recall / F1 at similarity threshold
  - Mean Average Precision (mAP)
  - Similarity distribution plot (same vs different)
  - t-SNE embedding visualisation
"""
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.manifold import TSNE
from tqdm import tqdm
from pathlib import Path
import csv

import config
from model import load_checkpoint
from dataset import EmbeddingDataset, TripletVideoDataset

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_all_embeddings(model, device) -> tuple[np.ndarray, list[str], list[str]]:
    """Returns (embeddings, video_ids, seg_paths)."""
    ds = EmbeddingDataset()
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=config.NUM_WORKERS)

    all_embs, all_vids, all_paths = [], [], []
    model.eval()

    for frames, video_ids, seg_paths in tqdm(loader, desc="Extracting embeddings"):
        embs = model(frames.to(device)).cpu().numpy()
        all_embs.append(embs)
        all_vids.extend(video_ids)
        all_paths.extend(seg_paths)

    return np.vstack(all_embs), all_vids, all_paths


# ── Recall@K ──────────────────────────────────────────────────────────────────

def recall_at_k(embeddings: np.ndarray, video_ids: list[str], k_values: list[int]) -> dict:
    """
    For each query segment, retrieve top-K and check if any result
    shares the same video_id (excluding the query itself).
    """
    n = len(embeddings)
    results = {k: 0 for k in k_values}

    if FAISS_OK:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        max_k = max(k_values) + 1
        _, indices = index.search(embeddings.astype(np.float32), max_k)
    else:
        # Numpy fallback
        sim_matrix = embeddings @ embeddings.T
        indices = np.argsort(-sim_matrix, axis=1)

    for i in range(n):
        query_vid = video_ids[i]
        # Exclude self (index 0 is always the query itself)
        retrieved = [j for j in indices[i] if j != i]

        for k in k_values:
            top_k = retrieved[:k]
            if any(video_ids[j] == query_vid for j in top_k):
                results[k] += 1

    return {f"Recall@{k}": round(results[k] / n, 4) for k in k_values}


# ── Precision / Recall / F1 ───────────────────────────────────────────────────

def precision_recall_at_threshold(
    embeddings: np.ndarray,
    video_ids: list[str],
    threshold: float = config.SIMILARITY_THRESHOLD,
) -> dict:
    """Compute P/R/F1 treating similarity >= threshold as 'same video' prediction."""
    n = len(embeddings)
    y_true, y_pred = [], []

    sim_matrix = embeddings @ embeddings.T

    for i in range(n):
        for j in range(i + 1, n):
            same = int(video_ids[i] == video_ids[j])
            pred = int(sim_matrix[i, j] >= threshold)
            y_true.append(same)
            y_pred.append(pred)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, [sim_matrix[i // (n-1), i % (n-1)] for i in range(len(y_true))])
    except Exception:
        auc = 0.0

    return {
        "precision": round(float(p), 4),
        "recall":    round(float(r), 4),
        "f1":        round(float(f1), 4),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_similarity_distribution(embeddings: np.ndarray, video_ids: list[str], out_dir: Path):
    """Plot cosine similarity distributions for same-video vs different-video pairs."""
    n = min(len(embeddings), 500)   # cap for speed
    embs = embeddings[:n]
    vids = video_ids[:n]

    sim_matrix = embs @ embs.T
    same_sims, diff_sims = [], []

    for i in range(n):
        for j in range(i + 1, n):
            s = sim_matrix[i, j]
            if vids[i] == vids[j]:
                same_sims.append(s)
            else:
                diff_sims.append(s)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(same_sims, bins=50, alpha=0.6, color="#6366f1", label="Same video")
    ax.hist(diff_sims, bins=50, alpha=0.6, color="#f43f5e", label="Different video")
    ax.axvline(config.SIMILARITY_THRESHOLD, color="orange", linestyle="--", label=f"Threshold ({config.SIMILARITY_THRESHOLD})")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Embedding Similarity Distribution")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "similarity_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_tsne(embeddings: np.ndarray, video_ids: list[str], out_dir: Path):
    """t-SNE visualisation of embeddings coloured by video_id."""
    n = min(len(embeddings), 300)
    embs = embeddings[:n]
    vids = video_ids[:n]

    unique_vids = sorted(set(vids))
    color_map = {v: i for i, v in enumerate(unique_vids)}
    colors = [color_map[v] for v in vids]

    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, n - 1), random_state=42, max_iter=1000)
    proj = tsne.fit_transform(embs)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=colors, cmap="tab20", alpha=0.7, s=20)
    ax.set_title("t-SNE of Segment Embeddings (coloured by video)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    path = out_dir / "tsne.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_training_history(out_dir: Path):
    history_path = config.LOGS_DIR / "history.json"
    if not history_path.exists():
        return
    with open(history_path) as f:
        history = json.load(f)

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    val_acc    = [h["val_acc"]    for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="Train Loss", color="#6366f1")
    ax1.plot(epochs, val_loss,   label="Val Loss",   color="#f43f5e")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(epochs, val_acc, label="Val Accuracy", color="#10b981")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    path = out_dir / "training_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(checkpoint_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = str(config.CHECKPOINTS / "best.pt")

    print(f"[Eval] Loading checkpoint: {checkpoint_path}")
    model, ckpt_meta = load_checkpoint(checkpoint_path, device)
    model.eval()

    print("[Eval] Extracting embeddings...")
    embeddings, video_ids, seg_paths = extract_all_embeddings(model, device)
    print(f"[Eval] {len(embeddings)} segments from {len(set(video_ids))} videos")

    out_dir = config.LOGS_DIR
    results = {}

    print("\n[Eval] Computing Recall@K...")
    recall = recall_at_k(embeddings, video_ids, config.TOP_K_VALUES)
    results.update(recall)
    for k, v in recall.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Eval] Computing Precision / Recall / F1...")
    prf = precision_recall_at_threshold(embeddings, video_ids)
    results.update(prf)
    for k, v in prf.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Eval] Generating plots...")
    plot_similarity_distribution(embeddings, video_ids, out_dir)
    plot_tsne(embeddings, video_ids, out_dir)
    plot_training_history(out_dir)

    # Save results JSON
    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] Results saved to {results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate(ckpt)
