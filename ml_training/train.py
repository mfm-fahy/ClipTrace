"""
Training Script
Phase 1 (epochs 1-5):   Backbone frozen  - train projection head only
Phase 2 (epochs 6-end): Backbone unfrozen - full fine-tuning with lower LR

Supports both triplet and contrastive loss modes via config.LOSS_TYPE.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import json
import time

import config
from model import EmbeddingNet, TripletLoss, ContrastiveLoss, build_model
from dataset import TripletVideoDataset, ContrastiveVideoDataset, split_csv

FREEZE_EPOCHS = 5   # epochs to train with frozen backbone


def train_epoch_triplet(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_d_pos = total_d_neg = 0.0

    for anchor, positive, negative in tqdm(loader, desc="  train", leave=False):
        anchor   = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss, d_pos, d_neg = criterion(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        total_d_pos += d_pos
        total_d_neg += d_neg

    n = len(loader)
    return total_loss / n, total_d_pos / n, total_d_neg / n


def train_epoch_contrastive(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for f1, f2, label in tqdm(loader, desc="  train", leave=False):
        f1, f2, label = f1.to(device), f2.to(device), label.to(device)
        emb1 = model(f1)
        emb2 = model(f2)
        loss = criterion(emb1, emb2, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate_triplet(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0.0

    for anchor, positive, negative in tqdm(loader, desc="  val  ", leave=False):
        anchor   = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss, d_pos, d_neg = criterion(emb_a, emb_p, emb_n)
        total_loss += loss.item()

        # Accuracy: d(a,p) < d(a,n)
        correct += (d_pos < d_neg)
        total   += 1

    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate_contrastive(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0.0

    for f1, f2, label in tqdm(loader, desc="  val  ", leave=False):
        f1, f2, label = f1.to(device), f2.to(device), label.to(device)
        emb1 = model(f1)
        emb2 = model(f2)
        loss = criterion(emb1, emb2, label)
        total_loss += loss.item()

        sim = (emb1 * emb2).sum(dim=1)
        pred = (sim >= config.SIMILARITY_THRESHOLD).float()
        correct += (pred == label).sum().item()
        total   += label.size(0)

    return total_loss / len(loader), correct / total


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, is_best=False):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "embedding_dim": config.EMBEDDING_DIM,
        "backbone": config.BACKBONE,
        "loss_type": config.LOSS_TYPE,
    }
    path = config.CHECKPOINTS / f"epoch_{epoch:03d}.pt"
    torch.save(state, path)
    if is_best:
        torch.save(state, config.CHECKPOINTS / "best.pt")
        print(f"  [BEST] New best model saved (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
    return path


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    print(f"[Train] Loss:   {config.LOSS_TYPE}")
    print(f"[Train] Backbone: {config.BACKBONE}")

    # ── Data ──────────────────────────────────────────────────────────────────
    triplets_csv = config.PAIRS_DIR / "triplets.csv"
    if not triplets_csv.exists():
        raise FileNotFoundError(f"Run prepare_dataset.py first. Missing: {triplets_csv}")

    train_csv, val_csv, test_csv = split_csv(triplets_csv)

    if config.LOSS_TYPE == "triplet":
        train_ds = TripletVideoDataset(train_csv, augment=True)
        val_ds   = TripletVideoDataset(val_csv,   augment=False)
        criterion = TripletLoss()
    else:
        train_ds = ContrastiveVideoDataset(train_csv, augment=True)
        val_ds   = ContrastiveVideoDataset(val_csv,   augment=False)
        criterion = ContrastiveLoss()

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=device.type == "cuda")

    print(f"[Train] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model + Optimizer ─────────────────────────────────────────────────────
    model = build_model(device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)

    writer = SummaryWriter(log_dir=str(config.LOGS_DIR / f"run_{int(time.time())}"))
    best_val_loss = float("inf")
    history = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, config.NUM_EPOCHS + 1):

        # Phase transition: unfreeze backbone after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n[Train] Epoch {epoch}: Unfreezing backbone for fine-tuning...")
            model.unfreeze_backbone()
            # Rebuild optimizer with lower LR for backbone
            optimizer = optim.AdamW([
                {"params": model.backbone.parameters(), "lr": config.LEARNING_RATE * 0.1},
                {"params": model.projector.parameters(), "lr": config.LEARNING_RATE},
            ], weight_decay=config.WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)

        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}  LR={scheduler.get_last_lr()}")

        if config.LOSS_TYPE == "triplet":
            train_loss, d_pos, d_neg = train_epoch_triplet(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_triplet(model, val_loader, criterion, device)
            print(f"  train_loss={train_loss:.4f}  d_pos={d_pos:.4f}  d_neg={d_neg:.4f}")
            writer.add_scalar("dist/positive", d_pos, epoch)
            writer.add_scalar("dist/negative", d_neg, epoch)
        else:
            train_loss = train_epoch_contrastive(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_contrastive(model, val_loader, criterion, device)
            print(f"  train_loss={train_loss:.4f}")

        print(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.add_scalar("acc/val",    val_acc,    epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, is_best)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        scheduler.step()

    writer.close()

    # Save history
    with open(config.LOGS_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Train] Done. Best val_loss={best_val_loss:.4f}")
    print(f"[Train] Best checkpoint: {config.CHECKPOINTS / 'best.pt'}")
    print(f"[Train] Run: python export_model.py  to export for ClipTrace")


if __name__ == "__main__":
    train()
