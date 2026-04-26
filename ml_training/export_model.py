"""
Model Export
Converts the best checkpoint to:
  1. TorchScript (.pt) — for production inference
  2. Copies embedder.py-compatible weights into backend/app/ml/
"""
import torch
import shutil
from pathlib import Path
import config
from model import load_checkpoint, EmbeddingNet


def export(checkpoint_path: str = None, export_name: str = "cliptrace_embedder"):
    device = torch.device("cpu")   # export on CPU for portability

    if checkpoint_path is None:
        checkpoint_path = str(config.CHECKPOINTS / "best.pt")

    print(f"[Export] Loading: {checkpoint_path}")
    model, meta = load_checkpoint(checkpoint_path, device)
    model.eval()

    # ── TorchScript export ────────────────────────────────────────────────────
    example = torch.randn(1, 3, *config.FRAME_SIZE)
    traced = torch.jit.trace(model, example)

    ts_path = config.EXPORTS_DIR / f"{export_name}.torchscript.pt"
    traced.save(str(ts_path))
    print(f"[Export] TorchScript saved: {ts_path}")

    # ── Raw state dict export ─────────────────────────────────────────────────
    state_path = config.EXPORTS_DIR / f"{export_name}.pt"
    torch.save({
        "model_state":   model.state_dict(),
        "embedding_dim": meta.get("embedding_dim", config.EMBEDDING_DIM),
        "backbone":      meta.get("backbone",      config.BACKBONE),
        "val_loss":      meta.get("val_loss"),
        "val_acc":       meta.get("val_acc"),
        "epoch":         meta.get("epoch"),
    }, str(state_path))
    print(f"[Export] State dict saved: {state_path}")

    # ── Copy into backend ─────────────────────────────────────────────────────
    backend_ml_dir = config.BASE_DIR.parent / "backend" / "app" / "ml"
    if backend_ml_dir.exists():
        dest = backend_ml_dir / f"{export_name}.pt"
        shutil.copy2(str(state_path), str(dest))
        print(f"[Export] Copied to backend: {dest}")

        # Patch embedder.py to load the trained weights
        _patch_embedder(backend_ml_dir, export_name, meta)
    else:
        print(f"[Export] Backend not found at {backend_ml_dir} — skipping copy")

    print("\n[Export] Done.")
    print(f"  TorchScript : {ts_path}")
    print(f"  State dict  : {state_path}")


def _patch_embedder(ml_dir: Path, export_name: str, meta: dict):
    """Update backend embedder.py to load the trained model weights."""
    embedder_path = ml_dir / "embedder.py"
    if not embedder_path.exists():
        return

    trained_weights_line = f'_TRAINED_WEIGHTS = Path(__file__).parent / "{export_name}.pt"'

    content = embedder_path.read_text()

    # Inject trained weights loader if not already present
    if "_TRAINED_WEIGHTS" not in content:
        injection = f"""
# ── Trained ClipTrace weights (auto-injected by export_model.py) ──────────────
from pathlib import Path as _Path
_TRAINED_WEIGHTS_PATH = _Path(__file__).parent / "{export_name}.pt"

def _load_trained_weights(model):
    if _TRAINED_WEIGHTS_PATH.exists():
        import torch as _torch
        ckpt = _torch.load(str(_TRAINED_WEIGHTS_PATH), map_location="cpu")
        # Load only backbone + projector if available
        state = ckpt.get("model_state", {{}})
        try:
            model.load_state_dict(state, strict=False)
            print(f"[Embedder] Loaded trained weights from {{_TRAINED_WEIGHTS_PATH.name}}")
        except Exception as e:
            print(f"[Embedder] Could not load trained weights: {{e}}")
    return model
"""
        # Insert after imports
        insert_at = content.find("\nTORCH_OK = ")
        if insert_at > 0:
            content = content[:insert_at] + injection + content[insert_at:]
            embedder_path.write_text(content)
            print(f"[Export] Patched {embedder_path}")


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    export(ckpt)
