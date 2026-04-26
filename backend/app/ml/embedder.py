"""
ML Embedding Module
Lazy-loads MobileNetV3-Small on first call to avoid blocking server startup.
Loads trained ClipTrace weights if available, otherwise uses ImageNet pretrained.
"""
import numpy as np
import cv2
from pathlib import Path

_model = None
_transform = None
_TORCH_OK = None
_WEIGHTS_PATH = Path(__file__).parent / "cliptrace_embedder.pt"


def _load_model():
    global _model, _transform, _TORCH_OK
    if _TORCH_OK is not None:
        return _TORCH_OK
    try:
        import torch
        import torchvision.transforms as T
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        model = mobilenet_v3_small(weights=None)
        model.classifier = torch.nn.Identity()

        if _WEIGHTS_PATH.exists():
            ckpt = torch.load(str(_WEIGHTS_PATH), map_location="cpu", weights_only=False)
            state = ckpt.get("model_state", {})
            # Load only backbone weights (strip projector keys)
            backbone_state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
            if backbone_state:
                model.load_state_dict(backbone_state, strict=False)
        else:
            # Fall back to ImageNet pretrained
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            model.classifier = torch.nn.Identity()

        model.eval()
        _model = model

        _transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _TORCH_OK = True
    except Exception as e:
        print(f"[Embedder] torch unavailable: {e}")
        _TORCH_OK = False
    return _TORCH_OK


def get_ml_embedding(frames: list[np.ndarray]) -> np.ndarray | None:
    """
    Returns a 576-dim L2-normalised float32 embedding, or None if torch unavailable.
    Model is loaded on first call (lazy init).
    """
    if not frames or not _load_model():
        return None

    import torch
    mid = frames[len(frames) // 2]
    rgb = cv2.cvtColor(mid, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        tensor = _transform(rgb).unsqueeze(0)
        feat = _model(tensor).squeeze().numpy()

    norm = np.linalg.norm(feat)
    return (feat / (norm + 1e-8)).astype(np.float32)
