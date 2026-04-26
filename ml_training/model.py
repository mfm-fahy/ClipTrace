"""
ClipTrace Embedding Model
Architecture:
  Backbone  → MobileNetV3-Small (pretrained, feature extractor)
  Pool      → AdaptiveAvgPool → flatten
  Projector → FC(576 → 512) → BN → ReLU → Dropout → FC(512 → EMBEDDING_DIM)
  Output    → L2-normalised embedding vector

The same network is used for all three branches of the triplet (shared weights).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import config


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = config.EMBEDDING_DIM, backbone: str = config.BACKBONE):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        if backbone == "mobilenet_v3_small":
            base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            feature_dim = 576
            # Keep only the feature extractor (drop classifier)
            self.backbone = nn.Sequential(base.features, base.avgpool)

        elif backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feature_dim = 512
            self.backbone = nn.Sequential(*list(base.children())[:-1])  # drop FC

        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = 1280
            self.backbone = nn.Sequential(base.features, base.avgpool)

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.feature_dim = feature_dim

        # ── Projection head ───────────────────────────────────────────────────
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        returns: (B, embedding_dim) L2-normalised
        """
        features = self.backbone(x)          # (B, feature_dim, 1, 1) or (B, feature_dim)
        embedding = self.projector(features) # (B, embedding_dim)
        return F.normalize(embedding, p=2, dim=1)

    def freeze_backbone(self):
        """Freeze backbone weights — train only the projection head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ── Loss Functions ────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    Triplet loss with L2-normalised embeddings (cosine distance via dot product).
    Loss = max(0, d(a,p) - d(a,n) + margin)
    where d = 1 - cosine_similarity (since embeddings are L2-normalised)
    """
    def __init__(self, margin: float = config.TRIPLET_MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        # Cosine distance = 1 - dot product (embeddings are L2-normalised)
        d_pos = 1.0 - (anchor * positive).sum(dim=1)   # (B,)
        d_neg = 1.0 - (anchor * negative).sum(dim=1)   # (B,)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean(), d_pos.mean().item(), d_neg.mean().item()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss:
      label=1 (same)      → minimise distance
      label=0 (different) → push apart up to margin
    """
    def __init__(self, margin: float = config.CONTRASTIVE_MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor):
        dist = 1.0 - (emb1 * emb2).sum(dim=1)   # cosine distance
        loss = label * dist.pow(2) + (1 - label) * F.relu(self.margin - dist).pow(2)
        return loss.mean()


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> EmbeddingNet:
    model = EmbeddingNet()
    model.freeze_backbone()   # start with frozen backbone
    return model.to(device)


def load_checkpoint(path: str, device: torch.device) -> tuple[EmbeddingNet, dict]:
    checkpoint = torch.load(path, map_location=device)
    model = EmbeddingNet(
        embedding_dim=checkpoint.get("embedding_dim", config.EMBEDDING_DIM),
        backbone=checkpoint.get("backbone", config.BACKBONE),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, checkpoint
