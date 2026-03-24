"""
WardrobeMultiHeadModel – Multi-head classifier built on a frozen/fine-tunable
ResNet-50 backbone.  Returns raw logits (no Sigmoid/Softmax) so the training
loop can use ``BCEWithLogitsLoss`` with NaN-masking (target == -1).
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)


class WardrobeMultiHeadModel(nn.Module):
    """ResNet-50 backbone with 4 independent classification heads.

    Parameters
    ----------
    num_category : int
        Number of category classes (default 148).
    num_color : int
        Number of colour classes (default 21).
    num_fabric : int
        Number of fabric / material classes (default 40).
    num_style : int
        Number of style classes (default 80).
    """

    def __init__(
        self,
        num_category: int = 148,
        num_color: int = 21,
        num_fabric: int = 40,
        num_style: int = 80,
    ) -> None:
        super().__init__()

        # ── backbone ──────────────────────────────────────────────
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features: int = backbone.fc.in_features          # 2048
        backbone.fc = nn.Identity()                         # strip classifier
        self.backbone = backbone

        # ── classification heads ──────────────────────────────────
        self.heads = nn.ModuleDict({
            "category": nn.Linear(in_features, num_category),
            "color":    nn.Linear(in_features, num_color),
            "fabric":   nn.Linear(in_features, num_fabric),
            "style":    nn.Linear(in_features, num_style),
        })

    # ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return **raw logits** for each head (no activation applied).

        Parameters
        ----------
        x : torch.Tensor
            Image batch of shape ``(B, 3, 224, 224)``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{'category': (B, 148), 'color': (B, 21),
               'fabric': (B, 40), 'style': (B, 80)}``
        """
        features = self.backbone(x)                         # (B, 2048)
        return {name: head(features) for name, head in self.heads.items()}


# ──────────────────────────── sanity-check ────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WardrobeMultiHeadModel().to(device)

    dummy = torch.randn(2, 3, 224, 224, device=device)
    logits = model(dummy)

    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Device          : {device}")
    logger.info(f"Total params    : {total_params:,}")
    logger.info(f"Trainable params: {trainable:,}")
    logger.info("Output shapes (expected [batch_size, num_classes]):")
    for name, t in logits.items():
        logger.info(f"  {name:10s} → {list(t.shape)}  "
                    f"dtype={t.dtype}  min={t.min().item():.3f}  max={t.max().item():.3f}")
    logger.info("Sanity-check passed ✓")
