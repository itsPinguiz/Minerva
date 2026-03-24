"""
train.py – Training loop for the WardrobeMultiHeadModel.

Features
--------
* **Masked BCEWithLogitsLoss**: NaN targets (encoded as -1) are masked out
  so they do not contribute to the gradient.
* **Mixed-precision** via ``torch.amp`` (autocast + GradScaler) for 11 GB VRAM.
* **AdamW + ReduceLROnPlateau** on total validation loss.
* **Per-head validation metrics** logged to TensorBoard + console.
* **Best-model checkpointing** (epoch, optimizer, scheduler, scaler state).
* **Resume Training**: Can pause and resume using --resume flag.

Usage
-----
::

    uv run python src/models/train.py                        # defaults
    uv run python src/models/train.py --epochs 10 --bs 64    # override
    uv run python src/models/train.py --resume               # resumes from last epoch
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from tqdm import tqdm

# ── project imports (run from repo root) ──────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.dataset import HEAD_NAMES, get_dataloaders          # noqa: E402
from src.models.model import WardrobeMultiHeadModel                # noqa: E402

logger = logging.getLogger("WardrobeML")


# ======================================================================
#  Custom masked multi-label loss
# ======================================================================

class MaskedBCEWithLogitsLoss(nn.Module):
    """Wrapper around ``BCEWithLogitsLoss(reduction='none')`` that masks out
    targets set to **-1** (missing / NaN labels).

    For each head the loss is:

    1. Compute element-wise BCE on the full ``(B, C)`` matrix.
    2. Build a boolean mask where ``target != -1``.
    3. Zero-out the invalid positions.
    4. Return the **mean of valid elements only** (or 0 if none are valid).
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(total_loss, {head_name: head_loss})``."""
        head_losses: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=next(iter(logits.values())).device)

        for head in HEAD_NAMES:
            pred = logits[head]
            tgt = targets[head]

            # Mask:  valid where target is NOT the sentinel (-1)
            mask = (tgt != -1).float()
            n_valid = mask.sum()

            # Clamp target to [0,1] before BCE (the -1 positions will be
            # zeroed out by the mask anyway, but BCE requires valid inputs)
            safe_tgt = tgt.clamp(min=0.0)

            loss_raw = self.bce(pred, safe_tgt)       # (B, C)
            loss_masked = (loss_raw * mask).sum()

            head_loss = loss_masked / n_valid if n_valid > 0 else loss_masked
            head_losses[head] = head_loss
            total = total + head_loss

        return total, head_losses


# ======================================================================
#  One epoch helpers
# ======================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: MaskedBCEWithLogitsLoss,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> tuple[float, dict[str, float]]:
    model.train()
    running_loss = 0.0
    head_sums: dict[str, float] = {h: 0.0 for h in HEAD_NAMES}

    pbar = tqdm(loader, desc=f"Train  [{epoch:02d}]", leave=False)
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss, loss_dict = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        for h in HEAD_NAMES:
            head_sums[h] += loss_dict[h].item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Intra-epoch TensorBoard logging (every 50 steps)
        global_step = (epoch - 1) * len(loader) + step
        if global_step % 50 == 0:
            writer.add_scalar("Train_Step/Total_Loss", loss.item(), global_step)

    n = len(loader)
    return running_loss / n, {h: v / n for h, v in head_sums.items()}


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader,
    criterion: MaskedBCEWithLogitsLoss,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    running_loss = 0.0
    head_sums: dict[str, float] = {h: 0.0 for h in HEAD_NAMES}

    for images, targets in tqdm(loader, desc="Valid      ", leave=False):
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss, loss_dict = criterion(logits, targets)

        running_loss += loss.item()
        for h in HEAD_NAMES:
            head_sums[h] += loss_dict[h].item()

    n = len(loader)
    return running_loss / n, {h: v / n for h, v in head_sums.items()}


# ======================================================================
#  Main training driver
# ======================================================================

def train_model(
    csv_path: str = "data/interim/master_dataset.csv",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    num_workers: int = 4,
    device_name: str = "cuda",
    run_dir: str = "runs/wardrobe",
    checkpoint_dir: str = "weights",
    resume: bool = False,
) -> None:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── logging ───────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/train.log", mode="a"),
        ],
    )
    writer = SummaryWriter(log_dir=run_dir)

    # ── data ──────────────────────────────────────────────────────
    logger.info("Loading data from %s ...", csv_path)
    train_loader, val_loader, _, encoders = get_dataloaders(
        csv_path=csv_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    head_sizes = {h: len(encoders[h].classes_) for h in HEAD_NAMES}
    logger.info("Head sizes: %s", head_sizes)

    # ── model ─────────────────────────────────────────────────────
    model = WardrobeMultiHeadModel(
        num_category=head_sizes["category"],
        num_color=head_sizes["color"],
        num_fabric=head_sizes["fabric"],
        num_style=head_sizes["style"],
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded → %s params on %s", f"{total_params:,}", device)

    # ── optimiser / scheduler / scaler / criterion ────────────────
    criterion = MaskedBCEWithLogitsLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True,
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Checkpoint Resuming Logic ─────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")
    last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")

    if resume and os.path.exists(last_ckpt_path):
        logger.info("Trovato checkpoint in %s. Ripristino in corso...", last_ckpt_path)
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") and scaler:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info("Training ripristinato con successo. Ripartiamo dall'epoca %d!", start_epoch)

    # ── training loop ─────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info(
        "Starting training  |  epochs=%d  batch=%d  lr=%.1e  device=%s",
        epochs, batch_size, lr, device,
    )
    logger.info("=" * 65)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # — train —
        train_loss, train_heads = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, writer,
        )

        # — validate —
        val_loss, val_heads = eval_one_epoch(
            model, val_loader, criterion, device,
        )

        # — scheduler step —
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # — TensorBoard epoch-level —
        writer.add_scalar("Epoch/Train_Loss", train_loss, epoch)
        writer.add_scalar("Epoch/Val_Loss", val_loss, epoch)
        writer.add_scalar("Epoch/LR", current_lr, epoch)
        for h in HEAD_NAMES:
            writer.add_scalar(f"Val_Head/{h}", val_heads[h], epoch)

        # — console / file log —
        head_str = "  ".join(f"{h[:3].upper()}={val_heads[h]:.4f}" for h in HEAD_NAMES)
        logger.info(
            "[Epoch %02d/%d  %4.0fs]  train=%.4f  val=%.4f  lr=%.1e  |  %s",
            epoch, epochs, elapsed, train_loss, val_loss, current_lr, head_str,
        )

        # — checkpoint best model —
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "head_sizes": head_sizes,
                },
                best_ckpt_path,
            )
            logger.info(
                "   -> New best val_loss=%.4f  – checkpoint saved to %s",
                best_val_loss, best_ckpt_path,
            )
            
        # — checkpoint last model (ALWAYS SAVE AT END OF EPOCH) —
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "head_sizes": head_sizes,
            },
            last_ckpt_path,
        )
        logger.info("   -> Last checkpoint saved for resuming to %s", last_ckpt_path)

    writer.close()
    logger.info("Training complete.  Best val loss: %.4f", best_val_loss)


# ======================================================================
#  CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train WardrobeMultiHeadModel")
    p.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    p.add_argument("--csv", help="Override CSV path")
    p.add_argument("--epochs", type=int, help="Override number of epochs")
    p.add_argument("--bs", type=int, help="Override batch size")
    p.add_argument("--lr", type=float, help="Override learning rate")
    p.add_argument("--device", help="Override device (cuda/cpu)")
    p.add_argument("--resume", action="store_true", help="Resume training from last_checkpoint.pth")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config)

    # CLI Overrides
    csv_path = args.csv or cfg.data.csv_path
    epochs = args.epochs or cfg.training.epochs
    batch_size = args.bs or cfg.training.batch_size
    lr = args.lr or cfg.training.learning_rate
    device_name = args.device or cfg.system.device

    train_model(
        csv_path=csv_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=cfg.training.weight_decay,
        num_workers=cfg.system.num_workers,
        device_name=device_name,
        run_dir=f"runs/{cfg.model.backbone}_{cfg.system.seed}",
        checkpoint_dir="weights",
        resume=args.resume,
    )