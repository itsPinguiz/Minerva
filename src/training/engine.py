import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import WardrobeDataset
from src.models.wardrobe_net import WardrobeMultiHeadModel

def setup_logger(log_dir="logs"):
    """
    Configura il logger standard di Python.
    - Console output: Livello INFO.
    - File output: Livello DEBUG, salvato in logs/train.log
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    logger = logging.getLogger("WardrobeML")
    logger.setLevel(logging.DEBUG)

    # Previene duplicazione degli handler in Notebooks/iterazioni calde
    if not logger.handlers:
        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)

        # File Handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

def create_dataloaders(df_train, df_val, batch_size=32, num_workers=4):
    train_dataset = WardrobeDataset(df_train, is_train=True)
    val_dataset = WardrobeDataset(df_val, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader

def total_loss_function(preds, targets, weights=None):
    """Calcola loss pesata e splittata."""
    criterion = nn.CrossEntropyLoss()
    if weights is None:
        weights = {"category": 1.0, "color": 1.0, "fabric": 1.0, "style": 1.0}
        
    loss_cat = criterion(preds["category"], targets["category"])
    loss_col = criterion(preds["color"], targets["color"])
    loss_fab = criterion(preds["fabric"], targets["fabric"])
    loss_sty = criterion(preds["style"], targets["style"])
    
    total = (weights["category"] * loss_cat + 
             weights["color"] * loss_col + 
             weights["fabric"] * loss_fab + 
             weights["style"] * loss_sty)
             
    loss_dict = {
        "total": total,
        "category": loss_cat,
        "color": loss_col,
        "fabric": loss_fab,
        "style": loss_sty
    }
    return total, loss_dict

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, writer, logger):
    model.train()
    running_loss = 0.0
    head_losses = {"category": 0.0, "color": 0.0, "fabric": 0.0, "style": 0.0}
    
    pbar = tqdm(dataloader, desc=f"Train Epoca {epoch}", leave=False)
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            preds = model(images)
            loss, loss_dict = total_loss_function(preds, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        for head in head_losses.keys():
            head_losses[head] += loss_dict[head].item()
            
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Logging granulare su Tensorboard (ogni 10 step) per avere le code intra-epocha
        global_step = (epoch - 1) * len(dataloader) + batch_idx
        if global_step % 10 == 0:
            writer.add_scalar("Train_GStep/Total_Loss", loss.item(), global_step)
            for head in head_losses.keys():
                writer.add_scalar(f"Train_GStep/Loss_{head.capitalize()}", loss_dict[head].item(), global_step)
        
    avg_loss = running_loss / len(dataloader)
    avg_head_losses = {k: v / len(dataloader) for k, v in head_losses.items()}
    
    return avg_loss, avg_head_losses

def eval_one_epoch(model, dataloader, device):
    """
    Fase di inferenza Validation Mode. Misura sia Loss che Accuracy Multi-Head.
    """
    model.eval()
    running_loss = 0.0
    head_losses = {"category": 0.0, "color": 0.0, "fabric": 0.0, "style": 0.0}
    
    # Contatori Accuracy
    total_samples = 0
    correct_preds = {"category": 0, "color": 0, "fabric": 0, "style": 0}
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            
            with autocast():
                preds = model(images)
                loss, loss_dict = total_loss_function(preds, targets)
                
            batch_size = images.size(0)
            total_samples += batch_size
            running_loss += loss.item()
            
            # Conta predizioni esatte e aggrega validation losses separate
            for head in ["category", "color", "fabric", "style"]:
                head_losses[head] += loss_dict[head].item()
                pred_classes = torch.argmax(preds[head], dim=1)
                correct_preds[head] += (pred_classes == targets[head]).sum().item()
                
    avg_loss = running_loss / len(dataloader)
    avg_head_losses = {k: v / len(dataloader) for k, v in head_losses.items()}
    accuracies = {k: v / total_samples for k, v in correct_preds.items()}
    
    return avg_loss, avg_head_losses, accuracies

def fit(model, train_loader, val_loader, epochs=50, lr=1e-4, device="cuda"):
    """Loop Principale MLOps Orchestrato"""
    logger = setup_logger()
    writer = SummaryWriter(log_dir="runs/wardrobe_experiment")
    os.makedirs("weights", exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("🚀 Inizio Sessione di Training (Wardrobe Multi-Head)")
    logger.info(f"Hardware Mappato: {device.upper()}")
    logger.info(f"Parametri Base: Epoche={epochs} | Batch Size={train_loader.batch_size} | Base LR={lr}")
    logger.info("=" * 60)
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # Checkpointing Tracker
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        logger.debug(f"==> Avviando epoca {epoch}/{epochs}")
        
        # 1. Fase Addestrativa
        train_loss, train_head_l = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, writer, logger)
        
        # 2. Fase Validativa e Raccolta Metriche
        val_loss, val_head_l, val_accs = eval_one_epoch(model, val_loader, device)
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        
        # 3. MLOps Experiment Tracking (Tensorboard)
        # Metriche Globali
        writer.add_scalar("Loss/1_Train_Overall", train_loss, epoch)
        writer.add_scalar("Loss/2_Valid_Overall", val_loss, epoch)
        writer.add_scalar("Config/Learning_Rate", current_lr, epoch)
        
        # Metriche Specifiche per le 4 Teste
        for head in ["category", "color", "fabric", "style"]:
            writer.add_scalar(f"Valid_Loss_Split/{head.capitalize()}", val_head_l[head], epoch)
            writer.add_scalar(f"Valid_Accuracy/{head.capitalize()}", val_accs[head], epoch)
            
        # 4. Standard Python Logging Console/File
        acc_str = ", ".join([f"{k[:3].upper()}: {v*100:.1f}%" for k, v in val_accs.items()])
        logger.info(f"[Epoca {epoch:02d}/{epochs} - {epoch_time:.0f}s] "
                    f"Tr_Loss: {train_loss:.4f} | Val_Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
        logger.info(f"   ∟ Accuracy (Validation): [{acc_str}]")
        
        # 5. Model Checkpointing Intelligente
        if val_loss < best_val_loss:
            logger.info(f"   ★ Validation Loss migliorata da {best_val_loss:.4f} a {val_loss:.4f}. Checkpoint salvato!")
            best_val_loss = val_loss
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, "weights/best_model.pth")
            
    writer.close()
    logger.info("✅ Training Completato con Successo.")
    logger.info(f"Il miglior modello ha registrato Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    from src.data.dataset import create_dummy_dataframe 
    import warnings
    warnings.filterwarnings("ignore")
    
    dumb_df = create_dummy_dataframe(num_samples=128, save_path="data/raw/dummy_wardrobe.csv")
    loader_t, loader_v = create_dataloaders(dumb_df, dumb_df, batch_size=32, num_workers=2)
    
    net = WardrobeMultiHeadModel()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fit(net, loader_t, loader_v, epochs=3, device=dev)
