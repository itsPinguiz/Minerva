import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import WardrobeDataset
from src.models.wardrobe_net import WardrobeMultiHeadModel

def create_dataloaders(df_train, df_val, batch_size=32, num_workers=4):
    """
    Crea i DataLoader ottimizzati per la RTX 2080 Ti e 16GB RAM.
    Il parametro pin_memory=True garantisce trasferimenti memory-to-memory
    asincroni diretti verso la GPU, bypassando colli di bottiglia della CPU.
    """
    train_dataset = WardrobeDataset(df_train, is_train=True)
    val_dataset = WardrobeDataset(df_val, is_train=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,  # Cruciale per massimizzare la velocità RAM (16GB) -> VRAM (11GB)
        drop_last=True    # Previene spike di VRAM o shape irregolari per l'outlier (ultimo batch più piccolo)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True 
    )
    
    return train_loader, val_loader

def total_loss_function(preds, targets, weights=None):
    """
    Calcola la somma pesata della CrossEntropyLoss per le 4 teste del modello.
    È ideale vista la natura mutualmente esclusiva tipica delle classi di ogni testa.
    """
    criterion = nn.CrossEntropyLoss()
    
    if weights is None:
        # Possiamo assegnare un "boost" dinamico se un task sta retrocedendo
        weights = {"category": 1.0, "color": 1.0, "fabric": 1.0, "style": 1.0}
        
    loss_cat = criterion(preds["category"], targets["category"])
    loss_col = criterion(preds["color"], targets["color"])
    loss_fab = criterion(preds["fabric"], targets["fabric"])
    loss_sty = criterion(preds["style"], targets["style"])
    
    total = (weights["category"] * loss_cat + 
             weights["color"] * loss_col + 
             weights["fabric"] * loss_fab + 
             weights["style"] * loss_sty)
             
    # Costruiamo un dizionario per eventuale log delle singole loss su Tensorboard / W&B
    loss_dict = {
        "total": total,
        "category": loss_cat.detach(),
        "color": loss_col.detach(),
        "fabric": loss_fab.detach(),
        "style": loss_sty.detach()
    }
    
    return total, loss_dict

def train_one_epoch(model, dataloader, optimizer, scaler, device):
    """
    Esegue una singola epoca di training sfruttando Mixed Precision (AMP).
    Permette di ridurre i consumi VRAM dal FP32 quasi del 45%.
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training Epoch", leave=False)
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # 1. Spostamento target su GPU (non_blocking=True massimizza pin_memory)
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        
        # 2. Azzera i gradienti (per default si accumulano)
        optimizer.zero_grad(set_to_none=True) # set_to_none è sensibilmente più VRAM-friendly
        
        # 3. Autocast per far viaggiare la forward prettamente in FP16 
        with autocast():
            preds = model(images)
            loss, loss_dict = total_loss_function(preds, targets)
            
        # 4. Backward. 
        # Lo scaler scala la loss dinamicamente: previene l'azzeramento algoritmico ("underflow") in FP16
        scaler.scale(loss).backward()
        
        # 5. Passo di ottimizzazione un-scalando prima i gradienti
        scaler.step(optimizer)
        scaler.update()
        
        # Computo statistico
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def fit(model, train_loader, val_loader, epochs=50, lr=1e-4, device="cuda"):
    """
    Loop principale per l'orchestrare l'intero addestramento.
    """
    model = model.to(device)
    
    # AdamW è standard de-facto in CV odierna per come decresce logicamente la weight_decay 
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Scheduler: T_max spinge il LR iterativamente a ~0. Decadimento Cosinoidale morbido.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Engine del Mixed Precision nativo su architetture Turing 2080 Ti
    scaler = GradScaler()
    
    print(f"[{device.upper()}] Avviando processo su RTX 2080 Ti || Batch size: {train_loader.batch_size}")
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- [Epoca {epoch}/{epochs}] ---")
        
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        
        # Update del learning rate alla fine dell'epoca
        scheduler.step()
        
        # Log Fine Iterazione
        current_lr = scheduler.get_last_lr()[0]
        print(f"-> Completo | Loss Totale Media: {avg_train_loss:.4f} | LR Attuale: {current_lr:.2e}")
        
        # ... qui va l'evaluation_one_epoch, calcolo dell'accuracy, early_stopping o checkpointing
        
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore") # Per prevenire spam warnings su CPU/Tensor vecchi se testato al volo

    from src.data.dataset import create_dummy_dataframe 
    # Sanity-check se questo file viene invocato dal terminale per assicurarsi che i batch reggano.
    print("Test hardware dell'engine di training (Dummy Dataset)...")
    
    dumb_df = create_dummy_dataframe(num_samples=64, save_path="data/raw/dummy_wardrobe.csv")
    loader_t, loader_v = create_dataloaders(dumb_df, dumb_df, batch_size=32, num_workers=2)
    
    net = WardrobeMultiHeadModel()
    test_device = "cuda" if torch.cuda.is_available() else "cpu"
    fit(net, loader_t, loader_v, epochs=2, device=test_device)
    print("Engine in salute!")
