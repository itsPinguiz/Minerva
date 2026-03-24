import torch
from torchvision import transforms
from PIL import Image
import os

def predict_item(image_path, model, transform, label_maps, device="cuda"):
    """
    Esegue l'inferenza su una singola immagine utilizzando un modello Multi-Head.
    """
    # 1. Carica l'immagine
    if not os.path.exists(image_path):
        print(f"Errore: l'immagine '{image_path}' non esiste.")
        return None
        
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Errore nel caricamento dell'immagine {image_path}: {e}")
        return None
        
    # 2. Applica la trasformazione (resize -> tensor -> normalize puro, zero augmentation)
    if transform:
        img_tensor = transform(image)
        # I modelli PyTorch si aspettano sempre input in forma di batch (B, C, H, W)
        # unsqueeze(0) aggiunge l'asse del batch per trasformarlo da (3, 224, 224) a (1, 3, 224, 224)
        img_tensor = img_tensor.unsqueeze(0)
    else:
        raise ValueError("È necessaria una pipeline di trasformazione valida.")
        
    # 3. Sposta i dati sulla GPU (se disponibile)
    img_tensor = img_tensor.to(device)
    model = model.to(device)
    
    # 4. Inferenza
    model.eval()  # Disabilita strati dinamici come Dropout e cristallizza le BatchNorm
    with torch.no_grad():  # Cruciale: disabilita il grafo dei gradienti per risparmiare VRAM e salire di velocità
        outputs = model(img_tensor)
        
    # 5. Elabora le predizioni (Argmax sui Logit)
    predictions = {}
    
    for head_name, logits in outputs.items():
        # Calcoliamo le probabilità col Softmax lungo l'asse delle classi (dim=1)
        probabilities = torch.softmax(logits, dim=1)
        
        # Troviamo l'indice con il punteggio fiduciario più alto
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        
        # Mappiamo l'id numerico alla stringa testuale, oppure usiamo un default di emergenza
        if head_name in label_maps and predicted_idx in label_maps[head_name]:
            predicted_label = label_maps[head_name][predicted_idx]
        else:
            predicted_label = f"Classe ID: {predicted_idx}"
            
        predictions[head_name] = predicted_label
        
    # 6. Formatta e stampa un output estremamente pulito
    print(f"\n=> Analisi File: {os.path.basename(image_path)}")
    print(f"   Predizione: {predictions.get('category', 'Sconosciuta')}, "
          f"Colore: {predictions.get('color', 'Sconosciuto')}, "
          f"Tessuto: {predictions.get('fabric', 'Sconosciuto')}, "
          f"Stile: {predictions.get('style', 'Sconosciuto')}")
          
    return predictions

if __name__ == "__main__":
    from src.models.wardrobe_net import WardrobeMultiHeadModel
    
    # Dizionari che mappano le reti logiche ai nomi per l'utente finale
    DUMMY_LABEL_MAPS = {
        "category": {0: "T-shirt", 1: "Pantaloni", 2: "Giacca", 3: "Sneakers", 14: "Camicia"},
        "color": {0: "Blu", 1: "Maculato", 2: "Nero", 3: "Rosso", 11: "Bianco"},
        "fabric": {0: "Cotone", 1: "Seta", 2: "Denim", 3: "Lino"},
        "style": {0: "Casual", 1: "Elegante", 2: "Sportivo", 3: "Invernale"}
    }
    
    # Pipeline di trasformazione strettamente da Inferenza (Normalizzazione Standard ImageNet)
    inf_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Setup del file d'esempio
    test_img = "data/raw/mia_maglietta.jpg"
    os.makedirs(os.path.dirname(test_img), exist_ok=True)
    if not os.path.exists(test_img):
        img_dummy = Image.new('RGB', (1024, 1024), color=(0, 0, 255)) # Maglietta Blu sintetica
        img_dummy.save(test_img)
        
    # Setup del Device e load del Modello
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Boot Inferenza su device: {dev.upper()}")
    
    model_instance = WardrobeMultiHeadModel()
    
    # IN UN CONTESTO REALE, qui caricheresti i pesi post-training:
    # model_instance.load_state_dict(torch.load("weights/model_epoch_50.pth"))
    
    # Richiamo della predizione astratta
    predict_item(
        image_path=test_img, 
        model=model_instance, 
        transform=inf_transform, 
        label_maps=DUMMY_LABEL_MAPS, 
        device=dev
    )
