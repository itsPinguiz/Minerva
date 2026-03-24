import argparse
import joblib
import logging
import os
import torch
import yaml
from torchvision import transforms
from PIL import Image

# Assicurati che l'import del modello punti al file corretto
from src.models.model import WardrobeMultiHeadModel

logger = logging.getLogger(__name__)

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def predict_item(image_path, model, transform, encoders, device="cuda", threshold=0.5):
    """
    Esegue l'inferenza multi-label su una singola immagine.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Errore: l'immagine '{image_path}' non esiste.")
        
    image = Image.open(image_path).convert("RGB")
    
    # 1. Preprocessing (Aggiungiamo la batch dimension con unsqueeze)
    img_tensor = transform(image).unsqueeze(0).to(device)
    model = model.to(device)
    
    # 2. Inferenza
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        
    predictions = {}
    
    # 3. Decoding Multi-Label
    for head_name, logits in outputs.items():
        # Applichiamo la Sigmoide per ottenere probabilità indipendenti tra 0 e 1
        probs = torch.sigmoid(logits)[0]  # [0] perché abbiamo un batch di 1
        
        # Troviamo tutte le classi che superano la soglia di confidenza
        mask = probs > threshold
        
        # Recuperiamo l'encoder specifico per questa testa
        encoder = encoders[head_name]
        
        # Estraiamo i nomi delle classi usando la maschera booleana
        predicted_classes = encoder.classes_[mask.cpu().numpy()]
        
        # Fallback di sicurezza: se nessuna classe supera la soglia, 
        # prendiamo quella con il punteggio più alto in assoluto (argmax)
        if len(predicted_classes) == 0:
            best_idx = torch.argmax(probs).item()
            # È già una lista Python, la salviamo direttamente
            predictions[head_name] = [encoder.classes_[best_idx]]
        else:
            # È un array NumPy, usiamo tolist()
            predictions[head_name] = predicted_classes.tolist()
        
    return predictions

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Testa il modello su una foto reale.")
    parser.add_argument("--image", type=str, required=True, help="Path dell'immagine da analizzare")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path del file di configurazione")
    parser.add_argument("--weights", type=str, default="weights/best_model.pth", help="Path dei pesi del modello")    
    parser.add_argument("--threshold", type=float, default=0.5, help="Soglia di confidenza (0.0 - 1.0)")
    args = parser.parse_args()

    # 1. Setup
    config = load_config(args.config)
    dev = config["system"]["device"] if torch.cuda.is_available() else "cpu"
    logger.info(f"Boot Inferenza su device: {dev.upper()}")

    # 2. Caricamento Encoder
    encoders_path = config["data"]["encoders_path"]
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Encoder non trovati in {encoders_path}. Hai lanciato il training?")
    encoders = joblib.load(encoders_path)
    
    # 3. Caricamento Modello e Pesi
    logger.info(f"Caricamento modello da {args.weights} ...")
    model_instance = WardrobeMultiHeadModel(
        num_category=config["model"]["heads"]["category"],
        num_color=config["model"]["heads"]["color"],
        num_fabric=config["model"]["heads"]["fabric"],
        num_style=config["model"]["heads"]["style"]
    )
    
    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=dev, weights_only=True)
        # Se il checkpoint contiene altre info (epoch, optimizer), estraiamo solo il model_state
        state_dict = checkpoint.get("model_state_dict", checkpoint) 
        model_instance.load_state_dict(state_dict)
    else:
        logger.warning("ATTENZIONE: best_model.pth non trovato. Verranno usati pesi casuali (non addestrati)!")

    # 4. Pipeline di trasformazione (Stessa risoluzione del training, ma senza augmentation)
    inf_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 5. Esegui la predizione
    preds = predict_item(
        image_path=args.image, 
        model=model_instance, 
        transform=inf_transform, 
        encoders=encoders, 
        device=dev,
        threshold=args.threshold
    )
    
    # 6. Stampa Risultati Formattati
    logger.info(f"\n{'='*40}")
    logger.info(f"👕 ANALISI CAPO: {os.path.basename(args.image)}")
    logger.info(f"{'='*40}")
    logger.info(f" - Categoria : {', '.join(preds['category']).title()}")
    logger.info(f" - Colore    : {', '.join(preds['color']).title()}")
    logger.info(f" - Tessuto   : {', '.join(preds['fabric']).title()}")
    logger.info(f" - Stile     : {', '.join(preds['style']).title()}")
    logger.info(f"{'='*40}\n")