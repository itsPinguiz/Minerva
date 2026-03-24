import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class WardrobeMultiHeadModel(nn.Module):
    def __init__(self, num_category=15, num_color=12, num_fabric=8, num_style=4):
        """
        Modello Multi-Head basato su ResNet-50 pre-addestrata.
        
        Args:
            num_category (int): Numero di classi per la testa Categoria.
            num_color (int): Numero di classi per la testa Colore.
            num_fabric (int): Numero di classi per la testa Tessuto.
            num_style (int): Numero di classi per la testa Stile.
        """
        super(WardrobeMultiHeadModel, self).__init__()
        
        # 1. Carichiamo la ResNet-50 pre-addestrata su ImageNet
        # Utilizziamo la nuova API di torchvision per i pesi (più raccomandato di pretrained=True)
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Estraiamo il numero di feature in ingresso all'ultimo strato (fc)
        # Per la ResNet-50, questo valore è tipicamente 2048
        in_features = resnet.fc.in_features
        
        # 2. Rimuoviamo l'ultimo strato fully connected (fc)
        # Creiamo un modulo sequenziale con tutti i layer originali tranne l'ultimo
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 3. Aggiungiamo le 4 teste (heads) distinte per ciascun task
        # Ogni testa riceve in input il vettore embeddato da 2048 feature
        self.head_category = nn.Linear(in_features, num_category)
        self.head_color = nn.Linear(in_features, num_color)
        self.head_fabric = nn.Linear(in_features, num_fabric)
        self.head_style = nn.Linear(in_features, num_style)

    def forward(self, x):
        """
        Passaggio in avanti (forward pass).
        
        Restituisce:
            Un dizionario contenente i logit non normalizzati (output grezzi)
            per ciascuna delle 4 teste. I logit devono essere raw poiché 
            verranno elaborati internamente dalla `nn.CrossEntropyLoss`.
        """
        # A. Passiamo le immagini attraverso il backbone
        # L'output di ResNet-50 prima della FC è (batch_size, 2048, 1, 1) per la presenza del Global Average Pooling
        features = self.backbone(x)
        
        # B. Appiattiamo (flatten) il tensore spaziale 1x1 in un semplice vettore
        # Forma finale: (batch_size, 2048)
        features = torch.flatten(features, 1)
        
        # C. Eseguiamo i layer indipendenti per dedurre il verdetto per ogni classe
        out_category = self.head_category(features)
        out_color = self.head_color(features)
        out_fabric = self.head_fabric(features)
        out_style = self.head_style(features)
        
        # D. Restituiamo i 4 tensori logit in un pratico dizionario
        return {
            "category": out_category,
            "color": out_color,
            "fabric": out_fabric,
            "style": out_style
        }

if __name__ == "__main__":
    # Test "Sanity Check" per l'architettura
    model = WardrobeMultiHeadModel()
    
    # Simula un mini-batch di 2 immagini RGB
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("Eseguendo il forward pass...\n")
    outputs = model(dummy_input)
    
    print("Modello inizializzato e funzionante!")
    print("\nShape di Output (Devono corrispondere a [batch_size, num_classes]):")
    for task_name, logits in outputs.items():
        print(f"  - {task_name.capitalize():<10}: {list(logits.shape)}")
