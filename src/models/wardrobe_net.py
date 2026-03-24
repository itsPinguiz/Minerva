import torch
import torch.nn as nn
import torchvision.models as models

class WardrobeMultiHeadNet(nn.Module):
    def __init__(self, config):
        super(WardrobeMultiHeadNet, self).__init__()
        
        # Determine backbone based on config
        backbone_name = config.model.backbone
        pretrained = config.model.pretrained
        
        if backbone_name == "resnet50":
            # using updated torchvision weights API
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            # Remove the original fully connected layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not supported yet.")

        # Multi-head classification layers
        self.head_category = nn.Linear(in_features, config.model.heads.category)
        self.head_color = nn.Linear(in_features, config.model.heads.color)
        self.head_season = nn.Linear(in_features, config.model.heads.season)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        # Forward pass through each head
        out_category = self.head_category(features)
        out_color = self.head_color(features)
        out_season = self.head_season(features)
        
        # Return a dictionary of predictions
        return {
            "category": out_category,
            "color": out_color,
            "season": out_season
        }
