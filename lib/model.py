import torch
import torch.nn as nn
import torchvision.models as models
from lib.logging_setup import logger

class AttentionPooling(nn.Module):
    """
    Attention pooling layer for Multiple Instance Learning (MIL).
    Aggregates features of variable number of tiles into a single slide-level embedding.
    """
    def __init__(self, input_dim=512, L=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, L),
            nn.Tanh(),
            nn.Linear(L, 1)
        )

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, num_tiles, input_dim)
        # attention weights
        attn_logits = self.attention(x) # (batch_size, num_tiles, 1)
        attn_weights = torch.softmax(attn_logits, dim=1) # (batch_size, num_tiles, 1)
        
        # Weighted sum of features
        out = torch.sum(attn_weights * x, dim=1) # (batch_size, input_dim)
        return out, attn_weights


class ProstateCADxModel(nn.Module):
    def __init__(self, backbone="resnet50", num_classes=6, tile_classes=4, aggregation="attention", pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        self.aggregation = aggregation
        
        # 1. Instantiate backbone
        if backbone == "resnet50":
            # For resnet50, features dimension is 2048
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity() # Remove classification layer
        elif backbone == "efficientnet_b0":
            # For efficientnet_b0, features dimension is 1280
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 2. Tile classifier
        self.tile_classifier = nn.Linear(self.feature_dim, tile_classes)

        # 3. Slide aggregator
        if aggregation == "attention":
            self.pool = AttentionPooling(input_dim=self.feature_dim)
        else:
            self.pool = None

        # 4. Slide classifier
        self.slide_classifier = nn.Linear(self.feature_dim, num_classes)

        # 5. Contrastive projection head (SimCLR style MLP)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x: torch.Tensor, return_attention=False, return_contrastive=False):
        # If input has shape (batch_size, num_tiles, channels, height, width)
        # we reshape it to batch_size * num_tiles to pass through backbone
        if x.dim() == 5:
            bs, num_tiles, c, h, w = x.shape
            x_flat = x.view(-1, c, h, w)
            features_flat = self.backbone(x_flat) # (bs * num_tiles, feature_dim)
            features = features_flat.view(bs, num_tiles, -1) # (bs, num_tiles, feature_dim)
            
            # Predict tile-level Gleason patterns
            tile_logits = self.tile_classifier(features_flat).view(bs, num_tiles, -1)
            
            # Aggregate features
            if self.aggregation == "attention":
                slide_features, attn_weights = self.pool(features)
            elif self.aggregation == "mean":
                slide_features = torch.mean(features, dim=1)
                attn_weights = torch.ones(bs, num_tiles, 1, device=x.device) / num_tiles
            elif self.aggregation == "max":
                slide_features, _ = torch.max(features, dim=1)
                attn_weights = torch.ones(bs, num_tiles, 1, device=x.device) / num_tiles
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
                
            slide_logits = self.slide_classifier(slide_features)
            
            if return_contrastive:
                proj = self.contrastive_proj(features_flat)
                if return_attention:
                    return slide_logits, tile_logits, attn_weights, proj
                return slide_logits, tile_logits, proj

            if return_attention:
                return slide_logits, tile_logits, attn_weights
            return slide_logits, tile_logits
            
        else:
            # Single image input (tile level)
            features = self.backbone(x) # (batch_size, feature_dim)
            tile_logits = self.tile_classifier(features)
            if return_contrastive:
                proj = self.contrastive_proj(features)
                return tile_logits, proj
            return tile_logits
