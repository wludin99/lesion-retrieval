"""Embedding extraction models."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm


class EmbeddingModel(nn.Module):
    """Model for extracting image embeddings.
    
    Supports ResNet, ViT, and DINOv2 backbones.
    """
    
    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        embedding_dim: int = 2048,
        pool_type: str = "avg",
    ):
        """Initialize embedding model.
        
        Args:
            backbone: Model backbone name. Options:
                - 'microsoft/resnet-50' (HuggingFace)
                - 'google/vit-base-patch16-224' (HuggingFace)
                - 'facebook/dinov2-base' (HuggingFace)
            pretrained: Whether to use pretrained weights.
            embedding_dim: Expected embedding dimension.
            pool_type: Pooling type. For ResNet: 'avg', 'max', 'adaptive'.
                For ViT/DINOv2: 'cls' (CLS token) or 'mean' (mean pooling).
        """
        super().__init__()
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self.pool_type = pool_type
        
        # Initialize backbone
        if "resnet" in backbone.lower():
            # HuggingFace ResNet (e.g., "microsoft/resnet-50")
            self._init_hf_resnet(backbone, pretrained)
        elif "vit" in backbone.lower() or "dinov2" in backbone.lower():
            self._init_transformer(backbone, pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use HuggingFace models like 'microsoft/resnet-50'")
    
    def _init_hf_resnet(self, backbone: str, pretrained: bool) -> None:
        """Initialize ResNet backbone from HuggingFace."""
        from transformers import ResNetModel
        
        self.backbone = ResNetModel.from_pretrained(backbone)
        
        # Get feature dimension from config - ResNet-50 has 2048 in final stage
        # Check by running a forward pass
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            outputs = self.backbone(dummy)
            # HuggingFace ResNet returns BaseModelOutput with last_hidden_state
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state
            else:
                # Fallback: might return tuple or tensor directly
                features = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            self.feature_dim = features.shape[1]
        
        # Add pooling layer
        if self.pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif self.pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif self.pool_type == "adaptive":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Unsupported pool_type for ResNet: {self.pool_type}")
    
    def _init_transformer(self, backbone: str, pretrained: bool) -> None:
        """Initialize ViT/DINOv2 transformer backbone."""
        self.backbone = AutoModel.from_pretrained(
            backbone,
            trust_remote_code=True,
        )
        
        # Disable pooler if it exists (we don't use it)
        # This prevents the warning about pooler weights not being initialized
        if hasattr(self.backbone, 'pooler') and self.backbone.pooler is not None:
            self.backbone.pooler = None
        
        self.feature_dim = self.backbone.config.hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Embedding tensor of shape (B, embedding_dim).
        """
        if "resnet" in self.backbone_name:
            # HuggingFace ResNet forward
            outputs = self.backbone(x)
            
            # HuggingFace ResNet returns BaseModelOutput with last_hidden_state
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state
            elif isinstance(outputs, torch.Tensor):
                features = outputs
            else:
                # Fallback: might be tuple
                features = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Pool spatial dimensions
            if len(features.shape) == 4:  # (B, C, H, W)
                features = self.pool(features)
                features = features.view(features.size(0), -1)
            
            return features
        
        else:
            # Transformer forward
            outputs = self.backbone(pixel_values=x)
            
            if self.pool_type == "cls":
                # Use CLS token
                embeddings = outputs.last_hidden_state[:, 0, :]
            else:
                # Mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings
    
    def extract_embeddings(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, List, List]:
        """Extract embeddings for all samples in a dataloader.
        
        Args:
            dataloader: DataLoader with images.
            device: Device to run inference on.
            normalize: Whether to L2-normalize embeddings.
            
        Returns:
            Tuple of (embeddings, lesion_ids, image_ids).
        """
        self.eval()
        embeddings_list = []
        lesion_ids_list = []
        image_ids_list = []
        
        with torch.no_grad():
            for images, lesion_ids, image_ids in tqdm(dataloader, desc="Extracting embeddings"):
                images = images.to(device)
                emb = self.forward(images)
                
                if normalize:
                    emb = nn.functional.normalize(emb, p=2, dim=1)
                
                embeddings_list.append(emb.cpu())
                lesion_ids_list.extend(lesion_ids)
                image_ids_list.extend(image_ids)
        
        embeddings = torch.cat(embeddings_list, dim=0)
        return embeddings, lesion_ids_list, image_ids_list
