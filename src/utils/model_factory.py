"""Factory functions for creating models and loss functions."""

from typing import Optional

import torch.nn as nn
from omegaconf import DictConfig

from models import EmbeddingModel
from losses import ContrastiveLoss, TripletLoss, InfoNCELoss


def create_model(cfg: DictConfig) -> EmbeddingModel:
    """Create embedding model based on configuration.
    
    Args:
        cfg: Hydra configuration object.
        
    Returns:
        Initialized EmbeddingModel.
    """
    freeze_backbone_config = cfg.model.get("freeze_backbone", None)
    projection_hidden_dim_config = cfg.model.get("projection_hidden_dim", None)
    
    # Phase 2 DINOv2 and ViT: automatically freeze and add projection head if not explicitly set
    if cfg.experiment.phase == 2:
        is_dinov2 = "dinov2" in cfg.model.backbone.lower()
        is_vit = "vit" in cfg.model.backbone.lower()
        
        if is_dinov2 or is_vit:
            model_name = "DINOv2" if is_dinov2 else "ViT"
            feature_dim = 768  # Both DINOv2-base and ViT-base feature dimension
            
            if freeze_backbone_config is None:
                freeze_backbone = True
                print(f"Phase 2 with {model_name}: Freezing pretrained backbone weights (default)")
            else:
                freeze_backbone = bool(freeze_backbone_config)
                if freeze_backbone:
                    print(f"Phase 2 with {model_name}: Freezing pretrained backbone weights (from config)")
            
            if projection_hidden_dim_config is None:
                projection_hidden_dim = min(256, feature_dim // 2)
                print(f"Phase 2 with {model_name}: Adding projection head (hidden_dim={projection_hidden_dim}, default)")
            else:
                projection_hidden_dim = projection_hidden_dim_config
                if projection_hidden_dim:
                    print(f"Phase 2 with {model_name}: Adding projection head (hidden_dim={projection_hidden_dim}, from config)")
        else:
            freeze_backbone = bool(freeze_backbone_config) if freeze_backbone_config is not None else False
            projection_hidden_dim = projection_hidden_dim_config
    else:
        freeze_backbone = bool(freeze_backbone_config) if freeze_backbone_config is not None else False
        projection_hidden_dim = projection_hidden_dim_config
    
    return EmbeddingModel(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        embedding_dim=cfg.model.embedding_dim,
        pool_type=cfg.model.pool_type,
        freeze_backbone=freeze_backbone,
        projection_hidden_dim=projection_hidden_dim,
    )


def create_loss_function(cfg: DictConfig) -> nn.Module:
    """Create loss function based on configuration.
    
    Args:
        cfg: Hydra configuration object.
        
    Returns:
        Loss function module.
    """
    loss_name = cfg.loss.name
    if loss_name == "contrastive":
        return ContrastiveLoss(
            margin=cfg.loss.margin,
            temperature=cfg.loss.temperature,
        )
    elif loss_name == "triplet":
        return TripletLoss(
            margin=cfg.loss.margin,
            sampling=cfg.loss.sampling,
        )
    elif loss_name == "infonce":
        return InfoNCELoss(
            temperature=cfg.loss.temperature,
            num_negatives=cfg.loss.get("num_negatives", None),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
