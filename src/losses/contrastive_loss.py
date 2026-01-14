"""Contrastive loss implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for metric learning.
    
    Pulls positive pairs together and pushes negative pairs apart
    if they are within the margin.
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.07):
        """Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs.
            temperature: Temperature scaling (not used in standard contrastive,
                but kept for consistency with other losses).
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: list[str],
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            embeddings: Embedding tensor of shape (N, D).
            labels: List of lesion_id labels for each embedding.
            
        Returns:
            Scalar loss value.
        """
        # Compute pairwise distances
        distances = self._pairwise_distance(embeddings)
        
        # Create positive mask (same lesion_id)
        labels_tensor = torch.tensor(
            [hash(label) for label in labels],
            device=embeddings.device
        )
        positive_mask = labels_tensor.unsqueeze(0) == labels_tensor.unsqueeze(1)
        
        # Remove diagonal (self-similarity)
        positive_mask = positive_mask & ~torch.eye(
            len(labels),
            device=embeddings.device,
            dtype=torch.bool
        )
        
        # Loss: pull positives together, push negatives apart if within margin
        positive_loss = (distances * positive_mask.float()).sum() / (
            positive_mask.sum().float() + 1e-8
        )
        
        negative_mask = ~positive_mask & ~torch.eye(
            len(labels),
            device=embeddings.device,
            dtype=torch.bool
        )
        negative_loss = F.relu(self.margin - distances) * negative_mask.float()
        negative_loss = negative_loss.sum() / (negative_mask.sum().float() + 1e-8)
        
        return positive_loss + negative_loss
    
    def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances.
        
        Args:
            embeddings: Embedding tensor of shape (N, D).
            
        Returns:
            Distance matrix of shape (N, N).
        """
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        dot_product = torch.mm(embeddings, embeddings.t())
        distances = 2 - 2 * dot_product  # Convert cosine to distance
        distances = torch.sqrt(torch.clamp(distances, min=0.0))
        
        return distances
