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
        labels: list,
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            embeddings: Embedding tensor of shape (N, D).
            labels: List of lesion_id labels for each embedding.
            
        Returns:
            Scalar loss value.
        """
        # Embeddings should already be normalized, but ensure they are
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
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
        
        For normalized embeddings, this computes cosine distance.
        Uses numerically stable computation to avoid NaN gradients.
        
        Args:
            embeddings: Embedding tensor of shape (N, D).
            
        Returns:
            Distance matrix of shape (N, N).
        """
        # Compute cosine similarity
        dot_product = torch.mm(embeddings, embeddings.t())
        
        # Clamp to avoid numerical issues when embeddings are very similar
        # Cosine similarity should be in [-1, 1], but clamp to [-0.9999, 0.9999] for stability
        dot_product = torch.clamp(dot_product, min=-0.9999, max=0.9999)
        
        # Convert to distance: 2 - 2 * cos(theta) = 2 * (1 - cos(theta))
        # This is the squared Euclidean distance for normalized vectors
        distances = 2 - 2 * dot_product
        
        # Clamp to ensure non-negative (should already be, but be safe)
        distances = torch.clamp(distances, min=1e-8)
        
        # Take square root for Euclidean distance
        distances = torch.sqrt(distances)
        
        return distances
