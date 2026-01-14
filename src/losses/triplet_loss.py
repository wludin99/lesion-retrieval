"""Triplet loss implementation."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet loss for metric learning.
    
    Ensures that anchor-positive distance is smaller than
    anchor-negative distance by at least a margin.
    """
    
    def __init__(
        self,
        margin: float = 0.5,
        sampling: Literal["random", "hard", "semi-hard"] = "random",
    ):
        """Initialize triplet loss.
        
        Args:
            margin: Margin between positive and negative pairs.
            sampling: Sampling strategy for triplets.
        """
        super().__init__()
        self.margin = margin
        self.sampling = sampling
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: list,
    ) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            embeddings: Embedding tensor of shape (N, D).
            labels: List of lesion_id labels for each embedding.
            
        Returns:
            Scalar loss value.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Create label tensor
        labels_tensor = torch.tensor(
            [hash(label) for label in labels],
            device=embeddings.device
        )
        
        # Compute pairwise distances
        distances = self._pairwise_distance(embeddings)
        
        # Create masks
        positive_mask = labels_tensor.unsqueeze(0) == labels_tensor.unsqueeze(1)
        negative_mask = ~positive_mask
        
        # Remove diagonal
        positive_mask = positive_mask & ~torch.eye(
            len(labels),
            device=embeddings.device,
            dtype=torch.bool
        )
        
        losses = []
        for i in range(len(embeddings)):
            # Positive distances for anchor i
            pos_distances = distances[i][positive_mask[i]]
            if len(pos_distances) == 0:
                continue
            
            # Negative distances for anchor i
            neg_distances = distances[i][negative_mask[i]]
            if len(neg_distances) == 0:
                continue
            
            # Sample triplets
            if self.sampling == "random":
                pos_dist = pos_distances[0]
                neg_dist = neg_distances[0]
            elif self.sampling == "hard":
                pos_dist = pos_distances.max()
                neg_dist = neg_distances.min()
            elif self.sampling == "semi-hard":
                pos_dist = pos_distances[0]
                # Find hardest negative that is still further than positive
                valid_negatives = neg_distances[neg_distances > pos_dist]
                if len(valid_negatives) > 0:
                    neg_dist = valid_negatives.min()
                else:
                    neg_dist = neg_distances.min()
            else:
                raise ValueError(f"Unknown sampling: {self.sampling}")
            
            # Triplet loss
            loss = F.relu(pos_dist - neg_dist + self.margin)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return torch.stack(losses).mean()
    
    def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances."""
        dot_product = torch.mm(embeddings, embeddings.t())
        distances = 2 - 2 * dot_product
        distances = torch.sqrt(torch.clamp(distances, min=0.0))
        return distances
