"""InfoNCE / NT-Xent loss implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE (NT-Xent) loss for contrastive learning.
    
    Maximizes agreement between positive pairs and minimizes
    agreement between negative pairs using a temperature-scaled
    softmax.
    """
    
    def __init__(self, temperature: float = 0.07, num_negatives: int | None = None):
        """Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature scaling parameter.
            num_negatives: Number of negatives to sample. If None, uses all negatives.
        """
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: list[str],
    ) -> torch.Tensor:
        """Compute InfoNCE loss.
        
        Args:
            embeddings: Embedding tensor of shape (N, D).
            labels: List of lesion_id labels for each embedding.
            
        Returns:
            Scalar loss value.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create positive mask (same lesion_id)
        labels_tensor = torch.tensor(
            [hash(label) for label in labels],
            device=embeddings.device
        )
        positive_mask = labels_tensor.unsqueeze(0) == labels_tensor.unsqueeze(1)
        
        # Remove diagonal
        positive_mask = positive_mask & ~torch.eye(
            len(labels),
            device=embeddings.device,
            dtype=torch.bool
        )
        
        # For each anchor, compute loss
        losses = []
        for i in range(len(embeddings)):
            # Positive similarities
            pos_similarities = similarity_matrix[i][positive_mask[i]]
            
            if len(pos_similarities) == 0:
                continue
            
            # Negative similarities
            neg_mask = ~positive_mask[i] & (torch.arange(len(labels), device=embeddings.device) != i)
            neg_similarities = similarity_matrix[i][neg_mask]
            
            if len(neg_similarities) == 0:
                continue
            
            # Sample negatives if needed
            if self.num_negatives is not None and len(neg_similarities) > self.num_negatives:
                indices = torch.randperm(len(neg_similarities), device=embeddings.device)[:self.num_negatives]
                neg_similarities = neg_similarities[indices]
            
            # Compute logits: [pos_1, pos_2, ..., neg_1, neg_2, ...]
            logits = torch.cat([pos_similarities, neg_similarities])
            
            # Labels: first len(pos_similarities) are positives (class 0)
            # In InfoNCE, we want to maximize similarity to positives
            # So we use cross-entropy where positives are the target
            # For simplicity, we average over all positive pairs
            loss = 0.0
            for pos_sim in pos_similarities:
                # Concatenate this positive with all negatives
                pos_neg_logits = torch.cat([pos_sim.unsqueeze(0), neg_similarities])
                # Positive is at index 0
                loss += F.cross_entropy(
                    pos_neg_logits.unsqueeze(0),
                    torch.tensor([0], device=embeddings.device)
                )
            
            losses.append(loss / len(pos_similarities))
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return torch.stack(losses).mean()
