"""Training loop for metric learning."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from evaluation.pairwise_metrics import evaluate_embeddings


class Trainer:
    """Trainer for metric learning models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: dict,
        output_dir: Path,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            loss_fn: Loss function.
            optimizer: Optimizer.
            device: Device to train on.
            config: Training configuration dictionary.
            output_dir: Directory to save checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_f1 = -1.0
        self.patience_counter = 0
        
        # Move model to device
        self.model.to(device)
    
    def train_epoch(self) -> dict:
        """Train for one epoch.
        
        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (images, lesion_ids, image_ids) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            embeddings = self.model(images)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            loss = self.loss_fn(embeddings, lesion_ids)
            
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("NaN/Inf in loss - stopping training")
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get("max_grad_norm", None) is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["max_grad_norm"]
                )
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    raise RuntimeError("NaN/Inf gradient norm - stopping training")
            
            self.optimizer.step()
            
            # Check for NaN/Inf in parameters (catches gradient and parameter issues)
            for name, param in self.model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    raise RuntimeError(f"NaN/Inf in gradients ({name}) - stopping training")
                if torch.isnan(param).any() or torch.isinf(param).any():
                    raise RuntimeError(f"NaN/Inf in parameters ({name}) - stopping training")
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
            
            # Log to wandb
            if wandb.run is not None and batch_idx % self.config.get("log_freq", 10) == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/epoch": self.current_epoch,
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "loss": avg_loss,
        }
    
    def validate(self) -> dict:
        """Validate the model.
        
        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        embeddings, lesion_ids, image_ids = self.model.extract_embeddings(
            self.val_loader,
            self.device,
            normalize=True,
        )
        
        # Evaluate
        eval_results = evaluate_embeddings(embeddings, lesion_ids)
        
        # Extract best metrics
        metrics = {
            "val/cosine_f1": eval_results["cosine_best"]["f1"],
            "val/cosine_precision": eval_results["cosine_best"]["precision"],
            "val/cosine_recall": eval_results["cosine_best"]["recall"],
            "val/cosine_threshold": eval_results["cosine_best"]["threshold"],
            "val/dbscan_f1": eval_results["dbscan_best"]["f1"],
            "val/dbscan_precision": eval_results["dbscan_best"]["precision"],
            "val/dbscan_recall": eval_results["dbscan_best"]["recall"],
            "val/dbscan_eps": eval_results["dbscan_best"]["eps"],
        }
        
        return metrics
    
    def train(self) -> None:
        """Run full training loop."""
        epochs = self.config.get("epochs", 50)
        early_stopping_patience = self.config.get("early_stopping", {}).get("patience", None)
        checkpoint_freq = self.config.get("checkpoint_freq", 5)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            self.model.train()  # Ensure model is in train mode
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Set back to train mode for next epoch
            self.model.train()
            
            # Log metrics
            log_dict = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
            }
            
            if wandb.run is not None:
                wandb.log(log_dict)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Val cosine F1: {val_metrics['val/cosine_f1']:.4f}")
            print(f"  Val DBSCAN F1: {val_metrics['val/dbscan_f1']:.4f}")
            if self.best_val_f1 >= 0:
                print(f"  Best val F1 so far: {self.best_val_f1:.4f}")
            print(f"  Patience: {self.patience_counter}/{early_stopping_patience if early_stopping_patience else 'N/A'}")
            
            # Checkpointing
            val_f1 = val_metrics["val/cosine_f1"]
            
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                self.save_checkpoint("best.pt")
                print(f"  ✓ New best model saved!")
            else:
                self.patience_counter += 1
            print("-"*100+"\n")
            
            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
            # Early stopping
            if early_stopping_patience is not None:
                if self.patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint_path = self.output_dir / filename
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
        }, checkpoint_path)
