"""Phase 1 and Phase 2 experiment runners."""

from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import wandb

from datasets import DermoscopicDataset
from evaluation import evaluate_embeddings
from evaluation.results import save_clustering_results
from models import EmbeddingModel
from training import Trainer
from utils.device import should_use_pin_memory
from utils.model_factory import create_model, create_loss_function


def run_phase1_baseline(
    model: EmbeddingModel,
    val_loader: DataLoader,
    device: torch.device,
    fold_idx: int,
    cfg: DictConfig,
) -> Dict:
    """Run Phase 1: Baseline embedding extraction.
    
    Args:
        model: Embedding model.
        val_loader: Validation data loader.
        device: Device to run on.
        fold_idx: Current fold index.
        cfg: Hydra configuration object.
        
    Returns:
        Dictionary with evaluation results.
    """
    print("\nPhase 1: Extracting baseline embeddings...")
    model.eval()
    
    # Extract embeddings
    val_embeddings, val_lesion_ids_list, val_image_ids = model.extract_embeddings(
        val_loader, device, normalize=True
    )
    
    # Evaluate on validation set
    print("Evaluating embeddings...")
    # Evaluate on validation set (misclassifications only saved for test set, not validation folds)
    eval_results = evaluate_embeddings(
        val_embeddings,
        val_lesion_ids_list,
        cosine_thresholds=None,
        dbscan_eps_values=None,
        dbscan_min_samples=cfg.evaluation.dbscan.min_samples,
        image_ids=None,  # Not tracking misclassifications for validation folds
        return_misclassified=False,  # Only save misclassifications for final test set
    )
    
    # Save clustering results (always save, uses best DBSCAN parameters)
    # Note: Misclassifications are only saved for final test set evaluation, not for validation folds
    output_dir = Path(cfg.paths.output_dir) / cfg.experiment.name / f"fold_{fold_idx}"
    if "cluster_labels" in eval_results["dbscan_best"]:
        save_clustering_results(
            cluster_labels=eval_results["dbscan_best"]["cluster_labels"],
            lesion_ids=val_lesion_ids_list,
            image_ids=val_image_ids,
            output_path=output_dir / "clustered_images.csv",
            eps=eval_results["dbscan_best"]["eps"],
            min_samples=cfg.evaluation.dbscan.min_samples,
        )
    
    # Extract results
    results = {
        "fold": fold_idx,
        "cosine_f1": eval_results["cosine_best"]["f1"],
        "cosine_precision": eval_results["cosine_best"]["precision"],
        "cosine_recall": eval_results["cosine_best"]["recall"],
        "cosine_threshold": eval_results["cosine_best"]["threshold"],
        "dbscan_f1": eval_results["dbscan_best"]["f1"],
        "dbscan_precision": eval_results["dbscan_best"]["precision"],
        "dbscan_recall": eval_results["dbscan_best"]["recall"],
        "dbscan_eps": eval_results["dbscan_best"]["eps"],
    }
    
    print(f"\nFold {fold_idx} Results:")
    print(f"  Cosine F1: {results['cosine_f1']:.4f} (threshold: {results['cosine_threshold']:.3f})")
    print(f"  DBSCAN F1: {results['dbscan_f1']:.4f} (eps: {results['dbscan_eps']:.3f})")
    
    if wandb.run is not None:
        wandb.log({
            f"fold_{fold_idx}/cosine_f1": results["cosine_f1"],
            f"fold_{fold_idx}/cosine_precision": results["cosine_precision"],
            f"fold_{fold_idx}/cosine_recall": results["cosine_recall"],
            f"fold_{fold_idx}/dbscan_f1": results["dbscan_f1"],
            f"fold_{fold_idx}/dbscan_precision": results["dbscan_precision"],
            f"fold_{fold_idx}/dbscan_recall": results["dbscan_recall"],
        })
    
    return results


def run_phase2_finetuning(
    model: EmbeddingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    fold_idx: int,
    cfg: DictConfig,
) -> Dict:
    """Run Phase 2: Metric learning fine-tuning.
    
    Args:
        model: Embedding model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Device to run on.
        fold_idx: Current fold index.
        cfg: Hydra configuration object.
        
    Returns:
        Dictionary with evaluation results.
    """
    print("\nPhase 2: Fine-tuning with metric learning...")
    
    # Create loss function and optimizer
    loss_fn = create_loss_function(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    
    # Create trainer
    output_dir = Path(cfg.paths.output_dir) / cfg.experiment.name / f"fold_{fold_idx}"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        config=cfg.training,
        output_dir=output_dir,
    )
    
    # Train
    trainer.train()
    
    # Load best model checkpoint for final evaluation
    best_checkpoint_path = output_dir / "best.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Final evaluation on validation set
    print("Final evaluation on validation set...")
    val_embeddings, val_lesion_ids_list, val_image_ids = model.extract_embeddings(
        val_loader, device, normalize=True
    )
    
    # Evaluate on validation set (misclassifications only saved for test set, not validation folds)
    eval_results = evaluate_embeddings(
        val_embeddings,
        val_lesion_ids_list,
        image_ids=None,  # Not tracking misclassifications for validation folds
        return_misclassified=False,  # Only save misclassifications for final test set
    )
    
    # Save clustering results (always save, uses best DBSCAN parameters)
    # Note: Misclassifications are only saved for final test set evaluation, not for validation folds
    if "cluster_labels" in eval_results["dbscan_best"]:
        save_clustering_results(
            cluster_labels=eval_results["dbscan_best"]["cluster_labels"],
            lesion_ids=val_lesion_ids_list,
            image_ids=val_image_ids,
            output_path=output_dir / "clustered_images.csv",
            eps=eval_results["dbscan_best"]["eps"],
            min_samples=cfg.evaluation.dbscan.min_samples,
        )
    
    # Extract results
    results = {
        "fold": fold_idx,
        "cosine_f1": eval_results["cosine_best"]["f1"],
        "cosine_precision": eval_results["cosine_best"]["precision"],
        "cosine_recall": eval_results["cosine_best"]["recall"],
        "cosine_threshold": eval_results["cosine_best"]["threshold"],
        "dbscan_f1": eval_results["dbscan_best"]["f1"],
        "dbscan_precision": eval_results["dbscan_best"]["precision"],
        "dbscan_recall": eval_results["dbscan_best"]["recall"],
        "dbscan_eps": eval_results["dbscan_best"]["eps"],
    }
    
    print(f"\nFold {fold_idx} Final Results (Validation Set):")
    print(f"  Cosine F1: {results['cosine_f1']:.4f} (threshold: {results['cosine_threshold']:.4f})")
    print(f"  DBSCAN F1: {results['dbscan_f1']:.4f} (eps: {results['dbscan_eps']:.4f})")
    
    return results


def train_final_model(
    model: EmbeddingModel,
    train_df: pd.DataFrame,
    device: torch.device,
    avg_cosine_threshold: float,
    avg_dbscan_eps: float,
    cfg: DictConfig,
) -> EmbeddingModel:
    """Train a final model on all training data using average hyperparameters.
    
    This function trains a model on the entire training set (all folds combined)
    and will be evaluated on the test set using average hyperparameters from CV.
    
    Args:
        model: Embedding model (will be re-initialized).
        train_df: Full training dataframe (all folds combined).
        device: Device to train on.
        avg_cosine_threshold: Average cosine threshold from CV folds.
        avg_dbscan_eps: Average DBSCAN eps from CV folds.
        cfg: Hydra configuration object.
        
    Returns:
        Trained model.
    """
    print(f"\n{'='*60}")
    print("Training Final Model on All Training Data")
    print(f"{'='*60}")
    print(f"Training on {len(train_df)} images from {train_df['lesion_id'].nunique()} lesions")
    print(f"Using average hyperparameters from CV:")
    print(f"  Cosine threshold: {avg_cosine_threshold:.4f}")
    print(f"  DBSCAN eps: {avg_dbscan_eps:.4f}")
    
    # Create a fresh model (re-initialize)
    final_model = create_model(cfg)
    final_model.to(device)
    
    # Only train if Phase 2
    if cfg.experiment.phase == 2:
        # Create dataset and loader
        train_dataset = DermoscopicDataset(
            train_df,
            images_dir=cfg.paths.images_dir,
            image_size=cfg.data.image_size,
            normalize=cfg.data.normalize,
        )
        use_pin_memory = should_use_pin_memory(device, cfg.data.pin_memory)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=use_pin_memory,
        )
        
        # Create loss function and optimizer
        loss_fn = create_loss_function(cfg)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, final_model.parameters()),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        
        # Create trainer (we'll train for the same number of epochs, but no validation)
        # Create a dummy validation loader (won't be used, but Trainer requires it)
        val_dataset = DermoscopicDataset(
            train_df.head(10),  # Dummy small dataset
            images_dir=cfg.paths.images_dir,
            image_size=cfg.data.image_size,
            normalize=cfg.data.normalize,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=use_pin_memory,
        )
        
        output_dir = Path(cfg.paths.output_dir) / cfg.experiment.name / "final_model"
        trainer = Trainer(
            model=final_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            config=cfg.training,
            output_dir=output_dir,
        )
        
        # Train for the configured number of epochs
        print(f"Training for {cfg.training.epochs} epochs...")
        trainer.train()
        
        # Load the best checkpoint (based on validation during training, but we're training on all data)
        best_checkpoint_path = output_dir / "best.pt"
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            final_model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded final model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print("Phase 1: Using frozen pretrained model (no training needed)")
    
    return final_model
