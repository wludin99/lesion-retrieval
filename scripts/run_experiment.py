"""Main entry point for running experiments."""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import hydra
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import wandb

from datasets import DermoscopicDataset, create_folds, create_train_test_split
from models import EmbeddingModel
from losses import ContrastiveLoss, TripletLoss, InfoNCELoss
from training import Trainer
from evaluation import evaluate_embeddings, save_misclassifications
from utils import set_seed


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def should_use_pin_memory(device: torch.device, config_pin_memory: bool) -> bool:
    """Determine if pin_memory should be used (MPS doesn't support it)."""
    return config_pin_memory and device.type != "mps"


def create_model(cfg: DictConfig) -> EmbeddingModel:
    """Create embedding model based on configuration.
    
    Args:
        cfg: Hydra configuration object.
        
    Returns:
        Initialized EmbeddingModel.
    """
    freeze_backbone_config = cfg.model.get("freeze_backbone", None)
    projection_hidden_dim_config = cfg.model.get("projection_hidden_dim", None)
    
    # Phase 2 DINOv2: automatically freeze and add projection head if not explicitly set
    if cfg.experiment.phase == 2 and "dinov2" in cfg.model.backbone.lower():
        if freeze_backbone_config is None:
            freeze_backbone = True
            print("Phase 2 with DINOv2: Freezing pretrained backbone weights (default)")
        else:
            freeze_backbone = bool(freeze_backbone_config)
            if freeze_backbone:
                print("Phase 2 with DINOv2: Freezing pretrained backbone weights (from config)")
        
        if projection_hidden_dim_config is None:
            feature_dim = 768  # DINOv2-base feature dimension
            projection_hidden_dim = min(256, feature_dim // 2)
            print(f"Phase 2 with DINOv2: Adding projection head (hidden_dim={projection_hidden_dim}, default)")
        else:
            projection_hidden_dim = projection_hidden_dim_config
            if projection_hidden_dim:
                print(f"Phase 2 with DINOv2: Adding projection head (hidden_dim={projection_hidden_dim}, from config)")
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


def save_misclassifications_if_enabled(
    eval_results: Dict,
    output_dir: Path,
    cfg: DictConfig,
) -> None:
    """Save misclassifications if enabled in config.
    
    Args:
        eval_results: Evaluation results dictionary.
        output_dir: Directory to save misclassification files.
        cfg: Hydra configuration object.
    """
    if not cfg.evaluation.get("save_misclassifications", False):
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cosine misclassifications
    if "misclassified" in eval_results.get("cosine_best", {}):
        save_misclassifications(
            eval_results["cosine_best"]["misclassified"],
            output_dir / "misclassifications_cosine.json",
            method="cosine",
            threshold=eval_results["cosine_best"]["threshold"],
        )
    
    # Save DBSCAN misclassifications
    if "misclassified" in eval_results.get("dbscan_best", {}):
        save_misclassifications(
            eval_results["dbscan_best"]["misclassified"],
            output_dir / "misclassifications_dbscan.json",
            method="dbscan",
            eps=eval_results["dbscan_best"]["eps"],
        )


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
    eval_results = evaluate_embeddings(
        val_embeddings,
        val_lesion_ids_list,
        cosine_thresholds=None,
        dbscan_eps_values=None,
        dbscan_min_samples=cfg.evaluation.dbscan.min_samples,
        image_ids=val_image_ids if cfg.evaluation.get("save_misclassifications", False) else None,
        return_misclassified=cfg.evaluation.get("save_misclassifications", False),
    )
    
    # Save misclassifications if enabled
    output_dir = Path(cfg.paths.output_dir) / cfg.experiment.name / f"fold_{fold_idx}"
    save_misclassifications_if_enabled(eval_results, output_dir, cfg)
    
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
    
    eval_results = evaluate_embeddings(
        val_embeddings,
        val_lesion_ids_list,
        image_ids=val_image_ids if cfg.evaluation.get("save_misclassifications", False) else None,
        return_misclassified=cfg.evaluation.get("save_misclassifications", False),
    )
    
    # Save misclassifications if enabled
    save_misclassifications_if_enabled(eval_results, output_dir, cfg)
    
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


def evaluate_test_set(
    model: EmbeddingModel,
    test_df: pd.DataFrame,
    best_fold_idx: int,
    best_cosine_threshold: float,
    best_dbscan_eps: float,
    cfg: DictConfig,
) -> None:
    """Evaluate model on test set.
    
    Args:
        model: Embedding model.
        test_df: Test set dataframe.
        best_fold_idx: Index of best performing fold.
        best_cosine_threshold: Best cosine threshold from validation.
        best_dbscan_eps: Best DBSCAN eps from validation.
        cfg: Hydra configuration object.
    """
    print(f"\n{'='*60}")
    print("Final Test Set Evaluation")
    print(f"{'='*60}")
    
    test_device = get_device()
    test_use_pin_memory = should_use_pin_memory(test_device, cfg.data.pin_memory)
    
    # Load best model if Phase 2
    if cfg.experiment.phase == 2:
        if best_fold_idx is None:
            best_fold_idx = 0
        
        best_model_path = Path(cfg.paths.output_dir) / cfg.experiment.name / f"fold_{best_fold_idx}" / "best.pt"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=test_device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from fold {best_fold_idx}")
            print(f"  Model path: {best_model_path}")
            print(f"  Model validation F1: {checkpoint.get('best_val_f1', 'N/A')}")
        else:
            print(f"Warning: Best model checkpoint not found at {best_model_path}")
            print(f"  Using model from last fold instead")
    else:
        print("Phase 1: Using frozen pretrained model (no checkpoint to load)")
    
    model.to(test_device)
    
    # Create test dataset and loader
    test_dataset = DermoscopicDataset(
        test_df,
        images_dir=cfg.paths.images_dir,
        image_size=cfg.data.image_size,
        normalize=cfg.data.normalize,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=test_use_pin_memory,
    )
    
    # Extract test embeddings
    model.eval()
    test_embeddings, test_lesion_ids_list, test_image_ids = model.extract_embeddings(
        test_loader, test_device, normalize=True
    )
    
    # Evaluate with best hyperparameters
    print(f"Evaluating test set with best hyperparameters...")
    print(f"  Cosine threshold: {best_cosine_threshold:.4f}")
    print(f"  DBSCAN eps: {best_dbscan_eps:.4f}")
    
    from evaluation.pairwise_metrics import compute_pairwise_f1, evaluate_with_dbscan
    
    test_cosine_results = compute_pairwise_f1(
        test_embeddings,
        test_lesion_ids_list,
        threshold=best_cosine_threshold,
        image_ids=test_image_ids if cfg.evaluation.get("save_misclassifications", False) else None,
        return_misclassified=cfg.evaluation.get("save_misclassifications", False),
    )
    test_dbscan_results = evaluate_with_dbscan(
        test_embeddings,
        test_lesion_ids_list,
        eps=best_dbscan_eps,
        min_samples=cfg.evaluation.dbscan.min_samples,
        image_ids=test_image_ids if cfg.evaluation.get("save_misclassifications", False) else None,
        return_misclassified=cfg.evaluation.get("save_misclassifications", False),
    )
    
    # Save test set misclassifications if enabled
    if cfg.evaluation.get("save_misclassifications", False):
        test_output_dir = Path(cfg.paths.output_dir) / cfg.experiment.name
        
        if "misclassified" in test_cosine_results:
            save_misclassifications(
                test_cosine_results["misclassified"],
                test_output_dir / "test_misclassifications_cosine.json",
                method="cosine",
                threshold=best_cosine_threshold,
            )
        
        if "misclassified" in test_dbscan_results:
            save_misclassifications(
                test_dbscan_results["misclassified"],
                test_output_dir / "test_misclassifications_dbscan.json",
                method="dbscan",
                eps=best_dbscan_eps,
            )
    
    print(f"\nTest Set Results:")
    print(f"  Cosine F1: {test_cosine_results['f1']:.4f}")
    print(f"  Cosine Precision: {test_cosine_results['precision']:.4f}")
    print(f"  Cosine Recall: {test_cosine_results['recall']:.4f}")
    print(f"  DBSCAN F1: {test_dbscan_results['f1']:.4f}")
    print(f"  DBSCAN Precision: {test_dbscan_results['precision']:.4f}")
    print(f"  DBSCAN Recall: {test_dbscan_results['recall']:.4f}")
    
    if wandb.run is not None:
        wandb.log({
            "test/cosine_f1": test_cosine_results["f1"],
            "test/cosine_precision": test_cosine_results["precision"],
            "test/cosine_recall": test_cosine_results["recall"],
            "test/dbscan_f1": test_dbscan_results["f1"],
            "test/dbscan_precision": test_dbscan_results["precision"],
            "test/dbscan_recall": test_dbscan_results["recall"],
        })


def compute_average_results(all_results: List[Dict]) -> Tuple[Dict, int, float, float]:
    """Compute average results across folds and find best fold.
    
    Args:
        all_results: List of result dictionaries, one per fold.
        
    Returns:
        Tuple of (average_results, best_fold_idx, best_cosine_threshold, best_dbscan_eps).
    """
    if len(all_results) == 0:
        return {}, 0, 0.5, 0.1
    
    if len(all_results) == 1:
        return {}, 0, all_results[0].get("cosine_threshold", 0.5), all_results[0].get("dbscan_eps", 0.1)
    
    # Compute averages
    avg_results = {
        "cosine_f1": sum(r["cosine_f1"] for r in all_results) / len(all_results),
        "cosine_precision": sum(r["cosine_precision"] for r in all_results) / len(all_results),
        "cosine_recall": sum(r["cosine_recall"] for r in all_results) / len(all_results),
        "dbscan_f1": sum(r["dbscan_f1"] for r in all_results) / len(all_results),
        "dbscan_precision": sum(r["dbscan_precision"] for r in all_results) / len(all_results),
        "dbscan_recall": sum(r["dbscan_recall"] for r in all_results) / len(all_results),
    }
    
    # Find best performing fold
    best_fold_idx = max(range(len(all_results)), key=lambda i: all_results[i]["cosine_f1"])
    best_fold_f1 = all_results[best_fold_idx]["cosine_f1"]
    
    # Use hyperparameters from best performing fold
    best_cosine_threshold = all_results[best_fold_idx].get("cosine_threshold", 0.5)
    best_dbscan_eps = all_results[best_fold_idx].get("dbscan_eps", 0.1)
    
    return avg_results, best_fold_idx, best_cosine_threshold, best_dbscan_eps


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run experiment.
    
    Args:
        cfg: Hydra configuration object.
    """
    # Set random seed
    set_seed(cfg.experiment.seed)
    
    # Development mode: automatically set epochs to 1
    if cfg.data.dev_mode:
        cfg.training.epochs = 1
    
    # Print configuration
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("="*60 + "\n")
    
    # Initialize W&B
    if cfg.logging.wandb.enabled:
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.experiment.name,
            tags=cfg.logging.wandb.tags,
            notes=cfg.logging.wandb.notes,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    
    # Load data and create splits
    print("Loading data...")
    df_full = pd.read_csv(cfg.paths.csv_path)
    
    max_lesions = cfg.data.max_lesions if cfg.data.dev_mode else None
    if cfg.data.dev_mode:
        print(f"Development mode enabled: Using max {max_lesions} lesions")
    
    # Create train/test split if enabled
    test_df = None
    if cfg.data.get("use_train_test_split", False):
        print("Creating train/test split...")
        train_lesion_ids_all, test_lesion_ids = create_train_test_split(
            df_full,
            test_size=cfg.data.get("test_size", 0.2),
            stratify_by=cfg.data.stratify_by,
            seed=cfg.experiment.seed,
        )
        df = df_full[df_full["lesion_id"].isin(train_lesion_ids_all)].copy()
        test_df = df_full[df_full["lesion_id"].isin(test_lesion_ids)].copy()
        print(f"Train/test split: {len(train_lesion_ids_all)} train lesions, {len(test_lesion_ids)} test lesions")
    else:
        df = df_full
    
    # Create folds
    print("Creating folds...")
    folds = create_folds(
        df,
        n_folds=cfg.data.n_folds,
        stratify_by=cfg.data.stratify_by,
        seed=cfg.experiment.seed,
        max_lesions=max_lesions,
    )
    
    # Filter dataframe if dev mode was used
    if max_lesions is not None:
        all_lesion_ids = set()
        for train_ids, val_ids in folds:
            all_lesion_ids.update(train_ids)
            all_lesion_ids.update(val_ids)
        df = df[df["lesion_id"].isin(all_lesion_ids)]
        print(f"Filtered dataset to {len(df)} images from {len(all_lesion_ids)} lesions")
    
    # Determine which folds to run
    if cfg.experiment.fold is not None:
        fold_indices = [cfg.experiment.fold]
    elif cfg.data.dev_mode:
        fold_indices = [0]
        print(f"Development mode: Running only fold 0 (out of {len(folds)} folds)")
    else:
        fold_indices = list(range(len(folds)))
    
    # Run experiments for each fold
    all_results = []
    model = None
    
    for fold_idx in fold_indices:
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{len(folds)}")
        print(f"{'='*60}")
        
        train_lesion_ids, val_lesion_ids = folds[fold_idx]
        
        # Split data
        train_df = df[df["lesion_id"].isin(train_lesion_ids)]
        val_df = df[df["lesion_id"].isin(val_lesion_ids)]
        
        print(f"Train lesions: {len(train_lesion_ids)}, Train images: {len(train_df)}")
        print(f"Val lesions: {len(val_lesion_ids)}, Val images: {len(val_df)}")
        
        # Create datasets
        train_dataset = DermoscopicDataset(
            train_df,
            images_dir=cfg.paths.images_dir,
            image_size=cfg.data.image_size,
            normalize=cfg.data.normalize,
        )
        val_dataset = DermoscopicDataset(
            val_df,
            images_dir=cfg.paths.images_dir,
            image_size=cfg.data.image_size,
            normalize=cfg.data.normalize,
        )
        
        # Get device and create data loaders
        device = get_device()
        print(f"Using device: {device}")
        use_pin_memory = should_use_pin_memory(device, cfg.data.pin_memory)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True if cfg.experiment.phase == 2 else False,
            num_workers=cfg.data.num_workers,
            pin_memory=use_pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=use_pin_memory,
        )
        
        # Create model (reuse for test evaluation if same config)
        if model is None or fold_idx == 0:
            model = create_model(cfg)
        model.to(device)
        
        # Run phase-specific logic
        if cfg.experiment.phase == 1:
            results = run_phase1_baseline(model, val_loader, device, fold_idx, cfg)
            all_results.append(results)
        elif cfg.experiment.phase == 2:
            results = run_phase2_finetuning(model, train_loader, val_loader, device, fold_idx, cfg)
            all_results.append(results)
        else:
            raise ValueError(f"Unknown phase: {cfg.experiment.phase}")
    
    # Compute average results and best hyperparameters
    avg_results, best_fold_idx, best_cosine_threshold, best_dbscan_eps = compute_average_results(all_results)
    
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Average Results Across Folds")
        print(f"{'='*60}")
        print(f"  Cosine F1: {avg_results['cosine_f1']:.4f}")
        print(f"  Cosine Precision: {avg_results['cosine_precision']:.4f}")
        print(f"  Cosine Recall: {avg_results['cosine_recall']:.4f}")
        print(f"  DBSCAN F1: {avg_results['dbscan_f1']:.4f}")
        print(f"  DBSCAN Precision: {avg_results['dbscan_precision']:.4f}")
        print(f"  DBSCAN Recall: {avg_results['dbscan_recall']:.4f}")
        print(f"  Best fold: {best_fold_idx} (cosine F1: {all_results[best_fold_idx]['cosine_f1']:.4f})")
        print(f"  Using hyperparameters from best fold:")
        print(f"    Cosine threshold: {best_cosine_threshold:.4f}")
        print(f"    DBSCAN eps: {best_dbscan_eps:.4f}")
        
        if wandb.run is not None:
            wandb.log({
                "avg/cosine_f1": avg_results["cosine_f1"],
                "avg/cosine_precision": avg_results["cosine_precision"],
                "avg/cosine_recall": avg_results["cosine_recall"],
                "avg/dbscan_f1": avg_results["dbscan_f1"],
                "avg/dbscan_precision": avg_results["dbscan_precision"],
                "avg/dbscan_recall": avg_results["dbscan_recall"],
                "best_fold_idx": best_fold_idx,
                "best_fold_cosine_f1": all_results[best_fold_idx]["cosine_f1"],
                "test_cosine_threshold": best_cosine_threshold,
                "test_dbscan_eps": best_dbscan_eps,
            })
    
    # Final test set evaluation if train/test split was used
    if test_df is not None and len(all_results) > 0:
        evaluate_test_set(
            model, test_df, best_fold_idx, best_cosine_threshold, best_dbscan_eps, cfg
        )
    
    # Save resolved config
    output_dir = Path(cfg.paths.output_dir) / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
