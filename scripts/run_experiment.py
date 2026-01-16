"""Main entry point for running experiments."""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import wandb

from datasets import DermoscopicDataset, create_folds, create_train_test_split
from evaluation import (
    compute_average_results,
    evaluate_test_set,
    save_fold_results,
    save_overall_results,
)
from training import run_phase1_baseline, run_phase2_finetuning, train_final_model
from utils import create_model, get_device, should_use_pin_memory, set_seed


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
    avg_results, best_fold_idx, best_cosine_threshold, best_dbscan_eps, avg_cosine_threshold, avg_dbscan_eps = \
        compute_average_results(all_results)
    
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
        print(f"\n  Average hyperparameters (will be used for final model):")
        print(f"    Cosine threshold: {avg_cosine_threshold:.4f}")
        print(f"    DBSCAN eps: {avg_dbscan_eps:.4f}")
        
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
                "avg_cosine_threshold": avg_cosine_threshold,
                "avg_dbscan_eps": avg_dbscan_eps,
            })
    
    # Train final model on all training data and evaluate on test set
    if test_df is not None and len(all_results) > 0 and not cfg.data.dev_mode:
        # Get full training dataframe (all folds combined, excluding test set)
        final_model = train_final_model(
            model, df, device, avg_cosine_threshold, avg_dbscan_eps, cfg
        )
        
        # Evaluate final model on test set using average hyperparameters
        evaluate_test_set(
            final_model, test_df, None, avg_cosine_threshold, avg_dbscan_eps, cfg
        )
    elif test_df is not None and len(all_results) > 0:
        # Dev mode: skip final model training, just evaluate on test with best fold model
        print("\nDev mode: Skipping final model training, using best fold model for test evaluation")
        evaluate_test_set(
            model, test_df, best_fold_idx, best_cosine_threshold, best_dbscan_eps, cfg
        )
    
    # Save resolved config
    output_dir = Path(cfg.paths.output_dir) / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)
    
    # Save results to CSV files
    save_fold_results(all_results, output_dir, cfg)
    save_overall_results(
        all_results, avg_results, best_fold_idx, best_cosine_threshold, best_dbscan_eps, output_dir
    )
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
