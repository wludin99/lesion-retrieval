"""Main entry point for running experiments."""

import os
from pathlib import Path
from typing import Optional

import hydra
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import wandb

from datasets import DermoscopicDataset, create_folds
from models import EmbeddingModel
from losses import ContrastiveLoss, TripletLoss, InfoNCELoss
from training import Trainer
from evaluation import evaluate_embeddings
from utils import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run experiment.
    
    Args:
        cfg: Hydra configuration object.
    """
    # Set random seed
    set_seed(cfg.experiment.seed)
    
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
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(cfg.paths.csv_path)
    
    # Create folds
    print("Creating folds...")
    folds = create_folds(
        df,
        n_folds=cfg.data.n_folds,
        stratify_by=cfg.data.stratify_by,
        seed=cfg.experiment.seed,
    )
    
    # Determine which folds to run
    if cfg.experiment.fold is not None:
        fold_indices = [cfg.experiment.fold]
    else:
        fold_indices = list(range(len(folds)))
    
    # Run experiments for each fold
    all_results = []
    
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
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True if cfg.experiment.phase == 2 else False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
        
        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = EmbeddingModel(
            backbone=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            embedding_dim=cfg.model.embedding_dim,
            pool_type=cfg.model.pool_type,
        )
        model.to(device)
        
        # Phase 1: Baseline embedding extraction
        if cfg.experiment.phase == 1:
            print("\nPhase 1: Extracting baseline embeddings...")
            model.eval()
            
            # Extract embeddings
            train_embeddings, train_lesion_ids_list, train_image_ids = model.extract_embeddings(
                train_loader, device, normalize=True
            )
            val_embeddings, val_lesion_ids_list, val_image_ids = model.extract_embeddings(
                val_loader, device, normalize=True
            )
            
            # Evaluate on validation set
            print("Evaluating embeddings...")
            eval_results = evaluate_embeddings(
                val_embeddings,
                val_lesion_ids_list,
                cosine_thresholds=None,  # Use defaults from config
                dbscan_eps_values=None,
                dbscan_min_samples=cfg.evaluation.dbscan.min_samples,
            )
            
            # Log results
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
            
            all_results.append(results)
            
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
        
        # Phase 2: Metric learning fine-tuning
        elif cfg.experiment.phase == 2:
            print("\nPhase 2: Fine-tuning with metric learning...")
            
            # Create loss function
            loss_name = cfg.loss.name
            if loss_name == "contrastive":
                loss_fn = ContrastiveLoss(
                    margin=cfg.loss.margin,
                    temperature=cfg.loss.temperature,
                )
            elif loss_name == "triplet":
                loss_fn = TripletLoss(
                    margin=cfg.loss.margin,
                    sampling=cfg.loss.sampling,
                )
            elif loss_name == "infonce":
                loss_fn = InfoNCELoss(
                    temperature=cfg.loss.temperature,
                    num_negatives=cfg.loss.get("num_negatives", None),
                )
            else:
                raise ValueError(f"Unknown loss: {loss_name}")
            
            # Create optimizer
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
            
            # Final evaluation
            print("Final evaluation...")
            val_embeddings, val_lesion_ids_list, val_image_ids = model.extract_embeddings(
                val_loader, device, normalize=True
            )
            
            eval_results = evaluate_embeddings(
                val_embeddings,
                val_lesion_ids_list,
            )
            
            results = {
                "fold": fold_idx,
                "cosine_f1": eval_results["cosine_best"]["f1"],
                "cosine_precision": eval_results["cosine_best"]["precision"],
                "cosine_recall": eval_results["cosine_best"]["recall"],
                "dbscan_f1": eval_results["dbscan_best"]["f1"],
                "dbscan_precision": eval_results["dbscan_best"]["precision"],
                "dbscan_recall": eval_results["dbscan_best"]["recall"],
            }
            
            all_results.append(results)
            
            print(f"\nFold {fold_idx} Final Results:")
            print(f"  Cosine F1: {results['cosine_f1']:.4f}")
            print(f"  DBSCAN F1: {results['dbscan_f1']:.4f}")
        
        else:
            raise ValueError(f"Unknown phase: {cfg.experiment.phase}")
    
    # Compute average results across folds
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Average Results Across Folds")
        print(f"{'='*60}")
        
        avg_results = {
            "cosine_f1": sum(r["cosine_f1"] for r in all_results) / len(all_results),
            "cosine_precision": sum(r["cosine_precision"] for r in all_results) / len(all_results),
            "cosine_recall": sum(r["cosine_recall"] for r in all_results) / len(all_results),
            "dbscan_f1": sum(r["dbscan_f1"] for r in all_results) / len(all_results),
            "dbscan_precision": sum(r["dbscan_precision"] for r in all_results) / len(all_results),
            "dbscan_recall": sum(r["dbscan_recall"] for r in all_results) / len(all_results),
        }
        
        print(f"  Cosine F1: {avg_results['cosine_f1']:.4f}")
        print(f"  Cosine Precision: {avg_results['cosine_precision']:.4f}")
        print(f"  Cosine Recall: {avg_results['cosine_recall']:.4f}")
        print(f"  DBSCAN F1: {avg_results['dbscan_f1']:.4f}")
        print(f"  DBSCAN Precision: {avg_results['dbscan_precision']:.4f}")
        print(f"  DBSCAN Recall: {avg_results['dbscan_recall']:.4f}")
        
        if wandb.run is not None:
            wandb.log({
                "avg/cosine_f1": avg_results["cosine_f1"],
                "avg/cosine_precision": avg_results["cosine_precision"],
                "avg/cosine_recall": avg_results["cosine_recall"],
                "avg/dbscan_f1": avg_results["dbscan_f1"],
                "avg/dbscan_precision": avg_results["dbscan_precision"],
                "avg/dbscan_recall": avg_results["dbscan_recall"],
            })
    
    # Save resolved config
    output_dir = Path(cfg.paths.output_dir) / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
