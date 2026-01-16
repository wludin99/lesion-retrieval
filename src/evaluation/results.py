"""Results computation and saving utilities."""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from datasets import DermoscopicDataset
from evaluation.pairwise_metrics import compute_pairwise_f1, evaluate_with_dbscan
from evaluation.misclassifications import save_misclassifications
from models import EmbeddingModel
from utils.device import get_device, should_use_pin_memory


def save_clustering_results(
    cluster_labels: List[int],
    lesion_ids: List[str],
    image_ids: List[str],
    output_path: Path,
    eps: float,
    min_samples: int,
) -> None:
    """Save clustering results to CSV.
    
    Args:
        cluster_labels: List of cluster labels (one per image).
        lesion_ids: List of lesion IDs (one per image).
        image_ids: List of image IDs (one per image).
        output_path: Path to save CSV file.
        eps: DBSCAN eps parameter used.
        min_samples: DBSCAN min_samples parameter used.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        "image_id": image_ids,
        "lesion_id": lesion_ids,
        "cluster_id": cluster_labels,
    })
    
    # Sort by cluster_id, then lesion_id for easier inspection
    df = df.sort_values(["cluster_id", "lesion_id", "image_id"])
    
    df.to_csv(output_path, index=False)
    print(f"Saved clustering results to {output_path}")
    print(f"  Total images: {len(df)}")
    print(f"  Total clusters: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    print(f"  Noise points: {list(cluster_labels).count(-1)}")
    print(f"  DBSCAN parameters: eps={eps:.4f}, min_samples={min_samples}")


def compute_average_results(all_results: List[Dict]) -> Tuple[Dict, int, float, float, float, float]:
    """Compute average results across folds and find best fold.
    
    Args:
        all_results: List of result dictionaries, one per fold.
        
    Returns:
        Tuple of (average_results, best_fold_idx, best_cosine_threshold, best_dbscan_eps,
                 avg_cosine_threshold, avg_dbscan_eps).
    """
    if len(all_results) == 0:
        return {}, 0, 0.5, 0.1, 0.5, 0.1
    
    if len(all_results) == 1:
        single_result = all_results[0]
        return {}, 0, single_result.get("cosine_threshold", 0.5), single_result.get("dbscan_eps", 0.1), \
               single_result.get("cosine_threshold", 0.5), single_result.get("dbscan_eps", 0.1)
    
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
    
    # Use hyperparameters from best performing fold (for backward compatibility/reporting)
    best_cosine_threshold = all_results[best_fold_idx].get("cosine_threshold", 0.5)
    best_dbscan_eps = all_results[best_fold_idx].get("dbscan_eps", 0.1)
    
    # Compute average hyperparameters across all folds (for final model training)
    avg_cosine_threshold = sum(r.get("cosine_threshold", 0.5) for r in all_results) / len(all_results)
    avg_dbscan_eps = sum(r.get("dbscan_eps", 0.1) for r in all_results) / len(all_results)
    
    return avg_results, best_fold_idx, best_cosine_threshold, best_dbscan_eps, avg_cosine_threshold, avg_dbscan_eps


def save_fold_results(all_results: List[Dict], output_dir: Path, cfg: DictConfig) -> None:
    """Save results for each fold to CSV files in their respective fold folders.
    
    Results are saved in transposed format (metrics as rows) for readability.
    
    Args:
        all_results: List of result dictionaries, one per fold.
        output_dir: Base output directory (experiment folder).
        cfg: Hydra configuration object.
    """
    for result in all_results:
        fold_idx = result["fold"]
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Transpose: convert dict to DataFrame with metrics as rows
        # Format: metric_name, value
        metrics_list = []
        for key, value in result.items():
            metrics_list.append({"metric": key, "value": value})
        
        fold_df = pd.DataFrame(metrics_list)
        
        # Save to CSV
        results_path = fold_dir / "results.csv"
        fold_df.to_csv(results_path, index=False)
        print(f"Saved fold {fold_idx} results to {results_path}")


def save_overall_results(
    all_results: List[Dict],
    avg_results: Dict,
    best_fold_idx: int,
    best_cosine_threshold: float,
    best_dbscan_eps: float,
    output_dir: Path,
) -> None:
    """Save overall/average results and best fold information to CSV.
    
    Results are saved in transposed format (metrics as rows) for readability.
    
    Args:
        all_results: List of result dictionaries, one per fold.
        avg_results: Dictionary with average metrics across folds.
        best_fold_idx: Index of best performing fold.
        best_cosine_threshold: Best cosine threshold from best fold.
        best_dbscan_eps: Best DBSCAN eps from best fold.
        output_dir: Base output directory (experiment folder).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create overall results dictionary
    overall_results = {
        "num_folds": len(all_results),
        "best_fold_idx": best_fold_idx,
        "best_fold_cosine_f1": all_results[best_fold_idx]["cosine_f1"] if len(all_results) > 0 else None,
        "avg_cosine_f1": avg_results.get("cosine_f1", None),
        "avg_cosine_precision": avg_results.get("cosine_precision", None),
        "avg_cosine_recall": avg_results.get("cosine_recall", None),
        "avg_dbscan_f1": avg_results.get("dbscan_f1", None),
        "avg_dbscan_precision": avg_results.get("dbscan_precision", None),
        "avg_dbscan_recall": avg_results.get("dbscan_recall", None),
        "best_cosine_threshold": best_cosine_threshold,
        "best_dbscan_eps": best_dbscan_eps,
    }
    
    # Add best fold individual metrics
    if len(all_results) > 0:
        best_result = all_results[best_fold_idx]
        overall_results.update({
            "best_fold_cosine_precision": best_result.get("cosine_precision", None),
            "best_fold_cosine_recall": best_result.get("cosine_recall", None),
            "best_fold_dbscan_f1": best_result.get("dbscan_f1", None),
            "best_fold_dbscan_precision": best_result.get("dbscan_precision", None),
            "best_fold_dbscan_recall": best_result.get("dbscan_recall", None),
        })
    
    # Transpose: convert dict to DataFrame with metrics as rows
    # Format: metric_name, value
    metrics_list = []
    for key, value in overall_results.items():
        metrics_list.append({"metric": key, "value": value})
    
    overall_df = pd.DataFrame(metrics_list)
    results_path = output_dir / "results.csv"
    overall_df.to_csv(results_path, index=False)
    print(f"Saved overall results to {results_path}")


def evaluate_test_set(
    model: EmbeddingModel,
    test_df: pd.DataFrame,
    best_fold_idx: Optional[int],
    best_cosine_threshold: float,
    best_dbscan_eps: float,
    cfg: DictConfig,
) -> None:
    """Evaluate model on test set.
    
    Args:
        model: Embedding model (should already be the trained model if best_fold_idx is None).
        test_df: Test set dataframe.
        best_fold_idx: Index of best performing fold, or None if using final trained model.
        best_cosine_threshold: Cosine threshold to use for evaluation.
        best_dbscan_eps: DBSCAN eps to use for evaluation.
        cfg: Hydra configuration object.
    """
    print(f"\n{'='*60}")
    print("Final Test Set Evaluation")
    print(f"{'='*60}")
    
    test_device = get_device()
    test_use_pin_memory = should_use_pin_memory(test_device, cfg.data.pin_memory)
    
    # Load best model if Phase 2 and best_fold_idx is specified (not using final trained model)
    if cfg.experiment.phase == 2 and best_fold_idx is not None:
        best_model_path = Path(cfg.paths.output_dir) / cfg.experiment.name / f"fold_{best_fold_idx}" / "best.pt"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=test_device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from fold {best_fold_idx}")
            print(f"  Model path: {best_model_path}")
            print(f"  Model validation F1: {checkpoint.get('best_val_f1', 'N/A')}")
        else:
            print(f"Warning: Best model checkpoint not found at {best_model_path}")
            print(f"  Using provided model instead")
    elif cfg.experiment.phase == 2:
        print("Using final trained model (trained on all training data)")
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
    
    # Save test set clustering results
    if "cluster_labels" in test_dbscan_results:
        save_clustering_results(
            cluster_labels=test_dbscan_results["cluster_labels"],
            lesion_ids=test_lesion_ids_list,
            image_ids=test_image_ids,
            output_path=test_output_dir / "test_clustered_images.csv",
            eps=best_dbscan_eps,
            min_samples=cfg.evaluation.dbscan.min_samples,
        )
    
    print(f"\nTest Set Results:")
    print(f"  Cosine F1: {test_cosine_results['f1']:.4f}")
    print(f"  Cosine Precision: {test_cosine_results['precision']:.4f}")
    print(f"  Cosine Recall: {test_cosine_results['recall']:.4f}")
    print(f"  DBSCAN F1: {test_dbscan_results['f1']:.4f}")
    print(f"  DBSCAN Precision: {test_dbscan_results['precision']:.4f}")
    print(f"  DBSCAN Recall: {test_dbscan_results['recall']:.4f}")
    
    import wandb
    if wandb.run is not None:
        wandb.log({
            "test/cosine_f1": test_cosine_results["f1"],
            "test/cosine_precision": test_cosine_results["precision"],
            "test/cosine_recall": test_cosine_results["recall"],
            "test/dbscan_f1": test_dbscan_results["f1"],
            "test/dbscan_precision": test_dbscan_results["precision"],
            "test/dbscan_recall": test_dbscan_results["recall"],
        })
