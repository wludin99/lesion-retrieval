"""Script to cluster images using a trained model and DBSCAN.

This script loads a trained model from an experiment folder, extracts embeddings
from images in a data folder, and clusters them using DBSCAN with optimal
hyperparameters from the experiment results.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import numpy as np
from omegaconf import OmegaConf
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import DermoscopicDataset
from models import EmbeddingModel
from evaluation.pairwise_metrics import compute_pairwise_f1, evaluate_with_dbscan


def load_model_from_experiment(
    experiment_dir: Path,
    fold_idx: int = 0,
    device: Optional[torch.device] = None,
) -> tuple[EmbeddingModel, dict, Path]:
    """Load model and config from experiment folder.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., experiments/finetune_resnet_infonce).
            Can also be a timestamped subfolder (e.g., experiments/finetune_resnet_infonce/2026-01-15_07-33-10).
        fold_idx: Index of fold to load model from (default: 0).
        device: Device to load model on. If None, auto-detects.
        
    Returns:
        Tuple of (model, config_dict, actual_experiment_dir).
    """
    experiment_dir = Path(experiment_dir)
    
    # Handle timestamped subfolders - check if resolved_config.yaml exists, if not check parent
    config_path = experiment_dir / "resolved_config.yaml"
    if not config_path.exists():
        # Try parent directory (in case user provided timestamped subfolder)
        parent_dir = experiment_dir.parent
        parent_config_path = parent_dir / "resolved_config.yaml"
        if parent_config_path.exists():
            experiment_dir = parent_dir
            config_path = parent_config_path
        else:
            raise FileNotFoundError(
                f"Config not found in {experiment_dir} or {parent_dir}. "
                f"Expected resolved_config.yaml in one of these locations."
            )
    
    cfg = OmegaConf.load(config_path)
    
    # Load model checkpoint
    checkpoint_path = experiment_dir / f"fold_{fold_idx}" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    # Create model
    model = EmbeddingModel(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        embedding_dim=cfg.model.embedding_dim,
        pool_type=cfg.model.pool_type,
        freeze_backbone=cfg.model.get("freeze_backbone", False),
        projection_hidden_dim=cfg.model.get("projection_hidden_dim", None),
    )
    
    # Load checkpoint (use CPU for loading to avoid MPS alignment issues)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load state dict with strict=False to handle any minor mismatches
    # but warn if there are missing or unexpected keys
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys when loading checkpoint: {missing_keys[:5]}...")  # Show first 5
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")  # Show first 5
    
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Model: {cfg.model.backbone}")
    print(f"  Device: {device}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'best_val_f1' in checkpoint:
        print(f"  Checkpoint best_val_f1: {checkpoint.get('best_val_f1', 'N/A'):.4f}")
    if 'cosine_threshold' in checkpoint:
        print(f"  Checkpoint cosine_threshold: {checkpoint.get('cosine_threshold', 'N/A'):.4f}")
    if 'dbscan_eps' in checkpoint:
        print(f"  Checkpoint dbscan_eps: {checkpoint.get('dbscan_eps', 'N/A'):.4f}")
    
    return model, OmegaConf.to_container(cfg, resolve=True), experiment_dir, checkpoint


def get_optimal_dbscan_params(
    cfg: dict, 
    experiment_dir: Path, 
    fold_idx: int = 0,
    checkpoint_path: Optional[Path] = None,
    checkpoint: Optional[dict] = None,
) -> tuple[float, int]:
    """Get optimal DBSCAN parameters from checkpoint, results CSV, or config.
    
    Tries to find the best hyperparameters from (in order):
    1. Checkpoint file (best.pt) - if provided, uses dbscan_eps from checkpoint
       (This is the cleanest method - hyperparameters saved with model)
    2. Fold-specific results.csv (if available) - uses dbscan_eps from that fold
    3. Overall results.csv (if available) - uses best_dbscan_eps from best fold
    4. Config defaults
    
    Args:
        cfg: Configuration dictionary.
        experiment_dir: Path to experiment directory.
        fold_idx: Fold index to get parameters from (if fold-specific results exist).
        checkpoint_path: Optional path to checkpoint file. If provided, will try to infer
            fold_idx from the checkpoint path if it's in a fold_* directory.
        checkpoint: Optional checkpoint dictionary (already loaded). If provided, will
            try to extract dbscan_eps directly.
        
    Returns:
        Tuple of (eps, min_samples).
    """
    min_samples = cfg.get("evaluation", {}).get("dbscan", {}).get("min_samples", 2)
    
    # FIRST: Try to load from checkpoint (cleanest method - hyperparameters saved with model)
    if checkpoint is not None and "dbscan_eps" in checkpoint:
        eps = float(checkpoint["dbscan_eps"])
        print(f"Using optimal DBSCAN eps from checkpoint: {eps:.4f}")
        return eps, min_samples
    
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            try:
                loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if "dbscan_eps" in loaded_checkpoint:
                    eps = float(loaded_checkpoint["dbscan_eps"])
                    print(f"Using optimal DBSCAN eps from checkpoint: {eps:.4f}")
                    print(f"  (Loaded from {checkpoint_path})")
                    return eps, min_samples
            except Exception as e:
                print(f"Warning: Could not read hyperparameters from checkpoint: {e}")
        
        # Try to infer fold_idx from checkpoint path if in fold_* directory
        parent = checkpoint_path.parent
        if parent.name.startswith("fold_"):
            try:
                inferred_fold = int(parent.name.split("_")[1])
                fold_idx = inferred_fold
                print(f"Inferred fold_idx={fold_idx} from checkpoint path: {checkpoint_path}")
            except (ValueError, IndexError):
                pass  # Use the provided fold_idx
    
    # SECOND: Try to load from fold-specific results.csv (transposed format)
    fold_results_path = experiment_dir / f"fold_{fold_idx}" / "results.csv"
    if fold_results_path.exists():
        try:
            fold_df = pd.read_csv(fold_results_path)
            # Check if transposed format (metric, value) or old format
            if "metric" in fold_df.columns and "value" in fold_df.columns:
                # Transposed format
                dbscan_eps_row = fold_df[fold_df["metric"] == "dbscan_eps"]
                if not dbscan_eps_row.empty and not pd.isna(dbscan_eps_row["value"].iloc[0]):
                    eps = float(dbscan_eps_row["value"].iloc[0])
                    print(f"Using optimal DBSCAN eps from fold {fold_idx} results: {eps:.4f}")
                    print(f"  (Loaded from {fold_results_path})")
                    return eps, min_samples
            elif "dbscan_eps" in fold_df.columns and not pd.isna(fold_df["dbscan_eps"].iloc[0]):
                # Old format (for backward compatibility)
                eps = float(fold_df["dbscan_eps"].iloc[0])
                print(f"Using optimal DBSCAN eps from fold {fold_idx} results: {eps:.4f}")
                print(f"  (Loaded from {fold_results_path})")
                return eps, min_samples
        except Exception as e:
            print(f"Warning: Could not read fold results.csv: {e}")
    
    # THIRD: Try to load from overall results.csv (transposed format)
    overall_results_path = experiment_dir / "results.csv"
    if overall_results_path.exists():
        try:
            overall_df = pd.read_csv(overall_results_path)
            # Check if transposed format (metric, value) or old format
            if "metric" in overall_df.columns and "value" in overall_df.columns:
                # Transposed format
                dbscan_eps_row = overall_df[overall_df["metric"] == "best_dbscan_eps"]
                if not dbscan_eps_row.empty and not pd.isna(dbscan_eps_row["value"].iloc[0]):
                    eps = float(dbscan_eps_row["value"].iloc[0])
                    print(f"Using optimal DBSCAN eps from overall results: {eps:.4f}")
                    print(f"  (Loaded from {overall_results_path})")
                    return eps, min_samples
            elif "best_dbscan_eps" in overall_df.columns and not pd.isna(overall_df["best_dbscan_eps"].iloc[0]):
                # Old format (for backward compatibility)
                eps = float(overall_df["best_dbscan_eps"].iloc[0])
                print(f"Using optimal DBSCAN eps from overall results: {eps:.4f}")
                print(f"  (Loaded from {overall_results_path})")
                return eps, min_samples
        except Exception as e:
            print(f"Warning: Could not read overall results.csv: {e}")
    
    # Fallback to config defaults
    eps_default = 0.5  # Default fallback
    dbscan_config = cfg.get("evaluation", {}).get("dbscan", {})
    if "eps_range" in dbscan_config:
        # Use middle of range as a reasonable default
        eps_range = dbscan_config["eps_range"]
        eps_default = (eps_range[0] + eps_range[1]) / 2
    
    print(f"Warning: No hyperparameters found in checkpoint or results.csv. Using default eps from config: {eps_default:.4f}")
    print(f"  Expected locations:")
    print(f"    - Checkpoint: {checkpoint_path if checkpoint_path else 'Not specified'}")
    print(f"    - {fold_results_path}")
    print(f"    - {overall_results_path}")
    print("  (This may not be optimal. Consider running validation first or manually specify --eps)")
    
    return eps_default, min_samples


def cluster_images_with_dbscan(
    embeddings: np.ndarray,
    eps: float,
    min_samples: int = 2,
) -> np.ndarray:
    """Cluster embeddings using DBSCAN.
    
    Args:
        embeddings: Normalized embedding array of shape (N, D).
        eps: DBSCAN eps parameter.
        min_samples: DBSCAN min_samples parameter.
        
    Returns:
        Array of cluster labels (shape N,).
    """
    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    cluster_labels = clustering.fit_predict(embeddings)
    
    return cluster_labels


def main():
    """Main function to cluster images."""
    parser = argparse.ArgumentParser(
        description="Cluster images using a trained model and DBSCAN"
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment directory (e.g., experiments/finetune_resnet_infonce)",
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to data directory with data.csv and images/ folder",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index to load model from (default: 0)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="DBSCAN eps parameter (overrides auto-detection)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="DBSCAN min_samples parameter (overrides auto-detection)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: experiment_dir/clustered_images.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction (default: 32)",
    )
    
    args = parser.parse_args()
    
    # Load model and config
    print("Loading model...")
    model, cfg, actual_experiment_dir, checkpoint = load_model_from_experiment(args.experiment_dir, args.fold)
    device = next(model.parameters()).device
    
    # Get checkpoint path to help determine optimal eps from the same fold
    checkpoint_path = actual_experiment_dir / f"fold_{args.fold}" / "best.pt"
    if not checkpoint_path.exists():
        # Fallback: try to find any best.pt in the experiment dir structure
        checkpoint_path = None
    
    # Get optimal DBSCAN parameters (prefer from checkpoint, then CSV, then config)
    eps_default, min_samples_default = get_optimal_dbscan_params(
        cfg, actual_experiment_dir, args.fold, checkpoint_path, checkpoint=checkpoint
    )
    eps = args.eps if args.eps is not None else eps_default
    min_samples = args.min_samples if args.min_samples is not None else min_samples_default
    
    print(f"\nDBSCAN parameters: eps={eps}, min_samples={min_samples}")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    csv_path = args.data_dir / "data.csv"
    images_dir = args.data_dir / "images"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images from {len(df['lesion_id'].unique())} lesions")
    
    # Create dataset and dataloader
    dataset = DermoscopicDataset(
        df,
        images_dir=images_dir,
        image_size=cfg["data"]["image_size"],
        normalize=cfg["data"]["normalize"],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type != "mps",
    )
    
    # Extract embeddings using the model's extract_embeddings method
    # This ensures consistent behavior with training/validation
    print("\nExtracting embeddings...")
    model.eval()  # Ensure model is in eval mode
    embeddings_tensor, lesion_ids_list, image_ids_list = model.extract_embeddings(
        dataloader, device, normalize=True
    )
    embeddings = embeddings_tensor.numpy()
    print(f"Extracted embeddings: shape {embeddings.shape}")
    
    # Diagnostic: Check embedding statistics
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    similarities = np.dot(embeddings_norm, embeddings_norm.T)
    np.fill_diagonal(similarities, 0)  # Remove diagonal for stats
    print(f"Embedding diagnostics:")
    print(f"  Norm range: {np.linalg.norm(embeddings, axis=1).min():.4f} to {np.linalg.norm(embeddings, axis=1).max():.4f}")
    print(f"  Similarity range (off-diagonal): {similarities.min():.4f} to {similarities.max():.4f}")
    print(f"  Mean similarity: {similarities.mean():.4f}")
    
    # Cluster with DBSCAN and evaluate using the same function as training/validation
    # This ensures we're using exactly the same evaluation logic
    print("\nClustering with DBSCAN...")
    cluster_labels = cluster_images_with_dbscan(embeddings, eps, min_samples)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"Found {n_clusters} clusters")
    print(f"  Noise points (unclustered): {n_noise}")
    
    # Evaluate using the same function as training/validation
    # This ensures we're using exactly the same evaluation logic
    print("\nEvaluating clustering performance...")
    eval_results = evaluate_with_dbscan(
        embeddings_tensor,  # Use tensor directly, function will handle normalization
        lesion_ids_list,
        eps=eps,
        min_samples=min_samples,
        image_ids=None,  # Not tracking misclassifications for now
        return_misclassified=False,
    )
    
    print(f"\nClustering Performance Metrics:")
    print(f"  Precision: {eval_results['precision']:.4f}")
    print(f"  Recall: {eval_results['recall']:.4f}")
    print(f"  F1 Score: {eval_results['f1']:.4f}")
    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        "image_id": image_ids_list,
        "lesion_id": lesion_ids_list,
        "cluster_id": cluster_labels,
    })
    
    # Sort by cluster_id, then lesion_id for easier inspection
    output_df = output_df.sort_values(["cluster_id", "lesion_id", "image_id"])
    
    # Save output (use actual experiment dir, not the potentially timestamped subfolder)
    if args.output is None:
        output_path = actual_experiment_dir / "clustered_images.csv"
    else:
        output_path = args.output
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved clustered images to {output_path}")
    print(f"  Total images: {len(output_df)}")
    print(f"  Total clusters: {n_clusters}")
    print(f"  Noise points: {n_noise}")
    
    # Print some statistics
    print("\nCluster statistics:")
    cluster_counts = output_df["cluster_id"].value_counts().sort_index()
    print(f"  Largest cluster: {cluster_counts.max()} images")
    print(f"  Smallest cluster: {cluster_counts[cluster_counts.index >= 0].min()} images")
    print(f"  Average cluster size: {cluster_counts[cluster_counts.index >= 0].mean():.1f} images")


if __name__ == "__main__":
    main()
