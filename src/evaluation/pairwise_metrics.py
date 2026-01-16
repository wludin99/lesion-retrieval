"""Pairwise F1 and other evaluation metrics."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def _track_misclassifications(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    similarities: np.ndarray,
    lesion_ids: List[str],
    image_ids: List[str],
    cluster_labels: Optional[np.ndarray] = None,
) -> Dict[str, List[Dict]]:
    """Track misclassified pairs from ground truth and predictions.
    
    Args:
        y_true: Ground truth boolean matrix (n, n), True if same lesion.
        y_pred: Prediction boolean matrix (n, n), True if predicted same.
        similarities: Similarity matrix (n, n).
        lesion_ids: List of lesion IDs for each sample.
        image_ids: List of image IDs for each sample.
        cluster_labels: Optional cluster labels for DBSCAN (to include in output).
        
    Returns:
        Dictionary with 'false_positives' and 'false_negatives' lists.
    """
    n = len(lesion_ids)
    false_positives = []  # Predicted same, actually different
    false_negatives = []  # Predicted different, actually same
    
    for i in range(n):
        for j in range(i + 1, n):
            true_same = y_true[i, j]
            pred_same = y_pred[i, j]
            
            if not true_same and pred_same:
                # False positive
                entry = {
                    "image_id_1": image_ids[i],
                    "image_id_2": image_ids[j],
                    "lesion_id_1": lesion_ids[i],
                    "lesion_id_2": lesion_ids[j],
                    "similarity": float(similarities[i, j]),
                }
                if cluster_labels is not None:
                    entry["cluster_1"] = int(cluster_labels[i])
                    entry["cluster_2"] = int(cluster_labels[j])
                false_positives.append(entry)
            elif true_same and not pred_same:
                # False negative
                entry = {
                    "image_id_1": image_ids[i],
                    "image_id_2": image_ids[j],
                    "lesion_id_1": lesion_ids[i],
                    "lesion_id_2": lesion_ids[j],
                    "similarity": float(similarities[i, j]),
                }
                if cluster_labels is not None:
                    entry["cluster_1"] = int(cluster_labels[i])
                    entry["cluster_2"] = int(cluster_labels[j])
                false_negatives.append(entry)
    
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def compute_pairwise_f1(
    embeddings: Union[torch.Tensor, np.ndarray],
    lesion_ids: List[str],
    threshold: float = 0.5,
    metric: str = "cosine",
    image_ids: Optional[List[str]] = None,
    return_misclassified: bool = False,
) -> Dict:
    """Compute pairwise F1 score using cosine similarity thresholding.
    
    Args:
        embeddings: Embedding tensor of shape (N, D).
        lesion_ids: List of lesion_id labels for each embedding.
        threshold: Similarity threshold for positive pairs.
        metric: Distance metric ('cosine' or 'euclidean').
        image_ids: Optional list of image IDs for tracking misclassifications.
        return_misclassified: If True, also return misclassified pairs.
        
    Returns:
        Dictionary with precision, recall, f1, accuracy, and optionally misclassified pairs.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute pairwise similarities
    if metric == "cosine":
        similarities = np.dot(embeddings, embeddings.T)
    else:
        # Euclidean distance converted to similarity
        distances = np.linalg.norm(
            embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :],
            axis=2
        )
        similarities = 1.0 / (1.0 + distances)
    
    # Ground truth: same lesion_id
    n = len(lesion_ids)
    y_true = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if lesion_ids[i] == lesion_ids[j]:
                y_true[i, j] = True
                y_true[j, i] = True
    
    # Predictions: similarity above threshold
    y_pred = similarities >= threshold
    
    # Remove diagonal (self-similarity)
    mask = ~np.eye(n, dtype=bool)
    y_true_flat = y_true[mask]
    y_pred_flat = y_pred[mask]
    
    # Compute metrics
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0.0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0.0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0.0)
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    
    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }
    
    # Track misclassified pairs if requested
    if return_misclassified and image_ids is not None:
        result["misclassified"] = _track_misclassifications(
            y_true, y_pred, similarities, lesion_ids, image_ids
        )
    
    return result


def evaluate_with_dbscan(
    embeddings: Union[torch.Tensor, np.ndarray],
    lesion_ids: List[str],
    eps: float = 0.5,
    min_samples: int = 2,
    image_ids: Optional[List[str]] = None,
    return_misclassified: bool = False,
) -> Dict:
    """Evaluate embeddings using DBSCAN clustering.
    
    Args:
        embeddings: Embedding tensor of shape (N, D).
        lesion_ids: List of lesion_id labels for each embedding.
        eps: DBSCAN eps parameter.
        min_samples: DBSCAN min_samples parameter.
        image_ids: Optional list of image IDs for tracking misclassifications.
        return_misclassified: If True, also return misclassified pairs.
        
    Returns:
        Dictionary with precision, recall, f1, accuracy, and optionally misclassified pairs.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Create pairwise predictions: same cluster = same lesion
    n = len(lesion_ids)
    y_pred = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_labels[i] == cluster_labels[j] and cluster_labels[i] != -1:
                y_pred[i, j] = True
                y_pred[j, i] = True
    
    # Ground truth: same lesion_id
    y_true = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if lesion_ids[i] == lesion_ids[j]:
                y_true[i, j] = True
                y_true[j, i] = True
    
    # Remove diagonal
    mask = ~np.eye(n, dtype=bool)
    y_true_flat = y_true[mask]
    y_pred_flat = y_pred[mask]
    
    # Compute metrics
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0.0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0.0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0.0)
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    
    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "cluster_labels": cluster_labels.tolist(),  # Always include cluster labels
    }
    
    # Track misclassified pairs if requested
    if return_misclassified and image_ids is not None:
        # Compute similarities for misclassification tracking
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(embeddings_norm, embeddings_norm.T)
        
        result["misclassified"] = _track_misclassifications(
            y_true, y_pred, similarities, lesion_ids, image_ids, cluster_labels
        )
    
    return result


def evaluate_embeddings(
    embeddings: Union[torch.Tensor, np.ndarray],
    lesion_ids: List[str],
    cosine_thresholds: Optional[List[float]] = None,
    dbscan_eps_values: Optional[List[float]] = None,
    dbscan_min_samples: int = 2,
    image_ids: Optional[List[str]] = None,
    return_misclassified: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Comprehensive evaluation of embeddings.
    
    Args:
        embeddings: Embedding tensor of shape (N, D).
        lesion_ids: List of lesion_id labels for each embedding.
        cosine_thresholds: List of cosine similarity thresholds to try.
        dbscan_eps_values: List of DBSCAN eps values to try.
        dbscan_min_samples: DBSCAN min_samples parameter.
        image_ids: Optional list of image IDs for tracking misclassifications.
        return_misclassified: If True, also return misclassified pairs in best results.
        
    Returns:
        Dictionary with results for each method and threshold.
    """
    results = {}
    
    # Cosine similarity thresholding
    if cosine_thresholds is None:
        cosine_thresholds = np.linspace(0.5, 0.99, 50)
    
    best_cosine_f1 = -1.0
    best_cosine_threshold = None
    
    for threshold in tqdm(cosine_thresholds, desc="Evaluating cosine thresholds"):
        metrics = compute_pairwise_f1(embeddings, lesion_ids, threshold=threshold)
        results[f"cosine_threshold_{threshold:.3f}"] = metrics
        
        if metrics["f1"] > best_cosine_f1:
            best_cosine_f1 = metrics["f1"]
            best_cosine_threshold = threshold
    
    results["cosine_best"] = {
        "threshold": best_cosine_threshold,
        **compute_pairwise_f1(
            embeddings,
            lesion_ids,
            threshold=best_cosine_threshold,
            image_ids=image_ids,
            return_misclassified=return_misclassified,
        ),
    }
    
    # DBSCAN clustering
    if dbscan_eps_values is None:
        dbscan_eps_values = np.linspace(0.1, 0.9, 20)
    
    best_dbscan_f1 = -1.0
    best_dbscan_eps = None
    
    for eps in tqdm(dbscan_eps_values, desc="Evaluating DBSCAN eps values"):
        metrics = evaluate_with_dbscan(
            embeddings, lesion_ids, eps=eps, min_samples=dbscan_min_samples
        )
        results[f"dbscan_eps_{eps:.3f}"] = metrics
        
        if metrics["f1"] > best_dbscan_f1:
            best_dbscan_f1 = metrics["f1"]
            best_dbscan_eps = eps
    
    results["dbscan_best"] = {
        "eps": best_dbscan_eps,
        **evaluate_with_dbscan(
            embeddings,
            lesion_ids,
            eps=best_dbscan_eps,
            min_samples=dbscan_min_samples,
            image_ids=image_ids,
            return_misclassified=return_misclassified,
        ),
    }
    
    return results
