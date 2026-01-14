"""Utilities for tracking and saving misclassified pairs."""

import json
from pathlib import Path
from typing import Dict, List, Optional


def save_misclassifications(
    misclassified_data: Dict,
    output_path: Path,
    method: str = "cosine",
    threshold: Optional[float] = None,
    eps: Optional[float] = None,
) -> None:
    """Save misclassified pairs to a JSON file.
    
    Args:
        misclassified_data: Dictionary with 'false_positives' and 'false_negatives' lists.
        output_path: Path to save the JSON file.
        method: Evaluation method ('cosine' or 'dbscan').
        threshold: Threshold used (for cosine method).
        eps: Eps parameter used (for DBSCAN method).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "method": method,
        "threshold": threshold,
        "eps": eps,
        "num_false_positives": len(misclassified_data.get("false_positives", [])),
        "num_false_negatives": len(misclassified_data.get("false_negatives", [])),
        "false_positives": misclassified_data.get("false_positives", []),
        "false_negatives": misclassified_data.get("false_negatives", []),
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved misclassifications to {output_path}")
    print(f"  False positives: {data['num_false_positives']}")
    print(f"  False negatives: {data['num_false_negatives']}")


def load_misclassifications(file_path: Path) -> Dict:
    """Load misclassified pairs from a JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Dictionary with misclassification data.
    """
    with open(file_path, "r") as f:
        return json.load(f)
