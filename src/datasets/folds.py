"""Cross-validation fold creation by lesion_id."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def create_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    stratify_by: Optional[str] = None,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create cross-validation folds by lesion_id.
    
    Important: Splits are done by lesion_id, not by image, to prevent
    data leakage. All images from the same lesion are kept together.
    
    Args:
        df: DataFrame with columns including 'lesion_id' and optionally
            a stratification column.
        n_folds: Number of folds for cross-validation.
        stratify_by: Column name to stratify by (e.g., 'anatom_site_general').
            If None, uses regular KFold.
        seed: Random seed for reproducibility.
        
    Returns:
        List of (train_lesion_ids, val_lesion_ids) tuples, one per fold.
    """
    # Get unique lesion IDs
    lesion_df = df.groupby("lesion_id").first().reset_index()
    
    if stratify_by is not None and stratify_by in lesion_df.columns:
        # Stratified split
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = skf.split(
            lesion_df["lesion_id"],
            lesion_df[stratify_by]
        )
    else:
        # Regular split
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = kf.split(lesion_df["lesion_id"])
    
    folds = []
    for train_idx, val_idx in splits:
        train_lesion_ids = lesion_df.iloc[train_idx]["lesion_id"].values
        val_lesion_ids = lesion_df.iloc[val_idx]["lesion_id"].values
        folds.append((train_lesion_ids, val_lesion_ids))
    
    return folds
