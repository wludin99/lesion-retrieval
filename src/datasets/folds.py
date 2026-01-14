"""Cross-validation fold creation by lesion_id."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


def create_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    stratify_by: Optional[str] = None,
    seed: int = 42,
    max_lesions: Optional[int] = None,
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
        max_lesions: Maximum number of lesions to use. If None, uses all lesions.
            Useful for development/testing with smaller datasets.
        
    Returns:
        List of (train_lesion_ids, val_lesion_ids) tuples, one per fold.
    """
    # Get unique lesion IDs
    lesion_df = df.groupby("lesion_id").first().reset_index()
    
    # Limit to max_lesions if specified (for dev mode)
    total_lesions = len(lesion_df)
    if max_lesions is not None and total_lesions > max_lesions:
        # Set random seed for reproducible subset selection
        np.random.seed(seed)
        lesion_df = lesion_df.sample(n=max_lesions, random_state=seed).reset_index(drop=True)
        print(f"Development mode: Using {max_lesions} lesions out of {total_lesions} total")
    
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


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify_by: Optional[str] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create train/test split by lesion_id.
    
    Important: Split is done by lesion_id, not by image, to prevent
    data leakage. All images from the same lesion are kept together.
    
    Args:
        df: DataFrame with columns including 'lesion_id' and optionally
            a stratification column.
        test_size: Proportion of lesions to use for test set.
        stratify_by: Column name to stratify by (e.g., 'anatom_site_general').
            If None, uses regular split.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_lesion_ids, test_lesion_ids).
    """
    # Get unique lesion IDs
    lesion_df = df.groupby("lesion_id").first().reset_index()
    
    if stratify_by is not None and stratify_by in lesion_df.columns:
        # Stratified split
        train_lesion_df, test_lesion_df = train_test_split(
            lesion_df,
            test_size=test_size,
            stratify=lesion_df[stratify_by],
            random_state=seed,
        )
    else:
        # Regular split
        train_lesion_df, test_lesion_df = train_test_split(
            lesion_df,
            test_size=test_size,
            random_state=seed,
        )
    
    train_lesion_ids = train_lesion_df["lesion_id"].values
    test_lesion_ids = test_lesion_df["lesion_id"].values
    
    return train_lesion_ids, test_lesion_ids
