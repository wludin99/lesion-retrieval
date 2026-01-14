"""Dataset classes for dermoscopic image matching."""

from .dermoscopic_dataset import DermoscopicDataset
from .folds import create_folds

__all__ = ["DermoscopicDataset", "create_folds"]
