"""PyTorch dataset for dermoscopic images."""

from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.paths import get_image_path


class DermoscopicDataset(Dataset):
    """Dataset for dermoscopic images with metadata.
    
    Each sample is an image with its lesion_id label.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        image_size: int = 224,
        normalize: Optional[dict] = None,
        transform: Optional[Callable] = None,
    ):
        """Initialize dataset.
        
        Args:
            df: DataFrame with columns 'image_id' and 'lesion_id'.
            images_dir: Directory containing image files.
            image_size: Target image size for resizing.
            normalize: Dict with 'mean' and 'std' for normalization.
                If None, uses ImageNet defaults.
            transform: Optional additional transforms to apply.
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        
        # Build transform pipeline
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        
        if normalize is None:
            normalize = {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }
        
        transform_list.append(
            transforms.Normalize(mean=normalize["mean"], std=normalize["std"])
        )
        
        self.base_transform = transforms.Compose(transform_list)
        self.transform = transform
        
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """Get a single sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (image_tensor, lesion_id, image_id).
        """
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        lesion_id = row["lesion_id"]
        
        # Load image
        image_path = get_image_path(self.images_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        image = self.base_transform(image)
        if self.transform is not None:
            image = self.transform(image)
        
        return image, lesion_id, image_id
