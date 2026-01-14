"""Path utilities."""

from pathlib import Path
from typing import Union


def get_image_path(images_dir: Union[str, Path], image_id: str) -> Path:
    """Get full path to an image file.
    
    Args:
        images_dir: Directory containing images.
        image_id: Image ID (e.g., 'ISIC_0028791').
        
    Returns:
        Path to the image file.
    """
    images_dir = Path(images_dir)
    return images_dir / f"{image_id}.jpg"
