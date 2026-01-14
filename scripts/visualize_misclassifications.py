"""Visualize misclassified pairs side-by-side."""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from PIL import Image

from evaluation import load_misclassifications
from utils.paths import get_image_path


def visualize_pairs(
    misclass_file: Path,
    images_dir: Path,
    output_dir: Optional[Path] = None,
    max_pairs: int = 50,
    error_type: Optional[str] = None,
) -> None:
    """Visualize misclassified pairs side-by-side.
    
    Args:
        misclass_file: Path to misclassifications JSON file.
        images_dir: Directory containing images.
        output_dir: Directory to save visualization images. If None, displays interactively.
        max_pairs: Maximum number of pairs to visualize.
        error_type: Type of errors to visualize ('false_positives', 'false_negatives', or None for both).
    """
    # Load misclassifications
    data = load_misclassifications(misclass_file)
    
    print(f"Loaded misclassifications from {misclass_file}")
    print(f"Method: {data['method']}")
    if data['method'] == 'cosine':
        print(f"Threshold: {data['threshold']:.4f}")
    else:
        print(f"Eps: {data['eps']:.4f}")
    print(f"False positives: {data['num_false_positives']}")
    print(f"False negatives: {data['num_false_negatives']}")
    
    # Determine which error types to visualize
    error_types = []
    if error_type is None:
        if data['num_false_positives'] > 0:
            error_types.append('false_positives')
        if data['num_false_negatives'] > 0:
            error_types.append('false_negatives')
    else:
        if error_type in ['false_positives', 'false_negatives']:
            error_types.append(error_type)
        else:
            raise ValueError(f"error_type must be 'false_positives' or 'false_negatives', got {error_type}")
    
    if not error_types:
        print("No misclassifications to visualize!")
        return
    
    # Create output directory if needed
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = Path(images_dir)
    
    # Visualize each error type
    for err_type in error_types:
        pairs = data[err_type][:max_pairs]
        
        if len(pairs) == 0:
            print(f"No {err_type} to visualize")
            continue
        
        print(f"\nVisualizing {len(pairs)} {err_type}...")
        
        # Create figure with subplots
        n_pairs = len(pairs)
        n_cols = 1
        n_rows = n_pairs
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
        if n_pairs == 1:
            axes = [axes]
        
        for idx, pair in enumerate(pairs):
            ax = axes[idx]
            
            # Load images
            img1_path = get_image_path(images_dir, pair['image_id_1'])
            img2_path = get_image_path(images_dir, pair['image_id_2'])
            
            try:
                img1 = Image.open(img1_path).convert("RGB")
                img2 = Image.open(img2_path).convert("RGB")
            except Exception as e:
                print(f"Error loading images for pair {idx}: {e}")
                ax.text(0.5, 0.5, f"Error loading images\n{pair['image_id_1']}\n{pair['image_id_2']}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # Create side-by-side visualization
            ax.axis('off')
            
            # Place images side by side
            ax.imshow(img1, extent=[0, 0.48, 0, 1])
            ax.imshow(img2, extent=[0.52, 1, 0, 1])
            
            # Add labels
            title = f"{err_type.replace('_', ' ').title()}\n"
            title += f"Image 1: {pair['image_id_1']} (Lesion: {pair['lesion_id_1']})\n"
            title += f"Image 2: {pair['image_id_2']} (Lesion: {pair['lesion_id_2']})\n"
            title += f"Similarity: {pair['similarity']:.4f}"
            
            ax.set_title(title, fontsize=10)
            ax.axvline(x=0.5, color='black', linewidth=2)
        
        plt.tight_layout()
        
        if output_dir is not None:
            output_path = output_dir / f"{err_type}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize misclassified pairs")
    parser.add_argument(
        "misclass_file",
        type=Path,
        help="Path to misclassifications JSON file",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save visualization images (if not provided, displays interactively)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=50,
        help="Maximum number of pairs to visualize per error type",
    )
    parser.add_argument(
        "--error-type",
        type=str,
        choices=["false_positives", "false_negatives"],
        default=None,
        help="Type of errors to visualize (default: both)",
    )
    
    args = parser.parse_args()
    
    visualize_pairs(
        args.misclass_file,
        args.images_dir,
        args.output_dir,
        args.max_pairs,
        args.error_type,
    )


if __name__ == "__main__":
    main()
