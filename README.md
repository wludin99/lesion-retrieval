# Skinbit: Instance-Level Image Matching for Dermoscopic Images

A complete, reproducible machine learning experiment repository for grouping dermoscopic images belonging to the same lesion using deep learning embeddings and metric learning.

## Problem Statement

**Input**: Folder of dermoscopic images + CSV metadata  
**Task**: Group images belonging to the same lesion (each lesion has 2-4 images)  
**Evaluation**: Pairwise F1 score (same lesion vs different lesion)

## Features

- ✅ **Lesion-based cross-validation**: All train/val splits done by `lesion_id`, preventing data leakage
- ✅ **5-fold cross-validation** with optional stratification
- ✅ **Train/test split**: Optional 80-20 stratified split before cross-validation
- ✅ **Baseline embeddings**: ResNet, ViT, DINOv2 with frozen ImageNet/self-supervised weights
- ✅ **Metric learning**: Fine-tuning with contrastive, triplet, and InfoNCE losses
- ✅ **Frozen backbone + projection head**: Default for ViT and DINOv2 in Phase 2
- ✅ **Comprehensive evaluation**: Cosine similarity thresholding + DBSCAN clustering
- ✅ **Final model training**: Trains on all training data using average hyperparameters from CV
- ✅ **Clustering results**: Automatically saved to CSV after each validation run
- ✅ **Misclassification tracking**: Save and visualize false positives/negatives
- ✅ **Experiment tracking**: Weights & Biases integration
- ✅ **Configuration management**: Hydra for all experiment settings
- ✅ **Reproducibility**: Random seed control and config versioning
- ✅ **Development mode**: Fast iteration with subset of data

## Setup

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Install dependencies
uv sync

# Activate virtual environment (if needed)
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### Weights & Biases Setup

1. Create a W&B account at https://wandb.ai
2. Login: `wandb login`
3. Update `configs/logging/default.yaml` with your W&B entity (optional)

## Project Structure

```
.
├── configs/              # Hydra configuration files
│   ├── config.yaml      # Main config
│   ├── data/            # Data loading configs
│   ├── model/           # Model configs (resnet, vit, dinov2)
│   ├── training/        # Training configs
│   ├── loss/            # Loss configs (none, contrastive, triplet, infonce)
│   ├── evaluation/      # Evaluation configs
│   └── logging/         # Logging configs
├── data/                # Data directory (images + CSV)
├── src/
│   ├── datasets/        # Dataset classes and fold splitting
│   ├── models/          # Model architectures
│   ├── losses/          # Loss functions
│   ├── training/        # Training loop
│   ├── evaluation/      # Evaluation metrics
│   └── utils/           # Utilities (seeding, paths)
├── scripts/
│   ├── run_experiment.py           # Main entry point
│   ├── run_all_experiments.py      # Run all Phase 1 + Phase 2 experiments
│   ├── cluster_images.py           # Cluster images using trained model
│   └── visualize_misclassifications.py  # Visualize misclassified pairs
├── experiments/         # Experiment outputs
├── pyproject.toml       # Project dependencies
└── README.md
```

## Configuration with Hydra

All experiment settings are managed through Hydra configs. The main config file is `configs/config.yaml`, which composes configs from subdirectories.

### Key Configuration Groups

- **`data/`**: Data loading, preprocessing, cross-validation settings, train/test split, dev mode
- **`model/`**: Model architecture (resnet, vit, dinov2)
- **`loss/`**: Loss function (none, contrastive, triplet, infonce)
- **`training/`**: Training hyperparameters
- **`evaluation/`**: Evaluation metrics and thresholds, misclassification saving
- **`logging/`**: W&B and logging settings

### Overriding Configs

You can override any config value from the command line:

```bash
# Change model
uv run python scripts/run_experiment.py model=vit

# Change loss
uv run python scripts/run_experiment.py loss=triplet

# Run specific fold
uv run python scripts/run_experiment.py experiment.fold=0

# Change experiment phase
uv run python scripts/run_experiment.py experiment.phase=2

# Enable train/test split
uv run python scripts/run_experiment.py data.use_train_test_split=true

# Enable development mode (30 lesions, 1 epoch, fold 0 only)
uv run python scripts/run_experiment.py data.dev_mode=true

# Combine multiple overrides
uv run python scripts/run_experiment.py model=dinov2 experiment.phase=1 experiment.fold=0 data.dev_mode=true
```

## Running Experiments

**Important**: Run all commands from the project root directory.

### Phase 1: Baseline Embedding Extraction

Extract embeddings from frozen pretrained models and evaluate with cosine similarity and DBSCAN.

#### ResNet Baseline

```bash
uv run python scripts/run_experiment.py \
    model=resnet \
    experiment.phase=1 \
    experiment.name=baseline_resnet
```

#### Vision Transformer Baseline

```bash
uv run python scripts/run_experiment.py \
    model=vit \
    experiment.phase=1 \
    experiment.name=baseline_vit
```

#### DINOv2 Baseline

```bash
uv run python scripts/run_experiment.py \
    model=dinov2 \
    experiment.phase=1 \
    experiment.name=baseline_dinov2
```

### Phase 2: Metric Learning Fine-Tuning

Fine-tune models using metric learning losses. **Note**: ViT and DINOv2 default to frozen backbone + trainable projection head in Phase 2.

#### Contrastive Loss

```bash
uv run python scripts/run_experiment.py \
    model=resnet \
    loss=contrastive \
    experiment.phase=2 \
    experiment.name=finetune_resnet_contrastive
```

#### Triplet Loss

```bash
uv run python scripts/run_experiment.py \
    model=resnet \
    loss=triplet \
    experiment.phase=2 \
    experiment.name=finetune_resnet_triplet
```

#### InfoNCE Loss

```bash
uv run python scripts/run_experiment.py \
    model=resnet \
    loss=infonce \
    experiment.phase=2 \
    experiment.name=finetune_resnet_infonce
```

### Development Mode

For fast iteration, use development mode (30 lesions max, 1 epoch, fold 0 only):

```bash
uv run python scripts/run_experiment.py \
    model=resnet \
    loss=contrastive \
    experiment.phase=2 \
    experiment.name=finetune_resnet_contrastive_dev \
    data.dev_mode=true
```

### Running All Experiments

Run all Phase 1 and Phase 2 experiments sequentially:

```bash
uv run python scripts/run_all_experiments.py
```

This runs:
- Phase 1: 3 baseline models (ResNet, ViT, DINOv2)
- Phase 2: 9 experiments (3 models × 3 losses)

Total: 12 experiments

## Cross-Validation and Evaluation Protocol

### Lesion-Based Splitting

**Critical**: All splits are done by `lesion_id`, not by image. This ensures:
- No data leakage: images from the same lesion never appear in both train and val
- Realistic evaluation: model must generalize to unseen lesions
- Proper grouping: all images of a lesion are kept together

### Train/Test Split

If enabled (`data.use_train_test_split=true`), the protocol is:

1. **Initial split**: 80-20 stratified split of lesions into training and test sets
2. **Cross-validation**: 5-fold CV performed on training set only
3. **Hyperparameter selection**: Average best hyperparameters computed across CV folds
4. **Final model training**: Train model on all training data (Phase 2 only)
5. **Test evaluation**: Evaluate final model on test set using average hyperparameters

This ensures unbiased final evaluation on held-out test data.

### Stratification

You can stratify folds by metadata columns (e.g., anatomical site):

```bash
uv run python scripts/run_experiment.py \
    data.stratify_by=anatom_site_general
```

### Fold Structure

- **5-fold cross-validation** by default
- Each fold splits lesions (not images) into train/val
- Results are computed per fold and averaged
- All metrics logged to W&B with fold-specific tags
- Per-fold results saved to `fold_{i}/results.csv`
- Overall results saved to `results.csv` in experiment root

## Evaluation Metrics

### Cosine Similarity Thresholding

- Computes pairwise cosine similarities between all embeddings
- Varies threshold from 0.5 to 0.99 (50 values by default)
- Reports best F1, precision, recall, accuracy
- Optimal threshold saved per fold

### DBSCAN Clustering

- Clusters embeddings using DBSCAN
- Varies `eps` parameter from 0.1 to 0.9 (20 values by default)
- Reports best F1, precision, recall, accuracy
- Optimal `eps` saved per fold
- Cluster assignments saved to `clustered_images.csv` per fold

### Pairwise F1 Score

The primary metric is **pairwise F1**:
- **Positive pairs**: Images from the same lesion
- **Negative pairs**: Images from different lesions
- F1 score computed over all pairwise comparisons

## Data Format

### CSV Structure

The `data/data.csv` file should have at minimum:
- `lesion_id`: Unique identifier for each lesion
- `image_id`: Unique identifier for each image (without .jpg extension)

Optional columns for stratification:
- `anatom_site_general`: Anatomical site
- `sex`: Patient sex
- `dermoscopic_type`: Dermoscopic imaging type
- Other metadata columns

### Image Files

Images should be stored in `data/images/` with naming:
- `{image_id}.jpg` (e.g., `ISIC_0028791.jpg`)

## Weights & Biases Integration

All experiments are automatically logged to W&B:

- **Metrics**: Training loss, validation F1, precision, recall (per fold and averaged)
- **Config**: Full Hydra config for reproducibility
- **Tags**: Experiment name, model, loss, phase
- **Test metrics**: Final test set evaluation (if train/test split enabled)

View results at: `https://wandb.ai/{entity}/{project}`

## Reproducibility

- **Random seeds**: Set via `experiment.seed` (default: 42)
- **Config versioning**: Resolved configs saved with each run
- **Deterministic operations**: PyTorch deterministic mode enabled
- **Fixed splits**: Cross-validation folds are deterministic given seed
- **Train/test split**: Deterministic given seed

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
uv run python scripts/run_experiment.py data.batch_size=16
```

### Slow Data Loading

Increase number of workers:
```bash
uv run python scripts/run_experiment.py data.num_workers=8
```

### W&B Not Logging

Check that W&B is enabled and you're logged in:
```bash
wandb status
```

Disable W&B if not needed:
```bash
uv run python scripts/run_experiment.py logging.wandb.enabled=false
```

### MPS (Apple Silicon) Issues

MPS doesn't support `pin_memory`, but this is automatically handled. If you encounter issues, ensure you're using a recent PyTorch version.

## Citation

If you use this codebase, please cite:

```bibtex
@software{skinbit2024,
  title={Skinbit: Instance-Level Image Matching for Dermoscopic Images},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/skinbit}
}
```

## License

[Your License Here]

## Contact

[Your Contact Information]
