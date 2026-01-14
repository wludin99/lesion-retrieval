# Skinbit: Instance-Level Image Matching for Dermoscopic Images

A complete, reproducible machine learning experiment repository for grouping dermoscopic images belonging to the same lesion using deep learning embeddings and metric learning.

## Problem Statement

**Input**: Folder of dermoscopic images + CSV metadata  
**Task**: Group images belonging to the same lesion (each lesion has 2-4 images)  
**Evaluation**: Pairwise F1 score (same lesion vs different lesion)

## Features

- ✅ **Lesion-based cross-validation**: All train/val splits done by `lesion_id`, preventing data leakage
- ✅ **5-fold cross-validation** with optional stratification
- ✅ **Baseline embeddings**: ResNet, ViT, DINOv2 with frozen ImageNet/self-supervised weights
- ✅ **Metric learning**: Fine-tuning with contrastive, triplet, and InfoNCE losses
- ✅ **Comprehensive evaluation**: Cosine similarity thresholding + DBSCAN clustering
- ✅ **Experiment tracking**: Weights & Biases integration
- ✅ **Configuration management**: Hydra for all experiment settings
- ✅ **Reproducibility**: Random seed control and config versioning

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
│   └── run_experiment.py  # Main entry point
├── experiments/         # Experiment outputs
├── pyproject.toml       # Project dependencies
└── README.md
```

## Configuration with Hydra

All experiment settings are managed through Hydra configs. The main config file is `configs/config.yaml`, which composes configs from subdirectories.

### Key Configuration Groups

- **`data/`**: Data loading, preprocessing, cross-validation settings
- **`model/`**: Model architecture (resnet, vit, dinov2)
- **`loss/`**: Loss function (none, contrastive, triplet, infonce)
- **`training/`**: Training hyperparameters
- **`evaluation/`**: Evaluation metrics and thresholds
- **`logging/`**: W&B and logging settings

### Overriding Configs

You can override any config value from the command line:

```bash
# Change model
python scripts/run_experiment.py model=vit

# Change loss
python scripts/run_experiment.py loss=triplet

# Run specific fold
python scripts/run_experiment.py experiment.fold=0

# Change experiment phase
python scripts/run_experiment.py experiment.phase=2

# Combine multiple overrides
python scripts/run_experiment.py model=dinov2 experiment.phase=1 experiment.fold=0
```

## Running Experiments

**Important**: Run all commands from the project root directory (`/Users/williamludington/Projects/skinbit`).

### Phase 1: Baseline Embedding Extraction

Extract embeddings from frozen pretrained models and evaluate with cosine similarity and DBSCAN.

#### ResNet Baseline

```bash
python scripts/run_experiment.py \
    model=resnet \
    experiment.phase=1 \
    experiment.name=baseline_resnet
```

#### Vision Transformer Baseline

```bash
python scripts/run_experiment.py \
    model=vit \
    experiment.phase=1 \
    experiment.name=baseline_vit
```

#### DINOv2 Baseline

```bash
python scripts/run_experiment.py \
    model=dinov2 \
    experiment.phase=1 \
    experiment.name=baseline_dinov2
```

#### Run All Folds

By default, all 5 folds are run. Results are averaged across folds and logged to W&B.

```bash
python scripts/run_experiment.py \
    model=resnet \
    experiment.phase=1 \
    experiment.fold=null  # null = all folds
```

### Phase 2: Metric Learning Fine-Tuning

Fine-tune the best backbone from Phase 1 using metric learning losses.

#### Contrastive Loss

```bash
python scripts/run_experiment.py \
    model=resnet \
    loss=contrastive \
    experiment.phase=2 \
    experiment.name=finetune_resnet_contrastive
```

#### Triplet Loss

```bash
python scripts/run_experiment.py \
    model=resnet \
    loss=triplet \
    experiment.phase=2 \
    experiment.name=finetune_resnet_triplet
```

#### InfoNCE Loss

```bash
python scripts/run_experiment.py \
    model=resnet \
    loss=infonce \
    experiment.phase=2 \
    experiment.name=finetune_resnet_infonce
```

## Cross-Validation Protocol

### Lesion-Based Splitting

**Critical**: All splits are done by `lesion_id`, not by image. This ensures:
- No data leakage: images from the same lesion never appear in both train and val
- Realistic evaluation: model must generalize to unseen lesions
- Proper grouping: all images of a lesion are kept together

### Stratification

You can stratify folds by metadata columns (e.g., anatomical site):

```bash
python scripts/run_experiment.py \
    data.stratify_by=anatom_site_general
```

### Fold Structure

- **5-fold cross-validation** by default
- Each fold splits lesions (not images) into train/val
- Results are computed per fold and averaged
- All metrics logged to W&B with fold-specific tags

## Evaluation Metrics

### Cosine Similarity Thresholding

- Computes pairwise cosine similarities between all embeddings
- Varies threshold from 0.5 to 0.99 (50 values by default)
- Reports best F1, precision, recall, accuracy

### DBSCAN Clustering

- Clusters embeddings using DBSCAN
- Varies `eps` parameter from 0.1 to 0.9 (20 values by default)
- Reports best F1, precision, recall, accuracy

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

## Experiment Outputs

All experiments save outputs to `experiments/{experiment_name}/`:

- `resolved_config.yaml`: Full resolved Hydra config for reproducibility
- `fold_{i}/`: Per-fold outputs (for Phase 2)
  - `best.pt`: Best model checkpoint
  - `checkpoint_epoch_{n}.pt`: Periodic checkpoints

## Weights & Biases Integration

All experiments are automatically logged to W&B:

- **Metrics**: Training loss, validation F1, precision, recall
- **Config**: Full Hydra config for reproducibility
- **Artifacts**: Model checkpoints (optional)
- **Tags**: Experiment name, model, loss, phase

View results at: `https://wandb.ai/{entity}/{project}`

## Reproducibility

- **Random seeds**: Set via `experiment.seed` (default: 42)
- **Config versioning**: Resolved configs saved with each run
- **Deterministic operations**: PyTorch deterministic mode enabled
- **Fixed splits**: Cross-validation folds are deterministic given seed

## Development

### Type Hints

All code uses Python type hints for clarity and IDE support.

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to public functions/classes
- Keep functions focused and clear

### Testing

```bash
# Run tests (when implemented)
uv run pytest
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python scripts/run_experiment.py data.batch_size=16
```

### Slow Data Loading

Increase number of workers:
```bash
python scripts/run_experiment.py data.num_workers=8
```

### W&B Not Logging

Check that W&B is enabled and you're logged in:
```bash
wandb status
```

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
