# Skinbit: Instance-Level Image Matching for Dermoscopic Images

A reproducible ML experiment repository for grouping dermoscopic images belonging to the same lesion using deep learning embeddings and metric learning.

## Problem Statement

**Input**: Folder of dermoscopic images + CSV metadata  
**Task**: Group images belonging to the same lesion (each lesion has 2-4 images)  
**Evaluation**: Pairwise F1 score (same lesion vs different lesion)

## Features

- ✅ **Lesion-based cross-validation**: All splits done by `lesion_id`, preventing data leakage
- ✅ **5-fold cross-validation** with optional stratification and train/test split
- ✅ **Baseline embeddings**: ResNet, ViT, DINOv2 with frozen pretrained weights
- ✅ **Metric learning**: Fine-tuning with contrastive, triplet, and InfoNCE losses
- ✅ **Frozen backbone + projection head**: Default for ViT and DINOv2 in Phase 2
- ✅ **Evaluation**: Cosine similarity thresholding + DBSCAN clustering
- ✅ **Final model training**: Trains on all training data using average hyperparameters from CV
- ✅ **Experiment tracking**: Weights & Biases integration
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
└── experiments/         # Experiment outputs
```

## Configuration

All settings are managed through Hydra configs. Override any config value from the command line:

```bash
# Change model, loss, phase
uv run python scripts/run_experiment.py model=vit loss=triplet experiment.phase=2

# Run specific fold
uv run python scripts/run_experiment.py experiment.fold=0

# Enable train/test split
uv run python scripts/run_experiment.py data.use_train_test_split=true

# Development mode (30 lesions, 1 epoch, fold 0 only)
uv run python scripts/run_experiment.py data.dev_mode=true
```

## Running Experiments

**Important**: Run all commands from the project root directory.

### Phase 1: Baseline Embedding Extraction

Extract embeddings from frozen pretrained models:

```bash
# ResNet
uv run python scripts/run_experiment.py model=resnet experiment.phase=1 experiment.name=baseline_resnet

# Vision Transformer
uv run python scripts/run_experiment.py model=vit experiment.phase=1 experiment.name=baseline_vit

# DINOv2
uv run python scripts/run_experiment.py model=dinov2 experiment.phase=1 experiment.name=baseline_dinov2
```

### Phase 2: Metric Learning Fine-Tuning

Fine-tune models using metric learning losses (ViT and DINOv2 default to frozen backbone + projection head):

```bash
# Contrastive Loss
uv run python scripts/run_experiment.py model=resnet loss=contrastive experiment.phase=2 experiment.name=finetune_resnet_contrastive

# Triplet Loss
uv run python scripts/run_experiment.py model=resnet loss=triplet experiment.phase=2 experiment.name=finetune_resnet_triplet

# InfoNCE Loss
uv run python scripts/run_experiment.py model=resnet loss=infonce experiment.phase=2 experiment.name=finetune_resnet_infonce
```

### Running All Experiments

Run all Phase 1 and Phase 2 experiments sequentially (12 total):

```bash
uv run python scripts/run_all_experiments.py
```

## Evaluation Protocol

### Lesion-Based Splitting

**Critical**: All splits are done by `lesion_id`, not by image. This ensures no data leakage and realistic evaluation.

### Train/Test Split (Optional)

If enabled (`data.use_train_test_split=true`):

1. **Initial split**: 80-20 stratified split of lesions into training and test sets
2. **Cross-validation**: 5-fold CV performed on training set only
3. **Hyperparameter selection**: Average best hyperparameters computed across CV folds
4. **Final model training**: Train model on all training data (Phase 2 only)
5. **Test evaluation**: Evaluate final model on test set using average hyperparameters

### Metrics

- **Cosine similarity thresholding**: Pairwise F1 with optimal threshold (0.5-0.99 range)
- **DBSCAN clustering**: Pairwise F1 with optimal `eps` (0.1-0.9 range)
- **Pairwise F1**: Primary metric computed over all image pairs (same lesion = positive, different lesion = negative)

Results are saved to CSV files (transposed format) per fold and overall.

## Data Format

### CSV Structure

The `data/data.csv` file should have:
- `lesion_id`: Unique identifier for each lesion
- `image_id`: Unique identifier for each image (without .jpg extension)

Optional columns for stratification (e.g., `anatom_site_general`).

### Image Files

Images should be stored in `data/images/` as `{image_id}.jpg` (e.g., `ISIC_0028791.jpg`).

## Utilities

### Cluster Images

Cluster images using a trained model:

```bash
uv run scripts/cluster_images.py experiments/finetune_resnet_contrastive data/ --fold 0
```

### Visualize Misclassifications

Generate side-by-side comparisons of misclassified pairs:

```bash
uv run python scripts/visualize_misclassifications.py experiments/finetune_resnet_contrastive/test_misclassifications_cosine.json
```

## Experiment Outputs

Each experiment saves:
- `resolved_config.yaml`: Full configuration used for the experiment
- `results.csv`: Overall results (transposed format)
- `fold_{i}/results.csv`: Per-fold results (transposed format)
- `fold_{i}/best.pt`: Best model checkpoint (includes hyperparameters)
- `fold_{i}/clustered_images.csv`: Cluster assignments
- `test_misclassifications_*.json`: Test set misclassifications (if enabled)

## Reproducibility

- Random seeds controlled via `experiment.seed` (default: 42)
- Resolved configs saved with each run
- PyTorch deterministic mode enabled
- All splits are deterministic given seed

## Troubleshooting

**CUDA Out of Memory**: Reduce batch size: `data.batch_size=16`  
**Slow Data Loading**: Increase workers: `data.num_workers=8`  
**W&B Not Logging**: Check `wandb status` or disable with `logging.wandb.enabled=false`
