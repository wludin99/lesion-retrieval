"""Run all experiments sequentially.

This script runs:
- Phase 1: 3 baseline models (ResNet, ViT, DINOv2)
- Phase 2: All models × 3 losses (Contrastive, Triplet, InfoNCE)

Total: 3 + 9 = 12 experiments

Note: After Phase 1, you can manually select the best model and modify
this script to only run Phase 2 with that model (3 experiments instead of 9).
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run all experiments sequentially."""
    
    # Phase 1: Baseline embeddings
    phase1_experiments = [
        ("resnet", "baseline_resnet"),
        ("vit", "baseline_vit"),
        ("dinov2", "baseline_dinov2"),
    ]
    
    # Phase 2: Metric learning (will use best model from Phase 1)
    # For now, we'll run all models with all losses
    # User can manually select best model after Phase 1
    phase2_experiments = [
        ("resnet", "contrastive", "finetune_resnet_contrastive"),
        ("resnet", "triplet", "finetune_resnet_triplet"),
        ("resnet", "infonce", "finetune_resnet_infonce"),
        ("vit", "contrastive", "finetune_vit_contrastive"),
        ("vit", "triplet", "finetune_vit_triplet"),
        ("vit", "infonce", "finetune_vit_infonce"),
        ("dinov2", "contrastive", "finetune_dinov2_contrastive"),
        ("dinov2", "triplet", "finetune_dinov2_triplet"),
        ("dinov2", "infonce", "finetune_dinov2_infonce"),
    ]
    
    print("="*60)
    print("Running All Experiments")
    print("="*60)
    print(f"Phase 1: {len(phase1_experiments)} experiments")
    print(f"Phase 2: {len(phase2_experiments)} experiments")
    print(f"Total: {len(phase1_experiments) + len(phase2_experiments)} experiments")
    print("="*60 + "\n")
    
    # Run Phase 1
    print("Phase 1: Baseline Embeddings")
    print("-"*60)
    for model, exp_name in phase1_experiments:
        print(f"\nRunning: {exp_name} ({model})")
        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            f"model={model}",
            f"experiment.phase=1",
            f"experiment.name={exp_name}",
        ]
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        if result.returncode != 0:
            print(f"ERROR: {exp_name} failed!")
            return result.returncode
    
    # Run Phase 2
    print("\n" + "="*60)
    print("Phase 2: Metric Learning Fine-tuning")
    print("-"*60)
    for model, loss, exp_name in phase2_experiments:
        print(f"\nRunning: {exp_name} ({model} + {loss})")
        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            f"model={model}",
            f"loss={loss}",
            f"experiment.phase=2",
            f"experiment.name={exp_name}",
        ]
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        if result.returncode != 0:
            print(f"ERROR: {exp_name} failed!")
            return result.returncode
    
    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
