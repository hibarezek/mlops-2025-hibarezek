"""CLI entry point for training pipeline."""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml_project.pipelines.pipeline import MLPipeline


def main():
    """Run training pipeline."""
    parser = argparse.ArgumentParser(description="Train ML pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--train-csv", type=str, default="src/ml_project/data/train.csv", help="Path to training CSV")
    parser.add_argument("--test-csv", type=str, default="src/ml_project/data/test.csv", help="Path to test CSV")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    try:
        pipeline = MLPipeline(config_path=args.config)
        results = pipeline.train_pipeline(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            val_size=args.val_size,
            seed=args.seed,
        )
        print(f"\n✓ Training completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
