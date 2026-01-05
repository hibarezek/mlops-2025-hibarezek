"""CLI entry point for inference pipeline."""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml_project.pipelines.pipeline import MLPipeline


def main():
    """Run inference pipeline."""
    parser = argparse.ArgumentParser(description="Run inference pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--test-csv", type=str, default="src/ml_project/data/test.csv", help="Path to test CSV")
    parser.add_argument("--model", type=str, default=None, help="Path to model (default: artifacts/best_model.pkl)")

    args = parser.parse_args()

    try:
        pipeline = MLPipeline(config_path=args.config)
        results = pipeline.inference_pipeline(
            test_csv=args.test_csv,
            model_path=args.model,
        )
        print(f"\n✓ Inference completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Inference failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
