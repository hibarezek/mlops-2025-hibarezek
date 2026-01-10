"""
SageMaker Batch Inference Pipeline for NYC Taxi Trip Duration Dataset
Uses SageMaker SDK 3.1.0+

This pipeline includes:
1. Preprocessing step - cleans and validates raw inference data
2. Feature engineering step - creates features using trained encoders/scalers
3. Batch inference step - loads model from registry and generates predictions
"""

import argparse

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep

# Configuration
INSTANCE_COUNT = 1
FRAMEWORK_VERSION = "1.2-1"


def _get_model_data_from_package(model_package_arn: str) -> str:
    """
    Resolve model.tar.gz S3 path from a model package ARN in the registry.
    
    Args:
        model_package_arn: ARN of the model package in SageMaker Model Registry
        
    Returns:
        S3 URI of the model.tar.gz file
    """
    sm = boto3.client("sagemaker")
    
    resp = sm.describe_model_package(ModelPackageName=model_package_arn)
    # Most common case: single container
    model_data_s3 = resp["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    return model_data_s3


def create_batch_inference_pipeline(
    pipeline_name: str,
    s3_bucket: str,
    model_package_arn: str,
    role_arn: str = None,
) -> Pipeline:
    """
    Create a SageMaker batch inference pipeline.

    The pipeline performs:
      1. Preprocess raw inference data (scripts/preprocess.py)
      2. Featurize preprocessed data (scripts/feature_engineer.py)
      3. Run batch inference via a ProcessingStep (scripts/inference.py),
         using the model artifacts from a Model Package in the registry.

    Args:
        pipeline_name: Name of the pipeline
        s3_bucket: S3 bucket for input/output data (e.g., s3://my-bucket)
        model_package_arn: ARN of the approved model package in the registry
        role_arn: IAM role ARN for SageMaker execution

    Returns:
        Pipeline object
    """

    if s3_bucket is None:
        raise ValueError("s3_bucket must be provided")

    session = sagemaker.Session()

    if role_arn is None:
        role_arn = sagemaker.get_execution_role()

    print(f"Using S3 bucket: {s3_bucket}")
    print(f"Using IAM role: {role_arn}")
    print(f"Using model package ARN: {model_package_arn}")

    # ----------------- Resolve model.tar.gz from the registry -----------------
    model_data_s3 = _get_model_data_from_package(model_package_arn)
    print(f"Resolved model artifacts at: {model_data_s3}")

    # ----------------- Shared SKLearnProcessor -----------------
    sklearn_processor = SKLearnProcessor(
        framework_version=FRAMEWORK_VERSION,
        role=role_arn,
        instance_count=INSTANCE_COUNT,
        instance_type="ml.t3.medium",
        sagemaker_session=session,
    )

    # ----------------- Step 1: Preprocess -----------------
    preprocess_step = ProcessingStep(
        name="Preprocess",
        processor=sklearn_processor,
        code="scripts/preprocess.py",
        inputs=[
            ProcessingInput(
                input_name="raw_data",
                source=f"{s3_bucket}/input/",
                destination="/opt/ml/processing/input",
            ),
            ProcessingInput(
                input_name="config",
                source="configs/config.yaml",
                destination="/opt/ml/processing/config",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="preprocessed",
                source="/opt/ml/processing/output",
            ),
        ],
        job_arguments=[
            "--input",
            "/opt/ml/processing/input/test.csv",
            "--output",
            "/opt/ml/processing/output/test_preprocessed.csv",
            "--mode",
            "inference",
            "--config",
            "/opt/ml/processing/config/config.yaml",
        ],
    )

    preprocessed_s3 = preprocess_step.properties.ProcessingOutputConfig.Outputs[
        "preprocessed"
    ].S3Output.S3Uri

    # ----------------- Step 2: Feature Engineering -----------------
    featurize_step = ProcessingStep(
        name="FeatureEngineering",
        processor=sklearn_processor,
        code="scripts/feature_engineer.py",
        inputs=[
            ProcessingInput(
                input_name="preprocessed",
                source=preprocessed_s3,
                destination="/opt/ml/processing/input",
            ),
            ProcessingInput(
                input_name="config",
                source="configs/config.yaml",
                destination="/opt/ml/processing/config",
            ),
            ProcessingInput(
                input_name="artifacts",
                source=f"{s3_bucket}/artifacts/",
                destination="/opt/ml/processing/artifacts",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="engineered",
                source="/opt/ml/processing/output",
            ),
        ],
        job_arguments=[
            "--input",
            "/opt/ml/processing/input/test_preprocessed.csv",
            "--output",
            "/opt/ml/processing/output/test_engineered.csv",
            "--mode",
            "inference",
            "--config",
            "/opt/ml/processing/config/config.yaml",
        ],
    )

    engineered_s3 = featurize_step.properties.ProcessingOutputConfig.Outputs[
        "engineered"
    ].S3Output.S3Uri

    # ----------------- Step 3: Batch Inference (ProcessingStep) -----------------
    inference_step = ProcessingStep(
        name="BatchInference",
        processor=sklearn_processor,
        code="scripts/inference.py",
        inputs=[
            # Engineered test data
            ProcessingInput(
                input_name="test_data",
                source=engineered_s3,
                destination="/opt/ml/processing/input",
            ),
            # Model artifacts from the registry
            ProcessingInput(
                input_name="model",
                source=model_data_s3,
                destination="/opt/ml/processing/model",
            ),
            # Config file
            ProcessingInput(
                input_name="config",
                source="configs/config.yaml",
                destination="/opt/ml/processing/config",
            ),
            # Artifacts (encoder, scaler)
            ProcessingInput(
                input_name="artifacts",
                source=f"{s3_bucket}/artifacts/",
                destination="/opt/ml/processing/artifacts",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="predictions",
                source="/opt/ml/processing/output",
                destination=f"{s3_bucket}/output/predictions/",
            ),
        ],
        job_arguments=[
            "--input",
            "/opt/ml/processing/input/test_engineered.csv",
            "--model",
            "/opt/ml/processing/model/best_model.pkl",
            "--output",
            "/opt/ml/processing/output/predictions.csv",
            "--config",
            "/opt/ml/processing/config/config.yaml",
        ],
    )

    # ----------------- Pipeline definition -----------------
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[preprocess_step, featurize_step, inference_step],
        sagemaker_session=session,
    )

    return pipeline


def main():
    """Execute the pipeline creation and submission."""

    parser = argparse.ArgumentParser(
        description="SageMaker batch inference pipeline for NYC Taxi Trip Duration dataset"
    )
    parser.add_argument(
        "--input-bucket",
        type=str,
        required=True,
        help="S3 bucket containing input data and artifacts (e.g., s3://my-bucket)",
    )
    parser.add_argument(
        "--model-arn",
        type=str,
        required=True,
        help="Deployed Model Package ARN from SageMaker Model Registry",
    )
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default="nyc-taxi-batch-inference-pipeline",
        help="Name of the pipeline (default: nyc-taxi-batch-inference-pipeline)",
    )
    parser.add_argument(
        "--role-arn",
        type=str,
        default=None,
        help="IAM role ARN for SageMaker execution (optional, uses default if not provided)",
    )
    args = parser.parse_args()

    # Create the pipeline
    pipeline = create_batch_inference_pipeline(
        pipeline_name=args.pipeline_name,
        s3_bucket=args.input_bucket,
        model_package_arn=args.model_arn,
        role_arn=args.role_arn,
    )

    # Define pipeline
    pipeline_definition = pipeline.definition()
    print("\n" + "=" * 80)
    print("Pipeline definition:")
    print("=" * 80)
    print(pipeline_definition)

    # Create or update the pipeline
    print("\n" + "=" * 80)
    print("Upserting pipeline...")
    print("=" * 80)
    pipeline.upsert(role_arn=args.role_arn or sagemaker.get_execution_role())

    # Submit the pipeline for execution
    print("\n" + "=" * 80)
    print("Starting pipeline execution...")
    print("=" * 80)
    execution = pipeline.start()

    print(f"\n✅ Pipeline '{pipeline.name}' submitted for execution")
    print(f"Execution ARN: {execution.arn}")

    # Wait for execution to complete
    print("\nWaiting for pipeline execution to complete...")
    execution.wait()

    print(f"\n✅ Pipeline execution status: {execution.describe()['PipelineExecutionStatus']}")


if __name__ == "__main__":
    main()
