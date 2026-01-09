"""
SageMaker Training Pipeline for NYC Taxi Trip Duration Dataset
Uses SageMaker SDK 3.1.0+

This pipeline includes:
1. Preprocessing step - cleans and validates raw data
2. Feature engineering step - creates features and applies encoding/scaling
3. Training step - trains multiple models and selects the best one
4. Model registration step - registers the best model in SageMaker Model Registry
"""

import argparse
import boto3
from botocore.exceptions import ClientError

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# Configuration
INSTANCE_COUNT = 1
FRAMEWORK_VERSION = "1.2-1"
MODEL_PACKAGE_GROUP_NAME = "nyc-taxi-model-group"


def create_training_pipeline(
    pipeline_name: str,
    s3_bucket: str,
    role_arn: str = None,
    session: sagemaker.session.Session | None = None,
):
    """
    Create a SageMaker training pipeline with preprocessing, feature engineering,
    training, and model registration steps.

    Args:
        pipeline_name: Name of the pipeline
        s3_bucket: S3 bucket for input/output data (e.g., s3://my-bucket)
        role_arn: IAM role ARN for SageMaker execution

    Returns:
        Pipeline object
    """

    # Initialize SageMaker session
    if session is None:
        session = sagemaker.Session()

    # Validate S3 bucket
    if s3_bucket is None:
        raise ValueError("s3_bucket must be provided")

    # Use provided role or get default execution role
    if role_arn is None:
        role_arn = sagemaker.get_execution_role()

    print(f"Using S3 bucket: {s3_bucket}")
    print(f"Using IAM role: {role_arn}")

    # Create SKLearnProcessor for preprocessing and feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version=FRAMEWORK_VERSION,
        role=role_arn,
        instance_count=INSTANCE_COUNT,
        instance_type="ml.t3.medium",
        sagemaker_session=session,
    )

    # ---------- Step 1: Preprocessing ----------
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
            "/opt/ml/processing/input/train.csv",
            "--output",
            "/opt/ml/processing/output/train_preprocessed.csv",
            "--mode",
            "train",
            "--config",
            "/opt/ml/processing/config/config.yaml",
        ],
    )

    preprocessed_s3 = preprocess_step.properties.ProcessingOutputConfig.Outputs[
        "preprocessed"
    ].S3Output.S3Uri

    # ---------- Step 2: Feature Engineering ----------
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
        ],
        outputs=[
            ProcessingOutput(
                output_name="engineered",
                source="/opt/ml/processing/output",
            ),
            ProcessingOutput(
                output_name="artifacts",
                source="/opt/ml/processing/artifacts",
            ),
        ],
        job_arguments=[
            "--input",
            "/opt/ml/processing/input/train_preprocessed.csv",
            "--output",
            "/opt/ml/processing/output/train_engineered.csv",
            "--mode",
            "train",
            "--config",
            "/opt/ml/processing/config/config.yaml",
        ],
    )

    engineered_s3 = featurize_step.properties.ProcessingOutputConfig.Outputs[
        "engineered"
    ].S3Output.S3Uri

    # ---------- Step 3: Training ----------
    estimator = SKLearn(
        entry_point="scripts/train.py",
        role=role_arn,
        instance_type="ml.m5.large",
        instance_count=INSTANCE_COUNT,
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=session,
        hyperparameters={
            "config": "configs/config.yaml",
        },
        dependencies=["configs/config.yaml"],
    )

    train_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=engineered_s3,
                content_type="text/csv",
            ),
        },
    )

    # ---------- Step 4: Register Model ----------
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="PendingManualApproval",
    )

    # ---------- Create Pipeline ----------
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[preprocess_step, featurize_step, train_step, register_step],
        sagemaker_session=session,
    )

    return pipeline


def main():
    """Execute the pipeline creation and submission."""

    parser = argparse.ArgumentParser(
        description="SageMaker training pipeline for NYC Taxi Trip Duration dataset"
    )
    parser.add_argument(
        "--input-bucket",
        type=str,
        required=True,
        help="S3 bucket containing input data (e.g., s3://my-bucket)",
    )
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default="nyc-taxi-training-pipeline",
        help="Name of the pipeline (default: nyc-taxi-training-pipeline)",
    )
    parser.add_argument(
        "--role-arn",
        type=str,
        default=None,
        help="IAM role ARN for SageMaker execution (optional, uses default if not provided)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region for SageMaker resources (default: us-east-1)",
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Create/update pipeline but don't execute it",
    )
    args = parser.parse_args()

    # Build explicit sessions for the requested region and show caller identity
    boto_sess = boto3.Session(region_name=args.region)
    sts = boto_sess.client("sts")
    try:
        ident = sts.get_caller_identity()
        print(f"AWS Caller Identity: Account={ident['Account']} Arn={ident['Arn']}")
    except Exception as e:
        print(f"⚠️  Could not retrieve caller identity: {e}")

    sm_session = sagemaker.Session(boto_session=boto_sess)

    # Create the pipeline
    pipeline = create_training_pipeline(
        pipeline_name=args.pipeline_name,
        s3_bucket=args.input_bucket,
        role_arn=args.role_arn,
        session=sm_session,
    )

    # Define pipeline
    pipeline_definition = pipeline.definition()
    print("\n" + "=" * 80)
    print("Pipeline definition:")
    print("=" * 80)
    print(pipeline_definition)

    # Create or update the pipeline via boto3 to avoid SDK hang and show rich errors
    print("\n" + "=" * 80)
    print("Upserting pipeline (via boto3)...")
    print("=" * 80)

    sm_client = boto_sess.client("sagemaker")
    definition = pipeline.definition()

    print(f"Region: {args.region}")
    print(f"Pipeline Name: {pipeline.name}")
    print(f"Execution Role: {args.role_arn or 'DEFAULT(sagemaker.get_execution_role())'}")

    # Decide create vs update
    exists = False
    try:
        sm_client.describe_pipeline(PipelineName=pipeline.name)
        exists = True
        print("Pipeline already exists. Updating...")
    except sm_client.exceptions.ResourceNotFound:
        print("Pipeline does not exist. Creating...")
    except ClientError as e:
        print(f"⚠️  Error checking pipeline existence: {e.response.get('Error', {})}")

    try:
        if exists:
            resp = sm_client.update_pipeline(
                PipelineName=pipeline.name,
                PipelineDefinition=definition,
                RoleArn=args.role_arn or sagemaker.get_execution_role(),
            )
        else:
            resp = sm_client.create_pipeline(
                PipelineName=pipeline.name,
                PipelineDefinition=definition,
                RoleArn=args.role_arn or sagemaker.get_execution_role(),
            )
        print("✅ Upsert complete")
        print(f"PipelineArn: {resp.get('PipelineArn')}")
    except ClientError as e:
        err = e.response.get('Error', {})
        print("❌ Boto3 error during upsert")
        print(f"Code: {err.get('Code')} Message: {err.get('Message')}")
        print(f"HTTPStatusCode: {e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')}")
        print("Suggested checks: permissions (sagemaker:CreatePipeline/UpdatePipeline), region, role trust policy")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        raise

    # If skip-execution flag is set, stop here
    if args.skip_execution:
        print("\n" + "=" * 80)
        print("Pipeline created successfully!")
        print("=" * 80)
        print(f"Pipeline name: {pipeline.name}")
        print("\nTo execute the pipeline, use AWS SageMaker Studio or:")
        print(f"aws sagemaker start-pipeline-execution --pipeline-name {pipeline.name} --region us-east-1")
        return

    # Submit the pipeline for execution
    print("\n" + "=" * 80)
    print("Starting pipeline execution...")
    print("=" * 80)
    try:
        execution = pipeline.start()
        print("Pipeline execution started!")
        print(f"Execution ARN: {execution.arn}")
    except Exception as e:
        print(f"❌ Error starting pipeline execution: {e}")
        raise

    # Wait for execution to complete
    print("\nWaiting for pipeline execution to complete...")
    print("(This may take 30+ minutes. You can check progress in SageMaker Studio.)")
    try:
        execution.wait()
        status = execution.describe()["PipelineExecutionStatus"]
        print(f"\n✅ Pipeline execution completed with status: {status}")
    except Exception as e:
        print(f"⚠️  Error waiting for execution or checking status: {e}")
        print(f"Execution ARN: {execution.arn}")
        print("Check AWS SageMaker Studio for pipeline status.")


if __name__ == "__main__":
    main()
