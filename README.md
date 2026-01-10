# NYC Taxi Trip Duration Prediction - MLOps Pipeline

An end-to-end MLOps pipeline for predicting NYC taxi trip durations using scikit-learn models with automated training, evaluation, and batch inference capabilities. Supports both local execution and cloud deployment via AWS SageMaker.

## Project Overview

This project implements a complete machine learning pipeline that:
- Preprocesses raw taxi data with validation and outlier removal
- Engineers features using distance calculations, temporal features, and categorical encoding
- Trains multiple models (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting)
- Evaluates models using MAE, RMSE, and R² metrics
- Registers the best model in SageMaker Model Registry
- Performs batch inference on new data using the trained model
- Supports local execution and AWS SageMaker orchestration
- Containerizes workflows using Docker for reproducibility

## Project Structure

```
src/ml_project/
├── data/              - Data loading (Base_DataLoader.py, DataLoader.py)
├── preprocess/        - Data preprocessing (Base_Preprocessor.py, preprocessor.py)
├── features/          - Feature engineering (Base_FeatureEngineer.py, FeatureEngineer.py)
├── train/             - Model training (Base_Trainer.py, trainer.py)
├── inference/         - Batch inference (Base_Inference.py, inference.py)
├── pipelines/         - Pipeline orchestration (pipeline.py)
└── utils/             - Utilities (helpers.py)

scripts/
├── train_cli.py       - Training entry point
├── inference_cli.py   - Inference entry point
├── preprocess.py      - SageMaker preprocessing
├── feature_engineer.py - SageMaker feature engineering
└── entrypoint.sh      - Docker entrypoint

configs/
└── config.yaml        - Configuration

artifacts/
├── best_model.pkl           - Trained model
├── model_metadata_*.json    - Model metadata
└── training_results_*.json  - Training metrics

outputs/
├── preprocessed/      - Preprocessed data
├── engineered/        - Feature-engineered data
├── predictions/       - Inference outputs
└── raw/               - Raw data copies

Root files:
├── pyproject.toml                      - Project metadata and dependencies
├── Dockerfile                          - Container definition
├── docker-compose.yml                  - Docker Compose services
├── run_training_pipeline.py            - SageMaker training pipeline
└── run_batch_inference_pipeline.py     - SageMaker inference pipeline
```

## Key Components

### Data Pipeline

- Preprocessing: Removes invalid coordinates, handles missing values, removes trip duration outliers (1st-99th percentile)
- Feature Engineering:
  - Distance: Haversine formula for great-circle distance
  - Datetime: Hour, day, weekday, month from pickup time
  - Categorical: One-hot encoding for vendor_id and store_and_fwd_flag
  - Scaling: StandardScaler for numeric features


Five regression models are trained and evaluated:
1. Linear Regression: Baseline linear model
2. Ridge Regression: L2 regularized linear model (α=1.0)
3. Lasso Regression: L1 regularized linear model (α=1.0, max_iter=5000)
4. Random Forest: 100 trees, max_depth=15 (best model in most runs)
5. Gradient Boosting0 trees, max_depth=15 (best model in most runs)
5. **Gradient Boosting**: 100 estimators, max_depth=5, learning_rate=0.1

### Evaluation Metrics

- MAE (Mean Absolute Error): Primary metric for model selection
- RMSE (Root Mean Squared Error): Penalizes larger errors
- R² Score: Variance explained by the model

## Selected Metrics and Justification

### Primary Metric: Mean Absolute Error (MAE)

- Selected: Yes - Used as the primary metric for model selection
- Justification:
  - Directly interpretable as average prediction error in seconds/minutes
  - Robust to outliers compared to RMSE
  - Domain-relevant for taxi drivers and passengers estimating trip duration
  - Less sensitive to extreme prediction errors than squared error metrics
  
### Secondary Metrics

- RMSE: Provides insight into penalizing larger errors
- R² Score: Indicates overall model fit and variance explained

### Training Results (Best Model: Random Forest)

```
Random Forest MAE: 197.32 seconds (3.3 minutes)
Random Forest RMSE: 284.47 seconds
Random Forest R²: 0.7546 (explains 75.46% of variance)
```

## Model Choices and Performance

### Why Random Forest?

The Random Forest model was selected as the best performer based on MAE (lowest error):

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Random Forest** | **197.32** | **284.47** | **0.7546** |
| Gradient Boosting | 281.06 | 382.11 | 0.5572 |
| Lasso | 289.66 | 389.44 | 0.5400 |
| Ridge | 289.68 | 389.68 | 0.5395 |
| Linear | 289.68 | 389.69 | 0.5395 |

**Advantages of Random Forest**:
- ✅ ~31% lower MAE than linear models (197 vs 290 seconds)
- ✅ Captures non-linear relationships in spatial/temporal features
- ✅ Handles feature interactions automatically
- ✅ Robust to missing values and outliers
- ✅ Provides feature importance scores
- ✅ Parallelizable training (n_jobs=-1)

**Model Hyperparameters**:
- `n_estimators=100`: 100 decision trees
- `max_depth=15`: Controls model complexity and overfitting
- `random_state=42`: Reproducibility
- `n_jobs=-1`: Use all CPU cores for parallel training

## How to Run Locally

### Prerequisites

- Python 3.12+
- `uv` package manager

### Installation

```bash
# Clone the repository
cd mlops-2025-hibarezek

# Install dependencies with uv
uv sync
```

### Training

```bash
# Run training pipeline with default config
python -m scripts.train_cli

# Or with custom paths
python -m scripts.train_cli \
  --config configs/config.yaml \
  --train-csv src/ml_project/data/train.csv \
  --test-csv src/ml_project/data/test.csv \
  --val-size 0.2 \
  --seed 42
```

**Output**:
- `artifacts/best_model.pkl` - Trained Random Forest model
- `artifacts/model_metadata_*.json` - Model metadata with timestamp
- `artifacts/training_results_*.json` - Training metrics for all models
- `outputs/preprocessed/*.csv` - Preprocessed data
- `outputs/engineered/*.csv` - Feature-engineered data

### Inference

```bash
# Run inference on test data
python -m scripts.inference_cli

# Or with custom model path
python -m scripts.inference_cli \
  --config configs/config.yaml \
  --test-csv src/ml_project/data/test.csv \
  --model artifacts/best_model.pkl
```

**Output**:
- `outputs/predictions/*.csv` - Predictions with timestamps

## How to Run with Docker

### Prerequisites

- Docker and Docker Compose installed
- No Python installation required (containerized)

### Building the Image

```bash
# Build the ML application image
docker-compose build app

# Verify build (optional)
docker images | grep ml-project
```

### Training with Docker

```bash
# Run training pipeline
docker-compose run app train

# With custom arguments
docker-compose run app train \
  --train-csv src/ml_project/data/train.csv \
  --test-csv src/ml_project/data/test.csv
```

### Inference with Docker

```bash
# Run inference
docker-compose run app inference

# With custom model
docker-compose run app inference --model artifacts/best_model.pkl
```

### Viewing Artifacts

Artifacts are mounted as volumes, so they persist on your host machine:
- `./artifacts/` - Training artifacts
- `./outputs/` - Predictions and processed data

### MLFlow Integration

An optional MLFlow server for experiment tracking is available in docker-compose.yml:

```bash
docker-compose up mlflow
# Access at http://localhost:5000
```

## How to Run SageMaker Pipeline

### Prerequisites
- AWS Account with SageMaker permissions
- S3 bucket for data and artifacts
- IAM role with SageMaker permissions
- AWS credentials configured (`aws configure`)

### Setup

```bash
# Install AWS dependencies
uv pip install sagemaker boto3

# Configure AWS credentials
aws configure
```

### Training Pipeline

```bash
# Create and execute training pipeline
python run_training_pipeline.py \
  --input-bucket s3://your-bucket \
  --pipeline-name nyc-taxi-training-pipeline \
  --region us-east-1

# Or create without executing (for review)
python run_training_pipeline.py \
  --input-bucket s3://your-bucket \
  --pipeline-name nyc-taxi-training-pipeline \
  --region us-east-1 \
  --skip-execution
```

**Pipeline Steps**:
1. Preprocessing: Data validation and cleaning (ProcessingStep)
2. Feature Engineering: Feature creation and encoding (ProcessingStep)
3. Training: Model training with scikit-learn (TrainingStep)
4. Model Registration: Register best model in Model Registry (RegisterModel)

**Outputs**:
- Model artifacts in S3 (`s3://your-bucket/output/model.tar.gz`)
- Model registered in SageMaker Model Registry under `nyc-taxi-model-group`

### Batch Inference Pipeline

```bash
# Create batch inference pipeline
python run_batch_inference_pipeline.py \
  --input-bucket s3://your-bucket \
  --pipeline-name nyc-taxi-inference-pipeline \
  --model-package-arn arn:aws:sagemaker:region:account:model-package/nyc-taxi-model-group/version \
  --region us-east-1

# Or with skip-execution
python run_batch_inference_pipeline.py \
  --input-bucket s3://your-bucket \
  --pipeline-name nyc-taxi-inference-pipeline \
  --model-package-arn arn:aws:sagemaker:region:account:model-package/nyc-taxi-model-group/version \
  --region us-east-1 \
  --skip-execution
```

**Pipeline Steps**:
1. Preprocessing: Clean inference data (ProcessingStep)
2. Feature Engineering: Apply training encoders/scalers (ProcessingStep)
3. Batch Inference: Generate predictions using registered model (ProcessingStep)

**Outputs**:
- Predictions in S3 (`s3://your-bucket/output/predictions.csv`)

### Configuration

Key SageMaker configuration parameters:
- INSTANCE_COUNT: Number of instances (default: 1)
- FRAMEWORK_VERSION: scikit-learn framework version (1.2-1)
- MODEL_PACKAGE_GROUP_NAME: Registry group for models (nyc-taxi-model-group)
- S3_BUCKET: Input/output data bucket
- ROLE_ARN: IAM execution role ARN

## Dependencies

### Core Dependencies

- numpy - Numerical computing
- pandas - Data manipulation
- scikit-learn - Machine learning models
- joblib - Model serialization
- pyyaml - Configuration files

### Optional Dependencies

- sagemaker (aws extra) - AWS SageMaker integration
- boto3 (aws extra) - AWS SDK
- mlflow (mlflow extra) - Experiment tracking

## Pipeline Architecture

```
Raw Data
   ↓
[Preprocessing] - Validation, outlier removal
   ↓
[Feature Engineering] - Distance, datetime, encoding, scaling
   ↓
[Train/Test Split] - 80/20 split (or custom)
   ↓
[Model Training] - 5 models trained in parallel
   ↓
[Model Evaluation] - MAE, RMSE, R² computed
   ↓
[Model Selection] - Best model selected by MAE
   ↓
[Inference] - Predictions on new data
```
