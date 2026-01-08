FROM python:3.12-slim

WORKDIR /app

# Environment variables for optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_HTTP_TIMEOUT=300 \
    UV_NO_PROGRESS=1 \
    UV_COMPILE_BYTECODE=1

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy project metadata
COPY pyproject.toml README.md ./
COPY src ./src

# Install only core dependencies using pip (skip uv.lock to avoid extras)
# This installs: numpy, pandas, scikit-learn, joblib, pyyaml  
# Skips: sagemaker, boto3, mlflow (and their 3GB+ of torch/CUDA deps)
RUN pip install --no-cache-dir \
        "numpy>=2.3.5" \
        "pandas>=2.3.3" \
        "scikit-learn>=1.8.0" \
        "joblib>=1.4.2" \
        "pyyaml>=6.0.2" && \
    pip install --no-cache-dir -e . --no-deps

# Copy configs, scripts and entrypoint
COPY configs/ ./configs
COPY scripts/ ./scripts
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && \
    sed -i 's/\r$//' /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]
