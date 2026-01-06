FROM python:3.11-slim

WORKDIR /app

# Install uv (PEP621 project manager)
RUN pip install --no-cache-dir uv

# Copy locking & project metadata first for better layer caching
COPY pyproject.toml uv.lock README.md ./

# Sync dependencies from uv.lock (locked install)
RUN uv sync --locked --no-install-project

# Copy package sources and perform non-editable install without dev deps
COPY src ./src
RUN uv sync --no-dev --no-editable

# Copy configs, scripts and entrypoint
COPY configs/ ./configs
COPY scripts/ ./scripts
COPY docker-compose.yml ./
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]
