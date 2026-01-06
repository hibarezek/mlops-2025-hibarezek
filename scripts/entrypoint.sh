#!/bin/sh
set -e

# entrypoint wrapper that maps simple commands to uv runs
# Usage: `docker-compose run app train` or `docker-compose run app inference`

if [ "$1" = "train" ]; then
  shift
  echo "Starting training pipeline..."
  uv run train "$@"
  exit $?
fi

if [ "$1" = "inference" ]; then
  shift
  echo "Starting inference pipeline..."
  uv run inference "$@"
  exit $?
fi

# fallback: run provided command
exec "$@"
