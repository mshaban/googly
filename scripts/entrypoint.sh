#!/bin/bash
source config/local.env
# Set the environment variables add python src to path
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo $SERVE_URL
echo "Starting FastAPI..."
pdm run uvicorn src.app.deployment.fastapi:app --reload --host ${FAST_HOST} --port ${FAST_PORT} &
echo "Installing Ray Serve..."
serve run src.app.deployment.ray:app --port ${SERVE_PORT} --host ${SERVE_URL}
