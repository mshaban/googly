default:
  just --list

cleanup:
  echo "Cleaning up..."
  serve shutdown -y && uvicorn shutdown
  echo "Cleanup done."

app:
  #!/bin/bash
  trap 'uvicorn shutdown' EXIT
  source config/local.env
  echo "Startin fastapi on ${FAST_HOST}:${FAST_PORT}"
  uvicorn src.app.deployment.fastapi:app --reload --host ${FAST_HOST} --port ${FAST_PORT}

# serve:
#   source config/local.env
#   echo "Starting Ray on ${RAY_URL}...${SERVE_URL} ${SERVE_PORT}"
#   serve run src.app.deployment.ray:app  --port ${SERVE_PORT} --host ${SERVE_URL}

run:
  #!/bin/bash
  trap 'just cleanup' EXIT
  echo "Exporting environment variables..."
  source config/local.env
  echo "Startin fastapi on ${FAST_HOST}:${FAST_PORT}"
  uvicorn src.app.deployment.fastapi:app --reload --host ${FAST_HOST} --port ${FAST_PORT} &

  # echo "Starting Ray on ${RAY_URL}..."
  # serve run src.app.deployment.ray:app  --port ${SERVE_PORT} --host ${SERVE_URL}

test:
  echo "Running Pytest tests..."
  pytest -s

lint:
  echo "Running flake8..."
  pdm run flake8 .
  pdm run black .
