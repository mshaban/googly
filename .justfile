default:
  just --list

cleanup:
  echo "Cleaning up..."
  ray stop


test:
  echo "Running Pytest tests..."
  pytest -s
