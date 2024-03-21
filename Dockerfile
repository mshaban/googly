# Start from Python 3.11 image
ARG PY_VERSION=3.11

# Build stage: build and install dependencies
# FROM python:${PY_VERSION} AS builder
FROM rayproject/ray:2.10


ARG VERSION=0.dev
ENV PDM_BUILD_SCM_VERSION=${VERSION}

WORKDIR /project


# install PDM
RUN pip install -U pip setuptools wheel
RUN pip install pdm pdm-dockerize
RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  --mount=type=bind,source=pdm.lock,target=pdm.lock \
  --mount=type=cache,target=$HOME/.cache\
  pdm dockerize --prod -v

###RUNTIME STAGE ###
FROM python:${PY_VERSION} AS runtime

# Set the environment variables
ENV FAST_PORT=8888

# Expose the ports
EXPOSE $FAST_PORT


# Set work directory
WORKDIR /googlify

#FIXME: dont't do that
COPY requirements.txt .
RUN pip install -r requirements.txt


# Fetch built dependencies
COPY --from=builder /project/dist/docker /googlify


# Create empty logs anqd out directories
RUN mkdir logs out

# Copy the necessary files
COPY README.md .
COPY justfile .
COPY artifacts/ ./artifacts/
COPY assets ./assets/
COPY config ./config/
COPY scripts ./scripts/

# Add src
COPY src/ ./src/

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$FAST_PORT/health || exit 1


# Entrypoint script
CMD ["./scripts/entrypoint.sh"]

# Optional labels
LABEL maintainer="shaban" \
  version="0.1" \
  description="Docker image for running a FastAPI app with Ray Serve"
