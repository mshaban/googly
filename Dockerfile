# Use a base image that includes Python
FROM rayproject/ray:2.10.0.6daa0a-aarch64

# Set work directory
WORKDIR /googlify

RUN pip install -U pip wheel setuptools
RUN pip install .

#FIXME: dont't do that
# COPY requirements.txt .
# RUN pip install -r requirements.txt

WORKDIR /googlify

# Install supervisord
RUN apt-get update && apt-get install -y supervisor

# Install any necessary Python packages
COPY pyproject.toml .
COPY README.md .
RUN pip install .
# RUN pip install pdm
# RUN pdm config python.use_venv False
# RUN pdm install --prod

# Create log directory for supervisord logs
RUN mkdir -p /var/log/supervisor

# Set the environment variables
ENV FAST_PORT=8888

# Expose the ports
EXPOSE $FAST_PORT

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$FAST_PORT/health || exit 1

# Create empty logs anqd out directories
RUN mkdir logs out

# Copy the necessary files
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY artifacts/ ./artifacts/
COPY scripts/ ./scripts/
COPY assets ./assets/
COPY config ./config/

# Add src
COPY src/ ./src/

# Start processes using supervisord
# CMD ["/usr/bin/supervisord"]
RUN chmod +x ./scripts/entrypoint.sh
CMD ["./scripts/entrypoint.sh"]

# Optional labels
LABEL maintainer="shaban" \
  version="0.1" \
  description="Docker image for running a FastAPI app with Ray Serve"
