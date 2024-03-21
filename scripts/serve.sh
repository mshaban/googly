#!/bin/bash
source config/local.env
serve run src.app.deployment.ray:app --port ${SERVE_PORT} --host ${SERVE_URL}
