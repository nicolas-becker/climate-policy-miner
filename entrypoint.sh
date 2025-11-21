#!/bin/bash
set -e

# Use PORT environment variable with fallback
PORT=${PORT:-10000}

# Start gunicorn with proper signal handling
exec gunicorn \
    --workers 1 \
    --threads 4 \
    --bind "0.0.0.0:$PORT" \
    --timeout 7200 \
    --graceful-timeout 120 \
    --keep-alive 600 \
    --max-requests 2000 \
    --max-requests-jitter 10 \
    --worker-class gthread \
    --worker-connections 1000 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --chdir /app/src \
    src.flask_app:app