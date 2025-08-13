#!/bin/bash
set -e

# Use PORT environment variable with fallback
PORT=${PORT:-10000}

# Start gunicorn with proper signal handling
exec gunicorn \
    --workers 1 \
    --bind "0.0.0.0:$PORT" \
    --timeout 3600 \
    --keep-alive 30 \
    --max-requests 2000 \
    --max-requests-jitter 10 \
    --worker-class sync \
    --worker-connections 1000 \
    --preload \
    --access-logfile - \
    --chdir /app/src \
    src.flask_app:app