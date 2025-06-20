#!/bin/bash
set -e

# Use PORT environment variable with fallback
PORT=${PORT:-10000}

# Start gunicorn with proper signal handling
exec gunicorn \
    --workers 1 \
    --bind "0.0.0.0:$PORT" \
    --timeout 1800 \
    --keep-alive 10 \
    --max-requests 100 \
    --worker-class sync \
    --pythonpath /app \
    src.flask_app:app