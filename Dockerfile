FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (much faster than conda)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        curl \
        gcc \
        g++ \
        libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements (much faster than conda)
COPY requirements_app.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements_app.txt

# Copy source code
COPY src/ ./src/

# Create __init__.py to make src a Python package
RUN touch src/__init__.py

# Add src to Python path for absolute imports
ENV PYTHONPATH=/app/src:/app

# Create necessary directories
RUN mkdir -p /app/data /app/results /app/uploads

# Set the PORT environment variable
EXPOSE ${PORT:-10000}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Create entrypoint script for better signal handling
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Use exec form with entrypoint
ENTRYPOINT ["/entrypoint.sh"]