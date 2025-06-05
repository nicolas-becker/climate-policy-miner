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
COPY src/quotation_utils.py .
COPY src/classification_utils.py .
COPY src/general_utils.py .

# Create necessary directories
RUN mkdir -p /app/data /app/results

# Set the PORT environment variable
EXPOSE ${PORT:-10000}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Run with gunicorn
CMD gunicorn -w 1 -b 0.0.0.0:${PORT:-10000} --timeout 600 --keep-alive 2 --max-requests 100 --worker-class sync src.flask_app:app