# Multi-stage build for FastAPI Microservice Backend
# Optimized for Python 3.13 with security hardening

# =================== BUILD STAGE ===================
FROM python:3.13-slim AS builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt


# =================== RUNTIME STAGE ===================
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies (Tesseract OCR for deskew service)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove

# Verify Tesseract is on PATH and working (build fails here if not)
RUN tesseract --version

# Create non-root user
RUN groupadd -r appuser -g 1001 && \
    useradd -r -u 1001 -g appuser -s /sbin/nologin -c "Application user" appuser

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser services/ ./services/
COPY --chown=appuser:appuser .env.example .env

# Set Python path
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
