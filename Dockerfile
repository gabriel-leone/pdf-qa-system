# Multi-stage Docker build for PDF Q&A System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs uploads chroma_db .cache && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "run.py", "server"]

# Development stage
FROM base as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio hypothesis pytest-cov

USER appuser

# Override command for development
CMD ["python", "main.py"]

# Production stage
FROM base as production

# Copy only necessary files for production
COPY --from=base /app /app

# Set production environment
ENV ENVIRONMENT=production \
    DEBUG=false \
    LOG_LEVEL=INFO

# Use production runner
CMD ["python", "run.py", "server"]