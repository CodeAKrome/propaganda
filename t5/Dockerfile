# Multi-stage build for smaller image
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml .
COPY bias_detector ./bias_detector

# Install package
RUN pip install --no-cache-dir --target=/app/deps .

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /app/deps /usr/local/lib/python3.11/site-packages

# Copy application code
COPY bias_detector ./bias_detector
COPY pyproject.toml .

# Install the package
RUN pip install --no-cache-dir .

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import bias_detector; print('OK')" || exit 1

# Default command
ENTRYPOINT ["bias-processor"]
CMD ["--help"]
