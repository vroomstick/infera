# Infera - SEC 10-K Risk Analysis Pipeline
# Multi-stage build for smaller image size

FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Create non-root user for security
RUN groupadd -r infera && useradd -r -g infera infera

# Copy installed packages from builder
COPY --from=builder /root/.local /home/infera/.local

# Make sure scripts in .local are usable
ENV PATH=/home/infera/.local/bin:$PATH

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY --chown=infera:infera . .

# Create directories for outputs with proper permissions
RUN mkdir -p reports data/clean data/scored data/segments data/summarized .sec_cache \
    && mkdir -p /home/infera/.cache \
    && chown -R infera:infera /app /home/infera

# Switch to non-root user
USER infera

# Pre-download the sentence-transformer model (optional, skip in CI to save space)
# Set PRELOAD_MODEL=true when building for production
ARG PRELOAD_MODEL=false
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"; \
    fi

# Expose API port
EXPOSE 8000

# Default command: run the API
CMD ["python", "-m", "api.main"]

