# Use Python 3.11 slim base image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

# Copy project files
COPY --chown=app:app . /app

# Install dependencies using uv
RUN uv pip install --system -e .

# Set environment variables with defaults
ENV LIGHTRAG_API_HOST=host.docker.internal
ENV LIGHTRAG_API_PORT=9621
ENV LIGHTRAG_API_KEY=no-key

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Set entrypoint
ENTRYPOINT ["python", "-m", "lightrag_mcp.main"]