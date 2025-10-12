# Use Python 3.11 slim base image for HTTP server deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for HTTP server
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

# Copy project files with proper ownership
COPY --chown=app:app . /app

# Copy .env file if it exists
COPY --chown=app:app .env* /app/

# Install dependencies using uv (includes starlette, uvicorn, anyio for HTTP transport)
RUN uv pip install --system -e .

# Set environment variables for HTTP transport mode
ENV TRANSPORT_MODE=http
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=3000
ENV LIGHTRAG_API_HOST=host.docker.internal
ENV LIGHTRAG_API_PORT=9621
ENV LIGHTRAG_API_KEY=no-key
ENV STATEFUL_MODE=true
ENV CORS_ORIGINS=["*"]

# Expose HTTP server port
EXPOSE 3000

# Switch to non-root user
USER app

# Health check for HTTP endpoints
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Set entrypoint for HTTP server mode
ENTRYPOINT ["python", "-m", "lightrag_mcp.main"]