"""
Entry point for LightRAG MCP server.
"""

import logging
import sys

from lightrag_mcp import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Main function for server startup."""
    try:
        log_level = getattr(logging, "INFO")
        logging.getLogger().setLevel(log_level)

        logger.info("Starting LightRAG MCP server")
        logger.info(
            f"LightRAG API server is expected to be already running and available at: {config.LIGHTRAG_API_BASE_URL}"
        )
        if config.LIGHTRAG_API_KEY:
            logger.info("API key is configured")
        else:
            logger.warning("No API key provided")

        # Choose transport mode based on configuration
        if config.TRANSPORT_MODE == "http":
            logger.info("Using HTTP transport mode")
            from lightrag_mcp.http_server import main as http_main
            http_main()
        else:
            logger.info("Using stdio transport mode")
            from lightrag_mcp.server import mcp
            mcp.run(transport="stdio")

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception(f"Error starting server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
