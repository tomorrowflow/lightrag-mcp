"""
Configuration module for LightRAG MCP server.
"""

import argparse
import os

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9621
DEFAULT_API_KEY = ""


def parse_args():
    """Parse command line arguments for LightRAG MCP server."""
    parser = argparse.ArgumentParser(description="LightRAG MCP Server")
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"LightRAG API host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"LightRAG API port (default: {DEFAULT_PORT})",
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="LightRAG API key (optional)")
    return parser.parse_args()


# Get configuration from environment variables or command line arguments
LIGHTRAG_API_HOST = os.getenv("LIGHTRAG_API_HOST", DEFAULT_HOST)
LIGHTRAG_API_PORT = int(os.getenv("LIGHTRAG_API_PORT", str(DEFAULT_PORT)))
LIGHTRAG_API_KEY = os.getenv("LIGHTRAG_API_KEY", DEFAULT_API_KEY)

# Override with command line arguments if provided
args = parse_args()
if args.host != DEFAULT_HOST:
    LIGHTRAG_API_HOST = args.host
if args.port != DEFAULT_PORT:
    LIGHTRAG_API_PORT = args.port
if args.api_key != DEFAULT_API_KEY:
    LIGHTRAG_API_KEY = args.api_key

LIGHTRAG_API_BASE_URL = f"http://{LIGHTRAG_API_HOST}:{LIGHTRAG_API_PORT}"
