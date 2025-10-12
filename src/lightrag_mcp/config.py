"""
Configuration module for LightRAG MCP server.
"""

import argparse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9621
DEFAULT_MCP_HOST = "localhost"
DEFAULT_MCP_PORT = 8080
DEFAULT_API_KEY = ""
DEFAULT_TRANSPORT_MODE = "stdio"  # stdio, http
DEFAULT_STATEFUL_MODE = True
DEFAULT_EVENT_RETENTION_HOURS = 24
DEFAULT_MAX_EVENTS_PER_SESSION = 1000
DEFAULT_CORS_ORIGINS = ["*"]
DEFAULT_SESSION_TIMEOUT_MINUTES = 30

# Security configuration defaults
DEFAULT_ENABLE_AUTH = False
DEFAULT_API_KEY = ""
DEFAULT_BEARER_TOKEN_SECRET = ""
DEFAULT_RATE_LIMIT_REQUESTS = 100
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_MAX_REQUEST_SIZE_MB = 10
DEFAULT_ENABLE_SECURITY_HEADERS = True
DEFAULT_ENVIRONMENT = "development"  # development, staging, production
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_ENABLE_REQUEST_LOGGING = True
DEFAULT_ENABLE_ERROR_LOGGING = True
DEFAULT_CORS_ALLOW_CREDENTIALS = True
DEFAULT_CORS_ALLOW_METHODS = ["GET", "POST", "DELETE", "OPTIONS"]
DEFAULT_CORS_ALLOW_HEADERS = ["*"]
DEFAULT_CORS_EXPOSE_HEADERS = ["Mcp-Session-Id"]
DEFAULT_CORS_MAX_AGE = 86400


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

    # HTTP transport arguments
    parser.add_argument(
        "--mcp-host",
        default=DEFAULT_MCP_HOST,
        help=f"MCP server host (default: {DEFAULT_MCP_HOST})"
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=DEFAULT_MCP_PORT,
        help=f"MCP server port (default: {DEFAULT_MCP_PORT})",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=DEFAULT_TRANSPORT_MODE,
        help=f"Transport mode (default: {DEFAULT_TRANSPORT_MODE})",
    )
    parser.add_argument(
        "--stateful",
        action="store_true",
        default=DEFAULT_STATEFUL_MODE,
        help=f"Enable stateful sessions (default: {DEFAULT_STATEFUL_MODE})",
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Enable stateless mode (overrides --stateful)",
    )

    return parser.parse_args()


# Get configuration from environment variables or command line arguments
LIGHTRAG_API_HOST = os.getenv("LIGHTRAG_API_HOST", DEFAULT_HOST)
LIGHTRAG_API_PORT = int(os.getenv("LIGHTRAG_API_PORT", str(DEFAULT_PORT)))
LIGHTRAG_API_KEY = os.getenv("LIGHTRAG_API_KEY", DEFAULT_API_KEY)

# HTTP transport configuration from environment variables
MCP_HOST = os.getenv("MCP_HOST", DEFAULT_MCP_HOST)
MCP_PORT = int(os.getenv("MCP_PORT", str(DEFAULT_MCP_PORT)))
TRANSPORT_MODE = os.getenv("TRANSPORT_MODE", DEFAULT_TRANSPORT_MODE)
STATEFUL_MODE = os.getenv("STATEFUL_MODE", str(DEFAULT_STATEFUL_MODE)).lower() == "true"
EVENT_RETENTION_HOURS = int(os.getenv("EVENT_RETENTION_HOURS", str(DEFAULT_EVENT_RETENTION_HOURS)))
MAX_EVENTS_PER_SESSION = int(os.getenv("MAX_EVENTS_PER_SESSION", str(DEFAULT_MAX_EVENTS_PER_SESSION)))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)).split(",")
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", str(DEFAULT_SESSION_TIMEOUT_MINUTES)))

# Security configuration from environment variables
ENABLE_AUTH = os.getenv("ENABLE_AUTH", str(DEFAULT_ENABLE_AUTH)).lower() == "true"
API_KEY = os.getenv("API_KEY", DEFAULT_API_KEY)
BEARER_TOKEN_SECRET = os.getenv("BEARER_TOKEN_SECRET", DEFAULT_BEARER_TOKEN_SECRET)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", str(DEFAULT_RATE_LIMIT_REQUESTS)))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", str(DEFAULT_RATE_LIMIT_WINDOW_SECONDS)))
MAX_REQUEST_SIZE_MB = int(os.getenv("MAX_REQUEST_SIZE_MB", str(DEFAULT_MAX_REQUEST_SIZE_MB)))
ENABLE_SECURITY_HEADERS = os.getenv("ENABLE_SECURITY_HEADERS", str(DEFAULT_ENABLE_SECURITY_HEADERS)).lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", DEFAULT_ENVIRONMENT)
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS)))
LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
ENABLE_REQUEST_LOGGING = os.getenv("ENABLE_REQUEST_LOGGING", str(DEFAULT_ENABLE_REQUEST_LOGGING)).lower() == "true"
ENABLE_ERROR_LOGGING = os.getenv("ENABLE_ERROR_LOGGING", str(DEFAULT_ENABLE_ERROR_LOGGING)).lower() == "true"
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", str(DEFAULT_CORS_ALLOW_CREDENTIALS)).lower() == "true"
CORS_ALLOW_METHODS = os.getenv("CORS_ALLOW_METHODS", ",".join(DEFAULT_CORS_ALLOW_METHODS)).split(",")
CORS_ALLOW_HEADERS = os.getenv("CORS_ALLOW_HEADERS", ",".join(DEFAULT_CORS_ALLOW_HEADERS)).split(",")
CORS_EXPOSE_HEADERS = os.getenv("CORS_EXPOSE_HEADERS", ",".join(DEFAULT_CORS_EXPOSE_HEADERS)).split(",")
CORS_MAX_AGE = int(os.getenv("CORS_MAX_AGE", str(DEFAULT_CORS_MAX_AGE)))

# Override with command line arguments if provided
args = parse_args()
if args.host != DEFAULT_HOST:
    LIGHTRAG_API_HOST = args.host
if args.port != DEFAULT_PORT:
    LIGHTRAG_API_PORT = args.port
if args.api_key != DEFAULT_API_KEY:
    LIGHTRAG_API_KEY = args.api_key
if args.mcp_host != DEFAULT_MCP_HOST:
    MCP_HOST = args.mcp_host
if args.mcp_port != DEFAULT_MCP_PORT:
    MCP_PORT = args.mcp_port
if args.transport != DEFAULT_TRANSPORT_MODE:
    TRANSPORT_MODE = args.transport
if args.stateless:
    STATEFUL_MODE = False
elif args.stateful != DEFAULT_STATEFUL_MODE:
    STATEFUL_MODE = args.stateful

LIGHTRAG_API_BASE_URL = f"http://{LIGHTRAG_API_HOST}:{LIGHTRAG_API_PORT}"
MCP_SERVER_URL = f"http://{MCP_HOST}:{MCP_PORT}"
