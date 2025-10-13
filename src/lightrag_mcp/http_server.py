"""
HTTP transport server module for LightRAG MCP server.

This module provides HTTP transport capabilities using StreamableHTTPSessionManager
with proper integration to the existing LightRAG MCP server infrastructure.
"""

import asyncio
import json
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, AsyncIterator
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, Mount
from starlette.types import Receive, Scope, Send
import uvicorn

from . import config
from .event_store import EventStore, create_event_store
from .lightrag_client import LightRAGClient
from .security import (
    AuthenticationMiddleware, SecurityHeadersMiddleware, RateLimitMiddleware,
    RequestSizeLimitMiddleware, RequestLoggingMiddleware, TimeoutMiddleware
)
from .server import mcp as fastmcp_server, AppContext, app_lifespan


logger = logging.getLogger(__name__)


@dataclass
class HTTPServerContext:
    """Extended context for HTTP server with session manager and event store."""
    lightrag_client: LightRAGClient
    event_store: EventStore
    session_manager: Optional[StreamableHTTPSessionManager] = None


@asynccontextmanager
async def http_server_lifespan(app: Starlette) -> AsyncIterator[HTTPServerContext]:
    """
    Lifespan manager for HTTP server that integrates all components.

    Manages:
    - LightRAG client lifecycle
    - Event store initialization and cleanup
    - Session manager setup and teardown
    """
    # Initialize LightRAG client
    lightrag_client = LightRAGClient(
        base_url=config.LIGHTRAG_API_BASE_URL,
        api_key=config.LIGHTRAG_API_KEY,
    )

    # Initialize event store based on configuration
    storage_type = "file" if config.STATEFUL_MODE else "memory"
    event_store = create_event_store(
        storage_type=storage_type,
        max_events_per_session=config.MAX_EVENTS_PER_SESSION
    )

    # Create session manager with our FastMCP server and event store
    # FastMCP wraps a low-level MCP server, we need to pass the low-level server
    session_manager = StreamableHTTPSessionManager(
        app=fastmcp_server._mcp_server,  # Use the underlying MCP server
        event_store=event_store if config.STATEFUL_MODE else None,
        json_response=False,  # Use SSE streams by default
    )

    # Background cleanup task for old events
    cleanup_task = None
    if config.STATEFUL_MODE:
        async def cleanup_events_periodically():
            """Periodically cleanup old events."""
            while True:
                try:
                    await asyncio.sleep(3600)  # Cleanup every hour
                    cleaned_count = await event_store.cleanup_old_events()
                    if cleaned_count > 0:
                        logger.info(f"Cleaned up {cleaned_count} old events")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error during event cleanup: {e}")

        cleanup_task = asyncio.create_task(cleanup_events_periodically())

    try:
        # Start the session manager
        async with session_manager.run():
            context = HTTPServerContext(
                lightrag_client=lightrag_client,
                event_store=event_store,
                session_manager=session_manager
            )
            yield context
    finally:
        # Cancel cleanup task
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

        # Close LightRAG client
        await lightrag_client.close()
        logger.info("LightRAG HTTP Server stopped")


class LightRAGHttpServer:
    """
    HTTP server for LightRAG MCP with StreamableHTTPSessionManager integration.

    Provides both HTTP transport for MCP protocol and REST endpoints for
    health monitoring and server management.
    """

    def __init__(self):
        """Initialize the HTTP server."""
        self.app: Optional[Starlette] = None
        self.session_manager: Optional[StreamableHTTPSessionManager] = None
        self.server: Optional[uvicorn.Server] = None
        self._shutdown_event = asyncio.Event()

    def _create_asgi_app(self, lifespan_context: HTTPServerContext) -> Starlette:
        """Create the ASGI application with all routes and middleware."""

        # ASGI handler for streamable HTTP connections
        async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
            await lifespan_context.session_manager.handle_request(scope, receive, send)

        # Define routes
        routes = [
            Mount("/mcp", app=handle_streamable_http),  # MCP protocol endpoint
            Route("/health", self._handle_health, methods=["GET"]),
            Route("/status", self._handle_status, methods=["GET"]),
            Route("/", self._handle_root, methods=["GET"]),
        ]

        # Create Starlette app with CORS middleware
        app = Starlette(
            debug=config.MCP_HOST == "localhost",  # Enable debug only for localhost
            routes=routes,
            lifespan=self._create_lifespan_handler(lifespan_context),
        )

        # Add security middleware in correct order
        # Order matters: authentication -> rate limiting -> size limits -> logging -> timeout -> security headers

        # 1. Authentication middleware (first, as it may reject requests)
        if config.ENABLE_AUTH:
            app.add_middleware(AuthenticationMiddleware)

        # 2. Rate limiting middleware
        app.add_middleware(RateLimitMiddleware)

        # 3. Request size limiting middleware
        app.add_middleware(RequestSizeLimitMiddleware)

        # 4. Request logging middleware
        if config.ENABLE_REQUEST_LOGGING:
            app.add_middleware(RequestLoggingMiddleware)

        # 5. Timeout middleware
        app.add_middleware(TimeoutMiddleware)

        # 6. Security headers middleware
        if config.ENABLE_SECURITY_HEADERS:
            app.add_middleware(SecurityHeadersMiddleware)

        # 7. CORS middleware (last, as it needs to handle preflight requests)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.CORS_ORIGINS,
            allow_credentials=config.CORS_ALLOW_CREDENTIALS,
            allow_methods=config.CORS_ALLOW_METHODS,
            allow_headers=config.CORS_ALLOW_HEADERS,
            expose_headers=config.CORS_EXPOSE_HEADERS,
            max_age=config.CORS_MAX_AGE,
        )

        return app

    def _create_lifespan_handler(self, context: HTTPServerContext):
        """Create lifespan handler for Starlette app."""
        @asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncIterator[None]:
            """Handle ASGI lifespan events."""
            logger.info("HTTP server started with transport")
            try:
                yield
            finally:
                logger.info("HTTP server shutting down")
                self._shutdown_event.set()

        return lifespan


    async def _handle_health(self, request) -> Response:
        """Health check endpoint."""
        try:
            context = request.app.state.lifespan_context

            # Check LightRAG API health
            health_result = await context.lightrag_client.get_health()

            return JSONResponse({
                "status": "healthy",
                "lightrag_api": health_result,
                "event_store": {
                    "type": type(context.event_store).__name__,
                    "stateful": config.STATEFUL_MODE
                },
                "transport": "http"
            })

        except Exception as e:
            if config.ENABLE_ERROR_LOGGING:
                logger.error(f"Health check failed: {e}", exc_info=True)
            else:
                logger.error(f"Health check failed: {str(e)}")

            error_message = "Health check failed"
            if config.ENVIRONMENT == "development":
                error_message = f"Health check failed: {str(e)}"

            return JSONResponse(
                {"status": "unhealthy", "error": error_message, "type": "health_check_error"},
                status_code=503
            )

    async def _handle_status(self, request) -> Response:
        """Detailed status endpoint."""
        try:
            context = request.app.state.lifespan_context

            # Get session statistics if using stateful mode
            session_stats = None
            if config.STATEFUL_MODE:
                # This would need to be implemented in the event store
                # For now, return basic info
                session_stats = {"mode": "stateful"}

            return JSONResponse({
                "status": "running",
                "configuration": {
                    "host": config.MCP_HOST,
                    "port": config.MCP_PORT,
                    "stateful_mode": config.STATEFUL_MODE,
                    "cors_origins": config.CORS_ORIGINS,
                    "event_retention_hours": config.EVENT_RETENTION_HOURS,
                    "max_events_per_session": config.MAX_EVENTS_PER_SESSION,
                    "session_timeout_minutes": config.SESSION_TIMEOUT_MINUTES,
                    "security": {
                        "auth_enabled": config.ENABLE_AUTH,
                        "rate_limit_requests": config.RATE_LIMIT_REQUESTS,
                        "rate_limit_window_seconds": config.RATE_LIMIT_WINDOW_SECONDS,
                        "max_request_size_mb": config.MAX_REQUEST_SIZE_MB,
                        "security_headers_enabled": config.ENABLE_SECURITY_HEADERS,
                        "environment": config.ENVIRONMENT,
                        "request_timeout_seconds": config.REQUEST_TIMEOUT_SECONDS,
                        "request_logging_enabled": config.ENABLE_REQUEST_LOGGING,
                        "error_logging_enabled": config.ENABLE_ERROR_LOGGING,
                    },
                },
                "lightrag_api": {
                    "base_url": config.LIGHTRAG_API_BASE_URL,
                    "has_api_key": bool(config.LIGHTRAG_API_KEY)
                },
                "session_stats": session_stats,
                "transport": "http"
            })

        except Exception as e:
            if config.ENABLE_ERROR_LOGGING:
                logger.error(f"Status check failed: {e}", exc_info=True)
            else:
                logger.error(f"Status check failed: {str(e)}")

            error_message = "Status check failed"
            if config.ENVIRONMENT == "development":
                error_message = f"Status check failed: {str(e)}"

            return JSONResponse(
                {"status": "error", "error": error_message, "type": "status_check_error"},
                status_code=500
            )

    async def _handle_root(self, request) -> Response:
        """Root endpoint with server information."""
        return JSONResponse({
            "name": "LightRAG MCP Server",
            "version": "0.1.0",
            "transport": "http",
            "endpoints": {
                "mcp": "/mcp",
                "health": "/health",
                "status": "/status"
            },
            "documentation": "This is the LightRAG MCP server with HTTP transport support"
        })

    async def start(self) -> None:
        """Start the HTTP server."""
        try:
            # Create lifespan context
            async with http_server_lifespan(None) as context:
                # Create ASGI app
                self.app = self._create_asgi_app(context)

                # Store context for route handlers
                if self.app and hasattr(self.app, 'state'):
                    self.app.state.lifespan_context = context
                self.session_manager = context.session_manager

                # Create uvicorn server
                server = uvicorn.Server(uvicorn.Config(
                    app=self.app,
                    host=config.MCP_HOST,
                    port=config.MCP_PORT,
                    access_log=True,
                    log_level="info" if config.MCP_HOST != "localhost" else "debug"
                ))
                self.server = server

                logger.info(f"Starting LightRAG MCP HTTP server on {config.MCP_HOST}:{config.MCP_PORT}")
                logger.info(f"CORS origins: {config.CORS_ORIGINS}")
                logger.info(f"Stateful mode: {config.STATEFUL_MODE}")

                # Setup signal handlers for graceful shutdown
                def signal_handler(signum, frame):
                    logger.info(f"Received signal {signum}, shutting down...")
                    self._shutdown_event.set()

                signal.signal(signal.SIGTERM, signal_handler)
                signal.signal(signal.SIGINT, signal_handler)

                # Start server
                if self.server and hasattr(self.server, 'serve'):
                    await self.server.serve()

        except Exception as e:
            if config.ENABLE_ERROR_LOGGING:
                logger.error(f"Failed to start HTTP server: {e}", exc_info=True)
            else:
                logger.error(f"Failed to start HTTP server: {str(e)}")
            raise

    async def stop(self) -> None:
        """Stop the HTTP server."""
        logger.info("Stopping LightRAG MCP HTTP server...")

        if self.server:
            self.server.should_exit = True

        self._shutdown_event.set()

        # Wait for shutdown to complete
        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Shutdown timeout exceeded")


async def run_http_server() -> None:
    """Run the HTTP server as a standalone function."""
    server = LightRAGHttpServer()

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        await server.stop()


def main() -> None:
    """Main entry point for HTTP server mode."""
    # Configure logging based on environment
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Suppress some noisy loggers in production
    if config.ENVIRONMENT == "production":
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info("Starting LightRAG MCP HTTP Server")
    logger.info(f"LightRAG API: {config.LIGHTRAG_API_BASE_URL}")
    logger.info(f"MCP Server: {config.MCP_SERVER_URL}")
    logger.info(f"Stateful mode: {config.STATEFUL_MODE}")

    if config.LIGHTRAG_API_KEY:
        logger.info("API key configured")
    else:
        logger.warning("No API key provided")

    # Run the HTTP server
    asyncio.run(run_http_server())


if __name__ == "__main__":
    main()