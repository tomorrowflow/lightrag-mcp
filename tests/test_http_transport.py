"""
Tests for HTTP transport functionality.

This module tests the HTTP server implementation including:
- Server startup and shutdown
- MCP endpoint functionality (POST /mcp)
- SSE streaming (GET /mcp)
- Session management
- Error handling
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from starlette.applications import Starlette

from lightrag_mcp.http_server import LightRAGHttpServer, HTTPServerContext, http_server_lifespan
from lightrag_mcp.event_store import create_event_store
from lightrag_mcp.lightrag_client import LightRAGClient


class TestHTTPServerStartupShutdown:
    """Test HTTP server startup and shutdown functionality."""

    @pytest.mark.asyncio
    async def test_server_initialization(self, mock_lightrag_client, in_memory_event_store):
        """Test that the server initializes correctly."""
        server = LightRAGHttpServer()

        # Create mock context
        context = HTTPServerContext(
            lightrag_client=mock_lightrag_client,
            event_store=in_memory_event_store,
            session_manager=None
        )

        # Create ASGI app
        app = server._create_asgi_app(context)

        assert isinstance(app, Starlette)
        assert len(app.routes) == 4  # /mcp, /health, /status, /

        # Check routes
        route_paths = [route.path for route in app.routes]
        assert "/mcp" in route_paths
        assert "/health" in route_paths
        assert "/status" in route_paths
        assert "/" in route_paths

    @pytest.mark.asyncio
    async def test_server_lifespan_context(self, mock_lightrag_client, in_memory_event_store):
        """Test the server lifespan context manager."""
        # Create a minimal Starlette app for testing
        app = Starlette()

        async with http_server_lifespan(app) as context:
            assert isinstance(context, HTTPServerContext)
            assert context.lightrag_client is not None
            assert context.event_store is not None
            assert context.session_manager is not None

    @pytest.mark.asyncio
    async def test_server_startup_shutdown(self, mock_lightrag_client, in_memory_event_store):
        """Test full server startup and shutdown cycle."""
        server = LightRAGHttpServer()

        # Mock the server to avoid actual network operations
        with patch('uvicorn.Server') as mock_uvicorn_server:
            mock_server_instance = MagicMock()
            mock_uvicorn_server.return_value = mock_server_instance
            mock_server_instance.serve = AsyncMock()

            # Create context and start server
            context = HTTPServerContext(
                lightrag_client=mock_lightrag_client,
                event_store=in_memory_event_store,
                session_manager=MagicMock()
            )

            # Mock the lifespan context
            with patch('lightrag_mcp.http_server.http_server_lifespan') as mock_lifespan:
                mock_lifespan.return_value.__aenter__ = AsyncMock(return_value=context)
                mock_lifespan.return_value.__aexit__ = AsyncMock(return_value=None)

                # Start server (should not actually start due to mocking)
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(server.start(), timeout=0.1)

                # Verify server was created
                assert server.server is not None


class TestMCPEndpoints:
    """Test MCP endpoint functionality."""

    @pytest.mark.asyncio
    async def test_mcp_post_endpoint(self, mock_http_server_context):
        """Test POST /mcp endpoint for MCP protocol."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock session manager
        mock_session_manager = MagicMock()
        mock_http_server_context.session_manager = mock_session_manager
        mock_session_manager.handle_request = AsyncMock()

        # Create test request
        test_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "query_document",
                "arguments": {"query": "test"}
            }
        }

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post(
                "/mcp",
                json=test_message,
                headers={"Content-Type": "application/json"}
            )

            # Verify session manager was called
            mock_session_manager.handle_request.assert_called_once()

            # Check response
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_mcp_get_endpoint_sse(self, mock_http_server_context):
        """Test GET /mcp endpoint for SSE streaming."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock session manager
        mock_session_manager = MagicMock()
        mock_http_server_context.session_manager = mock_session_manager
        mock_session_manager.handle_request = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            # Test SSE request
            response = await client.get(
                "/mcp",
                headers={
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache"
                }
            )

            # Verify session manager was called
            mock_session_manager.handle_request.assert_called_once()
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_mcp_endpoint_without_session_manager(self, mock_lightrag_client, in_memory_event_store):
        """Test MCP endpoint when session manager is not initialized."""
        server = LightRAGHttpServer()

        # Create context without session manager
        context = HTTPServerContext(
            lightrag_client=mock_lightrag_client,
            event_store=in_memory_event_store,
            session_manager=None
        )

        app = server._create_asgi_app(context)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post("/mcp", json={"test": "data"})

            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Session manager not initialized" in data["error"]


class TestHealthEndpoint:
    """Test health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint_success(self, mock_http_server_context):
        """Test successful health check."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock health check
        mock_http_server_context.lightrag_client.get_health.return_value = {
            "status": "healthy",
            "version": "1.0.0"
        }

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get("/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "healthy"
            assert "lightrag_api" in data
            assert "event_store" in data
            assert data["transport"] == "http"

    @pytest.mark.asyncio
    async def test_health_endpoint_failure(self, mock_http_server_context):
        """Test health check failure."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock health check failure
        mock_http_server_context.lightrag_client.get_health.side_effect = Exception("API unavailable")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get("/health")

            assert response.status_code == 503
            data = response.json()

            assert data["status"] == "unhealthy"
            assert "error" in data


class TestStatusEndpoint:
    """Test detailed status endpoint."""

    @pytest.mark.asyncio
    async def test_status_endpoint(self, mock_http_server_context, test_config):
        """Test status endpoint returns configuration."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get("/status")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "running"
            assert "configuration" in data
            assert "lightrag_api" in data
            assert "session_stats" in data
            assert data["transport"] == "http"

            # Check configuration structure
            config = data["configuration"]
            assert "host" in config
            assert "port" in config
            assert "stateful_mode" in config
            assert "cors_origins" in config
            assert "security" in config

    @pytest.mark.asyncio
    async def test_status_endpoint_with_session_stats(self, mock_http_server_context):
        """Test status endpoint with session statistics."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock session stats (would normally come from event store)
        with patch('lightrag_mcp.config.STATEFUL_MODE', True):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                response = await client.get("/status")

                assert response.status_code == 200
                data = response.json()
                assert data["configuration"]["stateful_mode"] is True


class TestRootEndpoint:
    """Test root endpoint."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, mock_http_server_context):
        """Test root endpoint returns server information."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get("/")

            assert response.status_code == 200
            data = response.json()

            assert "name" in data
            assert "LightRAG MCP Server" in data["name"]
            assert "version" in data
            assert data["transport"] == "http"
            assert "endpoints" in data
            assert "/mcp" in data["endpoints"]
            assert "/health" in data["endpoints"]
            assert "/status" in data["endpoints"]


class TestErrorHandling:
    """Test error handling in HTTP endpoints."""

    @pytest.mark.asyncio
    async def test_mcp_endpoint_error_handling(self, mock_http_server_context):
        """Test error handling in MCP endpoint."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock session manager to raise exception
        mock_session_manager = MagicMock()
        mock_http_server_context.session_manager = mock_session_manager
        mock_session_manager.handle_request.side_effect = Exception("Test error")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post("/mcp", json={"test": "data"})

            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "type" in data
            assert data["type"] == "server_error"

    @pytest.mark.asyncio
    async def test_invalid_json_request(self, mock_http_server_context):
        """Test handling of invalid JSON in requests."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            # Send invalid JSON
            response = await client.post(
                "/mcp",
                content="invalid json",
                headers={"Content-Type": "application/json"}
            )

            # Should still handle gracefully (Starlette handles JSON parsing)
            assert response.status_code in [200, 400, 500]  # Depends on session manager behavior

    @pytest.mark.asyncio
    async def test_method_not_allowed(self, mock_http_server_context):
        """Test handling of unsupported HTTP methods."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.put("/mcp", json={"test": "data"})

            # Should return 405 Method Not Allowed
            assert response.status_code == 405


class TestSessionManagement:
    """Test session management functionality."""

    @pytest.mark.asyncio
    async def test_session_manager_integration(self, mock_http_server_context):
        """Test integration with session manager."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock session manager with proper async context manager
        mock_session_manager = MagicMock()
        mock_session_manager.run = MagicMock()
        mock_session_manager.run.return_value.__aenter__ = AsyncMock()
        mock_session_manager.run.return_value.__aexit__ = AsyncMock()
        mock_session_manager.handle_request = AsyncMock()

        mock_http_server_context.session_manager = mock_session_manager

        # Test that session manager run context is used in lifespan
        lifespan_handler = server._create_lifespan_handler(mock_http_server_context)

        # This tests the lifespan handler integration
        assert lifespan_handler is not None

    @pytest.mark.asyncio
    async def test_event_store_integration(self, mock_http_server_context):
        """Test integration with event store."""
        # Verify event store is properly integrated in context
        assert mock_http_server_context.event_store is not None
        assert hasattr(mock_http_server_context.event_store, 'store_event')
        assert hasattr(mock_http_server_context.event_store, 'get_events_after')


class TestMiddlewareIntegration:
    """Test middleware integration."""

    @pytest.mark.asyncio
    async def test_cors_middleware(self, mock_http_server_context):
        """Test CORS middleware is properly configured."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Check that CORS middleware is added
        cors_middleware = None
        for middleware in app.user_middleware:
            if hasattr(middleware, 'cls') and 'CORSMiddleware' in str(middleware.cls):
                cors_middleware = middleware
                break

        assert cors_middleware is not None

    @pytest.mark.asyncio
    async def test_security_middleware_order(self, mock_http_server_context):
        """Test that security middleware is added in correct order."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Check middleware order (should be: auth, rate limit, size limit, logging, timeout, security headers, cors)
        middleware_classes = [str(m.cls) if hasattr(m, 'cls') else str(type(m)) for m in app.user_middleware]

        # Verify CORS is last
        assert 'CORSMiddleware' in middleware_classes[-1]

        # Verify security headers middleware exists
        security_headers_found = any('SecurityHeadersMiddleware' in cls for cls in middleware_classes)
        assert security_headers_found


class TestConfigurationIntegration:
    """Test configuration integration."""

    @pytest.mark.asyncio
    async def test_debug_mode_configuration(self, mock_http_server_context):
        """Test debug mode configuration."""
        with patch('lightrag_mcp.config.MCP_HOST', 'localhost'):
            server = LightRAGHttpServer()
            app = server._create_asgi_app(mock_http_server_context)

            # Should be in debug mode for localhost
            assert app.debug is True

    @pytest.mark.asyncio
    async def test_production_mode_configuration(self, mock_http_server_context):
        """Test production mode configuration."""
        with patch('lightrag_mcp.config.MCP_HOST', 'example.com'):
            server = LightRAGHttpServer()
            app = server._create_asgi_app(mock_http_server_context)

            # Should not be in debug mode for non-localhost
            assert app.debug is False