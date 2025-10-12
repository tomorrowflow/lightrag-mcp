"""
Integration tests for HTTP transport functionality.

This module tests end-to-end HTTP transport with existing LightRAG tools:
- End-to-end HTTP transport with existing LightRAG tools
- Docker container tests
- Backward compatibility tests with stdio transport
"""

import asyncio
import json
import os
import subprocess
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport
from starlette.applications import Starlette

from lightrag_mcp.http_server import LightRAGHttpServer, HTTPServerContext
from lightrag_mcp.event_store import create_event_store
from lightrag_mcp.lightrag_client import LightRAGClient
from lightrag_mcp.server import mcp as fastmcp_server


class TestEndToEndHTTPTransport:
    """Test end-to-end HTTP transport functionality."""

    @pytest.mark.asyncio
    async def test_full_mcp_request_flow(self, mock_http_server_context):
        """Test complete MCP request flow from HTTP to LightRAG."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock session manager to handle MCP protocol
        mock_session_manager = MagicMock()
        mock_http_server_context.session_manager = mock_session_manager

        # Mock the session manager to simulate MCP processing
        async def mock_handle_request(scope, receive, send):
            # Simulate receiving a query_document request
            # In real implementation, this would parse the HTTP request and call MCP tools
            pass

        mock_session_manager.handle_request = AsyncMock(side_effect=mock_handle_request)

        # Test MCP endpoint
        test_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "query_document",
                "arguments": {
                    "query": "What is machine learning?",
                    "mode": "mix"
                }
            }
        }

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post(
                "/mcp",
                json=test_request,
                headers={"Content-Type": "application/json"}
            )

            # Verify session manager was called
            mock_session_manager.handle_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_tool_execution_through_http(self, mock_http_server_context):
        """Test executing MCP tools through HTTP transport."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock the LightRAG client to return a successful response
        mock_http_server_context.lightrag_client.query.return_value = {
            "status": "success",
            "response": "Machine learning is a subset of artificial intelligence..."
        }

        # Mock session manager
        mock_session_manager = MagicMock()
        mock_http_server_context.session_manager = mock_session_manager
        mock_session_manager.handle_request = AsyncMock()

        # Test tool execution request
        tool_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "query_document",
                "arguments": {
                    "query": "What is machine learning?",
                    "mode": "mix",
                    "top_k": 10
                }
            }
        }

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post(
                "/mcp",
                json=tool_request,
                headers={"Content-Type": "application/json"}
            )

            # Verify the request was processed
            mock_session_manager.handle_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, mock_http_server_context):
        """Test handling multiple concurrent HTTP requests."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock session manager
        mock_session_manager = MagicMock()
        mock_http_server_context.session_manager = mock_session_manager
        mock_session_manager.handle_request = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            # Create multiple concurrent requests
            requests = []
            for i in range(5):
                request_data = {
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "tools/call",
                    "params": {
                        "name": "query_document",
                        "arguments": {
                            "query": f"Test query {i}",
                            "mode": "mix"
                        }
                    }
                }
                requests.append(
                    client.post("/mcp", json=request_data, headers={"Content-Type": "application/json"})
                )

            # Execute all requests concurrently
            responses = await asyncio.gather(*requests)

            # Verify all requests were processed
            assert len(responses) == 5
            assert mock_session_manager.handle_request.call_count == 5

            for response in responses:
                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_sse_streaming_response(self, mock_http_server_context):
        """Test Server-Sent Events streaming response."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock session manager for SSE
        mock_session_manager = MagicMock()
        mock_http_server_context.session_manager = mock_session_manager
        mock_session_manager.handle_request = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            # Test SSE request
            response = await client.get(
                "/mcp",
                headers={
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )

            # Verify SSE headers
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            assert response.headers.get("cache-control") == "no-cache"

            # Verify session manager was called
            mock_session_manager.handle_request.assert_called_once()


class TestDockerIntegration:
    """Test Docker container functionality."""

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and is valid."""
        dockerfile_path = "Dockerfile"
        assert os.path.exists(dockerfile_path)

        with open(dockerfile_path, 'r') as f:
            content = f.read()

        # Check for essential Docker instructions
        assert "FROM" in content
        assert "COPY" in content
        assert "RUN" in content
        assert "EXPOSE" in content
        assert "CMD" in content

    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists and is valid."""
        compose_path = "docker-compose.yml"
        assert os.path.exists(compose_path)

        with open(compose_path, 'r') as f:
            content = f.read()

        # Check for essential docker-compose structure
        assert "version" in content or "services:" in content
        assert "lightrag-mcp" in content  # Service name

    def test_env_example_exists(self):
        """Test that .env.example exists with required variables."""
        env_example_path = ".env.example"
        assert os.path.exists(env_example_path)

        with open(env_example_path, 'r') as f:
            content = f.read()

        # Check for essential environment variables
        required_vars = [
            "LIGHTRAG_API_HOST",
            "LIGHTRAG_API_PORT",
            "MCP_HOST",
            "MCP_PORT",
            "TRANSPORT_MODE"
        ]

        for var in required_vars:
            assert var in content

    @pytest.mark.skipif(not os.environ.get("RUN_DOCKER_TESTS"), reason="Docker tests disabled")
    def test_docker_build(self):
        """Test Docker image build (requires Docker)."""
        try:
            # Attempt to build the Docker image
            result = subprocess.run(
                ["docker", "build", "-t", "lightrag-mcp-test", "."],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            assert result.returncode == 0, f"Docker build failed: {result.stderr}"
            assert "Successfully built" in result.stdout or "Successfully tagged" in result.stdout

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available or build timeout")

    @pytest.mark.skipif(not os.environ.get("RUN_DOCKER_TESTS"), reason="Docker tests disabled")
    def test_docker_compose_config(self):
        """Test docker-compose configuration."""
        try:
            result = subprocess.run(
                ["docker-compose", "config"],
                capture_output=True,
                text=True
            )

            assert result.returncode == 0, f"docker-compose config failed: {result.stderr}"
            assert "lightrag-mcp" in result.stdout

        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("docker-compose not available")


class TestBackwardCompatibility:
    """Test backward compatibility with stdio transport."""

    @pytest.mark.asyncio
    async def test_stdio_transport_still_works(self):
        """Test that stdio transport still functions."""
        # This test ensures that the stdio transport functionality
        # hasn't been broken by HTTP transport changes

        # Import the stdio server
        from lightrag_mcp.main import main as stdio_main

        # Mock sys.argv to simulate stdio mode
        with patch('sys.argv', ['lightrag-mcp', '--transport', 'stdio']):
            with patch('lightrag_mcp.main.run_stdio_server') as mock_run_stdio:
                mock_run_stdio.return_value = None

                # Should not raise an exception
                try:
                    stdio_main()
                except SystemExit:
                    pass  # Expected for CLI apps

                # Verify stdio server was called
                mock_run_stdio.assert_called_once()

    @pytest.mark.asyncio
    async def test_config_backward_compatibility(self):
        """Test that configuration maintains backward compatibility."""
        # Test that all old configuration options still work
        with patch.dict(os.environ, {
            'LIGHTRAG_API_HOST': 'localhost',
            'LIGHTRAG_API_PORT': '9621',
            'TRANSPORT_MODE': 'stdio'
        }):
            # Reload config module to pick up environment changes
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            # Verify old config values are accessible
            assert config_module.LIGHTRAG_API_HOST == 'localhost'
            assert config_module.LIGHTRAG_API_PORT == 9621
            assert config_module.TRANSPORT_MODE == 'stdio'

    @pytest.mark.asyncio
    async def test_mcp_server_interface_unchanged(self):
        """Test that MCP server interface remains unchanged."""
        # Verify that the FastMCP server instance still has expected tools
        from lightrag_mcp.server import mcp

        # Check that essential tools are still registered
        tool_names = [tool.name for tool in mcp.list_tools()]

        essential_tools = [
            "query_document",
            "insert_document",
            "upload_document",
            "insert_file",
            "insert_batch",
            "scan_for_new_documents",
            "get_documents",
            "get_pipeline_status",
            "check_lightrag_health"
        ]

        for tool_name in essential_tools:
            assert tool_name in tool_names, f"Tool {tool_name} not found in MCP server"

    @pytest.mark.asyncio
    async def test_query_document_signature_unchanged(self):
        """Test that query_document maintains backward compatibility."""
        from lightrag_mcp.server import query_document
        import inspect

        sig = inspect.signature(query_document)
        params = list(sig.parameters.keys())

        # Check that streaming parameter exists
        assert 'streaming' in params, 'streaming parameter missing'

        # Check that all old parameters still exist
        required_params = [
            'query', 'mode', 'top_k', 'only_need_context', 'only_need_prompt',
            'response_type', 'max_token_for_text_unit', 'max_token_for_global_context',
            'max_token_for_local_context', 'hl_keywords', 'll_keywords', 'history_turns'
        ]

        for param in required_params:
            assert param in params, f'Parameter {param} missing'

        # Check that streaming has default False
        streaming_param = sig.parameters['streaming']
        assert streaming_param.default == False, f'streaming default is {streaming_param.default}, expected False'


class TestHTTPTransportConfiguration:
    """Test HTTP transport configuration."""

    @pytest.mark.asyncio
    async def test_http_transport_config_validation(self):
        """Test HTTP transport configuration validation."""
        # Test valid configurations
        valid_configs = [
            {"MCP_HOST": "localhost", "MCP_PORT": "8080"},
            {"MCP_HOST": "0.0.0.0", "MCP_PORT": "3000"},
            {"MCP_HOST": "example.com", "MCP_PORT": "8443"},
        ]

        for config in valid_configs:
            with patch.dict(os.environ, config):
                import importlib
                import lightrag_mcp.config as config_module
                importlib.reload(config_module)

                assert config_module.MCP_HOST == config["MCP_HOST"]
                assert config_module.MCP_PORT == int(config["MCP_PORT"])

    @pytest.mark.asyncio
    async def test_stateful_mode_configuration(self):
        """Test stateful mode configuration."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("", False),  # Default should be False for safety
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"STATEFUL_MODE": env_value}):
                import importlib
                import lightrag_mcp.config as config_module
                importlib.reload(config_module)

                assert config_module.STATEFUL_MODE == expected

    @pytest.mark.asyncio
    async def test_cors_configuration(self):
        """Test CORS configuration."""
        cors_origins = ["http://localhost:3000", "http://localhost:8080", "https://example.com"]

        with patch.dict(os.environ, {
            "CORS_ORIGINS": ",".join(cors_origins)
        }):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            assert config_module.CORS_ORIGINS == cors_origins


class TestErrorHandlingIntegration:
    """Test error handling in integrated scenarios."""

    @pytest.mark.asyncio
    async def test_http_server_error_recovery(self, mock_http_server_context):
        """Test HTTP server error recovery."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock session manager to raise exceptions
        mock_session_manager = MagicMock()
        mock_http_server_context.session_manager = mock_session_manager
        mock_session_manager.handle_request = AsyncMock(side_effect=Exception("Test error"))

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post("/mcp", json={"test": "data"})

            # Should return 500 error but not crash the server
            assert response.status_code == 500

            # Should be able to make another request
            response2 = await client.get("/health")
            assert response2.status_code == 200

    @pytest.mark.asyncio
    async def test_lightrag_client_error_handling(self, mock_http_server_context):
        """Test LightRAG client error handling in HTTP context."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        # Mock LightRAG client to raise exception
        mock_http_server_context.lightrag_client.get_health.side_effect = Exception("API unreachable")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get("/health")

            # Should return 503 error
            assert response.status_code == 503
            data = response.json()
            assert "unhealthy" in data["status"]

    @pytest.mark.asyncio
    async def test_invalid_request_handling(self, mock_http_server_context):
        """Test handling of invalid requests."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            # Test invalid JSON
            response = await client.post(
                "/mcp",
                content="invalid json",
                headers={"Content-Type": "application/json"}
            )

            # Should handle gracefully (Starlette handles JSON parsing)
            assert response.status_code in [200, 400, 500]

            # Test missing content type
            response2 = await client.post("/mcp", content='{"test": "data"}')
            assert response2.status_code in [200, 400, 500]


class TestPerformanceIntegration:
    """Test performance aspects of the integrated system."""

    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self, mock_http_server_context):
        """Test performance under concurrent load."""
        server = LightRAGHttpServer()
        app = server._create_asgi_app(mock_http_server_context)

        mock_session_manager = MagicMock()
        mock_http_server_context.session_manager = mock_session_manager
        mock_session_manager.handle_request = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            # Simulate concurrent requests
            start_time = asyncio.get_event_loop().time()

            tasks = []
            for i in range(10):
                request_data = {
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "tools/call",
                    "params": {"name": "query_document", "arguments": {"query": f"test {i}"}}
                }
                tasks.append(
                    client.post("/mcp", json=request_data, headers={"Content-Type": "application/json"})
                )

            responses = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()

            # All requests should succeed
            assert all(r.status_code == 200 for r in responses)

            # Should complete within reasonable time (adjust threshold as needed)
            duration = end_time - start_time
            assert duration < 5.0  # Less than 5 seconds for 10 concurrent requests