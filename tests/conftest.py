"""
Test configuration and fixtures for LightRAG MCP tests.
"""

import asyncio
import os
import tempfile
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient
from starlette.testclient import TestClient

from lightrag_mcp.config import (
    DEFAULT_HOST, DEFAULT_PORT, DEFAULT_MCP_HOST, DEFAULT_MCP_PORT,
    DEFAULT_STATEFUL_MODE, DEFAULT_EVENT_RETENTION_HOURS,
    DEFAULT_MAX_EVENTS_PER_SESSION, DEFAULT_CORS_ORIGINS,
    DEFAULT_ENABLE_AUTH, DEFAULT_API_KEY, DEFAULT_BEARER_TOKEN_SECRET,
    DEFAULT_RATE_LIMIT_REQUESTS, DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    DEFAULT_MAX_REQUEST_SIZE_MB, DEFAULT_ENABLE_SECURITY_HEADERS,
    DEFAULT_ENVIRONMENT, DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_ENABLE_REQUEST_LOGGING, DEFAULT_ENABLE_ERROR_LOGGING
)
from lightrag_mcp.event_store import create_event_store
from lightrag_mcp.http_server import LightRAGHttpServer, HTTPServerContext
from lightrag_mcp.lightrag_client import LightRAGClient
from lightrag_mcp.security import RateLimiter


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_lightrag_client():
    """Mock LightRAG client for testing."""
    client = AsyncMock(spec=LightRAGClient)

    # Mock common methods
    client.get_health.return_value = {"status": "healthy", "version": "1.0.0"}
    client.query.return_value = {
        "response": "Mock response",
        "context": [],
        "entities": [],
        "relationships": []
    }
    client.insert_text.return_value = {"status": "success", "documents": ["doc1"]}
    client.close.return_value = None

    return client


@pytest.fixture
def in_memory_event_store():
    """Create an in-memory event store for testing."""
    return create_event_store("memory", max_events_per_session=100)


@pytest.fixture
def file_event_store(temp_dir):
    """Create a file-based event store for testing."""
    storage_path = os.path.join(temp_dir, "event_store")
    return create_event_store("file", storage_path=storage_path, max_events_per_session=100)


@pytest.fixture
def rate_limiter():
    """Create a rate limiter for testing."""
    return RateLimiter(max_requests=10, window_seconds=60)


@pytest.fixture
def test_config():
    """Test configuration dictionary."""
    return {
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "mcp_host": DEFAULT_MCP_HOST,
        "mcp_port": DEFAULT_MCP_PORT,
        "stateful_mode": DEFAULT_STATEFUL_MODE,
        "event_retention_hours": DEFAULT_EVENT_RETENTION_HOURS,
        "max_events_per_session": DEFAULT_MAX_EVENTS_PER_SESSION,
        "cors_origins": DEFAULT_CORS_ORIGINS,
        "enable_auth": DEFAULT_ENABLE_AUTH,
        "api_key": DEFAULT_API_KEY,
        "bearer_token_secret": DEFAULT_BEARER_TOKEN_SECRET,
        "rate_limit_requests": DEFAULT_RATE_LIMIT_REQUESTS,
        "rate_limit_window_seconds": DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
        "max_request_size_mb": DEFAULT_MAX_REQUEST_SIZE_MB,
        "enable_security_headers": DEFAULT_ENABLE_SECURITY_HEADERS,
        "environment": DEFAULT_ENVIRONMENT,
        "request_timeout_seconds": DEFAULT_REQUEST_TIMEOUT_SECONDS,
        "enable_request_logging": DEFAULT_ENABLE_REQUEST_LOGGING,
        "enable_error_logging": DEFAULT_ENABLE_ERROR_LOGGING,
    }


@pytest.fixture
async def mock_http_server_context(mock_lightrag_client, in_memory_event_store):
    """Create a mock HTTP server context."""
    context = HTTPServerContext(
        lightrag_client=mock_lightrag_client,
        event_store=in_memory_event_store,
        session_manager=None
    )
    return context


@pytest.fixture
def sample_mcp_message():
    """Sample MCP JSON-RPC message."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "query_document",
            "arguments": {
                "query": "test query",
                "mode": "mix"
            }
        }
    }


@pytest.fixture
def sample_sse_event():
    """Sample Server-Sent Event data."""
    return {
        "event": "message",
        "data": '{"jsonrpc": "2.0", "id": 1, "result": {"status": "success"}}',
        "id": "event-123"
    }


@pytest.fixture
def sample_stored_event(sample_mcp_message):
    """Sample stored event for testing."""
    from lightrag_mcp.event_store import StoredEvent
    from datetime import datetime

    return StoredEvent(
        event_id="test-event-123",
        session_id="test-session-456",
        message=sample_mcp_message,
        timestamp=datetime.utcnow(),
        stream_id="test-stream"
    )


@pytest.fixture
def test_request_data():
    """Test request data for HTTP endpoints."""
    return {
        "jsonrpc": "2.0",
        "id": "test-123",
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


@pytest.fixture
def test_headers():
    """Test headers for HTTP requests."""
    return {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    }


@pytest.fixture
def auth_headers():
    """Authentication headers for testing."""
    return {
        "X-API-Key": "test-api-key-123",
        "Authorization": "Bearer test-token-456"
    }


@pytest.fixture
def cors_headers():
    """CORS-related headers for testing."""
    return {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,authorization"
    }


@pytest.fixture
async def async_client():
    """Async HTTP client fixture using httpx."""
    async with AsyncClient() as client:
        yield client


@pytest.fixture
def sync_client():
    """Synchronous HTTP client fixture using requests-like interface."""
    # This would be replaced with actual HTTP client in real tests
    return MagicMock()


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up temporary files after each test."""
    yield
    # Cleanup logic would go here if needed


@pytest.fixture
def mock_environment(monkeypatch):
    """Mock environment variables for testing."""
    env_vars = {
        "LIGHTRAG_API_HOST": "localhost",
        "LIGHTRAG_API_PORT": "9621",
        "LIGHTRAG_API_KEY": "test-key",
        "MCP_HOST": "localhost",
        "MCP_PORT": "8080",
        "TRANSPORT_MODE": "http",
        "STATEFUL_MODE": "true",
        "EVENT_RETENTION_HOURS": "24",
        "MAX_EVENTS_PER_SESSION": "1000",
        "CORS_ORIGINS": '["http://localhost:3000", "http://localhost:8080"]',
        "ENABLE_AUTH": "true",
        "API_KEY": "test-api-key",
        "BEARER_TOKEN_SECRET": "test-secret",
        "RATE_LIMIT_REQUESTS": "100",
        "RATE_LIMIT_WINDOW_SECONDS": "60",
        "MAX_REQUEST_SIZE_MB": "10",
        "ENABLE_SECURITY_HEADERS": "true",
        "ENVIRONMENT": "development",
        "REQUEST_TIMEOUT_SECONDS": "30",
        "ENABLE_REQUEST_LOGGING": "true",
        "ENABLE_ERROR_LOGGING": "true",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    yield env_vars


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
async def mock_async_context_manager():
    """Mock async context manager for testing."""
    class MockAsyncContextManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    return MockAsyncContextManager()


@pytest.fixture
def sample_batch_data():
    """Sample batch data for testing batch operations."""
    return [
        {
            "entity_name": "Python",
            "entity_type": "PROGRAMMING_LANGUAGE",
            "description": "A high-level programming language",
            "source_id": "doc1"
        },
        {
            "entity_name": "JavaScript",
            "entity_type": "PROGRAMMING_LANGUAGE",
            "description": "A scripting language for the web",
            "source_id": "doc2"
        },
        {
            "entity_name": "Machine Learning",
            "entity_type": "TECHNOLOGY",
            "description": "A subset of artificial intelligence",
            "source_id": "doc3"
        }
    ]


@pytest.fixture
def sample_relationship_data():
    """Sample relationship data for testing."""
    return [
        {
            "source": "Python",
            "target": "Machine Learning",
            "description": "Python is commonly used for machine learning",
            "keywords": "programming, AI, data science",
            "source_id": "doc1",
            "weight": 0.8
        },
        {
            "source": "JavaScript",
            "target": "Web Development",
            "description": "JavaScript is used for web development",
            "keywords": "frontend, backend, fullstack",
            "source_id": "doc2",
            "weight": 0.9
        }
    ]


@pytest.fixture
def error_responses():
    """Common error responses for testing."""
    return {
        "invalid_request": {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32600,
                "message": "Invalid Request"
            }
        },
        "method_not_found": {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32601,
                "message": "Method not found"
            }
        },
        "internal_error": {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32603,
                "message": "Internal error"
            }
        },
        "rate_limit_exceeded": {
            "error": "Rate limit exceeded",
            "type": "rate_limit_error",
            "retry_after": 60
        },
        "auth_required": {
            "error": "Invalid API key",
            "type": "authentication_error"
        }
    }