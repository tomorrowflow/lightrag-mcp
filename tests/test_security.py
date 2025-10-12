"""
Tests for security functionality.

This module tests the security middleware and utilities including:
- Authentication middleware tests (API key and Bearer token)
- Rate limiting tests
- CORS configuration tests
- Security headers tests
- Request validation tests
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse

from lightrag_mcp.security import (
    AuthenticationMiddleware, SecurityHeadersMiddleware, RateLimitMiddleware,
    RequestSizeLimitMiddleware, RequestLoggingMiddleware, TimeoutMiddleware,
    RateLimiter
)
from lightrag_mcp.config import (
    ENABLE_AUTH, API_KEY, BEARER_TOKEN_SECRET, RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS, MAX_REQUEST_SIZE_MB, ENABLE_SECURITY_HEADERS,
    ENVIRONMENT, REQUEST_TIMEOUT_SECONDS, ENABLE_REQUEST_LOGGING,
    ENABLE_ERROR_LOGGING
)


class TestRateLimiter:
    """Test RateLimiter functionality."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 60
        assert len(limiter.clients) == 0

    def test_rate_limiter_first_request(self):
        """Test first request from a client."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        allowed, retry_after = limiter.is_allowed(MagicMock())
        assert allowed is True
        assert retry_after is None
        assert len(limiter.clients) == 1

    def test_rate_limiter_within_limit(self):
        """Test requests within rate limit."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"

        # Make requests within limit
        for i in range(3):
            allowed, retry_after = limiter.is_allowed(mock_request)
            assert allowed is True
            assert retry_after is None

        assert limiter.clients["127.0.0.1"].requests == 3

    def test_rate_limiter_exceeds_limit(self):
        """Test exceeding rate limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"

        # Use up the limit
        for i in range(2):
            allowed, retry_after = limiter.is_allowed(mock_request)
            assert allowed is True

        # Next request should be blocked
        allowed, retry_after = limiter.is_allowed(mock_request)
        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0

    def test_rate_limiter_window_reset(self):
        """Test rate limit window reset."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)  # Short window for testing
        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"

        # Use up the limit
        for i in range(2):
            limiter.is_allowed(mock_request)

        # Wait for window to expire
        time.sleep(1.1)

        # Should allow new requests
        allowed, retry_after = limiter.is_allowed(mock_request)
        assert allowed is True
        assert retry_after is None

    def test_rate_limiter_cleanup(self):
        """Test cleanup of expired entries."""
        limiter = RateLimiter(max_requests=10, window_seconds=1)

        # Add some entries
        for i in range(50):  # Force cleanup threshold
            mock_request = MagicMock()
            mock_request.client.host = f"client{i}"
            limiter.is_allowed(mock_request)

        initial_count = len(limiter.clients)
        assert initial_count > 40  # Should have many entries

        # Wait for cleanup
        time.sleep(1.1)

        # Next request should trigger cleanup
        mock_request = MagicMock()
        mock_request.client.host = "new_client"
        limiter.is_allowed(mock_request)

        # Should have cleaned up old entries
        assert len(limiter.clients) < initial_count


class TestAuthenticationMiddleware:
    """Test AuthenticationMiddleware functionality."""

    @pytest.mark.asyncio
    async def test_auth_disabled(self):
        """Test middleware when auth is disabled."""
        with patch('lightrag_mcp.security.ENABLE_AUTH', False):
            middleware = AuthenticationMiddleware(MagicMock())
            mock_request = MagicMock()

            # Should pass through without authentication
            result = await middleware.dispatch(mock_request, MagicMock(return_value=MagicMock()))
            assert result is not None

    @pytest.mark.asyncio
    async def test_api_key_auth_success(self):
        """Test successful API key authentication."""
        with patch('lightrag_mcp.security.ENABLE_AUTH', True), \
             patch('lightrag_mcp.security.API_KEY', 'test-key'):

            middleware = AuthenticationMiddleware(MagicMock())
            mock_request = MagicMock()
            mock_request.headers = {"X-API-Key": "test-key"}
            mock_request.url.path = "/mcp"

            mock_call_next = AsyncMock(return_value=MagicMock())

            result = await middleware.dispatch(mock_request, mock_call_next)
            mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_api_key_auth_missing(self):
        """Test missing API key."""
        with patch('lightrag_mcp.security.ENABLE_AUTH', True), \
             patch('lightrag_mcp.security.API_KEY', 'test-key'):

            middleware = AuthenticationMiddleware(MagicMock())
            mock_request = MagicMock()
            mock_request.headers = {}
            mock_request.url.path = "/mcp"

            mock_call_next = AsyncMock()

            result = await middleware.dispatch(mock_request, mock_call_next)

            # Should return 401 response
            assert hasattr(result, 'status_code')
            assert result.status_code == 401
            assert "WWW-Authenticate" in result.headers

    @pytest.mark.asyncio
    async def test_api_key_auth_invalid(self):
        """Test invalid API key."""
        with patch('lightrag_mcp.security.ENABLE_AUTH', True), \
             patch('lightrag_mcp.security.API_KEY', 'test-key'):

            middleware = AuthenticationMiddleware(MagicMock())
            mock_request = MagicMock()
            mock_request.headers = {"X-API-Key": "wrong-key"}
            mock_request.url.path = "/mcp"

            mock_call_next = AsyncMock()

            result = await middleware.dispatch(mock_request, mock_call_next)
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_bearer_token_auth_success(self):
        """Test successful Bearer token authentication."""
        with patch('lightrag_mcp.security.ENABLE_AUTH', True), \
             patch('lightrag_mcp.security.BEARER_TOKEN_SECRET', 'secret'):

            middleware = AuthenticationMiddleware(MagicMock())
            mock_request = MagicMock()
            mock_request.headers = {"Authorization": "Bearer test-token"}
            mock_request.url.path = "/mcp"

            # Mock the token verification
            with patch.object(middleware, '_verify_bearer_token', return_value=True):
                mock_call_next = AsyncMock(return_value=MagicMock())

                result = await middleware.dispatch(mock_request, mock_call_next)
                mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_bearer_token_auth_invalid(self):
        """Test invalid Bearer token."""
        with patch('lightrag_mcp.security.ENABLE_AUTH', True), \
             patch('lightrag_mcp.security.BEARER_TOKEN_SECRET', 'secret'):

            middleware = AuthenticationMiddleware(MagicMock())
            mock_request = MagicMock()
            mock_request.headers = {"Authorization": "Bearer invalid-token"}
            mock_request.url.path = "/mcp"

            mock_call_next = AsyncMock()

            result = await middleware.dispatch(mock_request, mock_call_next)
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_health_endpoint_skip_auth(self):
        """Test that health endpoint skips authentication."""
        with patch('lightrag_mcp.security.ENABLE_AUTH', True), \
             patch('lightrag_mcp.security.API_KEY', 'test-key'):

            middleware = AuthenticationMiddleware(MagicMock())
            mock_request = MagicMock()
            mock_request.headers = {}  # No auth headers
            mock_request.url.path = "/health"

            mock_call_next = AsyncMock(return_value=MagicMock())

            # Should skip auth for health endpoint
            result = await middleware.dispatch(mock_request, mock_call_next)
            mock_call_next.assert_called_once_with(mock_request)

    def test_verify_bearer_token_success(self):
        """Test Bearer token verification."""
        with patch('lightrag_mcp.security.BEARER_TOKEN_SECRET', 'secret'):
            middleware = AuthenticationMiddleware(MagicMock())

            # Mock hmac.compare_digest to return True
            with patch('hmac.compare_digest', return_value=True):
                result = middleware._verify_bearer_token("test-token")
                assert result is True

    def test_verify_bearer_token_failure(self):
        """Test Bearer token verification failure."""
        with patch('lightrag_mcp.security.BEARER_TOKEN_SECRET', 'secret'):
            middleware = AuthenticationMiddleware(MagicMock())

            # Mock hmac.compare_digest to return False
            with patch('hmac.compare_digest', return_value=False):
                result = middleware._verify_bearer_token("wrong-token")
                assert result is False


class TestSecurityHeadersMiddleware:
    """Test SecurityHeadersMiddleware functionality."""

    @pytest.mark.asyncio
    async def test_security_headers_disabled(self):
        """Test middleware when security headers are disabled."""
        with patch('lightrag_mcp.security.ENABLE_SECURITY_HEADERS', False):
            middleware = SecurityHeadersMiddleware(MagicMock())
            mock_request = MagicMock()
            mock_response = MagicMock()

            mock_call_next = AsyncMock(return_value=mock_response)

            result = await middleware.dispatch(mock_request, mock_call_next)
            assert result == mock_response
            # Should not add any headers
            assert len(mock_response.headers) == 0

    @pytest.mark.asyncio
    async def test_security_headers_production(self):
        """Test security headers in production environment."""
        with patch('lightrag_mcp.security.ENABLE_SECURITY_HEADERS', True), \
             patch('lightrag_mcp.security.ENVIRONMENT', 'production'):

            middleware = SecurityHeadersMiddleware(MagicMock())
            mock_request = MagicMock()
            mock_response = MagicMock()
            mock_response.headers = {}

            mock_call_next = AsyncMock(return_value=mock_response)

            result = await middleware.dispatch(mock_request, mock_call_next)

            # Check production security headers
            assert "Content-Security-Policy" in mock_response.headers
            assert "Strict-Transport-Security" in mock_response.headers
            assert "X-Frame-Options" in mock_response.headers
            assert "X-Content-Type-Options" in mock_response.headers
            assert "Referrer-Policy" in mock_response.headers
            assert "X-XSS-Protection" in mock_response.headers

    @pytest.mark.asyncio
    async def test_security_headers_development(self):
        """Test security headers in development environment."""
        with patch('lightrag_mcp.security.ENABLE_SECURITY_HEADERS', True), \
             patch('lightrag_mcp.security.ENVIRONMENT', 'development'):

            middleware = SecurityHeadersMiddleware(MagicMock())
            mock_request = MagicMock()
            mock_response = MagicMock()
            mock_response.headers = {}

            mock_call_next = AsyncMock(return_value=mock_response)

            result = await middleware.dispatch(mock_request, mock_call_next)

            # Check development security headers (relaxed CSP)
            assert "Content-Security-Policy" in mock_response.headers
            csp = mock_response.headers["Content-Security-Policy"]
            assert "'unsafe-inline'" in csp
            assert "'unsafe-eval'" in csp

            # Should not have HSTS in development
            assert "Strict-Transport-Security" not in mock_response.headers


class TestRateLimitMiddleware:
    """Test RateLimitMiddleware functionality."""

    @pytest.mark.asyncio
    async def test_rate_limit_allowed(self):
        """Test allowed request in rate limiting."""
        middleware = RateLimitMiddleware(MagicMock())
        mock_request = MagicMock()
        mock_request.url.path = "/mcp"
        mock_request.client.host = "127.0.0.1"

        mock_call_next = AsyncMock(return_value=MagicMock())

        result = await middleware.dispatch(mock_request, mock_call_next)
        mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Test rate limit exceeded."""
        middleware = RateLimitMiddleware(MagicMock())
        mock_request = MagicMock()
        mock_request.url.path = "/mcp"
        mock_request.client.host = "127.0.0.1"

        # Exhaust the rate limit
        limiter = middleware.rate_limiter
        for _ in range(limiter.max_requests):
            limiter.is_allowed(mock_request)

        mock_call_next = AsyncMock()

        result = await middleware.dispatch(mock_request, mock_call_next)

        # Should return 429 response
        assert result.status_code == 429
        assert "Retry-After" in result.headers

    @pytest.mark.asyncio
    async def test_rate_limit_skip_health_endpoints(self):
        """Test that health endpoints skip rate limiting."""
        middleware = RateLimitMiddleware(MagicMock())
        mock_request = MagicMock()
        mock_request.url.path = "/health"
        mock_request.client.host = "127.0.0.1"

        mock_call_next = AsyncMock(return_value=MagicMock())

        # Should not check rate limit for health endpoint
        result = await middleware.dispatch(mock_request, mock_call_next)
        mock_call_next.assert_called_once_with(mock_request)


class TestRequestSizeLimitMiddleware:
    """Test RequestSizeLimitMiddleware functionality."""

    @pytest.mark.asyncio
    async def test_request_size_allowed(self):
        """Test request within size limit."""
        middleware = RequestSizeLimitMiddleware(MagicMock())
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/mcp"
        mock_request.headers = {"Content-Length": "1024"}  # 1KB

        mock_call_next = AsyncMock(return_value=MagicMock())

        result = await middleware.dispatch(mock_request, mock_call_next)
        mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_request_size_exceeded(self):
        """Test request exceeding size limit."""
        middleware = RequestSizeLimitMiddleware(MagicMock())
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/mcp"
        mock_request.headers = {"Content-Length": str(100 * 1024 * 1024)}  # 100MB

        mock_call_next = AsyncMock()

        result = await middleware.dispatch(mock_request, mock_call_next)

        # Should return 413 response
        assert result.status_code == 413
        assert "Maximum size" in result.body.decode()

    @pytest.mark.asyncio
    async def test_request_size_skip_get_requests(self):
        """Test that GET requests skip size checking."""
        middleware = RequestSizeLimitMiddleware(MagicMock())
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/mcp"
        mock_request.headers = {"Content-Length": str(100 * 1024 * 1024)}

        mock_call_next = AsyncMock(return_value=MagicMock())

        result = await middleware.dispatch(mock_request, mock_call_next)
        mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_request_size_skip_health_endpoints(self):
        """Test that health endpoints skip size checking."""
        middleware = RequestSizeLimitMiddleware(MagicMock())
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/health"
        mock_request.headers = {"Content-Length": str(100 * 1024 * 1024)}

        mock_call_next = AsyncMock(return_value=MagicMock())

        result = await middleware.dispatch(mock_request, mock_call_next)
        mock_call_next.assert_called_once_with(mock_request)


class TestRequestLoggingMiddleware:
    """Test RequestLoggingMiddleware functionality."""

    @pytest.mark.asyncio
    async def test_request_logging_disabled(self):
        """Test middleware when logging is disabled."""
        with patch('lightrag_mcp.security.ENABLE_REQUEST_LOGGING', False):
            middleware = RequestLoggingMiddleware(MagicMock())
            mock_request = MagicMock()
            mock_response = MagicMock()

            mock_call_next = AsyncMock(return_value=mock_response)

            result = await middleware.dispatch(mock_request, mock_call_next)
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_request_logging_enabled(self):
        """Test request logging when enabled."""
        with patch('lightrag_mcp.security.ENABLE_REQUEST_LOGGING', True):
            middleware = RequestLoggingMiddleware(MagicMock())
            mock_request = MagicMock()
            mock_request.method = "POST"
            mock_request.url.path = "/mcp"
            mock_request.client = MagicMock()
            mock_request.client.host = "127.0.0.1"
            mock_request.headers = {"User-Agent": "TestClient/1.0"}

            mock_response = MagicMock()
            mock_response.status_code = 200

            mock_call_next = AsyncMock(return_value=mock_response)

            with patch('lightrag_mcp.security.logger') as mock_logger:
                result = await middleware.dispatch(mock_request, mock_call_next)

                # Should log both request and response
                assert mock_logger.info.call_count >= 2
                assert result == mock_response

    @pytest.mark.asyncio
    async def test_request_logging_error(self):
        """Test error logging in request middleware."""
        with patch('lightrag_mcp.security.ENABLE_REQUEST_LOGGING', True), \
             patch('lightrag_mcp.security.ENABLE_ERROR_LOGGING', True):

            middleware = RequestLoggingMiddleware(MagicMock())
            mock_request = MagicMock()
            mock_request.method = "POST"
            mock_request.url.path = "/mcp"

            mock_call_next = AsyncMock(side_effect=Exception("Test error"))

            with patch('lightrag_mcp.security.logger') as mock_logger:
                with pytest.raises(Exception):
                    await middleware.dispatch(mock_request, mock_call_next)

                # Should log the error
                mock_logger.error.assert_called()


class TestTimeoutMiddleware:
    """Test TimeoutMiddleware functionality."""

    @pytest.mark.asyncio
    async def test_request_within_timeout(self):
        """Test request completing within timeout."""
        middleware = TimeoutMiddleware(MagicMock())
        mock_request = MagicMock()
        mock_response = MagicMock()

        mock_call_next = AsyncMock(return_value=mock_response)

        result = await middleware.dispatch(mock_request, mock_call_next)
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_request_timeout(self):
        """Test request timeout."""
        middleware = TimeoutMiddleware(MagicMock())
        middleware.timeout = 0.1  # Very short timeout

        mock_request = MagicMock()

        # Mock call_next to take longer than timeout
        async def slow_call_next(request):
            await asyncio.sleep(1.0)  # Longer than timeout
            return MagicMock()

        mock_call_next = AsyncMock(side_effect=slow_call_next)

        result = await middleware.dispatch(mock_request, mock_call_next)

        # Should return 408 timeout response
        assert result.status_code == 408
        assert "timeout" in result.body.decode().lower()


class TestMiddlewareIntegration:
    """Test middleware integration and ordering."""

    @pytest.mark.asyncio
    async def test_middleware_order_in_app(self):
        """Test that middleware is added in correct order."""
        from lightrag_mcp.http_server import LightRAGHttpServer
        from lightrag_mcp.event_store import create_event_store
        from lightrag_mcp.lightrag_client import LightRAGClient

        # Create a minimal server setup
        server = LightRAGHttpServer()
        mock_client = AsyncMock(spec=LightRAGClient)
        mock_event_store = create_event_store("memory")

        context = MagicMock()
        context.lightrag_client = mock_client
        context.event_store = mock_event_store
        context.session_manager = None

        app = server._create_asgi_app(context)

        # Check that middleware is present
        assert len(app.user_middleware) > 0

        # Verify middleware types are present
        middleware_classes = [str(m.cls) if hasattr(m, 'cls') else str(type(m)) for m in app.user_middleware]

        # Should have authentication, rate limiting, size limits, logging, timeout, security headers, CORS
        assert any('AuthenticationMiddleware' in cls for cls in middleware_classes)
        assert any('RateLimitMiddleware' in cls for cls in middleware_classes)
        assert any('RequestSizeLimitMiddleware' in cls for cls in middleware_classes)
        assert any('RequestLoggingMiddleware' in cls for cls in middleware_classes)
        assert any('TimeoutMiddleware' in cls for cls in middleware_classes)
        assert any('SecurityHeadersMiddleware' in cls for cls in middleware_classes)
        assert any('CORSMiddleware' in cls for cls in middleware_classes)

    @pytest.mark.asyncio
    async def test_cors_middleware_configuration(self):
        """Test CORS middleware configuration."""
        from lightrag_mcp.http_server import LightRAGHttpServer

        server = LightRAGHttpServer()
        mock_client = AsyncMock()
        mock_event_store = MagicMock()

        context = MagicMock()
        context.lightrag_client = mock_client
        context.event_store = mock_event_store
        context.session_manager = None

        app = server._create_asgi_app(context)

        # Find CORS middleware
        cors_middleware = None
        for middleware in app.user_middleware:
            if hasattr(middleware, 'cls') and 'CORSMiddleware' in str(middleware.cls):
                cors_middleware = middleware
                break

        assert cors_middleware is not None

        # Check CORS options
        options = cors_middleware.options
        assert 'allow_origins' in options
        assert 'allow_credentials' in options
        assert 'allow_methods' in options
        assert 'allow_headers' in options