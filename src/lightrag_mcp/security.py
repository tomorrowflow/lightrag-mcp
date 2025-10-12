"""
Security middleware and utilities for LightRAG MCP HTTP server.

This module provides authentication, security headers, rate limiting,
and other security features for the HTTP transport implementation.
"""

import hashlib
import hmac
import logging
import time
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from .config import (
    ENABLE_AUTH, API_KEY, BEARER_TOKEN_SECRET, RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS, MAX_REQUEST_SIZE_MB, ENABLE_SECURITY_HEADERS,
    ENVIRONMENT, REQUEST_TIMEOUT_SECONDS, ENABLE_REQUEST_LOGGING,
    ENABLE_ERROR_LOGGING
)

logger = logging.getLogger(__name__)


@dataclass
class RateLimitInfo:
    """Rate limiting information for a client."""
    requests: int
    window_start: float
    blocked_until: Optional[float] = None


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients: Dict[str, RateLimitInfo] = {}

    def _get_client_key(self, request: Request) -> str:
        """Get client identifier from request."""
        # Use IP address as client identifier
        client_ip = request.client.host if request.client else "unknown"
        return client_ip

    def _cleanup_expired_entries(self) -> None:
        """Clean up expired rate limit entries."""
        current_time = time.time()
        expired_keys = []

        for client_key, info in self.clients.items():
            if current_time - info.window_start > self.window_seconds:
                expired_keys.append(client_key)

        for key in expired_keys:
            del self.clients[key]

    def is_allowed(self, request: Request) -> Tuple[bool, Optional[int]]:
        """
        Check if request is allowed under rate limiting.

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: Optional[int])
        """
        client_key = self._get_client_key(request)
        current_time = time.time()

        # Clean up expired entries periodically
        if len(self.clients) > 1000:  # Arbitrary cleanup threshold
            self._cleanup_expired_entries()

        if client_key not in self.clients:
            # First request from this client
            self.clients[client_key] = RateLimitInfo(
                requests=1,
                window_start=current_time
            )
            return True, None

        info = self.clients[client_key]

        # Check if window has expired
        if current_time - info.window_start >= self.window_seconds:
            # Reset window
            info.requests = 1
            info.window_start = current_time
            info.blocked_until = None
            return True, None

        # Check if client is currently blocked
        if info.blocked_until and current_time < info.blocked_until:
            retry_after = int(info.blocked_until - current_time)
            return False, retry_after

        # Check request count
        if info.requests >= self.max_requests:
            # Block client for the window duration
            info.blocked_until = info.window_start + self.window_seconds
            retry_after = int(info.blocked_until - current_time)
            return False, retry_after

        # Allow request and increment counter
        info.requests += 1
        return True, None


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API key and Bearer token authentication."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.api_key = API_KEY
        self.bearer_secret = BEARER_TOKEN_SECRET

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process authentication for incoming requests."""
        if not ENABLE_AUTH:
            return await call_next(request)

        # Skip authentication for health/status endpoints
        if request.url.path in ["/health", "/status", "/"]:
            return await call_next(request)

        # Check API key authentication
        if self.api_key:
            api_key_header = request.headers.get("X-API-Key")
            if not api_key_header or api_key_header != self.api_key:
                return JSONResponse(
                    {"error": "Invalid API key", "type": "authentication_error"},
                    status_code=401,
                    headers={"WWW-Authenticate": "X-API-Key"}
                )

        # Check Bearer token authentication
        if self.bearer_secret:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    {"error": "Bearer token required", "type": "authentication_error"},
                    status_code=401,
                    headers={"WWW-Authenticate": 'Bearer realm="LightRAG MCP"'}
                )

            token = auth_header[7:]  # Remove "Bearer " prefix
            if not self._verify_bearer_token(token):
                return JSONResponse(
                    {"error": "Invalid bearer token", "type": "authentication_error"},
                    status_code=401,
                    headers={"WWW-Authenticate": 'Bearer realm="LightRAG MCP"'}
                )

        return await call_next(request)

    def _verify_bearer_token(self, token: str) -> bool:
        """Verify Bearer token using HMAC."""
        if not self.bearer_secret:
            return False

        try:
            # Simple HMAC verification - in production, use proper JWT or OAuth
            expected = hmac.new(
                self.bearer_secret.encode(),
                "lightrag-mcp-token".encode(),
                hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(token, expected)
        except Exception:
            return False


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Add security headers to the response."""
        response = await call_next(request)

        if not ENABLE_SECURITY_HEADERS:
            return response

        # Content Security Policy
        if ENVIRONMENT == "production":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            )
        else:
            # Relaxed CSP for development
            response.headers["Content-Security-Policy"] = (
                "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https: http:; "
                "font-src 'self' data:; "
                "connect-src 'self' ws: wss: http: https:;"
            )

        # Security headers
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # HSTS for production
        if ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply rate limiting to requests."""
        # Skip rate limiting for health/status endpoints
        if request.url.path in ["/health", "/status", "/"]:
            return await call_next(request)

        allowed, retry_after = self.rate_limiter.is_allowed(request)

        if not allowed:
            response = JSONResponse(
                {
                    "error": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "retry_after": retry_after
                },
                status_code=429
            )
            if retry_after:
                response.headers["Retry-After"] = str(retry_after)
            return response

        return await call_next(request)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.max_size = MAX_REQUEST_SIZE_MB * 1024 * 1024  # Convert MB to bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        """Check request size before processing."""
        # Skip size check for GET requests and health endpoints
        if request.method == "GET" or request.url.path in ["/health", "/status", "/"]:
            return await call_next(request)

        # Check Content-Length header
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size:
                    return JSONResponse(
                        {
                            "error": f"Request too large. Maximum size: {MAX_REQUEST_SIZE_MB}MB",
                            "type": "size_limit_error"
                        },
                        status_code=413
                    )
            except ValueError:
                pass  # Invalid Content-Length, let it pass for now

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Log incoming requests and responses."""
        if not ENABLE_REQUEST_LOGGING:
            return await call_next(request)

        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"

        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {client_ip} - {request.headers.get('User-Agent', 'Unknown')}"
        )

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            logger.info(
                f"Response: {request.method} {request.url.path} "
                f"-> {response.status_code} ({duration:.3f}s)"
            )

            return response

        except Exception as e:
            duration = time.time() - start_time
            if ENABLE_ERROR_LOGGING:
                logger.error(
                    f"Request error: {request.method} {request.url.path} "
                    f"({duration:.3f}s) - {str(e)}"
                )
            raise


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeouts."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.timeout = REQUEST_TIMEOUT_SECONDS

    async def dispatch(self, request: Request, call_next) -> Response:
        """Enforce timeout on request processing."""
        import asyncio

        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                {
                    "error": f"Request timeout after {self.timeout} seconds",
                    "type": "timeout_error"
                },
                status_code=408
            )