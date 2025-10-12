"""
Tests for configuration functionality.

This module tests configuration management including:
- Environment variable configuration tests
- HTTP transport configuration validation
- Security settings tests
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from lightrag_mcp.config import (
    DEFAULT_HOST, DEFAULT_PORT, DEFAULT_MCP_HOST, DEFAULT_MCP_PORT,
    DEFAULT_STATEFUL_MODE, DEFAULT_EVENT_RETENTION_HOURS,
    DEFAULT_MAX_EVENTS_PER_SESSION, DEFAULT_CORS_ORIGINS,
    DEFAULT_ENABLE_AUTH, DEFAULT_API_KEY, DEFAULT_BEARER_TOKEN_SECRET,
    DEFAULT_RATE_LIMIT_REQUESTS, DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    DEFAULT_MAX_REQUEST_SIZE_MB, DEFAULT_ENABLE_SECURITY_HEADERS,
    DEFAULT_ENVIRONMENT, DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_ENABLE_REQUEST_LOGGING, DEFAULT_ENABLE_ERROR_LOGGING,
    parse_args, LIGHTRAG_API_HOST, LIGHTRAG_API_PORT, LIGHTRAG_API_KEY,
    MCP_HOST, MCP_PORT, TRANSPORT_MODE, STATEFUL_MODE,
    EVENT_RETENTION_HOURS, MAX_EVENTS_PER_SESSION, CORS_ORIGINS,
    ENABLE_AUTH, API_KEY, BEARER_TOKEN_SECRET, RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS, MAX_REQUEST_SIZE_MB,
    ENABLE_SECURITY_HEADERS, ENVIRONMENT, REQUEST_TIMEOUT_SECONDS,
    ENABLE_REQUEST_LOGGING, ENABLE_ERROR_LOGGING, CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS, CORS_ALLOW_HEADERS, CORS_EXPOSE_HEADERS,
    CORS_MAX_AGE, LIGHTRAG_API_BASE_URL, MCP_SERVER_URL
)


class TestDefaultConfiguration:
    """Test default configuration values."""

    def test_default_constants(self):
        """Test that default constants are properly defined."""
        assert DEFAULT_HOST == "localhost"
        assert DEFAULT_PORT == 9621
        assert DEFAULT_MCP_HOST == "localhost"
        assert DEFAULT_MCP_PORT == 8080
        assert DEFAULT_TRANSPORT_MODE == "stdio"
        assert DEFAULT_STATEFUL_MODE is True
        assert DEFAULT_EVENT_RETENTION_HOURS == 24
        assert DEFAULT_MAX_EVENTS_PER_SESSION == 1000
        assert DEFAULT_CORS_ORIGINS == ["*"]
        assert DEFAULT_ENABLE_AUTH is False
        assert DEFAULT_API_KEY == ""
        assert DEFAULT_BEARER_TOKEN_SECRET == ""
        assert DEFAULT_RATE_LIMIT_REQUESTS == 100
        assert DEFAULT_RATE_LIMIT_WINDOW_SECONDS == 60
        assert DEFAULT_MAX_REQUEST_SIZE_MB == 10
        assert DEFAULT_ENABLE_SECURITY_HEADERS is True
        assert DEFAULT_ENVIRONMENT == "development"
        assert DEFAULT_REQUEST_TIMEOUT_SECONDS == 30
        assert DEFAULT_ENABLE_REQUEST_LOGGING is True
        assert DEFAULT_ENABLE_ERROR_LOGGING is True
        assert DEFAULT_CORS_ALLOW_CREDENTIALS is True
        assert DEFAULT_CORS_ALLOW_METHODS == ["GET", "POST", "DELETE", "OPTIONS"]
        assert DEFAULT_CORS_ALLOW_HEADERS == ["*"]
        assert DEFAULT_CORS_EXPOSE_HEADERS == ["Mcp-Session-Id"]
        assert DEFAULT_CORS_MAX_AGE == 86400

    def test_derived_urls(self):
        """Test that derived URLs are constructed correctly."""
        # Test with default values
        expected_api_url = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
        expected_mcp_url = f"http://{DEFAULT_MCP_HOST}:{DEFAULT_MCP_PORT}"

        # These should be computed from the actual config values
        assert LIGHTRAG_API_BASE_URL.startswith("http://")
        assert MCP_SERVER_URL.startswith("http://")


class TestEnvironmentVariableConfiguration:
    """Test configuration from environment variables."""

    @pytest.fixture(autouse=True)
    def clean_environment(self):
        """Clean up environment variables before each test."""
        env_vars_to_clean = [
            'LIGHTRAG_API_HOST', 'LIGHTRAG_API_PORT', 'LIGHTRAG_API_KEY',
            'MCP_HOST', 'MCP_PORT', 'TRANSPORT_MODE', 'STATEFUL_MODE',
            'EVENT_RETENTION_HOURS', 'MAX_EVENTS_PER_SESSION', 'CORS_ORIGINS',
            'ENABLE_AUTH', 'API_KEY', 'BEARER_TOKEN_SECRET',
            'RATE_LIMIT_REQUESTS', 'RATE_LIMIT_WINDOW_SECONDS',
            'MAX_REQUEST_SIZE_MB', 'ENABLE_SECURITY_HEADERS', 'ENVIRONMENT',
            'REQUEST_TIMEOUT_SECONDS', 'ENABLE_REQUEST_LOGGING',
            'ENABLE_ERROR_LOGGING', 'CORS_ALLOW_CREDENTIALS',
            'CORS_ALLOW_METHODS', 'CORS_ALLOW_HEADERS',
            'CORS_EXPOSE_HEADERS', 'CORS_MAX_AGE'
        ]

        for var in env_vars_to_clean:
            os.environ.pop(var, None)

        yield

        # Clean up after test
        for var in env_vars_to_clean:
            os.environ.pop(var, None)

    def test_lightrag_api_configuration(self):
        """Test LightRAG API configuration from environment."""
        with patch.dict(os.environ, {
            'LIGHTRAG_API_HOST': 'api.example.com',
            'LIGHTRAG_API_PORT': '8080',
            'LIGHTRAG_API_KEY': 'test-key-123'
        }):
            # Reload the config module to pick up changes
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            assert config_module.LIGHTRAG_API_HOST == 'api.example.com'
            assert config_module.LIGHTRAG_API_PORT == 8080
            assert config_module.LIGHTRAG_API_KEY == 'test-key-123'
            assert config_module.LIGHTRAG_API_BASE_URL == 'http://api.example.com:8080'

    def test_http_transport_configuration(self):
        """Test HTTP transport configuration from environment."""
        with patch.dict(os.environ, {
            'MCP_HOST': 'mcp.example.com',
            'MCP_PORT': '3000',
            'TRANSPORT_MODE': 'http'
        }):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            assert config_module.MCP_HOST == 'mcp.example.com'
            assert config_module.MCP_PORT == 3000
            assert config_module.TRANSPORT_MODE == 'http'
            assert config_module.MCP_SERVER_URL == 'http://mcp.example.com:3000'

    def test_stateful_mode_configuration(self):
        """Test stateful mode configuration."""
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('1', True),
            ('0', False),
            ('', False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {'STATEFUL_MODE': env_value}):
                import importlib
                import lightrag_mcp.config as config_module
                importlib.reload(config_module)

                assert config_module.STATEFUL_MODE == expected

    def test_event_store_configuration(self):
        """Test event store configuration."""
        with patch.dict(os.environ, {
            'EVENT_RETENTION_HOURS': '48',
            'MAX_EVENTS_PER_SESSION': '500'
        }):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            assert config_module.EVENT_RETENTION_HOURS == 48
            assert config_module.MAX_EVENTS_PER_SESSION == 500

    def test_cors_configuration(self):
        """Test CORS configuration."""
        cors_origins = ['http://localhost:3000', 'https://example.com', 'http://app.example.com']

        with patch.dict(os.environ, {
            'CORS_ORIGINS': ','.join(cors_origins),
            'CORS_ALLOW_CREDENTIALS': 'false',
            'CORS_MAX_AGE': '3600'
        }):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            assert config_module.CORS_ORIGINS == cors_origins
            assert config_module.CORS_ALLOW_CREDENTIALS is False
            assert config_module.CORS_MAX_AGE == 3600

    def test_security_configuration(self):
        """Test security configuration."""
        with patch.dict(os.environ, {
            'ENABLE_AUTH': 'true',
            'API_KEY': 'test-api-key',
            'BEARER_TOKEN_SECRET': 'test-secret',
            'RATE_LIMIT_REQUESTS': '50',
            'RATE_LIMIT_WINDOW_SECONDS': '30',
            'MAX_REQUEST_SIZE_MB': '5',
            'ENABLE_SECURITY_HEADERS': 'false',
            'ENVIRONMENT': 'production',
            'REQUEST_TIMEOUT_SECONDS': '60',
            'ENABLE_REQUEST_LOGGING': 'false',
            'ENABLE_ERROR_LOGGING': 'false'
        }):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            assert config_module.ENABLE_AUTH is True
            assert config_module.API_KEY == 'test-api-key'
            assert config_module.BEARER_TOKEN_SECRET == 'test-secret'
            assert config_module.RATE_LIMIT_REQUESTS == 50
            assert config_module.RATE_LIMIT_WINDOW_SECONDS == 30
            assert config_module.MAX_REQUEST_SIZE_MB == 5
            assert config_module.ENABLE_SECURITY_HEADERS is False
            assert config_module.ENVIRONMENT == 'production'
            assert config_module.REQUEST_TIMEOUT_SECONDS == 60
            assert config_module.ENABLE_REQUEST_LOGGING is False
            assert config_module.ENABLE_ERROR_LOGGING is False


class TestCommandLineArgumentConfiguration:
    """Test configuration from command line arguments."""

    def test_parse_args_default(self):
        """Test parsing default command line arguments."""
        args = parse_args()

        assert args.host == DEFAULT_HOST
        assert args.port == DEFAULT_PORT
        assert args.api_key == DEFAULT_API_KEY
        assert args.mcp_host == DEFAULT_MCP_HOST
        assert args.mcp_port == DEFAULT_MCP_PORT
        assert args.transport == DEFAULT_TRANSPORT_MODE
        assert args.stateful == DEFAULT_STATEFUL_MODE
        assert args.stateless is False

    def test_parse_args_custom_values(self):
        """Test parsing custom command line arguments."""
        test_args = [
            '--host', 'custom-host',
            '--port', '8080',
            '--api-key', 'custom-key',
            '--mcp-host', 'mcp-host',
            '--mcp-port', '3000',
            '--transport', 'http',
            '--stateless'
        ]

        with patch('sys.argv', ['test'] + test_args):
            args = parse_args()

            assert args.host == 'custom-host'
            assert args.port == 8080
            assert args.api_key == 'custom-key'
            assert args.mcp_host == 'mcp-host'
            assert args.mcp_port == 3000
            assert args.transport == 'http'
            assert args.stateless is True

    def test_command_line_override_environment(self):
        """Test that command line arguments override environment variables."""
        with patch.dict(os.environ, {
            'LIGHTRAG_API_HOST': 'env-host',
            'LIGHTRAG_API_PORT': '9000'
        }):
            with patch('sys.argv', ['test', '--host', 'cli-host', '--port', '8000']):
                # Simulate the config loading process
                args = parse_args()

                # Command line should override environment
                assert args.host == 'cli-host'
                assert args.port == 8000


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_port_validation(self):
        """Test port number validation."""
        # Valid ports
        valid_ports = [1, 65535, 8080, 3000]

        for port in valid_ports:
            with patch.dict(os.environ, {'MCP_PORT': str(port)}):
                import importlib
                import lightrag_mcp.config as config_module
                importlib.reload(config_module)

                assert config_module.MCP_PORT == port

    def test_boolean_conversion(self):
        """Test boolean environment variable conversion."""
        boolean_configs = [
            ('ENABLE_AUTH', True, False),
            ('ENABLE_SECURITY_HEADERS', True, False),
            ('ENABLE_REQUEST_LOGGING', True, False),
            ('ENABLE_ERROR_LOGGING', True, False),
            ('CORS_ALLOW_CREDENTIALS', True, False),
        ]

        for env_var, true_value, false_value in boolean_configs:
            # Test true values
            for true_str in ['true', 'True', 'TRUE', '1']:
                with patch.dict(os.environ, {env_var: true_str}):
                    import importlib
                    import lightrag_mcp.config as config_module
                    importlib.reload(config_module)

                    assert getattr(config_module, env_var) == true_value

            # Test false values
            for false_str in ['false', 'False', 'FALSE', '0', '']:
                with patch.dict(os.environ, {env_var: false_str}):
                    import importlib
                    import lightrag_mcp.config as config_module
                    importlib.reload(config_module)

                    assert getattr(config_module, env_var) == false_value

    def test_list_conversion(self):
        """Test list environment variable conversion."""
        # Test CORS_ORIGINS
        origins = ['http://localhost:3000', 'https://example.com']
        origins_str = ','.join(origins)

        with patch.dict(os.environ, {'CORS_ORIGINS': origins_str}):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            assert config_module.CORS_ORIGINS == origins

        # Test CORS_ALLOW_METHODS
        methods = ['GET', 'POST', 'PUT']
        methods_str = ','.join(methods)

        with patch.dict(os.environ, {'CORS_ALLOW_METHODS': methods_str}):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            assert config_module.CORS_ALLOW_METHODS == methods

    def test_numeric_conversion(self):
        """Test numeric environment variable conversion."""
        numeric_configs = [
            ('EVENT_RETENTION_HOURS', 48),
            ('MAX_EVENTS_PER_SESSION', 500),
            ('RATE_LIMIT_REQUESTS', 50),
            ('RATE_LIMIT_WINDOW_SECONDS', 30),
            ('MAX_REQUEST_SIZE_MB', 5),
            ('REQUEST_TIMEOUT_SECONDS', 60),
            ('CORS_MAX_AGE', 3600),
        ]

        for env_var, expected_value in numeric_configs:
            with patch.dict(os.environ, {env_var: str(expected_value)}):
                import importlib
                import lightrag_mcp.config as config_module
                importlib.reload(config_module)

                assert getattr(config_module, env_var) == expected_value


class TestConfigurationEdgeCases:
    """Test configuration edge cases and error handling."""

    def test_missing_environment_variables(self):
        """Test behavior with missing environment variables."""
        # Ensure defaults are used when environment variables are missing
        import importlib
        import lightrag_mcp.config as config_module
        importlib.reload(config_module)

        assert config_module.LIGHTRAG_API_HOST == DEFAULT_HOST
        assert config_module.LIGHTRAG_API_PORT == DEFAULT_PORT
        assert config_module.MCP_HOST == DEFAULT_MCP_HOST
        assert config_module.MCP_PORT == DEFAULT_MCP_PORT

    def test_invalid_port_values(self):
        """Test handling of invalid port values."""
        # Invalid port should fall back to default
        with patch.dict(os.environ, {'MCP_PORT': 'invalid'}):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            # Should fall back to default due to int() conversion failure
            assert config_module.MCP_PORT == DEFAULT_MCP_PORT

    def test_empty_string_values(self):
        """Test handling of empty string environment variables."""
        with patch.dict(os.environ, {
            'LIGHTRAG_API_KEY': '',
            'API_KEY': '',
            'BEARER_TOKEN_SECRET': ''
        }):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            assert config_module.LIGHTRAG_API_KEY == ''
            assert config_module.API_KEY == ''
            assert config_module.BEARER_TOKEN_SECRET == ''

    def test_whitespace_handling(self):
        """Test handling of whitespace in environment variables."""
        with patch.dict(os.environ, {
            'LIGHTRAG_API_HOST': '  localhost  ',
            'CORS_ORIGINS': ' http://example.com , https://test.com '
        }):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            # Host should be stripped
            assert config_module.LIGHTRAG_API_HOST == 'localhost'

            # CORS origins should be split and stripped
            expected_origins = ['http://example.com', 'https://test.com']
            assert config_module.CORS_ORIGINS == expected_origins


class TestConfigurationIntegration:
    """Test configuration integration with other components."""

    def test_config_affects_event_store_creation(self):
        """Test that configuration affects event store creation."""
        with patch.dict(os.environ, {
            'STATEFUL_MODE': 'true',
            'MAX_EVENTS_PER_SESSION': '500'
        }):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            from lightrag_mcp.event_store import create_event_store

            # Should create file store when stateful mode is enabled
            store = create_event_store("file", max_events_per_session=config_module.MAX_EVENTS_PER_SESSION)
            assert store.max_events_per_session == 500

    def test_config_affects_security_middleware(self):
        """Test that configuration affects security middleware."""
        with patch.dict(os.environ, {
            'ENABLE_AUTH': 'true',
            'API_KEY': 'test-key',
            'RATE_LIMIT_REQUESTS': '50'
        }):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            from lightrag_mcp.security import RateLimiter

            # Rate limiter should use configured values
            limiter = RateLimiter(
                config_module.RATE_LIMIT_REQUESTS,
                config_module.RATE_LIMIT_WINDOW_SECONDS
            )
            assert limiter.max_requests == 50

    def test_config_affects_http_server(self):
        """Test that configuration affects HTTP server."""
        with patch.dict(os.environ, {
            'MCP_HOST': '0.0.0.0',
            'MCP_PORT': '3000',
            'ENVIRONMENT': 'production'
        }):
            import importlib
            import lightrag_mcp.config as config_module
            importlib.reload(config_module)

            # These values should be used by the HTTP server
            assert config_module.MCP_HOST == '0.0.0.0'
            assert config_module.MCP_PORT == 3000
            assert config_module.ENVIRONMENT == 'production'