"""Tests for MCP errors module."""

import pytest

from hrp.api.platform import NotFoundError, PermissionError, PlatformAPIError
from hrp.mcp.errors import (
    InvalidParameterError,
    MCPError,
    ToolNotFoundError,
    handle_api_error,
)


class TestHandleApiError:
    """Tests for handle_api_error decorator."""

    def test_handle_api_error_success(self):
        """Decorator passes through successful results."""

        @handle_api_error
        def successful_fn():
            return {"success": True, "data": "test"}

        result = successful_fn()
        assert result["success"] is True
        assert result["data"] == "test"

    def test_handle_permission_error(self):
        """Handle PermissionError gracefully."""

        @handle_api_error
        def permission_error_fn():
            raise PermissionError("Agents cannot approve deployments")

        result = permission_error_fn()
        assert result["success"] is False
        assert "Permission denied" in result["message"]
        assert "cannot approve" in result["error"]

    def test_handle_not_found_error(self):
        """Handle NotFoundError gracefully."""

        @handle_api_error
        def not_found_fn():
            raise NotFoundError("Hypothesis HYP-2026-999 not found")

        result = not_found_fn()
        assert result["success"] is False
        assert "not found" in result["message"].lower()
        assert "HYP-2026-999" in result["error"]

    def test_handle_value_error(self):
        """Handle ValueError gracefully."""

        @handle_api_error
        def value_error_fn():
            raise ValueError("symbols list cannot be empty")

        result = value_error_fn()
        assert result["success"] is False
        assert "Invalid input" in result["message"]
        assert "empty" in result["error"]

    def test_handle_platform_api_error(self):
        """Handle PlatformAPIError gracefully."""

        @handle_api_error
        def platform_error_fn():
            raise PlatformAPIError("Database connection failed")

        result = platform_error_fn()
        assert result["success"] is False
        assert "Platform error" in result["message"]

    def test_handle_unexpected_error(self):
        """Handle unexpected exceptions gracefully."""

        @handle_api_error
        def unexpected_error_fn():
            raise RuntimeError("Unexpected failure")

        result = unexpected_error_fn()
        assert result["success"] is False
        assert "unexpected error" in result["message"].lower()
        # Should not expose internal details
        assert "RuntimeError" in result["error"]

    def test_decorator_preserves_function_name(self):
        """Decorator preserves wrapped function metadata."""

        @handle_api_error
        def my_tool_function():
            """My docstring."""
            return {"success": True}

        assert my_tool_function.__name__ == "my_tool_function"
        assert "My docstring" in my_tool_function.__doc__


class TestMCPError:
    """Tests for MCPError base class."""

    def test_mcp_error_basic(self):
        """Create basic MCPError."""
        error = MCPError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.code == "MCP_ERROR"

    def test_mcp_error_with_code(self):
        """Create MCPError with custom code."""
        error = MCPError("Custom error", code="CUSTOM_CODE")
        assert error.code == "CUSTOM_CODE"


class TestToolNotFoundError:
    """Tests for ToolNotFoundError."""

    def test_tool_not_found_error(self):
        """Create ToolNotFoundError."""
        error = ToolNotFoundError("nonexistent_tool")
        assert "nonexistent_tool" in str(error)
        assert error.tool_name == "nonexistent_tool"
        assert error.code == "TOOL_NOT_FOUND"


class TestInvalidParameterError:
    """Tests for InvalidParameterError."""

    def test_invalid_parameter_error(self):
        """Create InvalidParameterError."""
        error = InvalidParameterError("symbols", "must not be empty")
        assert "symbols" in str(error)
        assert "must not be empty" in str(error)
        assert error.param_name == "symbols"
        assert error.code == "INVALID_PARAMETER"
