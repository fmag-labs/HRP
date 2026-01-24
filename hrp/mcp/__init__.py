"""
HRP MCP Server Package.

Provides Model Context Protocol (MCP) integration for the Hedge Fund Research Platform,
enabling Claude to interact with research tools, run backtests, and manage hypotheses.

Usage:
    # Start the server
    python -m hrp.mcp

    # Or import the server instance
    from hrp.mcp import mcp
"""

from hrp.mcp.research_server import mcp

__all__ = ["mcp"]
