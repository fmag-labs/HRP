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

__all__ = ["mcp"]


def __getattr__(name: str):
    """Lazily expose the FastMCP server instance.

    Imported lazily (PEP 562) so that importing sibling modules such as
    hrp.mcp.errors does not eagerly pull in research_server — which imports back
    from hrp.mcp.errors and would otherwise create a package-level circular
    import depending on collection/import order.
    """
    if name == "mcp":
        from hrp.mcp.research_server import mcp

        return mcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
