"""
Entry point for running the MCP server as a module.

Usage:
    python -m hrp.mcp
    python -m hrp.mcp.research_server

The server starts in stdio mode for MCP client connections.
"""

from hrp.mcp.research_server import main

if __name__ == "__main__":
    main()
