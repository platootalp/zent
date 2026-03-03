"""MCP (Model Context Protocol) tool integration for Zent.

This module provides integration with MCP servers, allowing Zent agents
to discover and use tools exposed via the Model Context Protocol.

Example:
    ```python
    from zent.integrations.tools import MCPClient, MCPToolAdapter

    # Connect to an MCP server and discover tools
    async with MCPClient(command="npx -y @modelcontextprotocol/server-filesystem /tmp") as client:
        tools = await MCPToolAdapter.create_tools_from_client(client)

        # Use with an agent
        agent = Agent(model=model, tools=tools)
        result = await agent.run("Read the file /tmp/test.txt")
    ```
"""

from zent.integrations.tools.mcp import (
    MCPClient,
    MCPConnectionError,
    MCPError,
    MCPProtocolError,
    MCPTool,
    MCPToolAdapter,
    MCPToolError,
    MCPToolInfo,
)

__all__ = [
    "MCPClient",
    "MCPTool",
    "MCPToolAdapter",
    "MCPToolInfo",
    "MCPError",
    "MCPConnectionError",
    "MCPProtocolError",
    "MCPToolError",
]
