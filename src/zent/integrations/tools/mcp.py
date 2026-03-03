"""MCP (Model Context Protocol) tool integration for Zent.

This module provides integration with MCP servers, allowing Zent agents
to use tools exposed via the Model Context Protocol.

Reference:
    - MCP Specification: https://modelcontextprotocol.io
    - MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk

Example:
    ```python
    # Connect to an MCP server via stdio
    client = MCPClient(command="npx -y @modelcontextprotocol/server-filesystem /tmp")
    await client.connect()

    # Discover and create tools
    tools = await MCPToolAdapter.create_tools_from_client(client)

    # Use tools with an agent
    agent = Agent(model=model, tools=tools)

    # Clean up
    await client.close()
    ```
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import aiohttp

from zent.core.tool import BaseTool

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base exception for MCP-related errors."""

    pass


class MCPConnectionError(MCPError):
    """Exception raised when connection to MCP server fails."""

    pass


class MCPProtocolError(MCPError):
    """Exception raised when MCP protocol communication fails."""

    pass


class MCPToolError(MCPError):
    """Exception raised when tool execution fails."""

    pass


@dataclass
class MCPToolInfo:
    """Information about an MCP tool.

    Attributes:
        name: The tool name.
        description: The tool description.
        input_schema: JSON Schema for tool parameters.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


class MCPTransport(ABC):
    """Abstract base class for MCP transports.

    Handles the low-level communication with MCP servers.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to MCP server."""
        pass

    @abstractmethod
    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message.

        Args:
            message: The JSON-RPC message to send.
        """
        pass

    @abstractmethod
    async def receive(self) -> dict[str, Any]:
        """Receive a JSON-RPC message.

        Returns:
            The received JSON-RPC message.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""
        pass


class StdioTransport(MCPTransport):
    """Transport for MCP servers via stdio (subprocess).

    Communicates with MCP servers through stdin/stdout pipes.
    """

    def __init__(self, command: str, env: dict[str, str] | None = None) -> None:
        """Initialize stdio transport.

        Args:
            command: The command to run the MCP server.
            env: Optional environment variables for the subprocess.
        """
        self.command = command
        self.env = env
        self._process: subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Start the subprocess and establish connection."""
        try:
            # Parse command into args
            args = self.command.split()

            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**dict(asyncio.subprocess.DEVNULL), **(self.env or {})},
            )

            logger.info(f"Started MCP server process: {self.command}")

            # Wait a moment for server to initialize
            await asyncio.sleep(0.1)

            # Check if process is still running
            if self._process.returncode is not None:
                stderr = await self._process.stderr.read() if self._process.stderr else b""
                raise MCPConnectionError(
                    f"MCP server process exited immediately: {stderr.decode()}"
                )

        except Exception as e:
            raise MCPConnectionError(f"Failed to start MCP server: {e}") from e

    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message via stdin.

        Args:
            message: The JSON-RPC message to send.
        """
        if not self._process or self._process.stdin is None:
            raise MCPConnectionError("Not connected to MCP server")

        async with self._lock:
            data = json.dumps(message) + "\n"
            self._process.stdin.write(data.encode())
            await self._process.stdin.drain()

    async def receive(self) -> dict[str, Any]:
        """Receive a JSON-RPC message from stdout.

        Returns:
            The received JSON-RPC message.
        """
        if not self._process or self._process.stdout is None:
            raise MCPConnectionError("Not connected to MCP server")

        async with self._lock:
            line = await self._process.stdout.readline()
            if not line:
                raise MCPConnectionError("MCP server closed connection")

            try:
                return json.loads(line.decode().strip())
            except json.JSONDecodeError as e:
                raise MCPProtocolError(f"Invalid JSON from MCP server: {e}") from e

    async def close(self) -> None:
        """Terminate the subprocess."""
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception as e:
                logger.warning(f"Error closing MCP process: {e}")
            finally:
                self._process = None


class HTTPTransport(MCPTransport):
    """Transport for MCP servers via HTTP/SSE.

    Communicates with MCP servers through HTTP requests and
    Server-Sent Events (SSE) for streaming responses.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize HTTP transport.

        Args:
            url: The MCP server URL.
            headers: Optional HTTP headers.
            timeout: Request timeout in seconds.
        """
        self.url = url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        self._message_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._receive_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Establish HTTP connection and start SSE listener."""
        try:
            self._session = aiohttp.ClientSession(headers=self.headers)

            # Start SSE connection for receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())

            logger.info(f"Connected to MCP server at {self.url}")

        except Exception as e:
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def _receive_loop(self) -> None:
        """Background task to receive SSE messages."""
        if not self._session:
            return

        try:
            async with self._session.get(
                f"{self.url}/sse",
                headers={"Accept": "text/event-stream"},
                timeout=aiohttp.ClientTimeout(total=None),
            ) as response:
                async for line in response.content:
                    line = line.decode().strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        try:
                            message = json.loads(data)
                            await self._message_queue.put(message)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in SSE: {data}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SSE receive error: {e}")

    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message via HTTP POST.

        Args:
            message: The JSON-RPC message to send.
        """
        if not self._session:
            raise MCPConnectionError("Not connected to MCP server")

        try:
            async with self._session.post(
                f"{self.url}/message",
                json=message,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise MCPProtocolError(f"HTTP {response.status}: {text}")
        except aiohttp.ClientError as e:
            raise MCPConnectionError(f"Failed to send message: {e}") from e

    async def receive(self) -> dict[str, Any]:
        """Receive a JSON-RPC message from the queue.

        Returns:
            The received JSON-RPC message.
        """
        try:
            return await asyncio.wait_for(
                self._message_queue.get(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise MCPConnectionError("Timeout waiting for MCP server response")

    async def close(self) -> None:
        """Close HTTP session and cancel receive task."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()
            self._session = None


class MCPClient:
    """Client for connecting to MCP servers.

    Supports both stdio (subprocess) and HTTP transports.
    Implements JSON-RPC 2.0 for MCP protocol communication.

    Example:
        ```python
        # Stdio transport
        client = MCPClient(command="npx -y @modelcontextprotocol/server-filesystem /tmp")

        # HTTP transport
        client = MCPClient(url="http://localhost:3000")

        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})
        await client.close()
        ```
    """

    def __init__(
        self,
        command: str | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize MCP client.

        Args:
            command: Command to run MCP server (for stdio transport).
            url: URL of MCP server (for HTTP transport).
            env: Environment variables for stdio transport.
            headers: HTTP headers for HTTP transport.

        Raises:
            ValueError: If neither command nor url is provided.
        """
        if command:
            self._transport: MCPTransport = StdioTransport(command, env)
        elif url:
            self._transport = HTTPTransport(url, headers)
        else:
            raise ValueError("Either command or url must be provided")

        self._request_id = 0
        self._pending_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._receive_task: asyncio.Task | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to the MCP server.

        Raises:
            MCPConnectionError: If connection fails.
        """
        await self._transport.connect()
        self._connected = True

        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

        # Initialize session
        await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "zent-mcp-client",
                    "version": "0.4.0",
                },
            },
        )

    async def _receive_loop(self) -> None:
        """Background task to receive and route messages."""
        try:
            while self._connected:
                message = await self._transport.receive()

                # Handle responses
                if "id" in message:
                    request_id = str(message["id"])
                    if request_id in self._pending_requests:
                        future = self._pending_requests.pop(request_id)
                        if "error" in message:
                            future.set_exception(
                                MCPProtocolError(
                                    f"MCP error {message['error'].get('code')}: "
                                    f"{message['error'].get('message')}"
                                )
                            )
                        else:
                            future.set_result(message.get("result", {}))

                # Handle notifications (ignore for now)
                elif "method" in message:
                    logger.debug(f"Received notification: {message['method']}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
            # Fail all pending requests
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(MCPConnectionError(f"Connection lost: {e}"))
            self._pending_requests.clear()

    def _get_next_id(self) -> str:
        """Generate next request ID."""
        self._request_id += 1
        return str(self._request_id)

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for response.

        Args:
            method: The JSON-RPC method.
            params: Optional method parameters.

        Returns:
            The response result.

        Raises:
            MCPConnectionError: If not connected.
            MCPProtocolError: If request fails.
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        request_id = self._get_next_id()
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            message["params"] = params

        # Create future for response
        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            await self._transport.send(message)
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise MCPProtocolError(f"Request timeout for method: {method}")
        except Exception:
            self._pending_requests.pop(request_id, None)
            raise

    async def list_tools(self) -> list[MCPToolInfo]:
        """List available tools from the MCP server.

        Returns:
            List of tool information.

        Raises:
            MCPConnectionError: If not connected.
            MCPProtocolError: If request fails.
        """
        result = await self._send_request("tools/list")

        tools = []
        for tool_data in result.get("tools", []):
            tools.append(
                MCPToolInfo(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {"type": "object"}),
                )
            )

        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server.

        Args:
            name: The tool name.
            arguments: The tool arguments.

        Returns:
            The tool result as a string.

        Raises:
            MCPConnectionError: If not connected.
            MCPToolError: If tool execution fails.
        """
        result = await self._send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )

        if "content" in result:
            # Extract text content from result
            content = result["content"]
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                    else:
                        texts.append(str(item))
                return "\n".join(texts)
            elif isinstance(content, dict) and "text" in content:
                return content["text"]
            else:
                return str(content)
        elif "error" in result:
            raise MCPToolError(f"Tool execution failed: {result['error']}")
        else:
            return json.dumps(result)

    async def close(self) -> None:
        """Close the connection to the MCP server."""
        self._connected = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Cancel any pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        await self._transport.close()
        logger.info("MCP client disconnected")

    async def __aenter__(self) -> MCPClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


class MCPTool(BaseTool):
    """Tool wrapper for MCP tools.

    Wraps an MCP tool as a Zent BaseTool, allowing seamless
    integration with Zent agents.

    Attributes:
        name: The tool name.
        description: The tool description.
        client: The MCP client for executing the tool.
        _parameters: The tool parameter schema.

    Example:
        ```python
        tool = MCPTool(
            client=client,
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        )

        result = await tool.run(path="/tmp/test.txt")
        ```
    """

    def __init__(
        self,
        client: MCPClient,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Initialize MCP tool wrapper.

        Args:
            client: The MCP client for tool execution.
            name: The tool name.
            description: The tool description.
            parameters: JSON Schema for tool parameters.
        """
        self.client = client
        self.name = name
        self.description = description
        self._parameters = parameters

    @property
    def parameters(self) -> dict[str, Any]:
        """Get the tool parameter schema.

        Returns:
            JSON Schema for tool parameters.
        """
        return self._parameters

    async def run(self, **kwargs: Any) -> str:
        """Execute the MCP tool.

        Args:
            **kwargs: The tool arguments.

        Returns:
            The tool result as a string.

        Raises:
            MCPToolError: If tool execution fails.
        """
        try:
            return await self.client.call_tool(self.name, kwargs)
        except MCPError:
            raise
        except Exception as e:
            raise MCPToolError(f"Tool execution failed: {e}") from e


class MCPToolAdapter:
    """Adapter for converting MCP tools to Zent tools.

    Provides utilities for discovering MCP tools and converting
    their schemas to Zent-compatible formats.

    Example:
        ```python
        # Discover and create all tools from an MCP server
        async with MCPClient(command="npx @modelcontextprotocol/server-filesystem /tmp") as client:
            tools = await MCPToolAdapter.create_tools_from_client(client)

        # Use with an agent
        agent = Agent(model=model, tools=tools)
        ```
    """

    @staticmethod
    def convert_schema(mcp_schema: dict[str, Any]) -> dict[str, Any]:
        """Convert MCP tool schema to Zent parameter format.

        MCP uses JSON Schema directly, which is compatible with
        Zent's parameter format. This method ensures proper
        normalization.

        Args:
            mcp_schema: The MCP input schema.

        Returns:
            Normalized parameter schema.
        """
        # MCP already uses JSON Schema, so we mostly pass through
        # but ensure required fields are present
        schema = dict(mcp_schema)

        # Ensure type is set
        if "type" not in schema:
            schema["type"] = "object"

        # Ensure properties exists
        if "properties" not in schema:
            schema["properties"] = {}

        # Ensure required is a list
        if "required" not in schema:
            schema["required"] = []
        elif not isinstance(schema["required"], list):
            schema["required"] = list(schema["required"])

        return schema

    @staticmethod
    async def create_tools_from_client(client: MCPClient) -> list[MCPTool]:
        """Create MCPTool instances from an MCP client.

        Discovers all available tools from the MCP server and
        creates MCPTool wrappers for each.

        Args:
            client: Connected MCP client.

        Returns:
            List of MCPTool instances.

        Raises:
            MCPConnectionError: If not connected.
            MCPProtocolError: If tool discovery fails.
        """
        tool_infos = await client.list_tools()

        tools: list[MCPTool] = []
        for info in tool_infos:
            tool = MCPTool(
                client=client,
                name=info.name,
                description=info.description,
                parameters=MCPToolAdapter.convert_schema(info.input_schema),
            )
            tools.append(tool)

        logger.info(f"Created {len(tools)} MCP tools from client")
        return tools

    @staticmethod
    async def create_tools_from_server(
        command: str | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[list[MCPTool], MCPClient]:
        """Create tools and client from server configuration.

        Convenience method that creates a client, connects to the
        server, discovers tools, and returns both the tools and
        the client for management.

        Args:
            command: Command for stdio transport.
            url: URL for HTTP transport.
            env: Environment variables for stdio.
            headers: HTTP headers for HTTP transport.

        Returns:
            Tuple of (tools, client). Caller is responsible for
            closing the client.

        Example:
            ```python
            tools, client = await MCPToolAdapter.create_tools_from_server(
                command="npx @modelcontextprotocol/server-filesystem /tmp"
            )
            try:
                agent = Agent(model=model, tools=tools)
                # Use agent...
            finally:
                await client.close()
            ```
        """
        client = MCPClient(command=command, url=url, env=env, headers=headers)
        await client.connect()
        tools = await MCPToolAdapter.create_tools_from_client(client)
        return tools, client
