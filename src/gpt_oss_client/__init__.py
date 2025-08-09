"""GPT OSS Client - A client for interacting with GPT models via MCP servers."""

__version__ = "0.1.0"

from .lib import McpClient, StdioPipe

__all__ = ["McpClient", "StdioPipe"]