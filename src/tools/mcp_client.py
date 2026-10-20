"""In-memory and JSON-RPC Model Context Protocol (MCP) Client.

Enables agent nodes (such as `research_agent`) to inspect and execute financial
tools over the MCP JSON-RPC protocol interface, enforcing user role permissions
and recording OpenTelemetry spans directly within the active LangGraph execution context.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.tools.mcp_server import handle_mcp_jsonrpc

logger = logging.getLogger(__name__)


class MCPClientError(Exception):
    """Raised when an MCP tool invocation encounters an unhandled protocol failure."""

    def __init__(self, message: str, error_code: str | None = None, data: Any = None):
        super().__init__(message)
        self.error_code = error_code
        self.data = data


class MCPClient:
    """Client for invoking MCP-compliant tool services with institutional RBAC."""

    def __init__(self, user_role: str = "research_analyst", user_id: str = "agent"):
        self.user_role = user_role
        self.user_id = user_id

    def list_tools(self) -> list[dict[str, Any]]:
        """List available tools permitted for the client's configured user role."""
        payload = {
            "jsonrpc": "2.0",
            "id": "list-tools-req",
            "method": "tools/list",
            "params": {},
        }
        resp = handle_mcp_jsonrpc(payload, user_role=self.user_role, user_id=self.user_id)
        if "error" in resp:
            err = resp["error"]
            raise MCPClientError(err.get("message", "Failed to list tools"), error_code="RPC_ERROR", data=err)
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool with role permissions checking and structured response parsing.

        Returns a dictionary containing:
        - name: str
        - is_error: bool
        - data: parsed dict or raw string result
        - error_code: optional error code (e.g. 'PERMISSION_DENIED', 'CALCULATION_ERROR')
        - text: string representation of output or error message
        - latency_ms: execution time
        """
        payload = {
            "jsonrpc": "2.0",
            "id": f"call-{name}-req",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        resp = handle_mcp_jsonrpc(payload, user_role=self.user_role, user_id=self.user_id)
        if "error" in resp:
            err = resp["error"]
            return {
                "name": name,
                "is_error": True,
                "error_code": "JSON_RPC_ERROR",
                "message": err.get("message", "JSON-RPC error"),
                "text": err.get("message", "JSON-RPC error"),
                "data": None,
                "latency_ms": 0.0,
            }

        res = resp.get("result", {})
        is_error = res.get("isError", False)
        error_code = res.get("error_code")
        latency_ms = res.get("latency_ms", 0.0)
        content_items = res.get("content", [])
        text_content = content_items[0].get("text", "") if content_items else ""

        parsed_data = None
        if not is_error and text_content:
            try:
                parsed_data = json.loads(text_content)
            except Exception:
                parsed_data = text_content
        else:
            parsed_data = text_content

        return {
            "name": name,
            "is_error": is_error,
            "error_code": error_code,
            "data": parsed_data,
            "text": text_content,
            "latency_ms": latency_ms,
        }
