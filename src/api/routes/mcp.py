"""Model Context Protocol (MCP) FastAPI routes for investment tools.

Exposes standard JSON-RPC 2.0 MCP interface as well as developer-friendly
REST endpoints for tool discovery and execution with RBAC enforcement.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user
from src.models.auth import User
from src.tools.mcp_server import handle_mcp_jsonrpc, mcp_tool_registry

router = APIRouter(prefix="/mcp", tags=["mcp"])


class MCPJsonRpcRequest(BaseModel):
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version (must be '2.0')")
    id: Any = Field(default=None, description="Request identifier")
    method: str = Field(..., description="MCP method name: 'initialize', 'tools/list', or 'tools/call'")
    params: dict[str, Any] = Field(default_factory=dict, description="Method parameters")


class ToolExecutionRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool invocation parameters")


@router.post("", summary="Model Context Protocol JSON-RPC Endpoint")
async def mcp_jsonrpc_endpoint(
    request: MCPJsonRpcRequest,
    user: User = Depends(get_current_user),
):
    """Standard Model Context Protocol JSON-RPC 2.0 interface.

    Supports:
    - ``initialize``: Negotiate protocol and discover capabilities
    - ``tools/list``: List all available investment tools for caller's role
    - ``tools/call``: Execute a financial tool with argument validation and audit
    """
    response = handle_mcp_jsonrpc(
        payload=request.model_dump(),
        user_role=user.role,
        user_id=user.user_id,
    )
    return response


@router.get("/tools", summary="List Available MCP Tools")
async def list_mcp_tools(user: User = Depends(get_current_user)):
    """Return all investment tools accessible to the caller's role."""
    tools = mcp_tool_registry.list_tools(user_role=user.role)
    return {"tools": tools, "total": len(tools), "caller_role": user.role}


@router.post("/tools/{tool_name}", summary="Execute an MCP Tool Directly")
async def execute_mcp_tool(
    tool_name: str,
    request: ToolExecutionRequest,
    user: User = Depends(get_current_user),
):
    """Direct REST invocation of an MCP tool with audit logging and RBAC checks."""
    try:
        result = mcp_tool_registry.call_tool(
            name=tool_name,
            arguments=request.arguments,
            user_role=user.role,
            user_id=user.user_id,
        )
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
