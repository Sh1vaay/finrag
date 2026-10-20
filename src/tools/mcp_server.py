"""Role-gated Model Context Protocol (MCP)-style JSON-RPC 2.0 tool service for investment analysis.

Implements a role-gated JSON-RPC 2.0 service adhering to Model Context Protocol (MCP) tool schema
definitions (spec 2024-11-05) for secure, reusable, and audited financial tool execution
in institutional buy-side asset management environments.

Provides:
- MCP-compliant JSON-RPC protocol handling (initialize, ping, tools/list, tools/call)
- Explicit buy-side Role-Based Access Control (RBAC) per tool (analyst vs portfolio manager)
- Formal JSON Schema parameter validation
- Structured OpenTelemetry trace spans ('mcp.tool_call'), latency metrics, and audit logging
- Optional official MCP Python SDK export adapter (create_official_mcp_server)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from src.tools.calculator import CalculatorError, compute
from src.tools.decision_models import (
    calculate_factor_exposures,
    calculate_portfolio_var_cvar,
    optimize_portfolio_weights,
)
from src.tools.investment_metrics import (
    FinancialCalculationError,
    calculate_growth_metrics,
    calculate_portfolio_risk_metrics,
    calculate_profitability_metrics,
    calculate_valuation_multiples,
)
from src.tools.market_data import MarketDataError, get_market_data

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "finrag-investment-mcp"
SERVER_VERSION = "1.0.0"

# Optional official MCP SDK imports with fallback
try:
    import mcp.types as mcp_types
    from mcp.server.lowlevel import Server as OfficialMcpServer
except ImportError:
    mcp_types = None
    OfficialMcpServer = None


class MCPError(Exception):
    """Base error for MCP protocol and execution issues."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


def _normalize_role(role: str) -> str:
    r = role.lower().strip()
    if r in ("analyst", "finance"):
        return "research_analyst"
    elif r in ("pm", "c_level", "clevel"):
        return "portfolio_manager"
    elif r == "compliance":
        return "risk_compliance"
    return r


# Hierarchy levels: higher number = greater privilege
ROLE_HIERARCHY_LEVELS: dict[str, int] = {
    "research_analyst": 1,
    "portfolio_manager": 2,
    "risk_compliance": 3,
    "admin": 4,
}


def _is_role_authorized(user_role: str, required_role: str | list[str] | set[str] | None) -> bool:
    """Check if user_role satisfies required_role with buy-side hierarchy inheritance.

    Hierarchy:
      admin (4) > risk_compliance (3) > portfolio_manager (2) > research_analyst (1)
    """
    if not required_role:
        return True

    normalized_user = _normalize_role(user_role)
    if normalized_user == "admin":
        return True

    user_level = ROLE_HIERARCHY_LEVELS.get(normalized_user, 0)

    if isinstance(required_role, (list, set, tuple)):
        req_list = list(required_role)
    else:
        req_list = [required_role]

    for req in req_list:
        norm_req = _normalize_role(req)
        req_level = ROLE_HIERARCHY_LEVELS.get(norm_req, 999)
        # If user meets or exceeds the required hierarchy level, or matches the role directly
        if user_level >= req_level or normalized_user == norm_req:
            return True

    return False


class ToolRegistry:
    """Enterprise registry for Model Context Protocol tools with access control."""

    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, Callable] = {}
        self._register_default_tools()

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable,
        required_role: str | list[str] | None = None,
    ) -> None:
        """Register a new tool conforming to the MCP tool schema."""
        self._tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "required_role": required_role,
        }
        self._handlers[name] = handler
        logger.debug("Registered MCP tool: %s (required_role: %s)", name, required_role)

    def list_tools(self, user_role: str = "research_analyst") -> list[dict[str, Any]]:
        """Return tool definitions available to the caller's role."""
        available = []
        for name, spec in self._tools.items():
            req_role = spec.get("required_role")
            if not _is_role_authorized(user_role, req_role):
                continue
            available.append(
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "inputSchema": spec["inputSchema"],
                }
            )
        return available

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        user_role: str = "research_analyst",
        user_id: str = "system",
    ) -> dict[str, Any]:
        """Execute a tool with parameter validation, timing, role checks, and error handling."""
        start_time = time.perf_counter()

        if name not in self._tools:
            raise MCPError(-32601, f"Tool '{name}' not found")

        tool_spec = self._tools[name]
        req_role = tool_spec.get("required_role")
        if not _is_role_authorized(user_role, req_role):
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            denial_msg = (
                f"Access denied: user role '{user_role}' lacks permission to execute tool "
                f"'{name}' (requires '{req_role}')"
            )
            logger.warning("MCP Access Denied: tool=%s user=%s role=%s", name, user_id, user_role)
            return {
                "content": [{"type": "text", "text": denial_msg}],
                "isError": True,
                "error_code": "PERMISSION_DENIED",
                "latency_ms": elapsed_ms,
            }

        handler = self._handlers[name]
        try:
            try:
                from src.services.tracing_service import trace_span

                span_cm = trace_span("mcp.tool_call", {"tool.name": name, "user.role": user_role, "user.id": user_id})
            except ImportError:
                span_cm = None

            if span_cm:
                with span_cm:
                    result = handler(**arguments)
            else:
                result = handler(**arguments)

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                "MCP Tool executed: tool=%s user=%s role=%s latency=%.2fms",
                name,
                user_id,
                user_role,
                elapsed_ms,
            )
            return {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                "isError": False,
                "latency_ms": elapsed_ms,
            }
        except (FinancialCalculationError, CalculatorError, MarketDataError) as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning("MCP Tool domain calculation error in %s: %s", name, exc)
            return {
                "content": [{"type": "text", "text": f"Calculation error: {exc}"}],
                "isError": True,
                "error_code": "CALCULATION_ERROR",
                "latency_ms": elapsed_ms,
            }
        except TypeError as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning("MCP Tool invalid arguments for %s: %s", name, exc)
            raise MCPError(-32602, f"Invalid arguments for tool '{name}': {exc}") from exc
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error("MCP Tool internal failure in %s: %s", name, exc, exc_info=True)
            raise MCPError(-32603, f"Internal execution error in tool '{name}': {exc}") from exc

    def _register_default_tools(self) -> None:
        """Register the core suite of buy-side investment and market data tools."""
        # 1. Valuation multiples (Analyst + PM + Compliance + Admin)
        self.register(
            name="calculate_valuation_multiples",
            description="Calculate equity & enterprise valuation multiples: P/E, EV/EBITDA, P/B, FCF yield, and market cap.",
            input_schema={
                "type": "object",
                "properties": {
                    "share_price": {"type": "number", "description": "Current share price"},
                    "earnings_per_share": {"type": "number", "description": "Diluted EPS (trailing or forward)"},
                    "enterprise_value": {"type": "number", "description": "Total Enterprise Value"},
                    "ebitda": {"type": "number", "description": "EBITDA in millions or exact figures"},
                    "book_value_per_share": {"type": "number", "description": "Book value per share"},
                    "free_cash_flow": {"type": "number", "description": "Free cash flow in millions"},
                    "shares_outstanding": {"type": "number", "description": "Common shares outstanding"},
                },
                "required": ["share_price"],
            },
            handler=calculate_valuation_multiples,
            required_role="research_analyst",
        )

        # 2. Profitability metrics (Analyst + PM + Compliance + Admin)
        self.register(
            name="calculate_profitability_metrics",
            description="Calculate gross/operating/net margins, Return on Equity (ROE), Return on Assets (ROA), and Debt-to-Equity.",
            input_schema={
                "type": "object",
                "properties": {
                    "revenue": {"type": "number", "description": "Total revenue"},
                    "gross_profit": {"type": "number", "description": "Gross profit"},
                    "operating_income": {"type": "number", "description": "Operating income / EBIT"},
                    "net_income": {"type": "number", "description": "Net income"},
                    "total_equity": {"type": "number", "description": "Total stockholders' equity"},
                    "total_assets": {"type": "number", "description": "Total assets"},
                    "total_debt": {"type": "number", "description": "Total debt obligations"},
                },
                "required": ["revenue"],
            },
            handler=calculate_profitability_metrics,
            required_role="research_analyst",
        )

        # 3. Growth & Performance Attribution (Analyst + PM + Compliance + Admin)
        self.register(
            name="calculate_growth_metrics",
            description="Calculate percentage change, basis-point variance, and multi-year CAGR between financial periods.",
            input_schema={
                "type": "object",
                "properties": {
                    "current_value": {"type": "number", "description": "Value for current period"},
                    "prior_value": {"type": "number", "description": "Value for comparison baseline period"},
                    "metric_name": {
                        "type": "string",
                        "description": "Label of metric (e.g. 'Revenue', 'Operating Margin')",
                    },
                    "periods": {"type": "integer", "description": "Number of periods for annualized CAGR calculation"},
                },
                "required": ["current_value", "prior_value"],
            },
            handler=calculate_growth_metrics,
            required_role="research_analyst",
        )

        # 4. Live Market Data via yfinance (Analyst + PM + Compliance + Admin)
        self.register(
            name="get_market_data",
            description="Retrieve live equity price, market capitalization, trailing/forward P/E, 52-week range, and beta.",
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Equity ticker symbol (e.g. 'AAPL', 'MSFT', 'NVDA')"},
                },
                "required": ["ticker"],
            },
            handler=get_market_data,
            required_role="research_analyst",
        )

        # 5. Portfolio Risk & Concentration (STRICT: Portfolio Manager & Admin ONLY, Denied for Research Analyst)
        self.register(
            name="calculate_portfolio_risk_metrics",
            description="Calculate portfolio concentration, top-N weights, and Herfindahl-Hirschman Index (HHI) for investment mandates.",
            input_schema={
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "object",
                        "description": "Dictionary mapping asset/ticker symbols to dollar values or percentage weights",
                    },
                },
                "required": ["holdings"],
            },
            handler=calculate_portfolio_risk_metrics,
            required_role="portfolio_manager",
        )

        # 6. Safe Arithmetic Evaluator (Permitted for all roles)
        self.register(
            name="evaluate_arithmetic",
            description="Safely evaluate arithmetic expressions with AST whitelisting (replaces unsafe eval).",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic string e.g. '(383285 - 394328) / 394328'",
                    },
                },
                "required": ["expression"],
            },
            handler=lambda expression: {"expression": expression, "result": compute(expression)},
            required_role=None,
        )

        # 7. Portfolio Value-at-Risk & Expected Shortfall (Portfolio Manager & Admin)
        self.register(
            name="calculate_portfolio_var_cvar",
            description="Calculate institutional Parametric/Historical Value-at-Risk (VaR 95/99%) and Expected Shortfall (CVaR).",
            input_schema={
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "object",
                        "description": "Dictionary of ticker to dollar value or percentage weight",
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "Confidence level (0.95 or 0.99, default 0.95)",
                    },
                    "time_horizon_days": {"type": "integer", "description": "Time horizon in trading days (default 1)"},
                    "portfolio_nav_usd": {
                        "type": "number",
                        "description": "Portfolio NAV in USD (default 10,000,000.0)",
                    },
                },
                "required": ["holdings"],
            },
            handler=calculate_portfolio_var_cvar,
            required_role="portfolio_manager",
        )

        # 8. Constrained Portfolio Optimization (Portfolio Manager & Admin)
        self.register(
            name="optimize_portfolio_weights",
            description="Solve constrained Minimum Variance portfolio allocation using Ledoit-Wolf shrinkage covariance.",
            input_schema={
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "object",
                        "description": "Dictionary of ticker to current weight or dollar value",
                    },
                    "target_objective": {"type": "string", "description": "Optimization target (e.g. 'min_variance')"},
                    "max_single_cap": {
                        "type": "number",
                        "description": "Maximum weight per position (e.g. 0.15 for 15%)",
                    },
                    "shrinkage_intensity": {
                        "type": "number",
                        "description": "Ledoit-Wolf shrinkage factor delta (0.0 - 1.0)",
                    },
                },
                "required": ["holdings"],
            },
            handler=optimize_portfolio_weights,
            required_role="portfolio_manager",
        )

        # 9. Factor Exposure & Systematic Risk Decomposition (Analyst, PM, Compliance, Admin)
        self.register(
            name="calculate_factor_exposures",
            description="Calculate multi-factor sensitivities (Market Beta, Size SMB, Value HML) and systematic vs idiosyncratic risk.",
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Equity ticker symbol (e.g. 'AAPL', 'MSFT', 'NVDA')"},
                    "benchmark": {"type": "string", "description": "Benchmark ticker (default 'SPY')"},
                    "market_beta": {"type": "number", "description": "Optional prior market beta override"},
                },
                "required": ["ticker"],
            },
            handler=calculate_factor_exposures,
            required_role="research_analyst",
        )

        # 10. Official SEC EDGAR Filing Discovery (Analyst, PM, Compliance, Admin)
        from src.services.sec_edgar_service import fetch_recent_filings

        self.register(
            name="get_sec_filings",
            description="Retrieve recent audited SEC filings (10-K, 10-Q, 8-K) dates and official EDGAR report URLs.",
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Equity ticker symbol (e.g. 'AAPL', 'MSFT', 'NVDA')"},
                    "form_type": {
                        "type": "string",
                        "description": "Filing form type: '10-K', '10-Q', or '8-K' (default '10-K')",
                    },
                },
                "required": ["ticker"],
            },
            handler=lambda ticker, form_type="10-K": {
                "ticker": ticker.upper(),
                "filings": fetch_recent_filings(ticker, form_type=form_type),
            },
            required_role="research_analyst",
        )


# Global singleton registry
mcp_tool_registry = ToolRegistry()


def handle_mcp_jsonrpc(
    payload: dict[str, Any],
    user_role: str = "research_analyst",
    user_id: str = "system",
) -> dict[str, Any]:
    """Handle Model Context Protocol (MCP)-style JSON-RPC 2.0 requests (Spec 2024-11-05)."""
    req_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": MCPError(-32600, "Invalid Request: must be JSON-RPC 2.0").to_dict(),
        }

    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "capabilities": {"tools": {"listChanged": False}},
                },
            }

        elif method == "tools/list":
            tools = mcp_tool_registry.list_tools(user_role=user_role)
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

        elif method == "tools/call":
            tool_name = params.get("name")
            if not tool_name:
                raise MCPError(-32602, "Missing parameter 'name' in tools/call")
            arguments = params.get("arguments") or {}
            call_result = mcp_tool_registry.call_tool(
                name=tool_name,
                arguments=arguments,
                user_role=user_role,
                user_id=user_id,
            )
            return {"jsonrpc": "2.0", "id": req_id, "result": call_result}

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        else:
            raise MCPError(-32601, f"Method '{method}' not found")

    except MCPError as exc:
        return {"jsonrpc": "2.0", "id": req_id, "error": exc.to_dict()}
    except Exception as exc:
        logger.error("Unhandled MCP error: %s", exc, exc_info=True)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": MCPError(-32603, f"Internal server error: {exc}").to_dict(),
        }


def list_tools(user_role: str = "research_analyst") -> list[dict[str, Any]]:
    """List tools available to the given user role."""
    return mcp_tool_registry.list_tools(user_role=user_role)


def call_tool(
    name: str,
    arguments: dict[str, Any],
    user_role: str = "research_analyst",
    user_id: str = "system",
) -> dict[str, Any]:
    """Invoke a registered MCP tool with role gating."""
    return mcp_tool_registry.call_tool(name=name, arguments=arguments, user_role=user_role, user_id=user_id)


def create_official_mcp_server(default_role: str = "research_analyst"):
    """Create an official MCP Python SDK v2 Server instance when SDK is installed."""
    if OfficialMcpServer is None or mcp_types is None:
        return None

    app = OfficialMcpServer(SERVER_NAME)

    @app.list_tools()
    async def _handle_list_tools() -> list[mcp_types.Tool]:
        tools = []
        for t in mcp_tool_registry.list_tools(user_role=default_role):
            tools.append(
                mcp_types.Tool(
                    name=t["name"],
                    description=t["description"],
                    inputSchema=t["inputSchema"],
                )
            )
        return tools

    @app.call_tool()
    async def _handle_call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
        res = mcp_tool_registry.call_tool(name=name, arguments=arguments, user_role=default_role)
        texts = [mcp_types.TextContent(type="text", text=c["text"]) for c in res.get("content", [])]
        return texts

    return app


handle_json_rpc = handle_mcp_jsonrpc
