"""Quantitative investment analysis tools and Model Context Protocol (MCP) server."""

from src.tools.calculator import (
    CalculatorDisallowedError,
    CalculatorError,
    CalculatorParseError,
    CalculatorRuntimeError,
    compute,
)
from src.tools.decision_models import (
    DecisionModelError,
    calculate_altman_z_score,
    calculate_factor_exposures,
    calculate_piotroski_f_score,
    calculate_portfolio_var_cvar,
    evaluate_composite_investment_conviction,
    evaluate_portfolio_trade_decision,
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
from src.tools.mcp_client import MCPClient, MCPClientError
from src.tools.mcp_server import (
    MCPError,
    handle_mcp_jsonrpc,
    mcp_tool_registry,
)

__all__ = [
    "compute",
    "CalculatorError",
    "CalculatorParseError",
    "CalculatorRuntimeError",
    "CalculatorDisallowedError",
    "FinancialCalculationError",
    "calculate_valuation_multiples",
    "calculate_profitability_metrics",
    "calculate_growth_metrics",
    "calculate_portfolio_risk_metrics",
    "calculate_portfolio_var_cvar",
    "optimize_portfolio_weights",
    "calculate_factor_exposures",
    "calculate_piotroski_f_score",
    "calculate_altman_z_score",
    "evaluate_portfolio_trade_decision",
    "evaluate_composite_investment_conviction",
    "DecisionModelError",
    "get_market_data",
    "MarketDataError",
    "MCPError",
    "handle_mcp_jsonrpc",
    "mcp_tool_registry",
    "MCPClient",
    "MCPClientError",
]
