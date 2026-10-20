import secrets

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_current_user
from src.config.rbac_config import get_permissions
from src.config.settings import settings
from src.models.auth import User
from src.models.schemas import (
    LoginRequest,
    TokenResponse,
    UserMeResponse,
    UserPermissions,
)
from src.services.auth_service import create_token

router = APIRouter(prefix="/auth", tags=["auth"])

# In-memory development users aligned with Asset Management roles.
DEV_USERS = {
    "analyst": {
        "password": "analyst123",
        "name": "Senior Equity Analyst",
        "role": "research_analyst",
        "department": "Fundamental Equities",
    },
    "pm": {
        "password": "pm123",
        "name": "Lead Portfolio Manager",
        "role": "portfolio_manager",
        "department": "Global Asset Management",
    },
    "compliance": {
        "password": "compliance123",
        "name": "Risk & Compliance Officer",
        "role": "risk_compliance",
        "department": "Investment Compliance",
    },
    "admin": {
        "password": "admin123",
        "name": "Asset Management Lead Data Scientist",
        "role": "admin",
        "department": "Asset Management Data Science",
    },
    # Backwards-compatibility accounts:
    "finance": {
        "password": "finance123",
        "name": "Investment Associate",
        "role": "research_analyst",
        "department": "Fundamental Equities",
    },
    "hr": {"password": "hr123", "name": "Research Operations", "role": "hr", "department": "Research Operations"},
    "clevel": {
        "password": "clevel123",
        "name": "Chief Investment Officer",
        "role": "portfolio_manager",
        "department": "Executive Investment Committee",
    },
}


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    if settings.ENVIRONMENT == "production":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Development authentication provider is disabled in production.",
        )

    user = DEV_USERS.get(request.username)
    if not user or not secrets.compare_digest(user["password"], request.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_token(
        user_id=request.username,
        name=user["name"],
        role=user["role"],
        department=user["department"],
    )
    return TokenResponse(
        access_token=token,
        user_id=request.username,
        name=user["name"],
        role=user["role"],
        department=user["department"],
    )


@router.get("/me", response_model=UserMeResponse)
async def me(user: User = Depends(get_current_user)) -> UserMeResponse:
    """Return the current user's identity + role-derived permissions.

    The frontend calls this on app boot (or after a token refresh) to
    rehydrate the user state without keeping JWT payload state-of-truth
    in the client. The permissions block lets the UI gate admin nav,
    upload buttons, HITL approval surfaces, etc., without a separate
    config endpoint.
    """
    perms = get_permissions(user.role)
    return UserMeResponse(
        user_id=user.user_id,
        name=user.name,
        role=user.role,
        department=user.department,
        permissions=UserPermissions(
            allowed_doc_types=perms["allowed_doc_types"],
            allowed_confidentiality=perms["allowed_confidentiality"],
            max_results=perms["max_results"],
            requires_hitl_above=perms.get("requires_hitl_above"),
        ),
    )
