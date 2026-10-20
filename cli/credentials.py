"""JWT + base-URL storage with per-terminal profile isolation.

Profiles live at ~/.finrag/profiles/{profile}.json. Two terminals can have
different active identities without stepping on each other:

  Terminal 1:  export FB_PROFILE=admin    && finrag login -u admin
  Terminal 2:  export FB_PROFILE=finance  && finrag login -u finance

If FB_PROFILE is unset, profile name is "default" -- the single-terminal
case stays unchanged. Legacy profiles are migrated to profiles/default.json
on first read so existing installs don't need to re-login.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

CREDS_DIR = Path.home() / ".finrag"
PROFILES_DIR = CREDS_DIR / "profiles"
LEGACY_FILE = CREDS_DIR / "credentials.json"

_DEFAULT_PROFILE = "default"


def current_profile() -> str:
    return os.environ.get("FINRAG_PROFILE", os.environ.get("FB_PROFILE", _DEFAULT_PROFILE)).strip() or _DEFAULT_PROFILE


def _profile_path(profile: str | None = None) -> Path:
    return PROFILES_DIR / f"{profile or current_profile()}.json"


def save(token: str, user_id: str, base_url: str) -> None:
    PROFILES_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = _profile_path()
    payload = {"token": token, "user_id": user_id, "base_url": base_url}
    path.write_text(json.dumps(payload))
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def load() -> dict | None:
    path = _profile_path()
    if not path.exists():
        # One-time migration for installs predating the profile split: if
        # FB_PROFILE is unset (or 'default') and the legacy credentials.json
        # exists, read from it and back-fill the default profile.
        if current_profile() == _DEFAULT_PROFILE and LEGACY_FILE.exists():
            try:
                creds = json.loads(LEGACY_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                return None
            try:
                save(
                    token=creds.get("token", ""),
                    user_id=creds.get("user_id", ""),
                    base_url=creds.get("base_url", "http://localhost:8000"),
                )
                LEGACY_FILE.unlink()
            except OSError:
                pass
            return creds
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def clear() -> bool:
    path = _profile_path()
    if path.exists():
        path.unlink()
        return True
    if current_profile() == _DEFAULT_PROFILE and LEGACY_FILE.exists():
        LEGACY_FILE.unlink()
        return True
    return False
