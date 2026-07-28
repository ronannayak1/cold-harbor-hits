"""Shared Snowflake connection helper for Cold Harbor Hits.

Profiles:
- secrets/amg_research.env           → APP_STAT key-pair (Luminate / sandbox)
- secrets/amg_research_password.env  → SSO externalbrowser (DF_PROD_DAP_MISC)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import dotenv_values, load_dotenv
except ImportError:
    dotenv_values = None
    load_dotenv = None

SNOWFLAKE_ENV_PATH = Path("secrets/amg_research.env")
SNOWFLAKE_DAP_ENV_PATH = Path("secrets/amg_research_password.env")
SNOWFLAKE_BASE_ENV_VARS = (
    "SNOWFLAKE_USER",
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_ROLE",
)


def load_snowflake_env(env_path: Path = SNOWFLAKE_ENV_PATH) -> None:
    """Load a Snowflake profile into process env (override=True)."""
    if not env_path.exists():
        raise EnvironmentError(
            f"Snowflake env file not found: {env_path}. "
            "Create the secrets profile with connection settings."
        )
    if load_dotenv is not None:
        load_dotenv(env_path, override=True)


def _profile_values(env_path: Path) -> dict[str, str]:
    """Read a secrets profile without permanently mutating process env."""
    if not env_path.exists():
        raise EnvironmentError(
            f"Snowflake env file not found: {env_path}. "
            "Create the secrets profile with connection settings."
        )
    if dotenv_values is None:
        load_snowflake_env(env_path)
        return {key: value for key, value in os.environ.items() if key.startswith("SNOWFLAKE_")}

    raw = dotenv_values(env_path)
    return {
        str(key): str(value)
        for key, value in raw.items()
        if key and value is not None and str(value) != ""
    }


def _has_private_key_config(profile: dict[str, str]) -> bool:
    return bool(profile.get("SNOWFLAKE_PRIVATE_KEY_PATH") or profile.get("SNOWFLAKE_PRIVATE_KEY"))


def _load_snowflake_private_key(profile: dict[str, str]) -> bytes:
    """Load a PKCS#8 private key for Snowflake key-pair authentication."""
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:
        raise ImportError(
            "cryptography is required for Snowflake key-pair auth. "
            "Install with: pip install cryptography"
        ) from exc

    key_path = profile.get("SNOWFLAKE_PRIVATE_KEY_PATH")
    key_pem = profile.get("SNOWFLAKE_PRIVATE_KEY")
    passphrase = profile.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")

    if key_path:
        resolved = Path(key_path).expanduser()
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        if not resolved.exists():
            raise EnvironmentError(f"Snowflake private key file not found: {resolved}")
        key_data = resolved.read_bytes()
    elif key_pem:
        key_data = key_pem.replace("\\n", "\n").encode()
    else:
        raise EnvironmentError(
            "Key-pair auth requires SNOWFLAKE_PRIVATE_KEY_PATH or SNOWFLAKE_PRIVATE_KEY."
        )

    private_key = serialization.load_pem_private_key(
        key_data,
        password=passphrase.encode() if passphrase else None,
        backend=default_backend(),
    )
    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def get_snowflake_connection(env_path: Path = SNOWFLAKE_ENV_PATH):
    """Open a Snowflake connection from a secrets profile.

    Auth resolution order:
    1. SNOWFLAKE_AUTHENTICATOR / AUTH_METHOD = externalbrowser (SSO)
    2. Key-pair when AUTH_METHOD=key_pair or a private key is configured
    3. Password (SNOWFLAKE_PASSWORD)
    """
    try:
        import snowflake.connector
    except ImportError as exc:
        raise ImportError(
            "snowflake-connector-python is required. "
            f"Install with: {sys.executable} -m pip install snowflake-connector-python "
            f"(current interpreter: {sys.executable})"
        ) from exc

    profile = _profile_values(env_path)
    missing = [var for var in SNOWFLAKE_BASE_ENV_VARS if not profile.get(var)]
    if missing:
        raise EnvironmentError(
            f"Missing Snowflake environment variables in {env_path}: {', '.join(missing)}"
        )

    connect_kwargs: dict[str, Any] = {
        "user": profile["SNOWFLAKE_USER"],
        "account": profile["SNOWFLAKE_ACCOUNT"],
        "warehouse": profile["SNOWFLAKE_WAREHOUSE"],
        "role": profile["SNOWFLAKE_ROLE"],
    }

    database = profile.get("SNOWFLAKE_DATABASE")
    schema = profile.get("SNOWFLAKE_SCHEMA")
    if database:
        connect_kwargs["database"] = database
    if schema:
        connect_kwargs["schema"] = schema

    authenticator = profile.get("SNOWFLAKE_AUTHENTICATOR", "").strip().lower()
    auth_method = profile.get("SNOWFLAKE_AUTH_METHOD", "").strip().lower()

    if authenticator == "externalbrowser" or auth_method == "externalbrowser":
        connect_kwargs["authenticator"] = "externalbrowser"
    elif auth_method == "key_pair" or _has_private_key_config(profile):
        connect_kwargs["private_key"] = _load_snowflake_private_key(profile)
    else:
        password = profile.get("SNOWFLAKE_PASSWORD")
        if not password:
            raise EnvironmentError(
                "Set SNOWFLAKE_AUTHENTICATOR=externalbrowser, "
                "SNOWFLAKE_AUTH_METHOD=key_pair with a private key, "
                "or SNOWFLAKE_PASSWORD."
            )
        connect_kwargs["password"] = password

    return snowflake.connector.connect(**connect_kwargs)


def get_dap_snowflake_connection():
    """SSO connection for stitched DAP social (MISC TikTok + PROD YouTube)."""
    return get_snowflake_connection(SNOWFLAKE_DAP_ENV_PATH)
