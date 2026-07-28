"""Shared Snowflake connection helper for Cold Harbor Hits."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

SNOWFLAKE_ENV_PATH = Path("secrets/amg_research.env")
SNOWFLAKE_BASE_ENV_VARS = (
    "SNOWFLAKE_USER",
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_ROLE",
)


def load_snowflake_env(env_path: Path = SNOWFLAKE_ENV_PATH) -> None:
    """Load Snowflake credentials from secrets/amg_research.env."""
    if not env_path.exists():
        raise EnvironmentError(
            f"Snowflake env file not found: {env_path}. "
            "Create secrets/amg_research.env with connection settings."
        )
    if load_dotenv is not None:
        load_dotenv(env_path, override=True)


def _load_snowflake_private_key() -> bytes:
    """Load a PKCS#8 private key for Snowflake key-pair authentication."""
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:
        raise ImportError(
            "cryptography is required for Snowflake key-pair auth. "
            "Install with: pip install cryptography"
        ) from exc

    key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
    key_pem = os.getenv("SNOWFLAKE_PRIVATE_KEY")
    passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")

    if key_path:
        key_data = Path(key_path).expanduser().read_bytes()
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
    """Open a Snowflake connection using secrets/amg_research.env credentials.

    Auth resolution order:
    1. SNOWFLAKE_AUTHENTICATOR=externalbrowser (SSO)
    2. SNOWFLAKE_AUTH_METHOD=key_pair
    3. password (SNOWFLAKE_PASSWORD)
    """
    try:
        import snowflake.connector
    except ImportError as exc:
        raise ImportError(
            "snowflake-connector-python is required. "
            f"Install with: {sys.executable} -m pip install snowflake-connector-python "
            f"(current interpreter: {sys.executable})"
        ) from exc

    load_snowflake_env(env_path)
    missing = [var for var in SNOWFLAKE_BASE_ENV_VARS if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            f"Missing Snowflake environment variables: {', '.join(missing)}"
        )

    connect_kwargs: dict[str, Any] = {
        "user": os.environ["SNOWFLAKE_USER"],
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        "role": os.environ["SNOWFLAKE_ROLE"],
    }

    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")
    if database:
        connect_kwargs["database"] = database
    if schema:
        connect_kwargs["schema"] = schema

    authenticator = os.getenv("SNOWFLAKE_AUTHENTICATOR", "").strip().lower()
    auth_method = os.getenv("SNOWFLAKE_AUTH_METHOD", "password").strip().lower()

    if authenticator == "externalbrowser" or auth_method == "externalbrowser":
        connect_kwargs["authenticator"] = "externalbrowser"
    elif auth_method == "key_pair":
        connect_kwargs["private_key"] = _load_snowflake_private_key()
    else:
        password = os.getenv("SNOWFLAKE_PASSWORD")
        if not password:
            raise EnvironmentError(
                "Password auth requires SNOWFLAKE_PASSWORD, or set "
                "SNOWFLAKE_AUTHENTICATOR=externalbrowser / SNOWFLAKE_AUTH_METHOD=key_pair."
            )
        connect_kwargs["password"] = password

    return snowflake.connector.connect(**connect_kwargs)
