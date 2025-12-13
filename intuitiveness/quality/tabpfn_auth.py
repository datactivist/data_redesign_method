"""
TabPFN Authentication Utility

Handles authentication with the TabPFN cloud API (tabpfn-client).
Stores and retrieves access tokens for subsequent API calls.
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Default token storage location
TOKEN_FILE = Path.home() / ".tabpfn" / "token"


def get_stored_token() -> Optional[str]:
    """
    Retrieve stored TabPFN access token from file.

    Returns:
        Token string if found, None otherwise.
    """
    if TOKEN_FILE.exists():
        try:
            return TOKEN_FILE.read_text().strip()
        except Exception as e:
            logger.warning(f"Failed to read token file: {e}")
    return None


def store_token(token: str) -> None:
    """
    Store TabPFN access token to file.

    Args:
        token: The access token to store.
    """
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)  # Restrict permissions
    logger.info("TabPFN token stored successfully")


def is_authenticated() -> bool:
    """
    Check if TabPFN authentication is configured.

    Returns:
        True if a valid token is stored, False otherwise.
    """
    token = get_stored_token()
    return token is not None and len(token) > 0


def authenticate_interactive() -> str:
    """
    Perform interactive authentication with TabPFN API.
    Opens a browser for the user to authenticate.

    Returns:
        Access token on success.

    Raises:
        RuntimeError: If authentication fails.
    """
    try:
        import tabpfn_client

        # Check if already authenticated via tabpfn_client's own storage
        try:
            # Try to use existing auth
            from tabpfn_client import TabPFNClassifier
            clf = TabPFNClassifier()
            logger.info("TabPFN already authenticated via client library")
            return "existing"
        except Exception:
            pass

        # Perform new authentication
        logger.info("Starting TabPFN authentication (browser will open)...")
        token = tabpfn_client.get_access_token()
        tabpfn_client.set_access_token(token)
        store_token(token)
        logger.info("TabPFN authentication successful")
        return token

    except ImportError:
        raise RuntimeError(
            "tabpfn-client package not installed. "
            "Install with: pip install tabpfn-client"
        )
    except Exception as e:
        raise RuntimeError(f"TabPFN authentication failed: {e}")


def setup_authentication(token: Optional[str] = None) -> bool:
    """
    Setup TabPFN authentication.

    Args:
        token: Optional token to use directly. If None, checks stored token
               or prompts for interactive auth.

    Returns:
        True if authentication is set up successfully.
    """
    try:
        import tabpfn_client

        if token:
            tabpfn_client.set_access_token(token)
            store_token(token)
            return True

        # Try stored token first
        stored = get_stored_token()
        if stored:
            tabpfn_client.set_access_token(stored)
            return True

        # Need interactive auth
        return False

    except ImportError:
        logger.warning("tabpfn-client not installed, API features unavailable")
        return False
