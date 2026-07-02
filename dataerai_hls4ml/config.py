"""Configuration & credential resolution for the Dataerai ``hls4ml`` provenance layer.

This module figures out *where* to talk to Dataerai and *how* to authenticate,
reusing exactly what the ``dataerai`` CLI already stores — it never writes a
credential anywhere.

Resolution order
----------------
* **Server**   ``$DATAERAI_SERVER`` → the ``server_url`` saved at login → ``https://beta.dataerai.com``.
* **Token**    ``$DATAERAI_ACCESS_TOKEN`` → macOS Keychain (service ``dataerai``,
  account ``auth``) → the CLI config file (``os.UserConfigDir()/dataerai/credentials``).
  Before reading, ``dataerai auth status`` is shelled once so the CLI refreshes &
  re-persists an about-to-expire token (its own ``LoadAndRefreshCredentials``).
* **Owner**    ``$DATAERAI_PROJECT_ID`` / ``$DATAERAI_COLLECTION_ID``; otherwise the
  first project the user can see is auto-discovered at run time.

If no token can be found and dry-run wasn't explicitly requested, we *degrade to
dry-run* so a tutorial notebook never crashes just because Dataerai isn't set up.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_SERVER = "https://beta.dataerai.com"
_KEYCHAIN_SERVICE = "dataerai"
_KEYCHAIN_ACCOUNT = "auth"
_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def credentials_file() -> Path:
    """Mirror Go's ``credentialsFilePath()``: ``os.UserConfigDir()/dataerai/credentials``."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "dataerai" / "credentials"


def _decode_keyring_secret(raw: str) -> str:
    """The Go CLI stores creds via zalando/go-keyring, which base64-encodes values
    that aren't plain ASCII and prefixes them with ``go-keyring-base64:``."""
    prefix = "go-keyring-base64:"
    if raw.startswith(prefix):
        return base64.b64decode(raw[len(prefix):]).decode("utf-8")
    return raw


def read_credentials() -> dict:
    """Best-effort read of the ``dataerai`` CLI credentials (keychain first, then file)."""
    # 1) macOS Keychain — where the Go CLI stores creds by default.
    if sys.platform == "darwin" and shutil.which("security"):
        try:
            out = subprocess.run(
                ["security", "find-generic-password",
                 "-s", _KEYCHAIN_SERVICE, "-a", _KEYCHAIN_ACCOUNT, "-w"],
                capture_output=True, text=True, timeout=15,
            )
            if out.returncode == 0 and out.stdout.strip():
                return json.loads(_decode_keyring_secret(out.stdout.strip()))
        except (json.JSONDecodeError, OSError, subprocess.SubprocessError, ValueError):
            pass
    # 2) Config-file fallback.
    try:
        return json.loads(credentials_file().read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def refresh_credentials(binary: Optional[str]) -> None:
    """Trigger the CLI's own load+refresh+persist so the stored token is fresh."""
    binary = binary or os.environ.get("DATAERAI_BINARY") or shutil.which("dataerai")
    if not binary:
        return
    try:
        subprocess.run([binary, "auth", "status"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass


@dataclass
class Settings:
    """Resolved runtime configuration for one provenance session."""

    server: str
    dry_run: bool
    project_id: Optional[str]
    collection_id: Optional[str]
    binary_path: Optional[str]
    access_token: Optional[str]
    user_email: Optional[str] = None
    timeout: float = 60.0
    verify_tls: bool = True
    #: True when we silently fell back to dry-run because no token was found.
    degraded: bool = False

    @property
    def owner_type(self) -> Optional[str]:
        if self.project_id:
            return "project"
        if self.collection_id:
            return "collection"
        return None

    @property
    def owner_id(self) -> Optional[str]:
        return self.project_id or self.collection_id


def load_settings(*, dry_run: Optional[bool] = None, refresh: bool = True) -> Settings:
    """Resolve :class:`Settings` from the environment and the ``dataerai`` CLI."""
    binary = os.environ.get("DATAERAI_BINARY") or shutil.which("dataerai")
    token = os.environ.get("DATAERAI_ACCESS_TOKEN")

    if dry_run is None:
        dry_run = _env_flag("DATAERAI_DRY_RUN", default=False)

    creds: dict = {}
    if not dry_run and not token:
        if refresh:
            refresh_credentials(binary)
        creds = read_credentials()
        token = creds.get("access_token")

    server = (
        os.environ.get("DATAERAI_SERVER")
        or creds.get("server_url")
        or DEFAULT_SERVER
    ).rstrip("/")

    degraded = False
    if not dry_run and not token:
        # No credentials anywhere — keep notebooks working by recording locally.
        dry_run = True
        degraded = True

    return Settings(
        server=server,
        dry_run=dry_run,
        project_id=os.environ.get("DATAERAI_PROJECT_ID"),
        collection_id=os.environ.get("DATAERAI_COLLECTION_ID"),
        binary_path=binary,
        access_token=token,
        user_email=creds.get("user_email"),
        timeout=float(os.environ.get("DATAERAI_TIMEOUT", "60")),
        verify_tls=_env_flag("DATAERAI_VERIFY_TLS", default=True),
        degraded=degraded,
    )


def enabled() -> bool:
    """Whether provenance instrumentation should run at all (default **on**).

    Set ``DATAERAI_PROVENANCE=0`` to make every instrumented notebook cell a no-op.
    """
    return _env_flag("DATAERAI_PROVENANCE", default=True)
