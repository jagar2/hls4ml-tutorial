"""Thin REST client for the beta provenance endpoints the SDK/daemon does not proxy.

The Dataerai Python SDK (``dataerai.DataEraiClient``) handles *preservation*
(upload/download/metadata) through the Go daemon. The **lineage-run**,
**asset-relationship** and **identity/DID** APIs are plain authenticated REST
endpoints with no daemon path, so this module calls them directly with the
OAuth2 bearer token resolved by :mod:`dataerai_hls4ml.config`.

Every contract here was verified against ``origin/beta``:

* ``POST /api/assets/<pk>/relationships/``           (Layer 2 — lineage graph)
* ``POST /api/lineage/runs/``                          (Layer 3 — sealed run)
* ``POST /api/lineage/runs/<id>/{edges,trained-on,close}/``
* ``GET  /api/identity/resolve/<did>/``                (Layer 4 — citation)
* ``GET  /api/projects/`` · ``GET /api/collections/``  (owner discovery)
"""
from __future__ import annotations

import uuid
import time
from typing import Any, Iterable, Optional

from .config import Settings, read_credentials, refresh_credentials


class RestError(RuntimeError):
    def __init__(self, status: int, message: Any, path: str):
        super().__init__(f"HTTP {status} on {path}: {message}")
        self.status = status
        self.path = path
        self.body = message


class RestClient:
    """Minimal authenticated JSON client. Requires the ``requests`` package."""

    def __init__(self, settings: Settings):
        self.settings = settings
        try:
            import requests  # lazy: keep import cost off the no-network path
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "the 'requests' package is required for live Dataerai provenance; "
                "`pip install requests` or run with --dry-run"
            ) from exc
        self._requests = requests
        self._session = requests.Session()
        self._refreshed_once = False

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.settings.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.settings.server}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, *, json_body=None, params=None,
                 retry_auth: bool = True) -> Any:
        for attempt in range(3):
            resp = self._session.request(
                method, self._url(path),
                headers=self._headers(), json=json_body, params=params,
                timeout=self.settings.timeout, verify=self.settings.verify_tls,
            )
            if resp.status_code not in {502, 503, 504} or attempt == 2:
                break
            time.sleep(2 ** attempt)
        # On 401, let the CLI refresh+persist the token once, then retry.
        if resp.status_code == 401 and retry_auth and not self._refreshed_once:
            self._refreshed_once = True
            refresh_credentials(self.settings.binary_path)
            token = read_credentials().get("access_token")
            if token:
                self.settings.access_token = token
            return self._request(method, path, json_body=json_body,
                                  params=params, retry_auth=False)
        if not (200 <= resp.status_code < 300):
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text[:500]
            raise RestError(resp.status_code, detail, path)
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    # ── owner discovery ────────────────────────────────────────────────────────
    def list_projects(self) -> Any:
        return self._request("GET", "/api/projects/")

    def list_collections(self) -> Any:
        return self._request("GET", "/api/collections/")

    def get_asset(self, asset_id: str) -> dict:
        return self._request("GET", f"/api/assets/{asset_id}/")

    # ── collections (one per notebook) ──────────────────────────────────────────
    def find_collection(self, project_id: str, title: str) -> Optional[str]:
        resp = self._request("GET", "/api/collections/",
                             params={"owner_type": "project", "owner_id": project_id, "q": title})
        items = resp.get("results", resp) if isinstance(resp, dict) else resp
        for c in (items if isinstance(items, list) else []):
            if c.get("title") == title:
                return c.get("id")
        return None

    def create_collection(self, project_id: str, title: str) -> str:
        body = {"title": title, "owner_type": "project", "owner_id": project_id}
        return self._request("POST", "/api/collections/", json_body=body).get("id")

    def get_or_create_collection(self, project_id: str, title: str) -> Optional[str]:
        return self.find_collection(project_id, title) or self.create_collection(project_id, title)

    # ── Layer 2: asset relationships (UUID-based lineage graph) ──────────────────
    def create_relationship(self, from_id: str, to_id: str, rel_type: str, *,
                            qualifier_note: Optional[str] = None,
                            qualifiers: Optional[dict] = None) -> dict:
        body: dict = {"to_asset_id": to_id, "type": rel_type}
        if qualifier_note:
            body["qualifier_note"] = qualifier_note[:5000]
        if qualifiers:
            body["qualifiers"] = qualifiers
        try:
            return self._request("POST", f"/api/assets/{from_id}/relationships/",
                                  json_body=body)
        except RestError as err:
            if err.status == 409:  # identical (from, to, type) already exists — fine
                return {"duplicate": True, "type": rel_type, "to_asset_id": to_id}
            raise

    # ── Layer 3: sealed, hash-chained lineage run ───────────────────────────────
    def open_run(self, kind: str, *, model_pid: Optional[str] = None,
                 params_hash: Optional[str] = None) -> dict:
        body: dict = {"kind": kind}
        if model_pid:
            body["model_pid"] = model_pid
        if params_hash:
            body["params_hash"] = params_hash
        return self._request("POST", "/api/lineage/runs/", json_body=body)

    def trained_on(self, run_id: str, data_dids: Iterable[str], *,
                   idempotency_key: Optional[str] = None) -> dict:
        body = {"idempotency_key": idempotency_key or str(uuid.uuid4()),
                "data_dids": [d for d in data_dids if d]}
        return self._request("POST", f"/api/lineage/runs/{run_id}/trained-on/",
                             json_body=body)

    def add_edges(self, run_id: str, edges: list, *,
                  idempotency_key: Optional[str] = None) -> dict:
        body = {"idempotency_key": idempotency_key or str(uuid.uuid4()), "edges": edges}
        return self._request("POST", f"/api/lineage/runs/{run_id}/edges/",
                             json_body=body)

    def close_run(self, run_id: str, *, run_signature: Optional[str] = None) -> dict:
        body: dict = {}
        if run_signature:
            body["run_signature"] = run_signature
        return self._request("POST", f"/api/lineage/runs/{run_id}/close/",
                             json_body=body)

    def get_run(self, run_id: str) -> dict:
        return self._request("GET", f"/api/lineage/runs/{run_id}/")

    def resolve_record_sk(self, asset_id: str, subject_kind: int = 1) -> Optional[int]:
        """Resolve an asset's ProvenanceRecord surrogate key (needed as an edge's
        dst_sk) via GET /api/lineage/trace/. Returns None if the async provenance
        record isn't minted yet (subject_kind=1 == asset)."""
        try:
            r = self._request("GET", "/api/lineage/trace/",
                             params={"subject_kind": subject_kind, "subject_id": asset_id})
        except RestError:
            return None
        candidates = []
        if isinstance(r, dict):
            if isinstance(r.get("start"), dict):
                candidates.append(r["start"])
            candidates.extend(n for n in (r.get("nodes") or []) if isinstance(n, dict))
        for c in candidates:
            sk = c.get("record_sk") or c.get("sk")
            if sk and (c.get("subject_id") in (asset_id, None)):
                try:
                    return int(sk)
                except (TypeError, ValueError):
                    return None
        return None

    # ── Layer 4: verifiable DID identity (citation) ─────────────────────────────
    def resolve_did(self, did: str) -> dict:
        # Public endpoint; no bearer needed, and 404 for gated/unminted DIDs.
        return self._request("GET", f"/api/identity/resolve/{did}/", retry_auth=False)

    def publish_did(self, did: str) -> dict:
        # Irreversible private→public — only called on explicit opt-in.
        return self._request("POST", f"/api/identity/{did}/publish/", json_body={})
