"""Deterministic synthetic identifiers for offline ``--dry-run`` mode.

When no Dataerai backend/credentials are available, the integration still records
the *complete* lineage locally (``provenance_manifest.json`` + ``PROVENANCE.md``).
Ids are derived deterministically with ``uuid5`` so repeated dry-runs over the
same artifacts produce stable, diff-able manifests (and unit tests can assert on
exact ids).
"""
from __future__ import annotations

import uuid

# Stable namespace ("da7ae9a1" ≈ "dataerai") so dry-run ids never collide with
# real backend UUIDs yet stay reproducible across runs/machines.
_NS = uuid.UUID("da7ae9a1-0000-5000-a000-000000000000")


def fake_asset_id(seed: str) -> str:
    return str(uuid.uuid5(_NS, f"asset:{seed}"))


def fake_content_id(seed: str) -> str:
    return str(uuid.uuid5(_NS, f"content:{seed}"))


def fake_did(asset_id: str) -> str:
    return f"did:dataerai:asset:{asset_id}"


def fake_run_id(label: str) -> str:
    return str(uuid.uuid5(_NS, f"run:{label}"))
