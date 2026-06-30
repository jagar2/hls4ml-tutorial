"""``ProvenanceRun`` — ties preservation + lineage edges + citation into one run.

A single run represents one pass of (part of) the hls4ml pipeline. It:

* opens a sealed **lineage run** on beta (best-effort, Layer 3);
* preserves artifacts via :class:`~dataerai_hls4ml.artifacts.Preserver` (Layer 1);
* records typed edges between artifacts as **asset relationships** (Layer 2) and,
  for ``trained_on``, feeds dataset DIDs to the lineage run;
* resolves **DID citations** (Layer 4, best-effort) and, on close, writes
  ``provenance_manifest.json`` + ``PROVENANCE.md``.

State (run id + the artifact registry) is persisted to ``.dataerai_run.json`` in
the working directory so the tutorial's notebooks — each its own process — share
**one** run and can link to artifacts produced by earlier parts.
"""
from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

from . import dryrun
from .artifacts import Artifact, Preserver
from .config import Settings, load_settings

STATE_FILE = ".dataerai_run.json"
_TOOLS = ["tensorflow", "hls4ml", "qkeras", "conifer", "numpy", "sklearn", "pysr", "xgboost"]
_current: Optional["ProvenanceRun"] = None
PathLike = Union[str, Path]


def _warn(msg: str) -> None:
    print(f"[dataerai] {msg}", file=sys.stderr)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha(cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd),
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _tool_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in _TOOLS:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            continue
    return versions


def _aid(x: Union[Artifact, str]) -> str:
    return x.asset_id if isinstance(x, Artifact) else str(x)


class ProvenanceRun:
    """One instrumented pipeline run. Usable directly or as a context manager."""

    def __init__(self, label: str = "hls4ml-pipeline", kind: str = "training", *,
                 settings: Optional[Settings] = None, base_dir: PathLike = ".",
                 fresh: bool = False):
        self.label = label
        self.kind = kind
        self.settings = settings or load_settings()
        self.base_dir = Path(base_dir)
        self.rest = None
        self.preserver: Optional[Preserver] = None
        self.run_id: Optional[str] = None
        self.run_sk = None
        self.integrity_root: Optional[str] = None
        self.closed = False
        self.artifacts: Dict[str, Artifact] = {}
        self.edges: List[dict] = []
        self.started_at = _utcnow()
        self.env = {
            "git_sha": _git_sha(self.base_dir),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tools": _tool_versions(),
            "user_email": self.settings.user_email,
        }
        if self.settings.degraded:
            _warn("no Dataerai credentials found — recording provenance locally "
                  "(dry-run). Run `dataerai auth login --server "
                  f"{self.settings.server}` and set DATAERAI_PROJECT_ID to push to beta.")
        self._init_backend()
        if not fresh:
            self._load_state()

    # ── backend / owner / run setup ─────────────────────────────────────────────
    def _init_backend(self) -> None:
        if self.settings.dry_run:
            self.preserver = Preserver(self.settings, rest=None)
            self.run_id = dryrun.fake_run_id(self.label)
            return
        from .rest import RestClient
        self.rest = RestClient(self.settings)
        self._resolve_owner()
        self.preserver = Preserver(self.settings, self.rest)
        try:
            resp = self.rest.open_run(self.kind)
            self.run_id = resp.get("run_id")
            self.run_sk = resp.get("run_sk")
        except Exception as exc:
            _warn(f"lineage run open failed ({exc}); continuing with the "
                  "relationship graph (Layer 2) only.")

    def _resolve_owner(self) -> None:
        if self.settings.owner_id:
            return
        try:
            projects = self.rest.list_projects()
            items = projects.get("results", projects) if isinstance(projects, dict) else projects
            if items:
                first = items[0]
                self.settings.project_id = first.get("id") or first.get("project_id")
                _warn(f"DATAERAI_PROJECT_ID not set; using first visible project "
                      f"{self.settings.project_id!r}.")
        except Exception as exc:
            raise RuntimeError(
                f"could not auto-discover an owner project ({exc}); set DATAERAI_PROJECT_ID."
            )
        if not self.settings.owner_id:
            raise RuntimeError("no owner project/collection available; set DATAERAI_PROJECT_ID.")

    # ── cross-notebook state ─────────────────────────────────────────────────────
    def _state_path(self) -> Path:
        return self.base_dir / STATE_FILE

    def _load_state(self) -> None:
        try:
            data = json.loads(self._state_path().read_text())
        except (OSError, json.JSONDecodeError):
            return
        # Reuse the lineage run id opened by an earlier part.
        if data.get("run_id") and not self.settings.dry_run:
            self.run_id = data["run_id"]
            self.run_sk = data.get("run_sk", self.run_sk)
        for name, raw in (data.get("artifacts") or {}).items():
            if name not in self.artifacts:
                self.artifacts[name] = Artifact(**raw)
        self.edges = data.get("edges", self.edges)

    def _save_state(self) -> None:
        payload = {
            "label": self.label,
            "kind": self.kind,
            "run_id": self.run_id,
            "run_sk": self.run_sk,
            "server": self.settings.server,
            "dry_run": self.settings.dry_run,
            "artifacts": {n: a.to_dict() for n, a in self.artifacts.items()},
            "edges": self.edges,
        }
        try:
            self._state_path().write_text(json.dumps(payload, indent=2))
        except OSError as exc:  # pragma: no cover
            _warn(f"could not persist run state: {exc}")

    # ── preserve (delegates to Preserver, then registers) ────────────────────────
    def _register(self, art: Artifact) -> Artifact:
        self.artifacts[art.name] = art
        self._save_state()
        return art

    def preserve_dataset(self, paths, **kw) -> Artifact:
        return self._register(self.preserver.preserve_dataset(paths, **kw))

    def preserve_keras_model(self, model_or_path, **kw) -> Artifact:
        return self._register(self.preserver.preserve_keras_model(model_or_path, **kw))

    def preserve_hls_project(self, path, **kw) -> Artifact:
        return self._register(self.preserver.preserve_hls_project(path, **kw))

    def preserve_dir(self, path, *, kind, **kw) -> Artifact:
        return self._register(self.preserver.preserve_dir(path, kind=kind, **kw))

    def preserve_array(self, path, **kw) -> Artifact:
        return self._register(self.preserver.preserve_array(path, **kw))

    def preserve_file(self, path, **kw) -> Artifact:
        return self._register(self.preserver.preserve_file(path, **kw))

    def get(self, name: str) -> Optional[Artifact]:
        """Look up an artifact preserved earlier (incl. by a previous notebook)."""
        return self.artifacts.get(name)

    # ── edges ────────────────────────────────────────────────────────────────────
    def add_edge(self, frm: Union[Artifact, str], to: Union[Artifact, str], rel_type: str, *,
                 step: Optional[str] = None, note: Optional[str] = None,
                 qualifiers: Optional[dict] = None) -> dict:
        """Record a directed, typed lineage edge ``frm --rel_type--> to``."""
        frm_id, to_id = _aid(frm), _aid(to)
        # Idempotent: an identical (from, to, type) edge is only recorded once.
        for existing in self.edges:
            if (existing["from"], existing["to"], existing["type"]) == (frm_id, to_id, rel_type):
                return existing
        quals = dict(qualifiers or {})
        if step:
            quals["step"] = step
        quals.setdefault("pipeline", self.label)
        if self.run_id:
            quals.setdefault("run_id", self.run_id)
        edge = {"from": frm_id, "to": to_id, "type": rel_type,
                "step": step, "note": note, "qualifiers": quals}
        self.edges.append(edge)
        if not self.settings.dry_run and self.rest is not None:
            try:
                self.rest.create_relationship(frm_id, to_id, rel_type,
                                              qualifier_note=note, qualifiers=quals)
            except Exception as exc:
                _warn(f"relationship {rel_type} {frm_id}->{to_id} failed: {exc}")
        self._save_state()
        return edge

    def trained_on(self, model: Union[Artifact, str], datasets) -> None:
        """Link a model to the dataset(s) it was trained on (``trained_on`` edges)."""
        if isinstance(datasets, (Artifact, str)):
            datasets = [datasets]
        for ds in datasets:
            self.add_edge(model, ds, "trained_on")
        dids = [d.did for d in datasets if isinstance(d, Artifact) and d.did]
        if self.run_id and dids and not self.settings.dry_run and self.rest is not None:
            try:
                self.rest.trained_on(self.run_id, dids)
            except Exception as exc:
                _warn(f"lineage trained-on failed ({exc}); relationship edges still recorded.")

    # ── citations / close / manifest ─────────────────────────────────────────────
    def _collect_citations(self) -> List[dict]:
        cites = []
        for art in self.artifacts.values():
            if not art.did:
                continue
            entry = {"name": art.name, "asset_id": art.asset_id, "did": art.did}
            if not self.settings.dry_run and self.rest is not None:
                try:
                    res = self.rest.resolve_did(art.did)
                    entry["verified"] = res.get("verified")
                except Exception:
                    entry["resolvable"] = False
            cites.append(entry)
        return cites

    def manifest(self) -> dict:
        return {
            "label": self.label,
            "kind": self.kind,
            "server": self.settings.server,
            "dry_run": self.settings.dry_run,
            "run_id": self.run_id,
            "integrity_root": self.integrity_root,
            "started_at": self.started_at,
            "closed_at": _utcnow() if self.closed else None,
            "environment": self.env,
            "artifacts": [a.to_dict() for a in self.artifacts.values()],
            "edges": self.edges,
            "citations": self._collect_citations(),
        }

    def save(self, *, close: bool = True) -> dict:
        """Close the lineage run (sealing it) and write the manifest + markdown."""
        if close and self.run_id and not self.settings.dry_run and self.rest is not None:
            try:
                res = self.rest.close_run(self.run_id)
                self.integrity_root = res.get("integrity_root")
                self.closed = True
            except Exception as exc:
                _warn(f"lineage close failed ({exc}); manifest still written.")
        elif close:
            self.closed = True
        man = self.manifest()
        (self.base_dir / "provenance_manifest.json").write_text(json.dumps(man, indent=2))
        (self.base_dir / "PROVENANCE.md").write_text(_render_markdown(man))
        self._save_state()
        return man

    # ── context manager ──────────────────────────────────────────────────────────
    def __enter__(self) -> "ProvenanceRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Don't seal the shared run on error; just persist what we have.
        self.save(close=exc_type is None)


def _render_markdown(man: dict) -> str:
    lines: List[str] = []
    lines.append(f"# Provenance — {man['label']}")
    lines.append("")
    mode = "dry-run (recorded locally)" if man["dry_run"] else f"live → {man['server']}"
    lines.append(f"- **Mode:** {mode}")
    lines.append(f"- **Lineage run:** `{man['run_id']}`")
    if man.get("integrity_root"):
        lines.append(f"- **Integrity root:** `{man['integrity_root']}`")
    env = man.get("environment", {})
    if env.get("git_sha"):
        lines.append(f"- **Git SHA:** `{env['git_sha']}`")
    if env.get("tools"):
        tools = ", ".join(f"{k} {v}" for k, v in env["tools"].items())
        lines.append(f"- **Tools:** {tools}")
    lines.append("")
    lines.append("## Lineage graph")
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph LR")
    short = {a["asset_id"]: f"n{i}" for i, a in enumerate(man["artifacts"])}
    for a in man["artifacts"]:
        nid = short[a["asset_id"]]
        lines.append(f'  {nid}["{a["name"]}<br/><i>{a["kind"]}</i>"]')
    for e in man["edges"]:
        s, d = short.get(e["from"], e["from"][:6]), short.get(e["to"], e["to"][:6])
        label = e["type"] + (f" ({e['step']})" if e.get("step") else "")
        lines.append(f"  {s} -->|{label}| {d}")
    lines.append("```")
    lines.append("")
    lines.append("## Preserved assets")
    lines.append("")
    lines.append("| Artifact | Kind | Asset ID | DID |")
    lines.append("|---|---|---|---|")
    for a in man["artifacts"]:
        lines.append(f"| {a['name']} | {a['kind']} | `{a['asset_id']}` | "
                     f"{('`' + a['did'] + '`') if a.get('did') else '—'} |")
    cites = man.get("citations") or []
    if cites:
        lines.append("")
        lines.append("## Citations (DIDs)")
        lines.append("")
        for c in cites:
            lines.append(f"- **{c['name']}** — `{c['did']}`")
    lines.append("")
    return "\n".join(lines)


# ── module-level convenience API (for terse notebook cells) ─────────────────────
def get_run(label: Optional[str] = None, kind: str = "training", *,
            fresh: bool = False, base_dir: PathLike = ".") -> ProvenanceRun:
    """Return the active run, creating/loading one if needed.

    Multiple notebooks executed in sequence share one run via ``.dataerai_run.json``.
    Pass ``fresh=True`` to start a brand-new run.
    """
    global _current
    if _current is not None and not fresh:
        return _current
    _current = ProvenanceRun(label or "hls4ml-pipeline", kind=kind,
                             base_dir=base_dir, fresh=fresh)
    return _current


def current() -> ProvenanceRun:
    if _current is None:
        return get_run()
    return _current


def preserve_dataset(paths, **kw) -> Artifact:
    return current().preserve_dataset(paths, **kw)


def preserve_keras_model(model_or_path, **kw) -> Artifact:
    return current().preserve_keras_model(model_or_path, **kw)


def preserve_hls_project(path, **kw) -> Artifact:
    return current().preserve_hls_project(path, **kw)


def preserve_dir(path, *, kind, **kw) -> Artifact:
    return current().preserve_dir(path, kind=kind, **kw)


def preserve_array(path, **kw) -> Artifact:
    return current().preserve_array(path, **kw)


def preserve_file(path, **kw) -> Artifact:
    return current().preserve_file(path, **kw)


def add_edge(frm, to, rel_type, **kw) -> dict:
    return current().add_edge(frm, to, rel_type, **kw)


def trained_on(model, datasets) -> None:
    return current().trained_on(model, datasets)


def save(**kw) -> dict:
    return current().save(**kw)


def capture(root: PathLike = ".", *, label: str = "hls4ml-pipeline",
            kind: str = "training", close: bool = False) -> dict:
    """One-call instrumentation: preserve every pipeline output present under *root*
    and link them into the shared lineage run, then (re)write the manifest + report.

    Safe to call from the end of *every* notebook — it's idempotent (already-preserved
    artifacts and edges are reused, not duplicated) and order-independent, so the
    lineage DAG accumulates as more parts run. Pass ``close=True`` to seal the run.
    """
    from .pipeline import preserve_and_link

    run = get_run(label, kind=kind, base_dir=root)
    preserve_and_link(run, root)
    return run.save(close=close)
