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
import base64
import hashlib
import os
import platform
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:  # pragma: no cover - unavailable on non-Unix Python builds
    import resource as _resource
except ImportError:  # pragma: no cover
    _resource = None

from . import dryrun
from .artifacts import Artifact, Preserver
from .config import Settings, load_settings

STATE_FILE = ".dataerai_run.json"
FIGURE_DIR = ".dataerai_figures"
_TOOLS = ["tensorflow", "hls4ml", "qkeras", "conifer", "numpy", "sklearn", "pysr", "xgboost"]
_ENVIRONMENT_FILES = [
    "requirements.txt",
    "environment.yml",
    "environment.yaml",
    "pyproject.toml",
    "poetry.lock",
    "setup.py",
    "setup.cfg",
]
_IMAGE_MIMES = {
    "image/png": ("png", True),
    "image/jpeg": ("jpg", True),
    "image/svg+xml": ("svg", False),
    "image/gif": ("gif", True),
}
_LOCAL_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".gif"}
_MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|<img\s+[^>]*src=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
_TOOL_VERSION_CACHE: Optional[Dict[str, str]] = None
_PIP_FREEZE_CACHE: Optional[List[str]] = None
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
    global _TOOL_VERSION_CACHE
    if _TOOL_VERSION_CACHE is not None:
        return dict(_TOOL_VERSION_CACHE)
    versions: Dict[str, str] = {}
    for name in _TOOLS:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            continue
    _TOOL_VERSION_CACHE = versions
    return dict(versions)


def _file_sha256(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _join_mime_value(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(v) for v in value)
    return "" if value is None else str(value)


def _run_command(args: List[str], *, timeout: int = 10) -> Optional[str]:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    text = out.stdout.strip()
    return text or None


def _pip_freeze() -> List[str]:
    global _PIP_FREEZE_CACHE
    if _PIP_FREEZE_CACHE is not None:
        return list(_PIP_FREEZE_CACHE)
    out = _run_command([sys.executable, "-m", "pip", "freeze"], timeout=30)
    _PIP_FREEZE_CACHE = out.splitlines() if out else []
    return list(_PIP_FREEZE_CACHE)


def _environment_files(root: Path) -> List[dict]:
    files = []
    for rel in _ENVIRONMENT_FILES:
        path = root / rel
        if not path.is_file():
            continue
        item = {"path": rel, "sha256": _file_sha256(path), "size": path.stat().st_size}
        try:
            if path.stat().st_size <= 200_000:
                item["text"] = path.read_text(errors="replace")
        except OSError:
            pass
        files.append(item)
    return files


def _total_memory_bytes() -> Optional[int]:
    if sys.platform == "darwin":
        out = _run_command(["sysctl", "-n", "hw.memsize"])
        return int(out) if out and out.isdigit() else None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    return None


def _cpu_brand() -> Optional[str]:
    if sys.platform == "darwin":
        return _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text(errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except (OSError, IndexError):
            return None
    return None


def _accelerators() -> List[dict]:
    out = _run_command([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ], timeout=5)
    if not out:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        gpu = {"name": parts[0]}
        if len(parts) > 1:
            gpu["memory_total_mb"] = parts[1]
        if len(parts) > 2:
            gpu["driver_version"] = parts[2]
        gpus.append(gpu)
    return gpus


def _hardware_inventory() -> dict:
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_brand": _cpu_brand(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _total_memory_bytes(),
        "accelerators": _accelerators(),
    }


def _resource_usage_children() -> Optional[dict]:
    if _resource is None:
        return None
    usage = _resource.getrusage(_resource.RUSAGE_CHILDREN)
    rss_unit = "bytes" if sys.platform == "darwin" else "kilobytes"
    return {
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "max_rss": usage.ru_maxrss,
        "max_rss_unit": rss_unit,
        "minor_page_faults": usage.ru_minflt,
        "major_page_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def begin_hardware_sample() -> dict:
    """Capture a lightweight before-sample for one notebook execution."""
    return {
        "started_at": _utcnow(),
        "monotonic_started": time.perf_counter(),
        "resource_usage_children": _resource_usage_children(),
    }


def _resource_delta(before: Optional[dict], after: Optional[dict]) -> Optional[dict]:
    if before is None or after is None:
        return None
    delta = {"max_rss": after.get("max_rss"), "max_rss_unit": after.get("max_rss_unit")}
    for key in [
        "user_cpu_seconds",
        "system_cpu_seconds",
        "minor_page_faults",
        "major_page_faults",
        "voluntary_context_switches",
        "involuntary_context_switches",
    ]:
        delta[key] = after.get(key, 0) - before.get(key, 0)
    before_rss, after_rss = before.get("max_rss"), after.get("max_rss")
    if isinstance(before_rss, (int, float)) and isinstance(after_rss, (int, float)):
        delta["max_rss_delta"] = max(after_rss - before_rss, 0)
    return delta


def write_notebook_hardware_metrics(
    root: PathLike,
    notebook: str,
    *,
    sample: Optional[dict] = None,
    success: Optional[bool] = None,
    error: Optional[str] = None,
    kernel: Optional[str] = None,
    source: str = "papermill",
) -> Path:
    """Write the hardware/execution metrics JSON consumed by preservation."""
    root = Path(root)
    now = _utcnow()
    after = _resource_usage_children()
    execution = {
        "source": source,
        "kernel": kernel,
        "success": success,
        "started_at": (sample or {}).get("started_at"),
        "ended_at": now,
        "wall_seconds": None,
        "resource_usage_children": after,
        "resource_usage_delta": _resource_delta((sample or {}).get("resource_usage_children"), after),
    }
    if sample and sample.get("monotonic_started") is not None:
        execution["wall_seconds"] = time.perf_counter() - sample["monotonic_started"]
    if error:
        execution["error"] = error
    payload = {
        "notebook": notebook,
        "recorded_at": now,
        "hardware": _hardware_inventory(),
        "execution": execution,
    }
    path = root / f"{notebook}.hardware.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _environment_snapshot(root: Path, notebook: str, run_env: dict) -> dict:
    conda = {
        "default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "prefix": os.environ.get("CONDA_PREFIX"),
    }
    return {
        "notebook": notebook,
        "recorded_at": _utcnow(),
        "run": run_env,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": getattr(sys, "base_prefix", sys.prefix),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "conda": {k: v for k, v in conda.items() if v},
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "tools": _tool_versions(),
        "pip_freeze": _pip_freeze(),
        "environment_files": _environment_files(root),
    }


def _write_environment_snapshot(root: Path, notebook: str, run_env: dict) -> Path:
    path = root / f"{notebook}.environment.json"
    path.write_text(json.dumps(_environment_snapshot(root, notebook, run_env), indent=2))
    return path


def _figure_destination(root: Path, notebook: str) -> Path:
    return root / FIGURE_DIR / notebook


def _write_mime_figure(dest: Path, filename: str, mime: str, value: Any) -> Path:
    ext, is_binary = _IMAGE_MIMES[mime]
    path = dest / f"{filename}.{ext}"
    raw = _join_mime_value(value)
    if is_binary:
        path.write_bytes(base64.b64decode(raw))
    else:
        path.write_text(raw)
    return path


def _markdown_image_sources(text: str) -> List[str]:
    sources = []
    for match in _MARKDOWN_IMAGE_RE.finditer(text):
        src = match.group(1) or match.group(2)
        if not src:
            continue
        src = src.strip().strip("'\"")
        if src.startswith(("http://", "https://", "attachment:")):
            continue
        sources.append(src)
    return sources


def _extract_output_figures(nb_path: PathLike, root: PathLike, notebook: str) -> List[dict]:
    """Extract notebook-visible figures as standalone files for preservation."""
    nb_path = Path(nb_path)
    root = Path(root)
    data = json.loads(nb_path.read_text())
    dest = _figure_destination(root, notebook)
    if dest.exists():
        for old in dest.glob("*"):
            if old.is_file():
                old.unlink()
    dest.mkdir(parents=True, exist_ok=True)

    figures: List[dict] = []
    for cell_i, cell in enumerate(data.get("cells", [])):
        for output_i, output in enumerate(cell.get("outputs", [])):
            bundle = output.get("data") or {}
            for mime in _IMAGE_MIMES:
                if mime not in bundle:
                    continue
                path = _write_mime_figure(
                    dest, f"output-cell-{cell_i:03d}-{output_i:02d}", mime, bundle[mime])
                figures.append({
                    "path": path,
                    "metadata": {
                        "notebook": notebook,
                        "figure_source": "cell-output",
                        "source_notebook": str(nb_path),
                        "cell_index": cell_i,
                        "output_index": output_i,
                        "execution_count": cell.get("execution_count"),
                        "mime_type": mime,
                    },
                })
        for attachment_name, bundle in (cell.get("attachments") or {}).items():
            for mime in _IMAGE_MIMES:
                if mime not in bundle:
                    continue
                safe_name = Path(attachment_name).stem.replace("/", "_")
                path = _write_mime_figure(
                    dest, f"attachment-cell-{cell_i:03d}-{safe_name}", mime, bundle[mime])
                figures.append({
                    "path": path,
                    "metadata": {
                        "notebook": notebook,
                        "figure_source": "notebook-attachment",
                        "source_notebook": str(nb_path),
                        "cell_index": cell_i,
                        "attachment_name": attachment_name,
                        "mime_type": mime,
                    },
                })
        if cell.get("cell_type") == "markdown":
            source = _join_mime_value(cell.get("source", []))
            for ref_i, src in enumerate(_markdown_image_sources(source)):
                local = (nb_path.parent / src).resolve()
                if local.suffix.lower() not in _LOCAL_IMAGE_EXTS or not local.is_file():
                    continue
                suffix = "jpg" if local.suffix.lower() == ".jpeg" else local.suffix.lower().lstrip(".")
                path = dest / f"markdown-cell-{cell_i:03d}-{ref_i:02d}.{suffix}"
                path.write_bytes(local.read_bytes())
                figures.append({
                    "path": path,
                    "metadata": {
                        "notebook": notebook,
                        "figure_source": "markdown-image",
                        "source_notebook": str(nb_path),
                        "cell_index": cell_i,
                        "source_path": str(local),
                    },
                })
    return figures


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
        self._batched = False  # whether any Layer-3 lineage batch was appended
        self.artifacts: Dict[str, Artifact] = {}
        self.edges: List[dict] = []
        self._collections: Dict[str, Optional[str]] = {}  # title -> collection id
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
        # The Layer-3 sealed run is opened lazily (see _ensure_run) — only when a
        # lineage batch is actually appended (trained-on with DIDs). Opening one
        # eagerly and closing it empty is a 409 ("no committed batches").

    def _ensure_run(self) -> Optional[str]:
        """Open the sealed lineage run on first real use; returns its id or None."""
        if self.run_id or self.settings.dry_run or self.rest is None:
            return self.run_id
        try:
            resp = self.rest.open_run(self.kind)
            self.run_id = resp.get("run_id")
            self.run_sk = resp.get("run_sk")
        except Exception as exc:
            _warn(f"lineage run open failed ({exc}); relationship graph (Layer 2) still recorded.")
        return self.run_id

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

    # ── per-notebook collections ─────────────────────────────────────────────────
    def use_collection(self, title: str) -> Optional[str]:
        """Route subsequently-preserved artifacts into the collection *title*
        (get-or-create), so each notebook's outputs live in their own collection."""
        if title in self._collections:
            cid = self._collections[title]
        elif self.settings.dry_run:
            cid = dryrun.fake_asset_id(f"collection:{title}")
        elif self.rest is not None and self.settings.project_id:
            try:
                cid = self.rest.get_or_create_collection(self.settings.project_id, title)
            except Exception as exc:
                _warn(f"collection '{title}' unavailable ({exc}); using project root.")
                cid = None
        else:
            cid = None
        self._collections[title] = cid
        self.settings.collection_id = cid  # Preserver routes uploads here
        return cid

    def preserve_notebook_record(self, root, notebook: str) -> None:
        """Static 'recording': preserve the notebook file + execution artifacts
        into the active collection. Prefers the papermill-executed copy."""
        root = Path(root)
        nb = next((p for p in (root / f"{notebook}.executed.ipynb", root / f"{notebook}.ipynb")
                   if p.exists()), None)
        if nb is None:
            return
        rec = self.preserve_file(nb, title=f"{notebook} — notebook", kind="notebook",
                                 metadata={"notebook": notebook})
        try:
            log_path = root / f"{notebook}.execution.log"
            log_path.write_text(_extract_execution_log(nb))
            log = self.preserve_file(log_path, title=f"{notebook} — execution log", kind="log",
                                     metadata={"notebook": notebook})
            self.add_edge(log, rec, "derived_from", step="execution-log")
        except Exception as exc:
            _warn(f"execution-log capture failed for {notebook} ({exc}).")
        try:
            env_path = _write_environment_snapshot(root, notebook, self.env)
            env = self.preserve_file(env_path, title=f"{notebook} — environment",
                                     kind="environment", metadata={"notebook": notebook})
            self.add_edge(env, rec, "derived_from", step="environment")
        except Exception as exc:
            _warn(f"environment capture failed for {notebook} ({exc}).")
        try:
            hardware_path = root / f"{notebook}.hardware.json"
            if not hardware_path.exists():
                hardware_path = write_notebook_hardware_metrics(
                    root, notebook, success=None, source="capture")
            hardware = self.preserve_file(
                hardware_path,
                title=f"{notebook} — hardware metrics",
                kind="hardware_metrics",
                metadata={"notebook": notebook},
            )
            self.add_edge(hardware, rec, "derived_from", step="hardware-metrics")
        except Exception as exc:
            _warn(f"hardware-metrics capture failed for {notebook} ({exc}).")
        try:
            for i, figure in enumerate(_extract_output_figures(nb, root, notebook), start=1):
                fig = self.preserve_file(
                    figure["path"],
                    title=f"{notebook} — figure {i:02d}",
                    kind="figure",
                    metadata=figure["metadata"],
                )
                self.add_edge(fig, rec, "derived_from", step="output-figure")
        except Exception as exc:
            _warn(f"figure capture failed for {notebook} ({exc}).")

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
        if dids and not self.settings.dry_run and self.rest is not None and self._ensure_run():
            try:
                self.rest.trained_on(self.run_id, dids)
                self._batched = True
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
        if close and self.run_id and self._batched and not self.settings.dry_run and self.rest is not None:
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
        (self.base_dir / "provenance_graph.dot").write_text(to_dot(man))
        self._save_state()
        return man

    # ── context manager ──────────────────────────────────────────────────────────
    def __enter__(self) -> "ProvenanceRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Don't seal the shared run on error; just persist what we have.
        self.save(close=exc_type is None)


_KIND_STYLE = {
    "dataset": ("#e8f0fe", "#5b8def"),
    "model": ("#f3e8fd", "#954c9d"),
    "hls_project": ("#e9f7ef", "#2e8b57"),
    "array": ("#fff4e5", "#e67e22"),
    "bitstream": ("#fde8e8", "#c0392b"),
    "notebook": ("#fff9db", "#d4a72c"),
    "log": ("#f1f3f5", "#868e96"),
    "environment": ("#e6fcf5", "#12b886"),
    "hardware_metrics": ("#e7f5ff", "#228be6"),
    "figure": ("#fff0f6", "#d6336c"),
    "artifact": ("#f0f0f0", "#888888"),
}


def _extract_execution_log(nb_path) -> str:
    """Render a notebook's saved cell outputs as a plain-text execution log."""
    data = json.loads(Path(nb_path).read_text())
    out = [f"# Execution log — {Path(nb_path).name}", ""]
    for i, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        out.append(f"## In[{cell.get('execution_count')}]  (cell {i})")
        if src.strip():
            out += ["```python", src.rstrip(), "```"]
        for o in cell.get("outputs", []):
            ot = o.get("output_type")
            if ot == "stream":
                out.append("".join(o.get("text", [])).rstrip())
            elif ot in ("execute_result", "display_data"):
                txt = o.get("data", {}).get("text/plain")
                if txt:
                    out.append("".join(txt).rstrip())
            elif ot == "error":
                out.append(f"[ERROR] {o.get('ename')}: {o.get('evalue')}")
        out.append("")
    return "\n".join(out)


def to_dot(man: dict) -> str:
    """Render a lineage manifest as a Graphviz DOT graph (``dot -Tsvg`` to view)."""
    out = [
        "digraph provenance {",
        f'  label="Dataerai lineage — {man["label"]}"; labelloc=t; fontname="Helvetica"; fontsize=14;',
        "  rankdir=LR;",
        '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10];',
        '  edge [fontname="Helvetica" fontsize=9 color="#555555"];',
    ]
    ids = {a["asset_id"]: f"n{i}" for i, a in enumerate(man["artifacts"])}
    for a in man["artifacts"]:
        fill, border = _KIND_STYLE.get(a["kind"], ("#f5f5f5", "#999999"))
        label = a["name"].replace('"', "'") + f"\\n({a['kind']})"
        out.append(f'  {ids[a["asset_id"]]} [label="{label}" fillcolor="{fill}" color="{border}"];')
    for e in man["edges"]:
        s, d = ids.get(e["from"]), ids.get(e["to"])
        if not s or not d:
            continue
        lbl = e["type"] + (f"\\n{e['step']}" if e.get("step") else "")
        out.append(f'  {s} -> {d} [label="{lbl}"];')
    out.append("}")
    return "\n".join(out)


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


def capture(root: PathLike = ".", *, notebook: Optional[str] = None,
            label: str = "hls4ml-pipeline", kind: str = "training",
            close: bool = False) -> dict:
    """One-call instrumentation: preserve this notebook's pipeline outputs, link them
    into the shared lineage run, and (re)write the manifest + report.

    When *notebook* is given, that notebook's artifacts — plus a static "recording"
    (the notebook file + its execution log) — are routed into a per-notebook Dataerai
    collection. Safe to call from the end of *every* notebook: idempotent
    (already-preserved artifacts/edges are reused) and order-independent, so the
    lineage DAG accumulates as more parts run. Pass ``close=True`` to seal the run.
    """
    from .pipeline import NOTEBOOK_ARTIFACTS, preserve_and_link

    run = get_run(label, kind=kind, base_dir=root)
    only = None
    if notebook:
        run.use_collection(f"hls4ml — {notebook}")
        only = NOTEBOOK_ARTIFACTS.get(notebook)
    preserve_and_link(run, root, only=only)
    if notebook:
        run.preserve_notebook_record(root, notebook)
    return run.save(close=close)
