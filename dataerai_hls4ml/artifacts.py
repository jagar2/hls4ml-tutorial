"""Preserve hls4ml pipeline artifacts as versioned Dataerai assets (Layer 1).

Each helper turns a file / model / directory into a single uploaded asset and
returns an :class:`Artifact` carrying the backend ``asset_id`` + ``content_id``
(its version) and, when minted, its ``did``. Datasets and HLS-project
directories are tarred to one archive first so they upload as a single
versioned asset.

In dry-run mode nothing is uploaded: a deterministic synthetic id is produced
from the artifact's name + a cheap (path, size) signature so manifests are
stable and diff-able.
"""
from __future__ import annotations

import atexit
import hashlib
import os
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from . import dryrun
from .config import Settings

PathLike = Union[str, os.PathLike]


@dataclass
class Artifact:
    """A preserved pipeline artifact (one Dataerai asset version)."""

    name: str
    kind: str  # dataset | model | hls_project | array | bitstream | report | file
    asset_id: str
    content_id: Optional[str] = None
    did: Optional[str] = None
    path: Optional[str] = None
    sha256: Optional[str] = None
    size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "asset_id": self.asset_id,
            "content_id": self.content_id,
            "did": self.did,
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "metadata": self.metadata,
        }


# ── hashing / archiving helpers ────────────────────────────────────────────────
def _sha256(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cheap_signature(paths: Iterable[PathLike]) -> str:
    """A fast, deterministic content signature (name + size) for dry-run ids."""
    parts = []
    for p in sorted(str(x) for x in paths):
        try:
            parts.append(f"{Path(p).name}:{os.path.getsize(p)}")
        except OSError:
            parts.append(f"{Path(p).name}:?")
    return "|".join(parts)


def _expand(paths: Iterable[PathLike]) -> list:
    """Expand files + directories to a flat, existing file list."""
    out: list = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.is_file()))
        elif p.exists():
            out.append(p)
    return out


def _tar(sources: Iterable[PathLike], dest_tar: PathLike, *, base: Optional[Path] = None) -> str:
    with tarfile.open(dest_tar, "w:gz") as tar:
        for src in sources:
            src = Path(src)
            arc = src.relative_to(base) if base else src.name
            tar.add(src, arcname=str(arc))
    return str(dest_tar)


def _npy_shape(path: PathLike) -> Optional[list]:
    """Read a .npy header's shape without loading the array (best-effort)."""
    try:
        import numpy as np
        with open(path, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            shape, _fortran, _dtype = np.lib.format._read_array_header(fh, version)
        return list(shape)
    except Exception:
        return None


# ── SDK client (lazy singleton) ────────────────────────────────────────────────
_CLIENT = None


def _get_client(settings: Settings):
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    try:
        import dataerai
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the Dataerai Python SDK is required for live preservation; "
            "`pip install -e <dataerai-repo>/sdk/python` or run with --dry-run"
        ) from exc
    Client = getattr(dataerai, "DataeraiClient", None) or getattr(dataerai, "DataEraiClient", None)
    if Client is None:
        raise RuntimeError("the Dataerai Python SDK does not expose a daemon client")
    client = Client(binary_path=settings.binary_path)
    client.connect()
    atexit.register(_close_client)
    _CLIENT = client
    return _CLIENT


def _close_client() -> None:  # pragma: no cover - process teardown
    global _CLIENT
    if _CLIENT is not None:
        try:
            _CLIENT.close()
        finally:
            _CLIENT = None


class Preserver:
    """Uploads artifacts as Dataerai assets, or synthesizes them in dry-run."""

    def __init__(self, settings: Settings, rest=None):
        self.settings = settings
        self.rest = rest
        self._tmp = Path(tempfile.mkdtemp(prefix="dataerai-hls4ml-"))

    # ── public preserve_* ──────────────────────────────────────────────────────
    def preserve_file(self, path: PathLike, *, title: Optional[str] = None,
                      kind: str = "file", tags=None, metadata=None) -> Artifact:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"cannot preserve missing file: {path}")
        return self._commit([path], single=path, title=title or path.name,
                            kind=kind, tags=tags, metadata=metadata)

    def preserve_array(self, path: PathLike, *, title=None, tags=None, metadata=None) -> Artifact:
        md = dict(metadata or {})
        shape = _npy_shape(path)
        if shape is not None:
            md.setdefault("shape", shape)
        return self.preserve_file(path, title=title, kind="array", tags=tags, metadata=md)

    def preserve_dataset(self, paths, *, title=None, tags=None, metadata=None) -> Artifact:
        """Preserve a bundle of dataset files (e.g. the jet-tagging ``.npy`` set) as one asset."""
        if isinstance(paths, dict):
            items = list(paths.values())
        else:
            items = list(paths)
        files = _expand(items)
        if not files:
            raise FileNotFoundError(f"no dataset files found among: {paths}")
        md = dict(metadata or {})
        md.setdefault("files", [Path(f).name for f in files])
        shapes = {Path(f).name: _npy_shape(f) for f in files if str(f).endswith(".npy")}
        shapes = {k: v for k, v in shapes.items() if v}
        if shapes:
            md.setdefault("shapes", shapes)
        title = title or "dataset"
        if self.settings.dry_run:
            return self._synth(title, "dataset", files, md)
        tar = _tar(files, self._tmp / "dataset.tar.gz")
        return self._upload(tar, title=title, kind="dataset", tags=tags, metadata=md)

    def preserve_keras_model(self, model_or_path, *, title=None, tags=None, metadata=None) -> Artifact:
        """Preserve a Keras model — accepts an ``.h5``/dir path or a live model object."""
        md = dict(metadata or {})
        if hasattr(model_or_path, "save") and not isinstance(model_or_path, (str, os.PathLike)):
            # A live Keras model: capture a little structure, then serialize.
            try:
                md.setdefault("param_count", int(model_or_path.count_params()))
                md.setdefault("layers", [l.__class__.__name__ for l in model_or_path.layers])
            except Exception:
                pass
            dest = self._tmp / f"{(title or 'model').replace('/', '_')}.h5"
            model_or_path.save(dest)
            path = dest
        else:
            path = Path(model_or_path)
            if not path.exists():
                raise FileNotFoundError(f"cannot preserve missing model: {path}")
        title = title or path.name
        if path.is_dir():  # SavedModel directory
            if self.settings.dry_run:
                return self._synth(title, "model", _expand([path]), md)
            tar = _tar(_expand([path]), self._tmp / f"{title.replace('/', '_')}.tar.gz", base=path.parent)
            return self._upload(tar, title=title, kind="model", tags=tags, metadata=md)
        return self._commit([path], single=path, title=title, kind="model", tags=tags, metadata=md)

    def preserve_dir(self, path: PathLike, *, kind: str, title=None, tags=None, metadata=None) -> Artifact:
        """Preserve a directory (tarred) as one asset with an arbitrary ``kind``."""
        path = Path(path)
        files = _expand([path])
        if not files:
            raise FileNotFoundError(f"no files found under directory: {path}")
        md = dict(metadata or {})
        md.setdefault("file_count", len(files))
        title = title or path.name
        if self.settings.dry_run:
            return self._synth(title, kind, files, md)
        tar = _tar(files, self._tmp / f"{title.replace('/', '_')}.tar.gz", base=path.parent)
        return self._upload(tar, title=title, kind=kind, tags=tags, metadata=md)

    def preserve_hls_project(self, path: PathLike, *, title=None, tags=None, metadata=None) -> Artifact:
        """Preserve an hls4ml project directory (tarred) as one asset."""
        return self.preserve_dir(path, kind="hls_project", title=title, tags=tags, metadata=metadata)

    # ── internals ──────────────────────────────────────────────────────────────
    def _commit(self, files, *, single, title, kind, tags, metadata) -> Artifact:
        md = dict(metadata or {})
        if self.settings.dry_run:
            return self._synth(title, kind, files, md)
        return self._upload(single, title=title, kind=kind, tags=tags, metadata=md)

    def _synth(self, title, kind, files, metadata) -> Artifact:
        sig = _cheap_signature(files)
        aid = dryrun.fake_asset_id(f"{title}:{sig}")
        size = sum((os.path.getsize(f) for f in files if os.path.exists(f)), 0)
        md = dict(metadata)
        md.update({"artifact_kind": kind, "signature": sig, "dry_run": True})
        return Artifact(name=title, kind=kind, asset_id=aid,
                        content_id=dryrun.fake_content_id(f"{title}:{sig}"),
                        did=dryrun.fake_did(aid),
                        path=str(files[0]) if files else None,
                        sha256=None, size=size, metadata=md)

    def _upload(self, file_path, *, title, kind, tags, metadata) -> Artifact:
        if not self.settings.owner_id:
            raise RuntimeError(
                "no owner project/collection resolved — set DATAERAI_PROJECT_ID"
            )
        file_path = str(file_path)
        sha = _sha256(file_path)
        size = os.path.getsize(file_path)
        md = dict(metadata or {})
        md.update({"artifact_kind": kind, "sha256": sha})
        kwargs = dict(
            title=title,
            owner_type=self.settings.owner_type,
            owner_id=self.settings.owner_id,
            tags=tags or [kind, "hls4ml"],
            metadata=md,
        )
        if self.settings.collection_id and self.settings.owner_type == "project":
            kwargs["collection_id"] = self.settings.collection_id
        client = _get_client(self.settings)
        result = client.upload(file_path, **kwargs)
        did = None
        if self.rest is not None:
            try:
                did = self.rest.get_asset(result.asset_id).get("did") or None
            except Exception:
                did = None
        return Artifact(name=title, kind=kind, asset_id=result.asset_id,
                        content_id=result.content_id, did=did, path=file_path,
                        sha256=sha, size=size, metadata=md)
