"""Track neural-network training runs to Dataerai.

The repo's shipped NN-training-logging primitive is the **Lineage-run API**
(`dataerai.ml.lineage.record_training_run` on beta: open a `training` run → one
`trained_on` edge carrying ``meta={params, metrics}`` → close). There is no Keras
callback in the platform, so this module provides one: :func:`keras_callback`
returns a ``tf.keras`` callback that, for every ``model.fit(...)``, captures the
hyperparameters + per-epoch metrics, preserves them as a **training-log** asset
(and the trained model) in the notebook's collection, and registers the training
run in Dataerai's lineage graph via ``record_training_run`` (falling back to the
same lineage REST calls when the ``dataerai[ml]`` SDK isn't installed).

Importing this module does NOT import TensorFlow — the callback subclass is
created lazily inside :func:`keras_callback`, so ``import dataerai_hls4ml`` stays
light.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from .artifacts import Artifact
from .provenance import ProvenanceRun, _utcnow, _warn, get_run

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "run"


def _num(v: Any) -> Any:
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _as_dataset_artifact(run: ProvenanceRun, dataset) -> Optional[Artifact]:
    """Resolve *dataset* (an Artifact, a registered title, or a bare asset UUID)."""
    if dataset is None:
        return None
    if isinstance(dataset, Artifact):
        return dataset
    got = run.get(dataset) or (run.get("Jet-tagging dataset") if dataset == "dataset" else None)
    if got is not None:
        return got
    if isinstance(dataset, str) and _UUID_RE.match(dataset):
        return Artifact(name=dataset, kind="dataset", asset_id=dataset)
    return None


def _record_lineage_training_run(run: ProvenanceRun, dataset, params: dict,
                                 metrics: dict, name: str) -> Optional[str]:
    """Register the training run in the lineage graph using the repo's tooling
    (``record_training_run`` if the ``dataerai[ml]`` SDK is installed, else the
    equivalent lineage REST calls). Best-effort — needs the dataset's async
    provenance ``record_sk``; skips cleanly if unavailable."""
    if run.settings.dry_run or run.rest is None:
        return None
    ds = _as_dataset_artifact(run, dataset)
    if ds is None:
        return None
    try:
        sk = run.rest.resolve_record_sk(ds.asset_id)
    except Exception:
        sk = None
    if not sk:
        _warn(f"training '{name}': dataset provenance record not ready (async) — "
              "lineage training-run edge skipped; training-log asset still preserved.")
        return None
    # Prefer the platform's own helper when present.
    try:
        from dataerai.ml.lineage import record_training_run  # beta `dataerai[ml]` SDK
        creds = {"access_token": run.settings.access_token, "server": run.settings.server}
        res = record_training_run(creds, dataset_record_sk=int(sk), params=params,
                                  metrics=metrics, idempotency_key=_slug(name), cred_path=None)
        return res.get("run_id")
    except ImportError:
        pass
    except Exception as exc:
        _warn(f"record_training_run failed for '{name}' ({exc}); trying direct lineage API.")
    # Fallback: the same open → trained_on edge → close, via our REST client.
    try:
        r = run.rest.open_run("training")
        edge = {"relation": "trained_on", "src_kind": "run", "src_sk": int(r["run_sk"]),
                "dst_kind": "record", "dst_sk": int(sk),
                "meta": {"params": params, "metrics": metrics, "name": name}}
        run.rest.add_edges(r["run_id"], [edge])
        run.rest.close_run(r["run_id"])
        return r["run_id"]
    except Exception as exc:
        _warn(f"lineage training-run record failed for '{name}': {exc}")
        return None


def finalize_training(name: str, params: dict, history: list, *, dataset=None, model=None,
                      model_path=None, model_title: Optional[str] = None,
                      notebook: Optional[str] = None, root=".",
                      final_metrics: Optional[dict] = None) -> dict:
    """Preserve the training log (params + per-epoch metrics) and, when available,
    the trained model; link them; and register the training run in the lineage
    graph. Returns ``{'training_log': Artifact, 'model': Artifact|None, 'run_id': str|None}``.

    TF-free and dry-run safe — this is the testable core of :func:`keras_callback`.
    """
    run = get_run(base_dir=root)
    if notebook:
        run.use_collection(f"hls4ml — {notebook}")
    final = final_metrics or (history[-1] if history else {})

    doc = {"name": name, "kind": "training_run", "params": params,
           "final_metrics": final, "n_epochs": len(history), "epochs": history,
           "recorded_at": _utcnow(), "environment": run.env}
    log_path = Path(root) / f"{_slug(name)}.training.json"
    log_path.write_text(json.dumps(doc, indent=2, default=str))
    log_art = run.preserve_file(log_path, title=f"{name} — training log", kind="training_log",
                                metadata={"params": params, "final_metrics": final,
                                          "n_epochs": len(history)})

    model_art = None
    if model is not None or model_path:
        try:
            src = model if model is not None else model_path
            model_art = run.preserve_keras_model(
                src, title=model_title or f"{name} — trained model",
                metadata={"params": params, "final_metrics": final})
            run.add_edge(log_art, model_art, "derived_from", step="training-log")
            ds = _as_dataset_artifact(run, dataset)
            if ds is not None:
                run.trained_on(model_art, ds)
        except Exception as exc:
            _warn(f"model preservation failed for '{name}': {exc}")

    run_id = _record_lineage_training_run(run, dataset, params, final, name)
    run.save(close=False)
    return {"training_log": log_art, "model": model_art, "run_id": run_id}


def keras_callback(name: str, *, dataset=None, model_path=None, model_title=None,
                   notebook=None, params: Optional[dict] = None, root="."):
    """Return a ``tf.keras`` callback that tracks this training run to Dataerai.

    Add it to any ``model.fit(callbacks=[...])``. Captures optimizer/lr/param-count
    + per-epoch metrics, then on train-end preserves the training log + the trained
    model into the notebook's collection and records the lineage training run.

    Raises ``ImportError`` if TensorFlow isn't installed (guard with try/except in
    notebooks) — created lazily so importing the package never pulls TensorFlow.
    """
    from tensorflow import keras  # lazy: TF only needed when actually training

    base_params = dict(params or {})
    history: list = []
    started: dict = {}

    class _DataeraiTrainingCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            started["t0"] = time.time()
            try:
                opt = self.model.optimizer
                base_params.setdefault("optimizer", type(opt).__name__)
                lr = getattr(opt, "learning_rate", None)
                if lr is not None:
                    base_params.setdefault("learning_rate", float(keras.backend.get_value(lr)))
                base_params.setdefault("n_params", int(self.model.count_params()))
            except Exception:
                pass

        def on_epoch_end(self, epoch, logs=None):
            row = {"epoch": int(epoch)}
            row.update({k: _num(v) for k, v in (logs or {}).items()})
            history.append(row)

        def on_train_end(self, logs=None):
            if "t0" in started:
                base_params.setdefault("train_seconds", round(time.time() - started["t0"], 1))
            final = {k: _num(v) for k, v in (logs or {}).items()}
            try:
                finalize_training(name, base_params, history, dataset=dataset, model=self.model,
                                  model_path=model_path, model_title=model_title,
                                  notebook=notebook, root=root, final_metrics=final or None)
            except Exception as exc:  # never let tracking break training
                _warn(f"training tracking failed for '{name}': {exc}")

    return _DataeraiTrainingCallback()
