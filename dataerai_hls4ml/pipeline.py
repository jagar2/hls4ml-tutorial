"""Canonical hls4ml-pipeline artifact layout + lineage edge map.

:func:`preserve_and_link` walks whatever pipeline outputs exist on disk and
records the full lineage DAG — so it works whether you've run a single notebook
or the whole tutorial. The notebooks call the fine-grained ``dp.preserve_*`` /
``dp.add_edge`` API inline; this module is the one-call equivalent used by the
orchestrator, the self-test, and anyone who'd rather instrument once at the end.

Artifact filenames/dirs match the upstream tutorial exactly (verified against the
notebooks): ``model_1/KERAS_check_best_model.h5``, ``model_3/hls4ml_prj_pynq``,
``model_3/y_hls.npy``, ``pruned_cnn/``, ``sr/`` …
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .artifacts import Artifact
from .provenance import ProvenanceRun

# name -> (kind, relative path(s), human title)
LAYOUT: List[Tuple[str, str, Union[str, List[str]], str]] = [
    ("dataset", "dataset",
     ["X_train_val.npy", "y_train_val.npy", "X_test.npy", "y_test.npy", "classes.npy"],
     "Jet-tagging dataset"),
    ("model_1", "model", "model_1/KERAS_check_best_model.h5", "Baseline MLP (part1)"),
    ("model_1_hls", "hls_project", "model_1/hls4ml_prj", "HLS project — baseline (part1)"),
    ("model_2", "model", "model_2/KERAS_check_best_model.h5", "Pruned MLP (part3)"),
    ("model_2_hls", "hls_project", "model_2/hls4ml_prj", "HLS project — pruned (part3)"),
    ("model_3", "model", "model_3/KERAS_check_best_model.h5", "Quantized QKeras MLP (part4)"),
    ("model_3_hls", "hls_project", "model_3/hls4ml_prj", "HLS project — quantized (part4)"),
    ("model_3_hls_pynq", "hls_project", "model_3/hls4ml_prj_pynq", "HLS project — PYNQ-Z2 (part7)"),
    ("y_hls", "array", "model_3/y_hls.npy", "HLS inference predictions (part7)"),
    ("cnn_pruned", "model", "pruned_cnn", "Pruned CNN (part6)"),
    ("cnn_quantized", "model", "quantized_pruned_cnn", "Quantized pruned CNN (part6)"),
    ("bdt", "model", "model_5/xgboost_model.json", "BDT — XGBoost (part5)"),
    ("bdt_hls", "hls_project", "model_5", "Conifer HLS project — BDT (part5)"),
    ("sr", "artifact", "sr", "Symbolic regression (part8)"),
]

# (from, to, relation, step) — an edge is created only if BOTH endpoints exist.
EDGES: List[Tuple[str, str, str, Optional[str]]] = [
    ("model_1", "dataset", "trained_on", None),
    ("model_1_hls", "model_1", "derived_from", "synthesize"),
    ("model_2", "model_1", "derived_from", "prune"),
    ("model_2", "dataset", "trained_on", None),
    ("model_2_hls", "model_2", "derived_from", "synthesize"),
    ("model_3", "model_2", "derived_from", "quantize"),
    ("model_3", "dataset", "trained_on", None),
    ("model_3_hls", "model_3", "derived_from", "synthesize"),
    ("model_3_hls_pynq", "model_3", "derived_from", "synthesize"),
    ("y_hls", "model_3_hls_pynq", "evaluated_on", "infer"),
    ("cnn_quantized", "cnn_pruned", "derived_from", "quantize"),
    ("bdt", "dataset", "trained_on", None),
    ("bdt_hls", "bdt", "derived_from", "synthesize"),
]


def _resolve(root: Path, spec) -> Optional[object]:
    _name, _kind, rel, _title = spec
    if isinstance(rel, list):
        existing = [root / r for r in rel if (root / r).exists()]
        return existing or None
    target = root / rel
    return target if target.exists() else None


def preserve_and_link(run: ProvenanceRun, root: Union[str, Path] = ".") -> Dict[str, Artifact]:
    """Preserve every pipeline output present under *root* and link them per the edge map."""
    from .provenance import _warn

    root = Path(root)
    preserved: Dict[str, Artifact] = {}
    for spec in LAYOUT:
        name, kind, _rel, title = spec
        # Idempotent across repeated/cross-notebook calls: reuse an already-preserved
        # artifact (same title) instead of re-uploading it.
        existing = run.get(title)
        if existing is not None:
            preserved[name] = existing
            continue
        target = _resolve(root, spec)
        if target is None:
            continue
        try:
            if kind == "dataset":
                art = run.preserve_dataset(target, title=title)
            elif kind == "model":
                model_path = target[0] if isinstance(target, list) else target
                art = run.preserve_keras_model(model_path, title=title)
            elif kind == "hls_project":
                art = run.preserve_hls_project(target, title=title)
            elif kind == "array":
                art = run.preserve_array(target, title=title)
            else:
                art = run.preserve_dir(target, kind=kind, title=title)
            preserved[name] = art
        except Exception as exc:
            _warn(f"could not preserve {name}: {exc}")

    for frm, to, rel_type, step in EDGES:
        if frm in preserved and to in preserved:
            run.add_edge(preserved[frm], preserved[to], rel_type, step=step)
    return preserved


def make_synthetic_workspace(root: Union[str, Path]) -> Path:
    """Create tiny placeholder files mirroring the pipeline outputs (self-test / tests)."""
    root = Path(root)

    def put(rel: str, content: bytes = b"placeholder\n") -> None:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

    for npy in ["X_train_val.npy", "y_train_val.npy", "X_test.npy", "y_test.npy", "classes.npy"]:
        put(npy, npy.encode() + b"\x00data")
    put("model_1/KERAS_check_best_model.h5", b"h5-model-1")
    put("model_1/hls4ml_prj/firmware/myproject.cpp", b"// hls baseline")
    put("model_1/hls4ml_prj/vitis_hls.log", b"synthesis log")
    put("model_2/KERAS_check_best_model.h5", b"h5-model-2-pruned")
    put("model_2/hls4ml_prj/firmware/myproject.cpp", b"// hls pruned")
    put("model_3/KERAS_check_best_model.h5", b"h5-model-3-quantized")
    put("model_3/hls4ml_prj/firmware/myproject.cpp", b"// hls quantized")
    put("model_3/hls4ml_prj_pynq/firmware/myproject_axi.cpp", b"// hls pynq")
    put("model_3/y_hls.npy", b"y_hls\x00preds")
    put("pruned_cnn/keras_model.h5", b"cnn-pruned")
    put("quantized_pruned_cnn/keras_model.h5", b"cnn-quantized")
    put("model_5/xgboost_model.json", b'{"bdt": true}')
    put("model_5/my_prj.json", b'{"conifer": true}')
    put("sr/best_equations.csv", b"complexity,equation\n1,x0\n")
    return root
