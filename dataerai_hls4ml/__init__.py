"""Dataerai data-preservation & provenance for the hls4ml tutorial.

Weaves Dataerai into the hls4ml ML→FPGA pipeline so every artifact — the dataset,
each model version, each HLS project, the bitstream and eval outputs — is
**preserved** as a versioned, citable Dataerai asset and **linked** into one
lineage DAG, using only the existing beta APIs (no backend/SDK changes).

Typical instrumented notebook cell::

    import dataerai_hls4ml as dp
    if dp.enabled():
        run = dp.get_run("part1_getting_started", kind="training")
        D  = dp.preserve_dataset(["X_train_val.npy", "y_train_val.npy",
                                  "X_test.npy", "y_test.npy", "classes.npy"])
        M1 = dp.preserve_keras_model("model_1/KERAS_check_best_model.h5",
                                     metadata={"accuracy": acc})
        dp.trained_on(M1, D)
        H1 = dp.preserve_hls_project("model_1/hls4ml_prj")
        dp.add_edge(H1, M1, "derived_from", step="synthesize")
        dp.save()

Run offline (no creds) with ``DATAERAI_DRY_RUN=1`` — the full lineage is still
recorded to ``provenance_manifest.json`` + ``PROVENANCE.md``.
"""
from __future__ import annotations

from .artifacts import Artifact
from .config import Settings, enabled, load_settings
from .pipeline import preserve_and_link
from .provenance import (
    ProvenanceRun,
    add_edge,
    capture,
    current,
    get_run,
    preserve_array,
    preserve_dataset,
    preserve_dir,
    preserve_file,
    preserve_hls_project,
    preserve_keras_model,
    reference_dataset,
    refresh,
    save,
    trained_on,
)
from .training import finalize_training, keras_callback

__version__ = "0.1.0"

__all__ = [
    "Artifact",
    "ProvenanceRun",
    "Settings",
    "__version__",
    "add_edge",
    "capture",
    "current",
    "enabled",
    "finalize_training",
    "get_run",
    "keras_callback",
    "load_settings",
    "preserve_and_link",
    "preserve_array",
    "preserve_dataset",
    "preserve_dir",
    "preserve_file",
    "preserve_hls_project",
    "preserve_keras_model",
    "reference_dataset",
    "refresh",
    "save",
    "trained_on",
]
