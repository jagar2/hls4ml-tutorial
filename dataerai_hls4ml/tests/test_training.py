import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from dataerai_hls4ml import provenance, training
from dataerai_hls4ml.provenance import ProvenanceRun


def _run(dry_settings, base):
    r = ProvenanceRun("t", settings=dry_settings, base_dir=base, fresh=True)
    provenance._current = r  # finalize_training resolves the active run via get_run()
    return r


def test_slug_and_num():
    assert training._slug("Part6 Pruned CNN!") == "part6-pruned-cnn"
    assert training._num("0.5") == 0.5
    assert training._num("abc") == "abc"


def test_finalize_training_dry_run(dry_settings, workspace):
    run = _run(dry_settings, workspace)
    hist = [{"epoch": 0, "loss": 1.0, "accuracy": 0.5},
            {"epoch": 1, "loss": 0.5, "accuracy": 0.82}]
    out = training.finalize_training(
        "part6 pruned CNN", {"optimizer": "Adam", "learning_rate": 3e-3}, hist,
        notebook="part6_cnns", root=str(workspace),
        model_path=str(workspace / "model_1" / "KERAS_check_best_model.h5"))

    log = out["training_log"]
    assert log.kind == "training_log"
    assert log.metadata["n_epochs"] == 2
    doc = json.loads((workspace / "part6-pruned-cnn.training.json").read_text())
    assert doc["params"]["optimizer"] == "Adam"
    assert len(doc["epochs"]) == 2
    assert doc["final_metrics"]["accuracy"] == 0.82
    # trained model preserved and linked: training log --derived_from--> model
    assert out["model"] is not None
    assert any(e["from"] == log.asset_id and e["to"] == out["model"].asset_id
               and e["type"] == "derived_from" for e in run.edges)


def test_finalize_without_model(dry_settings, workspace):
    run = _run(dry_settings, workspace)
    out = training.finalize_training("mlp", {"epochs": 3}, [{"epoch": 0, "loss": 0.1}],
                                     notebook="part1_getting_started", root=str(workspace))
    assert out["model"] is None
    assert out["training_log"].kind == "training_log"


def test_import_does_not_pull_tensorflow():
    # The callback is created lazily, so importing the package must not import TF.
    assert "tensorflow" not in sys.modules


def test_keras_callback_lazy_tf():
    if importlib.util.find_spec("tensorflow") is None:
        with pytest.raises(ImportError):
            training.keras_callback("x", notebook="part6_cnns")
    else:  # pragma: no cover - only when TF is installed
        import tensorflow as tf
        cb = training.keras_callback("x", notebook="part6_cnns")
        assert isinstance(cb, tf.keras.callbacks.Callback)


def _notebook_sources(name: str):
    path = Path(__file__).resolve().parents[2] / name
    nb = json.loads(path.read_text())
    return ["".join(cell.get("source", [])) for cell in nb.get("cells", [])]


def test_keras_training_cells_include_dataerai_callback():
    keras_fit = re.compile(r"\b(model|model_pruned|qmodel_pruned|aqmodel)\.fit\s*\(")
    notebooks = [
        "part1_getting_started.ipynb",
        "part3_compression.ipynb",
        "part4_quantization.ipynb",
        "part4.1_HG_quantization.ipynb",
        "part6_cnns.ipynb",
    ]
    missing = []
    for notebook in notebooks:
        for i, source in enumerate(_notebook_sources(notebook)):
            if keras_fit.search(source) and "keras_callback(" not in source:
                missing.append(f"{notebook}:cell {i}")
    assert missing == []


def test_all_callbacks_notebooks_append_to_inner_callback_list():
    # all_callbacks is an object whose Keras callback list lives on `.callbacks`;
    # assigning `callbacks = list(callbacks) + ...` silently disables tracking.
    for notebook in [
        "part1_getting_started.ipynb",
        "part3_compression.ipynb",
        "part4_quantization.ipynb",
    ]:
        source = "\n".join(_notebook_sources(notebook))
        assert "callbacks.callbacks.append(_dp.keras_callback" in source
        assert "callbacks = list(callbacks) + [_dp.keras_callback" not in source
