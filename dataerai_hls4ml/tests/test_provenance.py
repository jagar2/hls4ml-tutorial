import base64
import json

from dataerai_hls4ml.artifacts import Artifact
from dataerai_hls4ml.provenance import STATE_FILE, ProvenanceRun


def _run(settings, base):
    return ProvenanceRun("test-pipeline", kind="training", settings=settings, base_dir=base, fresh=True)


def test_preserve_edge_and_manifest(dry_settings, workspace):
    run = _run(dry_settings, workspace)
    ds = run.preserve_dataset([workspace / "X_train_val.npy", workspace / "classes.npy"], title="ds")
    m1 = run.preserve_keras_model(workspace / "model_1" / "KERAS_check_best_model.h5", title="m1")
    run.trained_on(m1, ds)
    man = run.save()

    assert man["dry_run"] is True
    assert man["run_id"]
    assert {a["name"] for a in man["artifacts"]} == {"ds", "m1"}
    assert any(e["type"] == "trained_on" and e["from"] == m1.asset_id and e["to"] == ds.asset_id
               for e in man["edges"])
    # files written
    assert (workspace / "provenance_manifest.json").exists()
    assert (workspace / "PROVENANCE.md").exists()
    assert "```mermaid" in (workspace / "PROVENANCE.md").read_text()


def test_save_writes_dot_graph(dry_settings, workspace):
    from dataerai_hls4ml.provenance import to_dot
    run = _run(dry_settings, workspace)
    ds = run.preserve_dataset([workspace / "X_train_val.npy"], title="ds")
    m1 = run.preserve_keras_model(workspace / "model_1" / "KERAS_check_best_model.h5", title="m1")
    run.trained_on(m1, ds)
    man = run.save()
    dot = to_dot(man)
    assert dot.startswith("digraph provenance {")
    assert "trained_on" in dot
    assert (workspace / "provenance_graph.dot").exists()


def test_extract_execution_log(tmp_path):
    from dataerai_hls4ml.provenance import _extract_execution_log
    nb = {"cells": [{"cell_type": "code", "execution_count": 1, "source": ["print('hi')\n"],
                     "outputs": [{"output_type": "stream", "name": "stdout", "text": ["hello-out\n"]}]}],
          "nbformat": 4, "nbformat_minor": 5, "metadata": {}}
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    log = _extract_execution_log(p)
    assert "hello-out" in log and "In[1]" in log


def test_use_collection_and_notebook_record(dry_settings, workspace, monkeypatch):
    monkeypatch.setattr("dataerai_hls4ml.provenance._pip_freeze", lambda: ["dataerai-hls4ml==test"])
    run = _run(dry_settings, workspace)
    cid = run.use_collection("hls4ml — partX")
    assert cid and run.settings.collection_id == cid
    image = base64.b64encode(b"pngdata").decode()
    (workspace / "partX.ipynb").write_text(json.dumps(
        {"cells": [{
            "cell_type": "code",
            "execution_count": 1,
            "source": ["x=1\n"],
            "outputs": [{
                "output_type": "display_data",
                "data": {"image/png": image, "text/plain": ["<Figure size 640x480>"]},
                "metadata": {},
            }],
        }],
         "nbformat": 4, "nbformat_minor": 5, "metadata": {}}))
    run.preserve_notebook_record(workspace, "partX")
    kinds = {a.kind for a in run.artifacts.values()}
    assert {"notebook", "log", "environment", "hardware_metrics", "figure"} <= kinds
    assert run.get("partX — notebook") and run.get("partX — execution log")
    assert run.get("partX — environment") and run.get("partX — hardware metrics")
    assert run.get("partX — figure 01")
    assert (workspace / "partX.environment.json").exists()
    assert (workspace / "partX.hardware.json").exists()
    assert (workspace / ".dataerai_figures" / "partX" / "output-cell-000-00.png").exists()
    assert {"execution-log", "environment", "hardware-metrics", "output-figure"} <= {
        e.get("step") for e in run.edges
    }


def test_environment_capture(dry_settings, workspace):
    run = _run(dry_settings, workspace)
    env = run.manifest()["environment"]
    for key in ("python", "platform", "host", "tools"):
        assert key in env
    assert isinstance(env["tools"], dict)


def test_add_edge_accepts_ids_and_artifacts(dry_settings, workspace):
    run = _run(dry_settings, workspace)
    a = Artifact(name="A", kind="model", asset_id="aaa")
    e1 = run.add_edge(a, "bbb", "derived_from", step="prune")
    assert e1["from"] == "aaa" and e1["to"] == "bbb"
    assert e1["qualifiers"]["step"] == "prune"
    assert e1["qualifiers"]["pipeline"] == "test-pipeline"


def test_cross_notebook_state_reuse(dry_settings, workspace):
    run1 = _run(dry_settings, workspace)
    m1 = run1.preserve_keras_model(workspace / "model_1" / "KERAS_check_best_model.h5", title="shared")
    run1._save_state()
    assert (workspace / STATE_FILE).exists()

    # A second "notebook" (new process simulated) loads the same registry.
    run2 = ProvenanceRun("test-pipeline", settings=dry_settings, base_dir=workspace, fresh=False)
    again = run2.get("shared")
    assert again is not None
    assert again.asset_id == m1.asset_id


def test_state_file_roundtrip_schema(dry_settings, workspace):
    run = _run(dry_settings, workspace)
    run.preserve_array(workspace / "model_3" / "y_hls.npy", title="y")
    data = json.loads((workspace / STATE_FILE).read_text())
    assert "artifacts" in data and "y" in data["artifacts"]
    assert data["dry_run"] is True
