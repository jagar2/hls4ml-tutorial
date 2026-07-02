import json

from dataerai_hls4ml.provenance import ProvenanceRun


def _run(dry_settings, base):
    return ProvenanceRun("t", settings=dry_settings, base_dir=base, fresh=True)


def test_reference_dataset_dry_run(dry_settings, workspace):
    run = _run(dry_settings, workspace)
    ds = run.reference_dataset("SVHN (svhn_cropped)", source="tfds:svhn_cropped",
                               metadata={"classes": 10})
    assert ds.kind == "dataset"
    assert ds.metadata["source"] == "tfds:svhn_cropped"
    assert ds.metadata["classes"] == 10
    f = workspace / "svhn-svhn-cropped.dataset.json"
    assert f.exists()
    doc = json.loads(f.read_text())
    assert doc["name"] == "SVHN (svhn_cropped)"
    assert doc["source"] == "tfds:svhn_cropped"


def test_reference_dataset_enables_trained_on(dry_settings, workspace):
    run = _run(dry_settings, workspace)
    ds = run.reference_dataset("SVHN", source="tfds:svhn_cropped")
    model = run.preserve_keras_model(workspace / "model_1" / "KERAS_check_best_model.h5", title="cnn")
    run.trained_on(model, ds)
    assert any(e["type"] == "trained_on" and e["from"] == model.asset_id and e["to"] == ds.asset_id
               for e in run.edges)
