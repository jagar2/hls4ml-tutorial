import hashlib

from dataerai_hls4ml import artifacts
from dataerai_hls4ml.artifacts import Preserver, _cheap_signature, _sha256


class _Dense:
    pass


class FakeModel:
    """Minimal stand-in for a Keras model (has count_params/layers/save)."""

    def count_params(self):
        return 4321

    @property
    def layers(self):
        return [_Dense(), _Dense()]

    def save(self, path):
        with open(path, "wb") as fh:
            fh.write(b"fake-keras-h5")


def test_sha256(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert _sha256(f) == hashlib.sha256(b"hello").hexdigest()


def test_cheap_signature_deterministic(tmp_path):
    a = tmp_path / "a"; a.write_bytes(b"123")
    b = tmp_path / "b"; b.write_bytes(b"4567")
    assert _cheap_signature([a, b]) == _cheap_signature([b, a])  # order-independent


def test_preserve_file_dry_run(dry_settings, tmp_path):
    f = tmp_path / "thing.txt"; f.write_text("x")
    p = Preserver(dry_settings)
    art = p.preserve_file(f, title="thing", kind="report")
    assert art.kind == "report"
    assert art.asset_id and art.did.endswith(art.asset_id)
    assert art.metadata["dry_run"] is True


def test_preserve_dataset_dry_run(dry_settings, workspace):
    p = Preserver(dry_settings)
    files = [workspace / n for n in ["X_train_val.npy", "y_train_val.npy", "X_test.npy",
                                     "y_test.npy", "classes.npy"]]
    art = p.preserve_dataset(files, title="jets")
    assert art.kind == "dataset"
    assert set(art.metadata["files"]) == {f.name for f in files}


def test_preserve_keras_model_object_dry_run(dry_settings, tmp_path):
    p = Preserver(dry_settings)
    art = p.preserve_keras_model(FakeModel(), title="model_x")
    assert art.kind == "model"
    assert art.metadata["param_count"] == 4321
    assert art.metadata["layers"] == ["_Dense", "_Dense"]


def test_preserve_hls_project_dry_run(dry_settings, workspace):
    p = Preserver(dry_settings)
    art = p.preserve_hls_project(workspace / "model_1" / "hls4ml_prj", title="hls1")
    assert art.kind == "hls_project"
    assert art.metadata["file_count"] >= 1


def test_preserve_dir_custom_kind(dry_settings, workspace):
    p = Preserver(dry_settings)
    art = p.preserve_dir(workspace / "sr", kind="artifact", title="sr")
    assert art.kind == "artifact"


def test_live_upload_uses_sdk(monkeypatch, live_settings, tmp_path):
    """In live mode the Preserver calls the SDK client.upload and reads the DID."""
    f = tmp_path / "m.h5"; f.write_bytes(b"weights")

    class FakeResult:
        asset_id = "asset-123"
        content_id = "content-456"

    captured = {}

    class FakeClient:
        def upload(self, path, **kwargs):
            captured.update(kwargs); captured["path"] = path
            return FakeResult()

    class FakeRest:
        def get_asset(self, asset_id):
            assert asset_id == "asset-123"
            return {"did": "did:dataerai:asset:asset-123"}

    monkeypatch.setattr(artifacts, "_get_client", lambda settings: FakeClient())
    p = Preserver(live_settings, rest=FakeRest())
    art = p.preserve_file(f, title="m", kind="model")
    assert art.asset_id == "asset-123"
    assert art.content_id == "content-456"
    assert art.did == "did:dataerai:asset:asset-123"
    assert captured["owner_type"] == "project"
    assert captured["owner_id"] == live_settings.project_id
    assert art.sha256 == hashlib.sha256(b"weights").hexdigest()
    assert captured["metadata"]["sha256"] == art.sha256
