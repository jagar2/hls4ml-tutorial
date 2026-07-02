from dataerai_hls4ml.provenance import ProvenanceRun


class FakeRest:
    """Minimal stand-in for RestClient exercising the DID-refresh + seal path."""

    def __init__(self, dids):
        self.dids = dids  # asset_id -> did (or None)
        self.calls = []

    def get_asset(self, asset_id):
        return {"did": self.dids.get(asset_id)}

    def open_run(self, kind):
        self.calls.append(("open", kind))
        return {"run_id": "RUN", "run_sk": 9}

    def trained_on(self, run_id, data_dids, **kw):
        self.calls.append(("trained_on", run_id, list(data_dids)))
        return {"seq": 1}

    def close_run(self, run_id, **kw):
        self.calls.append(("close", run_id))
        return {"integrity_root": "deadbeef", "status": "closed"}

    def resolve_did(self, did):
        return {"verified": True}


def _live_run(dry_settings, workspace, fake_rest):
    run = ProvenanceRun("t", settings=dry_settings, base_dir=workspace, fresh=True)
    ds = run.preserve_dataset([workspace / "X_train_val.npy"], title="ds")
    model = run.preserve_keras_model(workspace / "model_1" / "KERAS_check_best_model.h5", title="m")
    run.trained_on(model, ds)
    # Flip to "live" with a fake backend; clear the dry-run synthetic DIDs so the
    # refresh path has to resolve them (as it would right after a real upload).
    run.settings.dry_run = False
    run.rest = fake_rest
    run.run_id = None
    for art in run.artifacts.values():
        art.did = None
    return run, ds, model


def test_refresh_dids_updates_registry(dry_settings, workspace):
    fake = FakeRest({})  # will be filled per-asset below
    run, ds, model = _live_run(dry_settings, workspace, fake)
    fake.dids = {ds.asset_id: "did:dataerai:asset:DS", model.asset_id: "did:dataerai:asset:M"}
    n = run.refresh_dids()
    assert n == 2
    assert run.get("ds").did == "did:dataerai:asset:DS"
    assert run.get("m").did == "did:dataerai:asset:M"


def test_seal_lineage_run_feeds_dataset_dids_and_closes(dry_settings, workspace):
    fake = FakeRest({})
    run, ds, model = _live_run(dry_settings, workspace, fake)
    fake.dids = {ds.asset_id: "did:dataerai:asset:DS", model.asset_id: "did:dataerai:asset:M"}
    run.refresh_dids()
    run.seal_lineage_run()

    assert run.run_id == "RUN" and run.closed
    assert run.integrity_root == "deadbeef"
    assert any(c[0] == "trained_on" and c[2] == ["did:dataerai:asset:DS"] for c in fake.calls)
    # citations now populate from the resolved DIDs
    cites = run.manifest()["citations"]
    assert {c["did"] for c in cites} == {"did:dataerai:asset:DS", "did:dataerai:asset:M"}


def test_refresh_noop_until_dids_minted(dry_settings, workspace):
    fake = FakeRest({})  # backend returns no DIDs yet (async not caught up)
    run, ds, model = _live_run(dry_settings, workspace, fake)
    assert run.refresh_dids() == 0
    run.seal_lineage_run()
    assert run.integrity_root is None      # nothing sealed without a dataset DID
    assert not any(c[0] == "trained_on" for c in fake.calls)
