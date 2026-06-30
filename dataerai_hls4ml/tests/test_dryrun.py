from dataerai_hls4ml import dryrun


def test_ids_are_deterministic():
    assert dryrun.fake_asset_id("x") == dryrun.fake_asset_id("x")
    assert dryrun.fake_asset_id("x") != dryrun.fake_asset_id("y")


def test_did_format():
    aid = dryrun.fake_asset_id("model_1")
    assert dryrun.fake_did(aid) == f"did:dataerai:asset:{aid}"


def test_run_id_deterministic():
    assert dryrun.fake_run_id("pipe") == dryrun.fake_run_id("pipe")
