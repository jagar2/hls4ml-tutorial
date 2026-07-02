from dataerai_hls4ml import pipeline
from dataerai_hls4ml.provenance import ProvenanceRun


def _run(settings, base):
    return ProvenanceRun("pipe", settings=settings, base_dir=base, fresh=True)


def test_full_layout_preserved_and_linked(dry_settings, workspace):
    run = _run(dry_settings, workspace)
    preserved = pipeline.preserve_and_link(run, workspace)
    assert len(preserved) == 14
    for name in ["dataset", "model_1", "model_1_hls", "model_2", "model_3",
                 "model_3_hls_pynq", "y_hls", "cnn_pruned", "cnn_quantized",
                 "bdt", "bdt_hls", "sr"]:
        assert name in preserved
    assert len(run.edges) == 13
    # spot-check key lineage edges
    types = {(e["from"], e["to"], e["type"]) for e in run.edges}
    assert (preserved["model_1"].asset_id, preserved["dataset"].asset_id, "trained_on") in types
    assert (preserved["model_3"].asset_id, preserved["model_2"].asset_id, "derived_from") in types


def test_partial_pipeline_is_robust(dry_settings, workspace):
    # Simulate a pipeline where the quantized model wasn't produced.
    (workspace / "model_3" / "KERAS_check_best_model.h5").unlink()
    run = _run(dry_settings, workspace)
    preserved = pipeline.preserve_and_link(run, workspace)
    assert "model_3" not in preserved
    # HLS projects still preserved, but their edges to the missing model are dropped.
    assert "model_3_hls_pynq" in preserved
    assert len(run.edges) < 13
    assert all(e["to"] != "model_3" for e in run.edges)


def test_preserve_and_link_is_idempotent(dry_settings, workspace):
    run = _run(dry_settings, workspace)
    first = pipeline.preserve_and_link(run, workspace)
    n_art, n_edge = len(run.artifacts), len(run.edges)
    # Calling again (e.g. from a later notebook) must not duplicate anything.
    second = pipeline.preserve_and_link(run, workspace)
    assert len(run.artifacts) == n_art
    assert len(run.edges) == n_edge
    assert {k: v.asset_id for k, v in first.items()} == {k: v.asset_id for k, v in second.items()}


def test_only_filter_scopes_preservation_but_keeps_crossnotebook_edges(dry_settings, workspace):
    run = _run(dry_settings, workspace)
    p1 = pipeline.preserve_and_link(run, workspace, only=["dataset", "model_1", "model_1_hls"])
    assert set(p1) <= {"dataset", "model_1", "model_1_hls"}
    # A later "notebook" preserves only its own artifacts...
    p3 = pipeline.preserve_and_link(run, workspace, only=["model_2", "model_2_hls"])
    assert set(p3) == {"model_2", "model_2_hls"}
    # ...but the cross-notebook edge model_2 --derived_from--> model_1 still forms.
    m1, m2 = run.get("Baseline MLP (part1)"), run.get("Pruned MLP (part3)")
    assert any(e["from"] == m2.asset_id and e["to"] == m1.asset_id and e["type"] == "derived_from"
               for e in run.edges)


def test_make_synthetic_workspace_contents(tmp_path):
    root = pipeline.make_synthetic_workspace(tmp_path)
    assert (root / "model_1" / "KERAS_check_best_model.h5").exists()
    assert (root / "model_3" / "hls4ml_prj_pynq").is_dir()
    assert (root / "sr").is_dir()
