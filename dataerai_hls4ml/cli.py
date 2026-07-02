"""Command-line orchestrator for the Dataerai-instrumented hls4ml pipeline.

Subcommands
-----------
* ``selftest``  Build a synthetic workspace and record the full lineage DAG offline
  (no TensorFlow, no Vitis, no credentials) — proves the integration end-to-end.
* ``capture``   Run :func:`~dataerai_hls4ml.pipeline.preserve_and_link` over an
  existing checkout, preserving whatever pipeline outputs are on disk.
* ``run``       Execute the tutorial notebooks in order under one shared lineage
  run (via papermill); CPU-feasible parts run, synthesis parts are tolerated.

Examples
--------
    dataerai-hls4ml selftest --dry-run
    DATAERAI_PROJECT_ID=<uuid> dataerai-hls4ml capture --root .
    dataerai-hls4ml run --only part1_getting_started
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

NOTEBOOK_ORDER = [
    "part1_getting_started",
    "part2_advanced_config",
    "part3_compression",
    "part4_quantization",
    "part4.1_HG_quantization",
    "part5_bdt",
    "part6_cnns",
    "part7a_bitstream",
    "part7b_deployment",
    "part7c_validation",
    "part8_symbolic_regression",
]


def _apply_env(args) -> None:
    if getattr(args, "dry_run", False):
        os.environ["DATAERAI_DRY_RUN"] = "1"
    if getattr(args, "server", None):
        os.environ["DATAERAI_SERVER"] = args.server
    if getattr(args, "project", None):
        os.environ["DATAERAI_PROJECT_ID"] = args.project


def _summarize(manifest: dict) -> None:
    print(f"\n  lineage run : {manifest['run_id']}")
    print(f"  mode        : {'dry-run' if manifest['dry_run'] else manifest['server']}")
    if manifest.get("integrity_root"):
        print(f"  integrity   : {manifest['integrity_root']}")
    print(f"  artifacts   : {len(manifest['artifacts'])}")
    print(f"  edges       : {len(manifest['edges'])}")
    for e in manifest["edges"]:
        frm = next((a["name"] for a in manifest["artifacts"] if a["asset_id"] == e["from"]), e["from"][:8])
        to = next((a["name"] for a in manifest["artifacts"] if a["asset_id"] == e["to"]), e["to"][:8])
        step = f" [{e['step']}]" if e.get("step") else ""
        print(f"      {frm}  --{e['type']}{step}-->  {to}")


def cmd_selftest(args) -> int:
    _apply_env(args)
    from . import pipeline
    from .provenance import get_run

    workdir = Path(args.root) if args.root else Path(tempfile.mkdtemp(prefix="dataerai-selftest-"))
    pipeline.make_synthetic_workspace(workdir)
    run = get_run("hls4ml-selftest", kind="training", fresh=True, base_dir=workdir)
    pipeline.preserve_and_link(run, workdir)
    manifest = run.save()
    _summarize(manifest)
    print(f"\n  manifest    : {workdir / 'provenance_manifest.json'}")
    print(f"  report      : {workdir / 'PROVENANCE.md'}")
    # A complete self-test exercises the whole DAG: dataset + 3 model gens + HLS + eval.
    ok = len(manifest["artifacts"]) >= 10 and len(manifest["edges"]) >= 9
    print(f"\n  self-test   : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_capture(args) -> int:
    _apply_env(args)
    from . import pipeline
    from .provenance import get_run

    root = Path(args.root)
    run = get_run("hls4ml-pipeline", kind=args.kind, fresh=args.fresh, base_dir=root)
    preserved = pipeline.preserve_and_link(run, root)
    if not preserved:
        print(f"No pipeline outputs found under {root} — run a notebook first.", file=sys.stderr)
        return 1
    manifest = run.save()
    _summarize(manifest)
    return 0


def cmd_demo(args) -> int:
    """Per-notebook capture across all parts: one collection per notebook, each with
    its artifacts (those present on disk) plus a static notebook + execution-log
    recording. Great for exercising the full integration against a backend."""
    _apply_env(args)
    from . import pipeline
    from .provenance import capture, get_run

    root = Path(args.root)
    get_run("hls4ml-pipeline", kind="training", fresh=True, base_dir=root)
    manifest = None
    for stem in pipeline.NOTEBOOK_ARTIFACTS:
        if not (root / f"{stem}.ipynb").exists():
            print(f"skip (no notebook): {stem}", file=sys.stderr)
            continue
        print(f"▶ collection: hls4ml — {stem}")
        manifest = capture(str(root), notebook=stem)
    if manifest is None:
        print(f"No notebooks found under {root}.", file=sys.stderr)
        return 1
    _summarize(manifest)
    return 0


def cmd_refresh(args) -> int:
    """Re-fetch async-minted DIDs and seal the lineage run for the persisted run."""
    _apply_env(args)
    from .provenance import get_run

    root = Path(args.root)
    run = get_run("hls4ml-pipeline", base_dir=root)
    if not run.artifacts:
        print(f"No persisted run under {root} (.dataerai_run.json) — run capture/demo first.",
              file=sys.stderr)
        return 1
    new = run.refresh_dids(retries=args.retries, sleep_s=args.interval)
    run.seal_lineage_run()
    manifest = run.save(close=False)
    cites = manifest.get("citations") or []
    with_did = sum(1 for a in manifest["artifacts"] if a.get("did"))
    print(f"  DIDs newly resolved : {new}")
    print(f"  assets with a DID   : {with_did}/{len(manifest['artifacts'])}")
    print(f"  citations           : {len(cites)}")
    print(f"  lineage run         : {manifest['run_id']}  integrity={manifest.get('integrity_root')}")
    return 0


def cmd_run(args) -> int:
    _apply_env(args)
    try:
        import papermill as pm
    except ImportError:
        print("papermill is required for `run` (pip install papermill), "
              "or use `capture` after running notebooks manually.", file=sys.stderr)
        return 2

    root = Path(args.root)
    selected = args.only.split(",") if args.only else NOTEBOOK_ORDER
    os.environ.setdefault("DATAERAI_RUN_LABEL", "hls4ml-pipeline")
    from .provenance import begin_hardware_sample, capture, get_run, write_notebook_hardware_metrics

    get_run("hls4ml-pipeline", kind="training", fresh=True, base_dir=root)
    failures = []
    for stem in selected:
        nb = root / f"{stem}.ipynb"
        if not nb.exists():
            print(f"skip (missing): {nb.name}", file=sys.stderr)
            continue
        out = root / f"{stem}.executed.ipynb"
        print(f"▶ executing {nb.name} …")
        sample = begin_hardware_sample()
        success = True
        error = None
        try:
            pm.execute_notebook(str(nb), str(out), kernel_name=args.kernel,
                                progress_bar=False)
        except Exception as exc:  # synthesis/Vitis/heavy-train failures are expected
            success = False
            error = str(exc).splitlines()[-1][:160]
            failures.append((stem, error))
            print(f"  ! {stem} did not finish: {failures[-1][1]}", file=sys.stderr)
        finally:
            write_notebook_hardware_metrics(
                root, stem, sample=sample, success=success, error=error, kernel=args.kernel)
            capture(str(root), notebook=stem)

    run = get_run("hls4ml-pipeline", base_dir=root)
    manifest = run.save()
    _summarize(manifest)
    if failures:
        print("\n  parts that did not complete (expected without Vitis HLS / heavy deps):")
        for stem, why in failures:
            print(f"    - {stem}: {why}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Shared flags usable either before or after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--server", help="Dataerai backend (default: $DATAERAI_SERVER or beta)")
    common.add_argument("--project", help="owner project UUID (default: $DATAERAI_PROJECT_ID)")
    common.add_argument("--dry-run", action="store_true", help="record lineage locally, upload nothing")

    p = argparse.ArgumentParser(
        prog="dataerai-hls4ml", parents=[common],
        description="Dataerai preservation & provenance for the hls4ml pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("selftest", parents=[common],
                       help="prove the integration offline with synthetic artifacts")
    s.add_argument("--root", help="workspace dir (default: a temp dir)")
    s.set_defaults(func=cmd_selftest)

    c = sub.add_parser("capture", parents=[common],
                       help="preserve+link whatever outputs exist under --root")
    c.add_argument("--root", default=".", help="repo root (default: .)")
    c.add_argument("--kind", default="derivation", choices=["training", "derivation", "eval"])
    c.add_argument("--fresh", action="store_true", help="start a new lineage run")
    c.set_defaults(func=cmd_capture)

    d = sub.add_parser("demo", parents=[common],
                       help="per-notebook capture: one collection + recording per part")
    d.add_argument("--root", default=".", help="repo root (default: .)")
    d.set_defaults(func=cmd_demo)

    rf = sub.add_parser("refresh", parents=[common],
                        help="re-fetch async-minted DIDs + seal the lineage run")
    rf.add_argument("--root", default=".", help="repo root (default: .)")
    rf.add_argument("--retries", type=int, default=5, help="DID re-fetch attempts (default: 5)")
    rf.add_argument("--interval", type=float, default=3.0, help="seconds between attempts (default: 3)")
    rf.set_defaults(func=cmd_refresh)

    r = sub.add_parser("run", parents=[common],
                       help="papermill-execute notebooks under one lineage run")
    r.add_argument("--root", default=".", help="repo root (default: .)")
    r.add_argument("--only", help="comma-separated notebook stems (default: all, in order)")
    r.add_argument("--kernel", default="python3", help="Jupyter kernel name")
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
