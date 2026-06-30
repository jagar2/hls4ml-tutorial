#!/usr/bin/env python
"""Run the Dataerai-instrumented hls4ml pipeline.

Thin wrapper around :mod:`dataerai_hls4ml.cli` so you can drive everything without
installing the package:

    python run_pipeline.py selftest --dry-run          # prove it offline, no deps
    python run_pipeline.py capture --root .            # link whatever outputs exist
    python run_pipeline.py run --only part1_getting_started

See DATAERAI_PROVENANCE.md for setup (auth, project id) and how the lineage maps
to Dataerai's provenance graph + DID citations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataerai_hls4ml.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
