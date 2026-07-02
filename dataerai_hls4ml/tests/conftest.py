import sys
from pathlib import Path

import pytest

# Make the repo root importable so `import dataerai_hls4ml` works without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataerai_hls4ml.config import Settings  # noqa: E402


@pytest.fixture
def dry_settings():
    return Settings(
        server="https://beta.dataerai.com",
        dry_run=True,
        project_id=None,
        collection_id=None,
        binary_path=None,
        access_token=None,
    )


@pytest.fixture
def live_settings():
    return Settings(
        server="https://beta.dataerai.com",
        dry_run=False,
        project_id="11111111-1111-1111-1111-111111111111",
        collection_id=None,
        binary_path=None,
        access_token="test-token",
    )


@pytest.fixture
def workspace(tmp_path):
    from dataerai_hls4ml import pipeline
    return pipeline.make_synthetic_workspace(tmp_path)
