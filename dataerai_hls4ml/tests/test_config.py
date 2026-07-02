from dataerai_hls4ml import config
from dataerai_hls4ml.config import Settings, load_settings


def _clear_env(monkeypatch):
    for var in ["DATAERAI_DRY_RUN", "DATAERAI_SERVER", "DATAERAI_ACCESS_TOKEN",
                "DATAERAI_PROJECT_ID", "DATAERAI_COLLECTION_ID", "DATAERAI_PROVENANCE"]:
        monkeypatch.delenv(var, raising=False)


def test_explicit_dry_run_skips_credentials(monkeypatch):
    _clear_env(monkeypatch)
    called = {"creds": False}
    monkeypatch.setattr(config, "read_credentials", lambda: called.__setitem__("creds", True) or {})
    s = load_settings(dry_run=True)
    assert s.dry_run is True
    assert called["creds"] is False  # no credential read on the explicit dry-run path


def test_degrades_to_dry_run_without_credentials(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(config, "refresh_credentials", lambda binary: None)
    monkeypatch.setattr(config, "read_credentials", lambda: {})
    s = load_settings()
    assert s.dry_run is True
    assert s.degraded is True
    assert s.server == config.DEFAULT_SERVER


def test_live_when_token_present(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(config, "refresh_credentials", lambda binary: None)
    monkeypatch.setattr(config, "read_credentials",
                        lambda: {"access_token": "abc", "server_url": "https://dev.dataerai.com",
                                 "user_email": "me@example.com"})
    s = load_settings()
    assert s.dry_run is False
    assert s.access_token == "abc"
    # server_url from creds is used when no env override
    assert s.server == "https://dev.dataerai.com"
    assert s.user_email == "me@example.com"


def test_env_server_overrides_credentials(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DATAERAI_SERVER", "https://beta.dataerai.com")
    monkeypatch.setenv("DATAERAI_ACCESS_TOKEN", "tok")
    s = load_settings()
    assert s.server == "https://beta.dataerai.com"
    assert s.access_token == "tok"
    assert s.dry_run is False


def test_owner_resolution():
    proj = Settings("s", False, "p-id", None, None, "t")
    assert proj.owner_type == "project" and proj.owner_id == "p-id"
    coll = Settings("s", False, None, "c-id", None, "t")
    assert coll.owner_type == "collection" and coll.owner_id == "c-id"
    none = Settings("s", True, None, None, None, None)
    assert none.owner_type is None and none.owner_id is None


def test_decode_keyring_secret():
    import base64
    plain = '{"access_token":"abc"}'
    encoded = "go-keyring-base64:" + base64.b64encode(plain.encode()).decode()
    assert config._decode_keyring_secret(encoded) == plain
    assert config._decode_keyring_secret(plain) == plain  # non-prefixed passthrough


def test_enabled_flag(monkeypatch):
    _clear_env(monkeypatch)
    assert config.enabled() is True
    monkeypatch.setenv("DATAERAI_PROVENANCE", "0")
    assert config.enabled() is False
    monkeypatch.setenv("DATAERAI_PROVENANCE", "yes")
    assert config.enabled() is True
