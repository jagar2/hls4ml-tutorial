import pytest

from dataerai_hls4ml import rest as rest_mod
from dataerai_hls4ml.config import Settings
from dataerai_hls4ml.rest import RestClient, RestError


class FakeResp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body
        self.text = text
        self.content = b"x" if (body is not None or text) else b""

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, headers=None, json=None, params=None, timeout=None, verify=None):
        self.calls.append({"method": method, "url": url, "json": json, "headers": headers})
        return self.responses.pop(0)


def make_client(responses):
    c = RestClient.__new__(RestClient)  # bypass __init__ (no `requests` needed)
    c.settings = Settings("https://beta.dataerai.com", False, "p", None, None, "tok")
    c._session = FakeSession(responses)
    c._requests = None
    c._refreshed_once = False
    return c


def test_create_relationship_payload():
    c = make_client([FakeResp(201, {"id": "rel-1"})])
    out = c.create_relationship("A", "B", "derived_from", qualifiers={"step": "prune"})
    call = c._session.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/assets/A/relationships/")
    assert call["json"]["to_asset_id"] == "B"
    assert call["json"]["type"] == "derived_from"
    assert call["json"]["qualifiers"]["step"] == "prune"
    assert out == {"id": "rel-1"}


def test_create_relationship_duplicate_409():
    c = make_client([FakeResp(409, {"detail": "exists"})])
    out = c.create_relationship("A", "B", "trained_on")
    assert out["duplicate"] is True


def test_open_run_and_close_payloads():
    c = make_client([FakeResp(201, {"run_id": "r", "run_sk": 7, "status": "open"})])
    out = c.open_run("training")
    assert c._session.calls[0]["url"].endswith("/api/lineage/runs/")
    assert c._session.calls[0]["json"] == {"kind": "training"}
    assert out["run_id"] == "r"

    c2 = make_client([FakeResp(200, {"status": "closed", "integrity_root": "deadbeef"})])
    out2 = c2.close_run("r")
    assert c2._session.calls[0]["url"].endswith("/api/lineage/runs/r/close/")
    assert out2["integrity_root"] == "deadbeef"


def test_trained_on_includes_idempotency_and_dids():
    c = make_client([FakeResp(202, {"seq": 1})])
    c.trained_on("r", ["did:dataerai:asset:1", "did:dataerai:asset:2"])
    body = c._session.calls[0]["json"]
    assert body["data_dids"] == ["did:dataerai:asset:1", "did:dataerai:asset:2"]
    assert "idempotency_key" in body


def test_error_raises_resterror():
    c = make_client([FakeResp(500, {"detail": "boom"})])
    with pytest.raises(RestError) as exc:
        c.get_asset("A")
    assert exc.value.status == 500


def test_401_triggers_refresh_and_retry(monkeypatch):
    monkeypatch.setattr(rest_mod, "refresh_credentials", lambda binary: None)
    monkeypatch.setattr(rest_mod, "read_credentials", lambda: {"access_token": "fresh"})
    c = make_client([FakeResp(401, {"detail": "expired"}), FakeResp(200, {"ok": True})])
    out = c.get_asset("A")
    assert out == {"ok": True}
    assert c.settings.access_token == "fresh"
    assert len(c._session.calls) == 2
