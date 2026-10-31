"""API, agent, ingestion and scenario tests (AC1,4-6,10-16, E-cases, scenarios 1,4,5,6)."""
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from tests.conftest import HDR, make_weather
from data.schemas import ActionStatus, Provenance, SignalKind, Status


def run_cycle(client, **body):
    r = client.post("/api/cycles", json=body, headers=HDR)
    assert r.status_code == 202, r.text
    cid = r.json()["cycle_id"]
    for _ in range(100):
        c = client.get(f"/api/cycles/{cid}").json()
        if c["status"] != "running":
            return c
        import time; time.sleep(0.1)
    raise AssertionError("cycle did not finish")


def test_ac1_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_ac15_token_and_e16(client):
    r = client.post("/api/cycles", json={})
    assert r.status_code == 401 and r.json()["error"]["message"] == "Operator token required."


def test_e13_offset_validation(client):
    r = client.post("/api/cycles", json={"stress_offset_c": 9}, headers=HDR)
    assert r.status_code == 422 and r.json()["error"]["message"] == "Stress offset must be between 0 and 8 °C."


def test_e12_not_found(client):
    assert client.get("/api/cycles/nope").status_code == 404
    assert client.post("/api/actions/nope/approve", headers=HDR).status_code == 404


def test_scenario1_happy_path_and_ac14_ac16(client, service):
    c = run_cycle(client, mode="approval_required", stress_offset_c=4.0)
    assert c["status"] == "completed" and c["counterfactual"] is True            # AC17 flag
    assert c["deterministic_mode"] is True and "Deterministic mode (no LLM)." in c["notices"]   # scenario 6
    assert c["stress_windows"], "peak tariff window should be detected"
    assert c["comparison"]["cost_inr"]["baseline"] > 0
    assert all(t["citations"] for t in c["trace"])                               # AC12
    act = c["actions"][0]; assert act["status"] == "awaiting_approval"
    before = service.d.twin.snapshot()
    r = client.post(f"/api/actions/{act['action_id']}/approve", headers=HDR)
    assert r.status_code == 200 and r.json()["status"] == "applied"
    r2 = client.post(f"/api/actions/{act['action_id']}/approve", headers=HDR)    # AC14
    assert r2.status_code == 409 and r2.json()["error"]["message"] == "This action is no longer awaiting approval."
    assert service.d.twin.snapshot() != before
    assert client.post(f"/api/actions/{act['action_id']}/rollback", headers=HDR).json()["status"] == "rolled_back"
    assert service.d.twin.snapshot() == before                                    # AC16 exact restore


def test_e14_concurrent_cycle(client, service):
    assert service._run_lock.acquire(blocking=False)
    try:
        r = client.post("/api/cycles", json={}, headers=HDR)
        assert r.status_code == 409 and r.json()["error"]["message"] == "A control cycle is already running."
    finally:
        service._run_lock.release()


def test_autonomous_applies_and_scenario5_rollback_on_deviation(client, service):
    c = run_cycle(client, mode="autonomous")
    assert c["actions"][0]["status"] == "applied"
    # Force a verification mismatch: IT telemetry jumps between planning and application (scenario 5)
    from data.schemas import ItLoadSample
    calls = {"n": 0}
    def jumpy():
        calls["n"] += 1
        return ItLoadSample(ts=datetime.now(timezone.utc), cpu_pct=99, mem_pct=99, it_load_kw=5000.0, scaling_formula="test", host_label="test")
    service.d.sample_it_load = jumpy
    snap = service.d.twin.snapshot()
    c = run_cycle(client, mode="autonomous")
    assert any(a["status"] == "rolled_back" for a in c["actions"])
    assert "Twin response deviated from prediction; action rolled back." in c["notices"]


def test_ac5_cold_start_and_e3(env):
    from api.main import build_service
    from fastapi.testclient import TestClient
    from api.main import create_app
    svc = build_service()  # real clients; sandbox has no route to the live APIs
    cl = TestClient(create_app(svc))
    j = cl.get("/api/inputs").json()
    assert j["sources"]["weather"]["provenance"]["status"] == "FAILED" and j["sources"]["weather"]["data"] is None
    c = run_cycle(cl)
    assert c["status"] == "held_safe" and any("Holding current safe state" in n for n in c["notices"])
    assert c["plan"] is None


def test_ac4_ac6_stale_then_failed(service):
    ing = service.d.ingestor
    first = ing.fetch_all(); assert first.status("weather") == Status.LIVE
    ing.chains["weather"] = [lambda: (_ for _ in ()).throw(RuntimeError("down"))]
    stale = ing.fetch_all()
    assert stale.status("weather") == Status.STALE and stale.states["weather"].provenance.stale_reason
    # AC6: age beyond max age => FAILED, not STALE
    lkg = service.d.store.load_lkg("weather")
    payload, prov = lkg
    prov["fetched_at"] = (datetime.now(timezone.utc) - timedelta(minutes=200)).isoformat()
    service.d.store.save_lkg("weather", payload, prov)
    assert ing.fetch_all().status("weather") == Status.FAILED


def test_ac4_fallback_live(service):
    ing = service.d.ingestor
    good = ing.chains["weather"][0]
    ing.chains["weather"] = [lambda: (_ for _ in ()).throw(RuntimeError("down")), good]
    inp = ing.fetch_all()
    assert inp.status("weather") == Status.FALLBACK_LIVE and "Primary weather source unavailable. Using backup source." in inp.notices


def test_ac7_published_tariff_not_labeled_live(service):
    m = service.d.ingestor.fetch_all().states["market"]
    assert m.provenance.signal_kind == SignalKind.PUBLISHED_SCHEDULE


def test_tariff_unconfigured_is_failed(env, monkeypatch):
    (env / "tariff.yaml").write_text('order_citation: ""\norder_url: ""\nbands: []\n')
    from api.main import build_service
    svc = build_service()
    assert svc.d.ingestor.fetch_all().status("market") == Status.FAILED


# ---------------------------------------------------------------- agent / LLM
class ScriptedLLM:
    configured = True
    def __init__(self, script): self.script, self.i = script, 0
    def chat(self, messages, tools=None):
        r = self.script[min(self.i, len(self.script) - 1)]; self.i += 1; return r
    def parse_alerts(self, *a, **k): return []


def call(name, args=None, i=0):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}]}


def test_ut10_disallowed_tool_refused(service):
    from agent.context import CycleContext
    from agent.tools import Tools
    from data.schemas import Mode
    ctx = CycleContext(deps=service.d, mode=Mode.approval_required)
    ctx.inputs = service.d.ingestor.fetch_all()
    out = Tools(ctx).dispatch("set_setpoint", {"value": 12})
    assert "not allowed" in out["error"]
    out = Tools(ctx).dispatch("optimize_plan", {"setpoint": 1})   # numeric args are rejected by schema
    assert "invalid arguments" in out["error"]


def test_ac11_iteration_cap(client, service):
    service.d.llm = ScriptedLLM([call("get_inputs")])          # never finishes
    c = run_cycle(client)
    assert c["status"] == "held_safe"
    assert "Agent stopped after 8 steps. No action applied." in c["notices"]


def test_llm_orchestrated_cycle(client, service):
    service.d.llm = ScriptedLLM([call("get_inputs"), call("detect_stress"), call("optimize_plan"),
                                 {"role": "assistant", "content": "Done: pre-cooling planned."}])
    c = run_cycle(client)
    assert c["status"] == "completed" and c["deterministic_mode"] is False
    assert c["actions"] and c["plan"] is not None                # pipeline completed deterministically after LLM finished


def test_ac10_invalid_json_retry_includes_validation_error(env):
    from agent.llm import ConstraintParseError, LLMClient
    from config.settings import get_settings
    from data.schemas import AlertItem
    s = get_settings().model_copy(update={"LLM_API_KEY": "k", "LLM_MODEL": "m", "LLM_PROVIDER": "groq"})
    bodies = []
    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "not json"}}]})
    llm = LLMClient(s, transport=httpx.MockTransport(handler))
    with pytest.raises(ConstraintParseError):
        llm.parse_alerts([AlertItem(title="t", summary="s", published="p", link="l", source_feed="f")], datetime.now(timezone.utc))
    assert len(bodies) == 3                                    # 1 attempt + 2 retries
    assert "failed schema validation" in bodies[1]["messages"][-1]["content"]


def test_alert_parse_valid_and_conflict(env):
    from agent.llm import LLMClient
    from config.settings import get_settings
    from data.schemas import AlertItem
    from forecast.stress import resolve_conflicts
    s = get_settings().model_copy(update={"LLM_API_KEY": "k", "LLM_MODEL": "m"})
    good = {"constraints": [{"type": "heatwave", "severity": 2, "start": None, "end": None, "region_match": True, "evidence": "x"},
                            {"type": "heatwave", "severity": 3, "start": None, "end": None, "region_match": True, "evidence": "y"}]}
    llm = LLMClient(s, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": json.dumps(good)}}]})))
    cons = llm.parse_alerts([AlertItem(title="t", summary="s", published="p", link="l", source_feed="f")], datetime.now(timezone.utc))
    kept, conflict = resolve_conflicts(cons)
    assert conflict and len(kept) == 1 and kept[0].severity == 3


def test_stale_data_widens_margin(service):
    ing = service.d.ingestor
    ing.fetch_all()
    ing.chains["weather"] = [lambda: (_ for _ in ()).throw(RuntimeError("down"))]
    inp = ing.fetch_all()
    assert inp.any_stale() and any("Safety margin widened" in n for n in inp.notices)


def test_backtest_metrics_function():
    from forecast.backtest import evaluate
    import math
    series = [20 + 5 * math.sin(2 * math.pi * h / 24) for h in range(24 * 12)]
    r = evaluate(series)
    assert r["same_hour_yesterday"] < r["persistence"]


def test_body_size_limit(client):
    r = client.post("/api/cycles", content=b"x" * 70000, headers={**HDR, "content-type": "application/json"})
    assert r.status_code == 413


def test_sse_replays_trace(client):
    c = run_cycle(client)
    with client.stream("GET", f"/api/cycles/{c['cycle_id']}/events") as r:
        text = "".join(r.iter_text())
    assert text.count("data: {\"n\"") == len(c["trace"]) and "event: end" in text
