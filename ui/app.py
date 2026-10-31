"""GridSentinel dashboard (Streamlit). Talks to the backend over HTTP only; the operator token stays server-side."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import httpx
import plotly.graph_objects as go
import streamlit as st

API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
TOKEN = os.environ.get("OPERATOR_TOKEN", "")
HEAD = {"X-Operator-Token": TOKEN}
CHIP = {"LIVE": "#1a7f37", "FALLBACK_LIVE": "#2457D6", "STALE": "#b7791f", "FAILED": "#c62828", "PUBLISHED": "#1a7f37",
        "applied": "#1a7f37", "rejected": "#c62828", "awaiting_approval": "#b7791f", "rolled_back": "#6b7280",
        "proposed": "#6b7280", "failed": "#c62828"}

st.set_page_config(page_title="GridSentinel", layout="wide")
st.markdown("""<style>
.stApp{background:#F4F7FB;color:#172033;font-family:-apple-system,"Segoe UI",sans-serif}
.card{background:#fff;border:1px solid #DCE2EA;border-radius:15px;padding:14px 16px;margin-bottom:10px}
.card h4{margin:0 0 6px 0;font-size:.95rem}.muted{color:#5d687b;font-size:.82rem}
.chip{display:inline-block;color:#fff;border-radius:999px;padding:2px 10px;font-size:.72rem;font-weight:700;margin-right:6px}
.tag{display:inline-block;border:1px solid #DCE2EA;border-radius:6px;padding:1px 7px;font-size:.7rem;color:#5d687b}
.badge{border:2px solid #c62828;color:#c62828;border-radius:10px;padding:6px 12px;font-weight:800;display:inline-block}
.banner{border-radius:10px;padding:10px 14px;margin:6px 0;font-size:.9rem}
.warn{background:#fff7e6;border:1px solid #f0c36d}.err{background:#fdecea;border:1px solid #f5a5a0}.info{background:#edf2ff;border:1px solid #cbd8f9}
.trace{border-left:3px solid #2457D6;background:#fff;border-radius:8px;padding:8px 12px;margin-bottom:8px;font-size:.85rem}
</style>""", unsafe_allow_html=True)


def call(method: str, path: str, **kw: Any) -> Tuple[bool, Any]:
    try:
        r = httpx.request(method, API + path, headers=HEAD if method != "GET" else None, timeout=15, **kw)
        data = r.json()
        return r.status_code < 400, data
    except Exception:
        return False, {"error": {"code": "unreachable", "message": "Cannot reach the controller. Retrying in 10 s."}}


def chip(text: str) -> str:
    return f'<span class="chip" style="background:{CHIP.get(text, "#6b7280")}">{text}</span>'


def banner(kind: str, text: str) -> None:
    st.markdown(f'<div class="banner {kind}">{text}</div>', unsafe_allow_html=True)


def fmt(ts: Optional[str]) -> str:
    if not ts:
        return "—"
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ts


# ---------------------------------------------------------------------------------- header
ok_s, settings = call("GET", "/api/health")
c1, c2, c3, c4 = st.columns([3, 3, 3, 2])
with c1:
    st.markdown("### GridSentinel Preemptive Controller")
with c2:
    st.markdown('<div class="badge">SIMULATED ACTUATOR — DIGITAL TWIN</div>', unsafe_allow_html=True)
with c3:
    cur_mode = "approval_required"
    ok_m, _ = call("GET", "/api/twin/state")
    mode = st.radio("Mode", ["approval_required", "autonomous"], horizontal=True, index=0, key="mode",
                    help="autonomous applies validated actions to the twin without approval")
with c4:
    run = st.button("Run cycle now", type="primary", width="stretch")
offset = st.slider("Counterfactual: applied on top of live data (°C added to the live forecast)", 0.0, 8.0, 0.0, 0.5, key="offset")
if offset > 0:
    banner("info", f"Counterfactual: +{offset} °C applied on top of live data")

if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode
if st.session_state.last_mode != mode:
    call("PUT", "/api/settings", json={"mode": mode}); st.session_state.last_mode = mode

if run:
    ok, resp = call("POST", "/api/cycles", json={"mode": mode, "stress_offset_c": offset})
    if not ok:
        banner("err", resp.get("error", {}).get("message", "Request failed."))
    else:
        cid = resp["cycle_id"]
        st.subheader("Agent reasoning (streaming)")
        box = st.container()
        try:
            with httpx.stream("GET", f"{API}/api/cycles/{cid}/events", timeout=120) as r:
                for line in r.iter_lines():
                    if line.startswith("data: {") and '"n"' in line:
                        t = json.loads(line[6:])
                        box.markdown(f'<div class="trace"><b>{t["n"]}. {t["tool"]}</b><br>observed: {t["observed"]}<br>'
                                     f'decided: {t["decided"]}<br>why: {t["why"]}<br>result: {t["tool_result_summary"]}<br>'
                                     f'<span class="muted">{" · ".join(c["source_name"] + " @ " + fmt(c["ts"]) for c in t["citations"][:3])}</span></div>',
                                     unsafe_allow_html=True)
        except Exception:
            banner("err", "Cannot reach the controller. Retrying in 10 s.")


@st.fragment(run_every=10)
def dashboard() -> None:
    ok_i, inputs = call("GET", "/api/inputs")
    if not ok_i:
        banner("err", inputs["error"]["message"]); return
    ok_c, cyc = call("GET", "/api/cycles/latest")
    cyc = cyc if ok_c else None
    src = inputs["sources"]

    # banners ------------------------------------------------------------------------------
    for n in inputs.get("notices", []):
        banner("warn", n)
    if cyc:
        for n in cyc.get("notices", []):
            banner("info" if n.startswith(("Deterministic", "Counterfactual")) else "warn", n)
    if inputs.get("waiting_for_live_data"):
        banner("info", "Waiting for live data… nothing is shown until a real source responds.")

    # live inputs ---------------------------------------------------------------------------
    st.markdown("#### Live inputs")
    cols = st.columns(5)
    def card(col, title: str, key: str, body: str, tag: str = "") -> None:
        s = src[key]; p = s["provenance"]
        status = "PUBLISHED" if p["signal_kind"] == "PUBLISHED_SCHEDULE" and p["status"] != "FAILED" else p["status"]
        col.markdown(f'<div class="card"><h4>{title}</h4>{chip(status)}<span class="tag">{tag or p["signal_kind"]}</span><br>'
                     f'<div style="font-size:1.05rem;font-weight:700;margin:6px 0">{body}</div>'
                     f'<div class="muted">{p["source_name"]}<br>{fmt(p["fetched_at"]) if s["data"] else "no data"}'
                     f'{"<br>" + p["stale_reason"][:90] if p.get("stale_reason") else ""}</div></div>', unsafe_allow_html=True)
    w, m, g, a, t = (src[k]["data"] for k in ("weather", "market", "grid_signal", "alerts", "telemetry"))
    card(cols[0], "Weather", "weather", f'{w["points"][0]["temp_dry_c"]:.1f} °C · {w["points"][0]["rh_pct"]:.0f}% RH' if w else "—")
    card(cols[1], "Electricity price", "market", f'{m["points"][0]["tariff_inr_per_kwh"]:.2f} INR/kWh (published tariff)' if m else "—", "PUBLISHED_SCHEDULE")
    card(cols[2], "Live grid signal", "grid_signal", f'{g["value"]:.0f} {g["unit"]}' if g else "—")
    card(cols[3], "Alerts", "alerts", f'{len(a["alerts"])} bulletins' if a else "—")
    card(cols[4], "IT load", "telemetry", f'{t["it_load_kw"]:.0f} kW ({t["host_label"]})' if t else "—", "MEASURED→MODEL_DERIVED")

    if not cyc:
        st.info("No control cycle has run yet. Press “Run cycle now”.")
    else:
        fc = cyc.get("forecast")
        # forecast chart ---------------------------------------------------------------------
        st.markdown("#### 12-hour forecast and stress windows")
        if fc:
            fig = go.Figure()
            fig.add_scatter(x=fc["ts"], y=fc["temp_dry_c"], name="Ambient °C (live forecast" + (" + counterfactual)" if cyc["counterfactual"] else ")"))
            fig.add_scatter(x=fc["ts"], y=fc["tariff_inr_per_kwh"], name="Published tariff INR/kWh", yaxis="y2", line=dict(dash="dot"))
            for w_ in cyc["stress_windows"]:
                fig.add_vrect(x0=w_["start"], x1=w_["end"], fillcolor="#c62828", opacity=0.12, line_width=0,
                              annotation_text="/".join(w_["indicators"]), annotation_position="top left")
            if cyc.get("plan"):
                acts = [s for s in cyc["plan"]["steps"] if s["q_mech_kw"] + s["q_evap_kw"] > 1]
                fig.add_scatter(x=[fc["ts"][s["step"]] for s in acts], y=[fc["temp_dry_c"][s["step"]] for s in acts], mode="markers",
                                marker=dict(size=5, color="#2457D6"), name="Cooling action steps")
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#F4F7FB", plot_bgcolor="#fff",
                              yaxis2=dict(overlaying="y", side="right"), legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")
        if cyc.get("message"):
            st.info(cyc["message"])

        # plan + trace -------------------------------------------------------------------------
        left, right = st.columns(2)
        with left:
            st.markdown("#### Plan (first 12 steps)")
            if cyc.get("plan"):
                rows = [{"time": fc["ts"][s["step"]][11:16] if fc else s["step"], "mode": s["cooling_mode"],
                         "cooling kW": round(s["q_mech_kw"] + s["q_evap_kw"], 1), "setpoint °C": round(s["air_setpoint_c"], 2),
                         "power kW": round(s["expected_power_kw"], 1), "water L": round(s["expected_water_l"], 1)} for s in cyc["plan"]["steps"][:12]]
                st.dataframe(rows, width="stretch", hide_index=True)
                if cyc["plan"]["relaxed_constraints"]:
                    st.caption("Relaxed: " + ", ".join(cyc["plan"]["relaxed_constraints"]))
            else:
                st.caption("No plan (cycle held the safe state).")
        with right:
            st.markdown("#### Reasoning trace")
            for t_ in cyc["trace"]:
                st.markdown(f'<div class="trace"><b>{t_["n"]}. {t_["tool"]}</b><br>observed: {t_["observed"]}<br>decided: {t_["decided"]}<br>'
                            f'why: {t_["why"]}<br>result: {t_["tool_result_summary"]}<br><span class="muted">'
                            f'{" · ".join(c["source_name"] + " @ " + fmt(c["ts"]) for c in t_["citations"][:3])}</span></div>', unsafe_allow_html=True)

        # comparison -----------------------------------------------------------------------------
        st.markdown("#### Reactive vs preemptive (simulated)")
        if cyc.get("comparison"):
            names = {"cost_inr": ("Cost (INR)", 0), "peak_kw": ("Peak (kW)", 1), "water_l": ("Water (L)", 0), "max_temp_c": ("Max temp (°C)", 2)}
            mc = st.columns(4)
            for col, (k, (label, dp)) in zip(mc, names.items()):
                v = cyc["comparison"][k]; sv = "—" if v["savings_pct"] is None else f'{v["savings_pct"]:+.1f}% saved'
                col.markdown(f'<div class="card"><h4>{label}</h4>reactive <b>{v["baseline"]:.{dp}f}</b><br>preemptive <b>{v["plan"]:.{dp}f}</b>'
                             f'<br><span class="muted">Δ {v["delta"]:+.{dp}f} · {sv}</span></div>', unsafe_allow_html=True)
            tr = cyc.get("trajectories")
            if tr:
                f2 = go.Figure()
                f2.add_scatter(y=tr["baseline"]["t_air"], name="Reactive T_air °C"); f2.add_scatter(y=tr["plan"]["t_air"], name="Preemptive T_air °C")
                f2.add_scatter(y=tr["baseline"]["power_kw"], name="Reactive kW", yaxis="y2", line=dict(dash="dot"))
                f2.add_scatter(y=tr["plan"]["power_kw"], name="Preemptive kW", yaxis="y2", line=dict(dash="dot"))
                f2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis2=dict(overlaying="y", side="right"),
                                 xaxis_title="15-minute step", legend=dict(orientation="h"), paper_bgcolor="#F4F7FB", plot_bgcolor="#fff")
                st.plotly_chart(f2, width="stretch")
            st.caption("Water may rise under the preemptive plan: evaporative assist trades water for electricity.")
        else:
            st.caption("Comparison appears after a cycle produces a plan.")

    # action log --------------------------------------------------------------------------------------
    st.markdown("#### Action log")
    ok_a, acts = call("GET", "/api/actions")
    if ok_a:
        for ac in acts["actions"][:8]:
            r1, r2, r3 = st.columns([5, 1, 1])
            r1.markdown(f'{chip(ac["status"])} <b>{ac["action_id"]}</b> · plan {ac["plan_id"]} · {fmt(ac["created_at"])}'
                        f'<br><span class="muted">{"; ".join(ac["rejection_reasons"])[:200] or ac.get("note") or ""}</span>', unsafe_allow_html=True)
            if ac["status"] == "awaiting_approval":
                if r2.button("Approve", key="ap" + ac["action_id"]):
                    ok, resp = call("POST", f"/api/actions/{ac['action_id']}/approve"); st.rerun()
                if r3.button("Reject", key="rj" + ac["action_id"]):
                    call("POST", f"/api/actions/{ac['action_id']}/reject"); st.rerun()
            elif ac["status"] == "applied":
                if r2.button("Rollback", key="rb" + ac["action_id"]):
                    call("POST", f"/api/actions/{ac['action_id']}/rollback"); st.rerun()
        if not acts["actions"]:
            st.caption("No actions yet.")
        if acts["operator_alerts"]:
            with st.expander("Operator alerts (in-app)"):
                for al in acts["operator_alerts"][:10]:
                    st.write(f'{fmt(al["ts"])} — {al["message"]}')

    # data health ---------------------------------------------------------------------------------------
    st.markdown("#### Data health")
    ok_h, h = call("GET", "/api/data-health")
    if ok_h:
        st.dataframe([{"source": x["source"], "status": x["status"], "kind": x["signal_kind"], "latency ms": x["latency_ms"],
                       "last success": fmt(x["last_success"]), "age s": x["age_s"], "max age s": x["max_age_s"], "note": x["stale_reason"] or ""}
                      for x in h["sources"]], width="stretch", hide_index=True)

    ok_p, perf = call("GET", "/api/performance")
    if ok_p:
        st.markdown(f'<div class="muted">Last cycle: {perf.get("runtime_ms")} ms · {perf.get("memory_mb")} MB · '
                    f'The facility is a physics-based digital twin (SIMULATED ACTUATOR); it never controls real hardware.</div>', unsafe_allow_html=True)
    with st.expander("Assumed model constants (must be calibrated)"):
        st.dataframe([{"name": c["name"], "value": c["value"], "unit": c["unit"]} for c in inputs.get("assumed_constants", [])], hide_index=True)


dashboard()
