# GridSentinel Preemptive Controller

Autonomous, preemptive cooling controller for a data center. It ingests **live** weather, a published tariff schedule, an optional live grid signal,
alert text and real host telemetry; forecasts 12 hours; finds stress windows; computes a plan with a deterministic optimizer (LP / receding horizon);
validates every action with a deterministic safety validator; and applies it to a **physics-based digital twin — `SIMULATED ACTUATOR`**.
An LLM agent orchestrates the tools and parses alerts; **it never sets numbers**.

> **SIMULATED ACTUATOR.** The facility is a digital twin. Nothing here controls real hardware.

## Architecture
```mermaid
flowchart LR
  W[Open-Meteo / OpenWeatherMap] --> I[data/ingest: fallback chain + LKG + max-age]
  T[Tariff order (PUBLISHED_SCHEDULE)] --> I
  G[Live grid signal (optional)] --> I
  A[Alert RSS] --> I
  H[psutil host telemetry] --> I
  I --> S[forecast/stress S1-S4]
  I --> AG[agent: LLM tool loop or deterministic sequence]
  AG -->|allowlisted tools| O[optimizer: LP + reactive baseline]
  O --> V[safety validator V1-V6]
  V -->|only if ok| TW[twin: 2-node RC + plant + water]
  TW --> API[FastAPI /api/*] --> UI[Streamlit dashboard]
```
Folders: `config/ data/ forecast/ twin/ optimizer/ safety/ agent/ api/ ui/ tests/`. All LLM access is in `agent/llm.py`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill FACILITY_LAT, FACILITY_LON, FACILITY_TIMEZONE, OPERATOR_TOKEN (>= 8 chars)
```
**Required before the controller can plan** (the app refuses to invent data):
1. **Tariff:** fill `config/tariff_schedule.yaml` from your state electricity regulator's tariff order (citation, URL, bands covering 00:00-24:00 local time).
   Ask the organizers to confirm that a published tariff schedule counts as an official dataset. Until then the market source is `FAILED` and cycles hold the safe state (E3).
2. **Calibrate `config/constants.yaml`:** every `assumed` constant is a team assumption (listed in the UI). Replace with cited/derived values for your facility.
3. Optional: `OPENWEATHER_API_KEY` (fallback), `ALERT_FEED_URL`, `GRID_PROVIDER/GRID_API_KEY/GRID_ZONE`, and `LLM_*` (`LLM_MODEL` is required for LLM mode; `LLM_BASE_URL` for non-Groq/OpenAI providers). Without an LLM the app runs in deterministic mode (E7).

## Run
```bash
set -a; source .env; set +a
uvicorn api.main:app_factory --factory --port 8000      # backend
streamlit run ui/app.py                                # dashboard (needs OPERATOR_TOKEN, API_BASE_URL in env)
# or one container: docker build -t gridsentinel . && docker run -p 7860:7860 --env-file .env gridsentinel
```
Hugging Face Spaces: Docker Space; set the variables as secrets. The disk is ephemeral, so cold start without network is a supported state (E17).

## Test
```bash
python -m pytest tests -q      # UT1-UT10 + AC/E/scenario tests (inputs are built inside tests/ only)
```

## UNVERIFIED — test in hour 0 (nothing below was run against the live services)
- Open-Meteo `minutely_15` / `forecast_minutely_15` parameters, first timestamp within 15 min of now; where no native 15-minute model exists Open-Meteo may interpolate hourly data itself.
- Open-Meteo archive endpoint (backtest), OpenWeatherMap response fields, Electricity Maps URL/zone/free-tier fields, your alert feed's format, LLM base URL and model name.
- Groq/OpenAI tool-calling behaviour with your chosen model.

## Measured results (fill from YOUR runs — none are claimed here)
| Metric | Value |
|---|---|
| Backtest MAE: persistence / same-hour-yesterday / mean-prev-3-days | _run `GET /api/forecast/backtest`_ |
| Runtime per cycle (ms) | _see `GET /api/performance`_ |
| Memory (MB) | _see `GET /api/performance`_ |
| Cost / peak / water: reactive vs preemptive | _from your live cycles (dashboard comparison)_ |

## Data sources
| Input | Source | signal_kind |
|---|---|---|
| Weather | Open-Meteo (primary), OpenWeatherMap (fallback; no radiation invented) | MEASURED (wet-bulb MODEL_DERIVED) |
| Price | Regulator's tariff order in `config/tariff_schedule.yaml` | PUBLISHED_SCHEDULE (never shown as live) |
| Grid signal | Optional Electricity Maps adapter; S2 uses the app's own stored observations (7 days) | MEASURED |
| Alerts | `ALERT_FEED_URL` (RSS) | MEASURED |
| IT load | psutil on the **backend host** (labeled `backend_host:<name>`), mapped to kW by a documented formula | MEASURED → MODEL_DERIVED |

## Design decisions to know before judging
- Exact (matrix-exponential) discretisation: explicit Euler would be unstable for the default constants (dt/(C_air·R_mass) ≈ 1.13).
- Optimizer is an LP (HiGHS via SciPy), so cooling is two continuous variables (mechanical, evaporative); mode is derived. Relaxation order: peak cap → water budget → recommended maximum; hard limits (18–27 °C) never.
- Fair comparison: the plan's final thermal-mass temperature may not be warmer than the baseline's.
- Validator is the only gate to the twin; `execute_step` re-validates even if the agent already did. Plan hashes stop edited plans (V1).
- Telemetry is re-sampled when an action is applied; a >1.0 °C deviation from the prediction triggers rollback (E15).
- Twin state changes only when an action is applied; holding the safe state leaves it unchanged.

## Limitations
Digital twin, not hardware. Constants are assumptions until calibrated. Cold-weather lower-bound infeasibility (no heating model) holds the safe state. Alert relevance needs `FACILITY_REGION_NAME`. The generic "demo data" lines in the spec-builder template were intentionally **not** followed (Q9 forbids dummy data).
