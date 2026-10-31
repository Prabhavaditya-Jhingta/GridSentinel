"""Test fixtures. Inputs built here are TEST-ONLY and never imported by runtime code."""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TARIFF_YAML = """order_citation: "TEST ONLY - not a real tariff order"
order_url: "https://example.invalid/test-order"
bands:
  - {start: "00:00", end: "12:00", rate_inr_per_kwh: 6.0, is_peak: false}
  - {start: "12:00", end: "18:00", rate_inr_per_kwh: 12.0, is_peak: true}
  - {start: "18:00", end: "24:00", rate_inr_per_kwh: 6.0, is_peak: false}
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "tariff.yaml").write_text(TARIFF_YAML)
    for k, v in {"FACILITY_LAT": "13.0", "FACILITY_LON": "80.2", "FACILITY_TIMEZONE": "UTC", "OPERATOR_TOKEN": "test-token-123",
                 "DB_PATH": str(tmp_path / "t.db"), "TARIFF_PATH": str(tmp_path / "tariff.yaml"), "AUTO_CYCLE": "false",
                 "HTTP_MAX_RETRIES": "1", "HTTP_TIMEOUT_SECONDS": "1", "LLM_API_KEY": "", "LLM_MODEL": ""}.items():
        monkeypatch.setenv(k, v)
    from config.settings import get_settings
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def make_weather(offset: float = 0.0):
    """TEST-ONLY synthetic forecast used to exercise the pipeline offline."""
    from data.schemas import WeatherForecast, WeatherPoint
    from forecast.grid import floor15, step_times, wet_bulb_stull
    pts = []
    for k, t in enumerate(step_times(floor15(datetime.now(timezone.utc)))):
        td = 31.0 + offset + 7.0 * np.sin(np.pi * k / 48)
        pts.append(WeatherPoint(ts=t, temp_dry_c=float(td), rh_pct=60.0, temp_wet_c=wet_bulb_stull(td, 60.0), solar_w_m2=None))
    return WeatherForecast(points=pts)


@pytest.fixture
def service(env):
    from api.main import build_service
    svc = build_service()
    from data.schemas import SignalKind
    svc.d.ingestor.chains["weather"] = [lambda: (make_weather(), "TEST weather", "test://weather", SignalKind.MEASURED)]
    svc.d.ingestor.chains["alerts"] = [lambda: (_ for _ in ()).throw(RuntimeError("no feed in tests"))]
    svc.d.ingestor.chains["grid_signal"] = [lambda: (_ for _ in ()).throw(RuntimeError("not configured"))]
    return svc


@pytest.fixture
def client(service):
    from fastapi.testclient import TestClient
    from api.main import create_app
    return TestClient(create_app(service))


HDR = {"X-Operator-Token": "test-token-123"}
