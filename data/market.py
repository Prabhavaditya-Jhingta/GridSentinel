"""Market sources: (1) published time-of-day tariff schedule (PUBLISHED_SCHEDULE, never 'live'),
(2) optional live grid signal (UNVERIFIED Electricity Maps adapter)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import yaml

from config.settings import Settings
from data.http import HttpFailure, get_json
from data.schemas import GridSignal, MarketData, MarketPoint, SignalKind
from forecast.grid import floor15, step_times


class TariffNotConfigured(Exception):
    pass


def _hhmm(s: str) -> int:
    h, m = str(s).split(":")
    return int(h) * 60 + int(m)


def load_tariff_bands(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise TariffNotConfigured(f"Tariff schedule file not found: {path}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not str(raw.get("order_citation", "")).strip() or not str(raw.get("order_url", "")).strip():
        raise TariffNotConfigured("Tariff schedule not configured: cite the regulator's order (order_citation, order_url)")
    bands = raw.get("bands") or []
    if not bands:
        raise TariffNotConfigured("Tariff schedule not configured: no bands defined")
    spans = sorted((_hhmm(b["start"]), _hhmm(b["end"]), float(b["rate_inr_per_kwh"]), bool(b.get("is_peak", False))) for b in bands)
    cursor = 0
    for st, en, rate, _ in spans:
        if st != cursor or en <= st or rate <= 0:
            raise TariffNotConfigured("Tariff bands must be contiguous, positive and cover 00:00-24:00")
        cursor = en
    if cursor != 24 * 60:
        raise TariffNotConfigured("Tariff bands must cover 00:00-24:00")
    return {"spans": spans, "citation": raw["order_citation"], "url": raw["order_url"]}


def fetch_tariff(s: Settings, now: datetime | None = None) -> Tuple[MarketData, str, str, SignalKind]:
    cfg = load_tariff_bands(s.TARIFF_PATH)
    grid = step_times(floor15(now or datetime.now(timezone.utc)))
    pts: List[MarketPoint] = []
    for t in grid:
        local = t.astimezone(s.tz)
        minute = local.hour * 60 + local.minute
        for st, en, rate, peak in cfg["spans"]:
            if st <= minute < en:
                pts.append(MarketPoint(ts=t, tariff_inr_per_kwh=rate, is_peak_window=peak))
                break
    return MarketData(points=pts, order_citation=cfg["citation"], order_url=cfg["url"]), \
        "Published tariff schedule", cfg["url"], SignalKind.PUBLISHED_SCHEDULE


def fetch_grid_signal(s: Settings) -> Tuple[GridSignal, str, str, SignalKind]:
    """UNVERIFIED adapter: Electricity Maps latest carbon intensity. Verify URL/zone/free-tier fields first."""
    if s.GRID_PROVIDER.lower() != "electricitymaps" or not s.GRID_API_KEY or not s.GRID_ZONE:
        raise HttpFailure("Live grid signal not configured (set GRID_PROVIDER, GRID_API_KEY, GRID_ZONE)")
    url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
    d = get_json(url, params={"zone": s.GRID_ZONE}, headers={"auth-token": s.GRID_API_KEY},
                 timeout=s.HTTP_TIMEOUT_SECONDS, retries=s.HTTP_MAX_RETRIES)
    return GridSignal(name="carbon_intensity", value=float(d["carbonIntensity"]), unit="gCO2eq/kWh",
                      observed_at=datetime.fromisoformat(str(d["datetime"]).replace("Z", "+00:00"))), \
        "Electricity Maps carbon-intensity/latest", url, SignalKind.MEASURED
