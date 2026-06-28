"""Settings endpoints backed by the ``user_profiles`` table (active profile)."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends

from hrp.api.http.deps import get_api
from hrp.api.http.schemas import Settings, SettingsUpdate

router = APIRouter(prefix="/settings", tags=["settings"])

_SELECT_ACTIVE = "SELECT * FROM user_profiles WHERE active = TRUE ORDER BY created_at DESC LIMIT 1"


def _csv_to_list(value: Any) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in str(value).split(",") if s.strip()]


def _list_to_csv(value: list[str] | None) -> str | None:
    if value is None:
        return None
    return ",".join(value)


def get_active_profile(api) -> dict | None:
    """Return the active user profile row as a dict, or None if absent."""
    df = api.query_readonly(_SELECT_ACTIVE, [])
    if df is None or df.empty:
        return None
    return df.iloc[0].to_dict()


def upsert_profile(api, updates: dict[str, Any]) -> dict:
    """Update the active profile in place, or create one if none exists."""
    existing = get_active_profile(api)
    if existing:
        sets, params = [], []
        for col, val in updates.items():
            sets.append(f"{col} = ?")
            params.append(val)
        if sets:
            params.append(existing["profile_id"])
            api.execute_write(
                f"UPDATE user_profiles SET {', '.join(sets)} WHERE profile_id = ?",
                params,
            )
    else:
        profile_id = uuid.uuid4().hex
        row = {
            "profile_id": profile_id,
            "name": updates.get("name", "Default"),
            "risk_tolerance": updates.get("risk_tolerance", 3),
            "portfolio_value": updates.get("portfolio_value", 100000.0),
            "max_positions": updates.get("max_positions", 20),
            "max_position_pct": updates.get("max_position_pct", 0.10),
            "excluded_sectors": updates.get("excluded_sectors"),
            "excluded_symbols": updates.get("excluded_symbols"),
            "preferred_horizon": updates.get("preferred_horizon", "medium"),
        }
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        api.execute_write(
            f"INSERT INTO user_profiles ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )
    return get_active_profile(api) or {}


def _to_settings(row: dict) -> Settings:
    return Settings(
        profile_id=row.get("profile_id"),
        name=row.get("name") or "Default",
        risk_tolerance=int(row.get("risk_tolerance") or 3),
        portfolio_value=float(row.get("portfolio_value") or 100000.0),
        max_positions=int(row.get("max_positions") or 20),
        max_position_pct=float(row.get("max_position_pct") or 0.10),
        excluded_symbols=_csv_to_list(row.get("excluded_symbols")),
        excluded_sectors=_csv_to_list(row.get("excluded_sectors")),
        preferred_horizon=row.get("preferred_horizon") or "medium",
    )


@router.get("", response_model=Settings)
def read_settings(api=Depends(get_api)) -> Settings:
    profile = get_active_profile(api)
    return _to_settings(profile) if profile else Settings()


@router.put("", response_model=Settings)
def update_settings(body: SettingsUpdate, api=Depends(get_api)) -> Settings:
    updates = body.model_dump(exclude_none=True)
    # CSV-encode list fields for the VARCHAR columns.
    for field in ("excluded_symbols", "excluded_sectors"):
        if field in updates:
            updates[field] = _list_to_csv(updates[field])
    saved = upsert_profile(api, updates)
    return _to_settings(saved)
