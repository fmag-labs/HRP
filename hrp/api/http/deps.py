"""Dependencies and serialization helpers for the HTTP API."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pandas as pd


def get_api():
    """FastAPI dependency returning a PlatformAPI instance.

    Overridden in tests via ``app.dependency_overrides``. Imported lazily so the
    heavy data layer is not pulled in at module import time.
    """
    from hrp.api.platform import PlatformAPI

    return PlatformAPI()


def df_to_records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records (NaN/NaT -> null, dates -> ISO)."""
    if df is None or getattr(df, "empty", True):
        return []
    # pandas to_json handles NaN->null and Timestamp->ISO robustly.
    return json.loads(df.to_json(orient="records", date_format="iso"))


def to_jsonable(value: Any) -> Any:
    """Best-effort scalar coercion for non-JSON-native types."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
