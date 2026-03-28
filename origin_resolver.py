"""
origin_resolver.py
------------------
Resolves the trip-origin coordinate for a given vehicle plate.

Look-up priority
----------------
1. vehicle_origins.json  (project root, optional) — per-plate overrides
   Format: { "<plate>": [lat, lng], ... }
   The file is read once at first call and cached for the process lifetime.

2. DEPOT_LAT / DEPOT_LNG environment variables  — site-wide override

3. Hard-coded default  (10.802417, 106.501501)  — original company depot

The `source` field in the returned OriginResolution identifies which tier was
used so callers can surface this in reports for audit purposes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import maps_config

LatLng = Tuple[float, float]

_REGISTRY_PATH = Path(__file__).resolve().parent / "vehicle_origins.json"

# Module-level cache — populated lazily on first call.
_registry: Optional[Dict[str, Dict[str, Optional[LatLng]]]] = None


def _parse_latlng(value) -> Optional[LatLng]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
    ):
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _load_registry() -> Dict[str, Dict[str, Optional[LatLng]]]:
    """Read vehicle_origins.json once and cache the result.

    Supported formats:
      1) Legacy:
         { "<plate>": [lat, lng] }

      2) New:
         {
           "<plate>": {
             "start": [lat, lng],
             "end": [lat, lng]   # optional
           }
         }

    Rules:
      - legacy [lat, lng] -> start = end = that point
      - new object with start only -> end = start
      - new object with start+end -> use both
      - malformed rows are skipped
    """
    global _registry
    if _registry is not None:
        return _registry

    if not _REGISTRY_PATH.exists():
        _registry = {}
        return _registry

    try:
        raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        parsed: Dict[str, Dict[str, Optional[LatLng]]] = {}

        for plate, value in raw.items():
            # bỏ qua metadata như _comment, _format
            if str(plate).startswith("_"):
                continue

            plate_key = str(plate).strip()

            # --- Legacy format: "<plate>": [lat, lng]
            legacy_point = _parse_latlng(value)
            if legacy_point is not None:
                parsed[plate_key] = {
                    "start": legacy_point,
                    "end": legacy_point,
                }
                continue

            # --- New format: "<plate>": {"start": [...], "end": [...]}
            if isinstance(value, dict):
                start_point = _parse_latlng(value.get("start"))
                end_point = _parse_latlng(value.get("end"))

                if start_point is None:
                    # không có start hợp lệ thì bỏ qua entry
                    continue

                if end_point is None:
                    end_point = start_point

                parsed[plate_key] = {
                    "start": start_point,
                    "end": end_point,
                }
                continue

            # format khác thì bỏ qua
            continue

        _registry = parsed

    except Exception as exc:
        print(f"[WARN] origin_resolver: không đọc được vehicle_origins.json: {exc}")
        _registry = {}

    return _registry

@dataclass
class OriginResolution:
    start_lat: float
    start_lng: float
    end_lat: Optional[float]
    end_lng: Optional[float]
    source: str  # "vehicle_registry" | "global_default"

    def start_as_latlng(self) -> LatLng:
        return (self.start_lat, self.start_lng)

    def end_as_latlng(self) -> Optional[LatLng]:
        if self.end_lat is None or self.end_lng is None:
            return None
        return (self.end_lat, self.end_lng)


def resolve_trip_origin(plate: str) -> OriginResolution:
    """
    Return the configured trip endpoints for *plate*.

    Rules:
      - if vehicle_origins.json has plate -> use it
      - else use maps_config depot as both start and end
    """
    registry = _load_registry()

    if plate in registry:
        row = registry[plate]
        start = row.get("start")
        end = row.get("end") or start

        if start is not None:
            return OriginResolution(
                start_lat=start[0],
                start_lng=start[1],
                end_lat=end[0] if end else None,
                end_lng=end[1] if end else None,
                source="vehicle_registry",
            )

    depot = maps_config.get_depot_origin()
    return OriginResolution(
        start_lat=depot[0],
        start_lng=depot[1],
        end_lat=depot[0],
        end_lng=depot[1],
        source="global_default",
    )


def reload_registry() -> None:
    """Force reload of vehicle_origins.json (useful in tests / long-running processes)."""
    global _registry
    _registry = None
