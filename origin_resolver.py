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
_registry: Optional[Dict[str, LatLng]] = None


def _load_registry() -> Dict[str, LatLng]:
    """Read vehicle_origins.json once and cache the result."""
    global _registry
    if _registry is not None:
        return _registry
    if not _REGISTRY_PATH.exists():
        _registry = {}
        return _registry
    try:
        raw: dict = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        parsed: Dict[str, LatLng] = {}
        for plate, coords in raw.items():
            if (
                isinstance(coords, (list, tuple))
                and len(coords) == 2
            ):
                try:
                    parsed[str(plate)] = (float(coords[0]), float(coords[1]))
                except (TypeError, ValueError):
                    pass  # skip malformed entry
        _registry = parsed
    except Exception as exc:
        print(f"[WARN] origin_resolver: không đọc được vehicle_origins.json: {exc}")
        _registry = {}
    return _registry


@dataclass
class OriginResolution:
    """Result of resolving a trip origin for one plate."""
    lat: float
    lng: float
    source: str  # "vehicle_registry" | "global_default"

    def as_latlng(self) -> LatLng:
        return (self.lat, self.lng)


def resolve_trip_origin(plate: str) -> OriginResolution:
    """
    Return the trip-origin coordinate for *plate*.

    Parameters
    ----------
    plate : str
        Vehicle plate number, exactly as it appears as a key in
        vehicle_origins.json (and as passed through trip_pipeline).

    Returns
    -------
    OriginResolution
        .lat / .lng : coordinate to use as trip start/end
        .source     : "vehicle_registry" if a per-plate entry was found,
                      "global_default"   if the system-wide depot is used
    """
    registry = _load_registry()
    if plate in registry:
        lat, lng = registry[plate]
        return OriginResolution(lat=lat, lng=lng, source="vehicle_registry")

    depot = maps_config.get_depot_origin()
    return OriginResolution(lat=depot[0], lng=depot[1], source="global_default")


def reload_registry() -> None:
    """Force reload of vehicle_origins.json (useful in tests / long-running processes)."""
    global _registry
    _registry = None
