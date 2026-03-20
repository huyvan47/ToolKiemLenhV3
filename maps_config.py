"""
maps_config.py
--------------
Single source of truth for all external API configuration.

All values are read from environment variables.  Missing variables never
crash the system at import time — callers receive None or safe defaults.

Optional: place a `.env` file in the project root.  If `python-dotenv` is
installed it will be loaded automatically.  If not installed, env vars must
be set by the shell or CI before running.

Environment variables
---------------------
GOOGLE_MAPS_API_KEY
    Google Maps API key used by Geocoding, Directions, Roads, and Routes APIs.
    Required for any Google API call.  When absent, all Google calls are
    skipped and the system falls back to OSRM / straight-line.

MAP_MATCH_PROVIDER
    Which provider to use for GPS map-matching.
    Values: "osrm" (default) | "google_roads"
    "google_roads" requires GOOGLE_MAPS_API_KEY.

ROUTE_PROVIDER
    Which provider to use for corridor route fetching.
    Values: "google_directions" (default) | "google_routes" | "osrm"
    Both google_* values require GOOGLE_MAPS_API_KEY.

CORRIDOR_BUFFER_M
    Lateral tolerance (metres) around each route polyline for corridor check.
    Default: 200.0

DEPOT_LAT / DEPOT_LNG
    Latitude and longitude of the company depot (trip start and end point).
    Default: 10.802417, 106.501501
    All trips begin at the depot (leg 0) and end at the depot (return leg).

ENABLE_RETURN_LEG
    Whether to add a return-to-origin leg when building corridors.
    Default: "true" — every delivery loop must return to the depot.
    Set to "false" only for one-way trips where the vehicle does not return.

ENABLE_AVOID_TOLLS_ROUTE
    Request toll-free route alternative when building corridors.
    Values: "false" (default) | "true"

ENABLE_AVOID_HIGHWAYS_ROUTE
    Request no-highway route alternative when building corridors.
    Values: "false" (default) | "true"

ENABLE_AVOID_FERRIES_ROUTE
    Request no-ferry route alternative when building corridors.
    Values: "false" (default) | "true"
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Optional: load .env file if python-dotenv is installed.
# override=False means existing shell env vars take precedence.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore[import]
    _load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on shell environment


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def get_api_key() -> Optional[str]:
    """Return Google Maps API key from env, or None if not set."""
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    return key if key else None


def has_api_key() -> bool:
    """True if GOOGLE_MAPS_API_KEY is present and non-empty."""
    return get_api_key() is not None


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

_VALID_MAP_MATCH = ("osrm", "google_roads")
_VALID_ROUTE = ("google_directions", "google_routes", "osrm")


def get_map_match_provider() -> str:
    """
    Return the map-matching provider.
    Values: 'osrm' (default) | 'google_roads'
    Falls back to 'osrm' on unrecognised value.
    """
    val = os.environ.get("MAP_MATCH_PROVIDER", "osrm").strip().lower()
    return val if val in _VALID_MAP_MATCH else "osrm"


def get_route_provider() -> str:
    """
    Return the route-building provider.
    Values: 'google_directions' (default) | 'google_routes' | 'osrm'
    Falls back to 'google_directions' on unrecognised value.
    """
    val = os.environ.get("ROUTE_PROVIDER", "google_directions").strip().lower()
    return val if val in _VALID_ROUTE else "google_directions"


# ---------------------------------------------------------------------------
# Corridor geometry settings
# ---------------------------------------------------------------------------

def get_depot_origin() -> Tuple[float, float]:
    """
    Return the depot coordinate (lat, lng) used as trip start and end point.

    Reads DEPOT_LAT and DEPOT_LNG env vars.
    Default: (10.802417, 106.501501) — company depot.

    Falls back to the default if either value is missing or cannot be parsed.
    """
    _DEFAULT = (10.802417, 106.501501)
    lat_str = os.environ.get("DEPOT_LAT", "").strip()
    lng_str = os.environ.get("DEPOT_LNG", "").strip()
    if not lat_str or not lng_str:
        return _DEFAULT
    try:
        return (float(lat_str), float(lng_str))
    except ValueError:
        return _DEFAULT


def get_corridor_buffer_m() -> float:
    """
    Return lateral corridor buffer in metres.
    Default: 200.0.  Falls back to default on invalid value.
    """
    raw = os.environ.get("CORRIDOR_BUFFER_M", "").strip()
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return 200.0


# ---------------------------------------------------------------------------
# Corridor feature flags
# ---------------------------------------------------------------------------

def _bool_env(name: str, default: bool) -> bool:
    """Read a boolean env var ('true'/'false').  Returns default on missing/invalid."""
    raw = os.environ.get(name, "").strip().lower()
    if raw == "true":
        return True
    if raw == "false":
        return False
    return default


def is_return_leg_enabled() -> bool:
    """
    Whether to add a return-to-origin leg when building trip corridors.

    Default: True — every delivery loop starts and ends at the depot.
    The return leg (stopN → depot) is built as a corridor leg and GPS points
    from the return trip are assigned to it (not to the last delivery leg).

    Set ENABLE_RETURN_LEG=false only for one-way trips where the vehicle
    does not return to the depot within the same GPS recording.

    NOTE: expected_distance_km and detour_ratio always reflect delivery legs
    only (origin → stop1 → … → stopN), regardless of this setting.
    The return leg affects corridor_compliance_pct but not detour_ratio.
    """
    return _bool_env("ENABLE_RETURN_LEG", True)


def is_avoid_tolls_enabled() -> bool:
    """Request toll-free route alternative. Default: False."""
    return _bool_env("ENABLE_AVOID_TOLLS_ROUTE", False)


def is_avoid_highways_enabled() -> bool:
    """Request no-highway route alternative. Default: False."""
    return _bool_env("ENABLE_AVOID_HIGHWAYS_ROUTE", False)


def is_avoid_ferries_enabled() -> bool:
    """Request no-ferry route alternative. Default: False."""
    return _bool_env("ENABLE_AVOID_FERRIES_ROUTE", False)
