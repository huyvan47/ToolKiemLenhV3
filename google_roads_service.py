"""
google_roads_service.py
-----------------------
Thin, stateless wrapper around the Google Roads API snapToRoads endpoint.

Responsibilities
----------------
- snap_path_to_roads(): chunk + snap a GPS path to the road network.
- _snap_chunk()       : single API call for one chunk (≤ 100 points).
- GoogleRoadsError    : raised on any API-level or HTTP failure.

Does NOT
--------
- Decide whether to use Google Roads or fall back to OSRM — that is the
  caller's responsibility (see trace_reconstructor._do_map_match).
- Import any other project module (no circular dependencies).
- Read API keys or config — caller passes api_key explicitly.

Google Roads snapToRoads limits
--------------------------------
- Max 100 points per request.
- When interpolate=True the response may contain MORE points than sent
  (the API inserts intermediate road points).
- Each snapped point has a 'location' with 'latitude' and 'longitude'.
  Interpolated points omit the 'originalIndex' field.
"""
from __future__ import annotations

import logging
import math
from typing import List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

LatLng = Tuple[float, float]

ROADS_SNAP_URL: str = "https://roads.googleapis.com/v1/snapToRoads"

# Google Roads API hard limit per request.
ROADS_MAX_POINTS: int = 100

# Default chunk size used by snap_path_to_roads (≤ ROADS_MAX_POINTS).
_DEFAULT_CHUNK_SIZE: int = 100

# Points from the previous chunk included at the start of the next chunk.
# Helps the API produce smoother transitions at chunk boundaries.
_CHUNK_OVERLAP: int = 5

# Distance threshold (metres) for removing duplicate boundary points
# that appear in both the tail of one snapped chunk and the head of the next.
_DEDUP_THRESHOLD_M: float = 10.0


# ---------------------------------------------------------------------------
# Module-level HTTP session with retry (no project imports).
# ---------------------------------------------------------------------------

def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


_SESSION = _build_session()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GoogleRoadsError(Exception):
    """Raised when the Google Roads API returns an error or empty result."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _haversine_m(p1: LatLng, p2: LatLng) -> float:
    """Haversine distance in metres between two (lat, lng) points."""
    R = 6_371_000.0
    lat1 = math.radians(p1[0])
    lat2 = math.radians(p2[0])
    dlat = math.radians(p2[0] - p1[0])
    dlng = math.radians(p2[1] - p1[1])
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _snap_chunk(
    chunk: List[LatLng],
    api_key: str,
    interpolate: bool,
    timeout: int,
) -> List[LatLng]:
    """
    Call snapToRoads for one chunk of ≤ ROADS_MAX_POINTS points.

    Parameters
    ----------
    chunk       : (lat, lng) points, len ≤ ROADS_MAX_POINTS
    api_key     : Google Maps API key
    interpolate : passed as-is to the API
    timeout     : HTTP timeout in seconds

    Returns
    -------
    List[LatLng] snapped to the road network.

    Raises
    ------
    GoogleRoadsError  on HTTP error, API error body, or empty result.
    """
    path_str = "|".join(f"{lat},{lng}" for lat, lng in chunk)
    params = {
        "path": path_str,
        "interpolate": "true" if interpolate else "false",
        "key": api_key,
    }

    try:
        resp = _SESSION.get(ROADS_SNAP_URL, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise GoogleRoadsError(f"Roads API HTTP error: {exc}") from exc

    data = resp.json()

    # The API returns {"error": {...}} on failure (even with HTTP 200).
    if "error" in data:
        err = data["error"]
        raise GoogleRoadsError(
            f"Roads API error {err.get('code')}: {err.get('message', 'unknown')}"
        )

    snapped = data.get("snappedPoints") or []
    if not snapped:
        raise GoogleRoadsError("Roads API returned no snappedPoints")

    result: List[LatLng] = []
    for pt in snapped:
        loc = pt.get("location", {})
        lat = loc.get("latitude")
        lng = loc.get("longitude")
        if lat is not None and lng is not None:
            result.append((float(lat), float(lng)))

    if not result:
        raise GoogleRoadsError("Roads API snappedPoints contained no valid locations")

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def snap_path_to_roads(
    points: List[LatLng],
    api_key: str,
    interpolate: bool = True,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    timeout: int = 30,
) -> List[LatLng]:
    """
    Snap a GPS path to the road network using Google Roads snapToRoads API.

    Handles chunking automatically (API hard limit: 100 points per request).
    Consecutive chunks overlap by _CHUNK_OVERLAP points for continuity.
    Duplicate boundary points are removed by distance-based deduplication.

    Parameters
    ----------
    points      : (lat, lng) GPS breadcrumbs to snap
    api_key     : Google Maps API key (caller's responsibility to provide)
    interpolate : if True the API inserts intermediate road points —
                  output may be longer than input
    chunk_size  : points per request; must be ≤ ROADS_MAX_POINTS (100)
    timeout     : HTTP timeout in seconds per chunk request

    Returns
    -------
    List[LatLng] snapped to the road network.
    Output length ≥ input length when interpolate=True.
    Returns [] for empty input.

    Raises
    ------
    ValueError        if chunk_size > ROADS_MAX_POINTS
    GoogleRoadsError  if any chunk call fails
    """
    if not points:
        return []

    if chunk_size > ROADS_MAX_POINTS:
        raise ValueError(
            f"chunk_size={chunk_size} exceeds Roads API limit of {ROADS_MAX_POINTS}"
        )

    # Fast path: single request, no chunking needed.
    if len(points) <= chunk_size:
        logger.debug("Roads API: single chunk (%d points)", len(points))
        return _snap_chunk(points, api_key, interpolate, timeout)

    # Multi-chunk path.
    all_snapped: List[LatLng] = []
    start = 0

    while start < len(points):
        end = min(start + chunk_size, len(points))
        chunk = list(points[start:end])

        logger.debug(
            "Roads API: chunk [%d:%d] (%d points, total so far: %d)",
            start, end, len(chunk), len(all_snapped),
        )

        snapped_chunk = _snap_chunk(chunk, api_key, interpolate, timeout)

        # Remove boundary duplicate: if the first point of the new snapped
        # chunk is within _DEDUP_THRESHOLD_M of the last accumulated point,
        # drop it to avoid a stutter at the chunk boundary.
        if all_snapped and snapped_chunk:
            if _haversine_m(all_snapped[-1], snapped_chunk[0]) < _DEDUP_THRESHOLD_M:
                snapped_chunk = snapped_chunk[1:]

        all_snapped.extend(snapped_chunk)

        if end >= len(points):
            break
        start = end - _CHUNK_OVERLAP

    logger.debug(
        "Roads API: %d raw points → %d snapped points", len(points), len(all_snapped)
    )
    return all_snapped
