"""
google_routes_service.py
------------------------
Thin, stateless wrapper around the Google Routes API v2 computeRoutes endpoint.

Responsibilities
----------------
- fetch_routes() : POST to computeRoutes, parse response into RoutesApiRoute list.
- RoutesApiRoute : lightweight result dataclass (independent of corridor_builder).
- GoogleRoutesError : raised on any API-level or HTTP failure.

Does NOT
--------
- Import any other project module (no circular dependencies).
- Read API keys or config — caller passes api_key explicitly.
- Decide when to call or fallback — that is corridor_builder's responsibility.

Google Routes API v2 notes
--------------------------
- Endpoint : POST https://routes.googleapis.com/directions/v2:computeRoutes
- Auth      : X-Goog-Api-Key header (preferred) or key= query param.
- REQUIRED  : X-Goog-FieldMask header — without it the API returns empty {} or 400.
- Polyline  : routes[].polyline.encodedPolyline uses Google precision-5 encoding
              (same as Directions API), decoded by the `polyline` library directly.
- Alternatives: controlled by computeAlternativeRoutes boolean in request body.
  Returns up to 3 routes per request.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import polyline as _polyline_codec
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

LatLng = Tuple[float, float]

ROUTES_API_URL: str = "https://routes.googleapis.com/directions/v2:computeRoutes"

# Minimal field mask: request only the fields we actually consume.
# Without this header the API returns an empty body or HTTP 400.
_FIELD_MASK: str = "routes.distanceMeters,routes.polyline.encodedPolyline"


# ---------------------------------------------------------------------------
# Result type (independent of corridor_builder.RouteOption)
# ---------------------------------------------------------------------------

@dataclass
class RoutesApiRoute:
    """
    One decoded route returned by computeRoutes.

    Mirrors the fields of corridor_builder.RouteOption so that conversion
    is a trivial one-liner in the caller.  Defined here to avoid circular
    imports between google_routes_service and corridor_builder.
    """
    source: str          # e.g. "google_routes_v2_0", "google_routes_v2_notolls_0"
    coords: List[LatLng] # (lat, lng) tuples decoded from encodedPolyline
    distance_m: float    # metres


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GoogleRoutesError(Exception):
    """Raised when the Google Routes API returns an error or empty result."""


# ---------------------------------------------------------------------------
# Module-level HTTP session with retry
# ---------------------------------------------------------------------------

def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


_SESSION = _build_session()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_request_body(
    origin: LatLng,
    dest: LatLng,
    compute_alternatives: bool,
    avoid_tolls: bool,
    avoid_highways: bool,
    avoid_ferries: bool,
    routing_preference: str,
) -> Dict[str, Any]:
    """Build the JSON request body for computeRoutes."""
    return {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": origin[0],
                    "longitude": origin[1],
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": dest[0],
                    "longitude": dest[1],
                }
            }
        },
        "travelMode": "DRIVE",
        "routingPreference": routing_preference,
        "computeAlternativeRoutes": compute_alternatives,
        "routeModifiers": {
            "avoidTolls": avoid_tolls,
            "avoidHighways": avoid_highways,
            "avoidFerries": avoid_ferries,
        },
        "polylineQuality": "OVERVIEW",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_routes(
    origin: LatLng,
    dest: LatLng,
    api_key: str,
    compute_alternatives: bool = True,
    avoid_tolls: bool = False,
    avoid_highways: bool = False,
    avoid_ferries: bool = False,
    routing_preference: str = "TRAFFIC_UNAWARE",
    source_prefix: str = "google_routes_v2",
    timeout: int = 30,
) -> List[RoutesApiRoute]:
    """
    POST to computeRoutes and return decoded routes.

    Parameters
    ----------
    origin               : (lat, lng) trip-leg origin
    dest                 : (lat, lng) trip-leg destination
    api_key              : Google Maps API key (caller supplies)
    compute_alternatives : if True the API returns up to 3 route alternatives
    avoid_tolls          : request a toll-free route
    avoid_highways       : request a no-highway route
    avoid_ferries        : request a no-ferry route
    routing_preference   : "TRAFFIC_UNAWARE" (default, no real-time traffic),
                           "TRAFFIC_AWARE", or "TRAFFIC_AWARE_OPTIMAL"
    source_prefix        : prefix for RoutesApiRoute.source labels;
                           appended with route index: "{source_prefix}_{idx}"
    timeout              : HTTP timeout in seconds

    Returns
    -------
    List[RoutesApiRoute] — one entry per successfully parsed route.

    Raises
    ------
    GoogleRoutesError  on HTTP error, API error body, or all routes un-parseable.
    """
    body = _build_request_body(
        origin, dest,
        compute_alternatives, avoid_tolls, avoid_highways, avoid_ferries,
        routing_preference,
    )
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": _FIELD_MASK,
    }

    try:
        resp = _SESSION.post(ROUTES_API_URL, json=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise GoogleRoutesError(f"Routes API HTTP error: {exc}") from exc

    data = resp.json()

    # API-level error (may arrive with HTTP 200 or 4xx).
    if "error" in data:
        err = data["error"]
        raise GoogleRoutesError(
            f"Routes API error {err.get('code')}: {err.get('message', 'unknown')}"
        )

    raw_routes = data.get("routes") or []
    if not raw_routes:
        raise GoogleRoutesError("Routes API returned no routes")

    results: List[RoutesApiRoute] = []
    for idx, route in enumerate(raw_routes):
        dist_m = route.get("distanceMeters")
        encoded = route.get("polyline", {}).get("encodedPolyline")

        if not encoded or dist_m is None:
            logger.warning(
                "Routes API v2: route[%d] missing encodedPolyline or distanceMeters — skipped",
                idx,
            )
            continue

        coords: List[LatLng] = _polyline_codec.decode(encoded)
        results.append(RoutesApiRoute(
            source=f"{source_prefix}_{idx}",
            coords=coords,
            distance_m=float(dist_m),
        ))

    if not results:
        raise GoogleRoutesError(
            "Routes API returned routes but none had parseable polyline/distance"
        )

    logger.debug(
        "Routes API v2: %d route(s) [prefix=%s] from %s to %s",
        len(results), source_prefix, origin, dest,
    )
    return results
