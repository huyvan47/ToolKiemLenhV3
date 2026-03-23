"""
corridor_builder.py
-------------------
Builds a "feasible corridor" for each leg of a truck trip by collecting
multiple route alternatives from Google Directions and OSRM, then merging
them into a single CorridorLeg object.

A CorridorLeg answers one key question per GPS point:
  "Is this point within `buffer_m` metres of ANY feasible route?"

No shapely dependency — uses the same haversine distance_to_route helper
that already exists in vtracking_tool.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import polyline as polyline_codec

import google_routes_service
import maps_config
from vtracking_tool import (
    SESSION,
    OSRM_ROUTE,
    GOOGLE_ROUTE,
    haversine,
    distance_to_route,
)

LatLng = Tuple[float, float]

# Default lateral buffer around each route polyline (metres).
# Can be overridden via CORRIDOR_BUFFER_M env var (read through maps_config).
CORRIDOR_BUFFER_M: float = maps_config.get_corridor_buffer_m()

# Depot coordinate: trip start and mandatory return point.
# Configured via DEPOT_LAT / DEPOT_LNG env vars (see maps_config).
FIXED_ORIGIN: LatLng = maps_config.get_depot_origin()
# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RouteOption:
    """One candidate route between an origin and a destination."""
    source: str            # e.g. "google_0", "osrm_1"
    coords: List[LatLng]   # (lat, lng) pairs
    distance_m: float


@dataclass
class CorridorLeg:
    """
    All feasible routes for one leg of the trip, combined into a corridor.

    The corridor is represented implicitly: a point is "inside" the corridor
    if it is within `buffer_m` metres of at least one route's polyline.
    """
    leg_idx: int
    origin: LatLng
    dest: LatLng
    routes: List[RouteOption] = field(default_factory=list)
    buffer_m: float = CORRIDOR_BUFFER_M

    @property
    def min_distance_m(self) -> float:
        return min((r.distance_m for r in self.routes), default=0.0)

    @property
    def max_distance_m(self) -> float:
        return max((r.distance_m for r in self.routes), default=0.0)

    def distance_to_corridor(self, lat: float, lng: float) -> float:
        """
        Minimum distance from point to any route polyline in the corridor.

        Uses vertex-to-point distance (same as the existing distance_to_route
        helper).  This is accurate when route polylines are dense (vertices
        every ~50–200 m), which is always the case for Google/OSRM output.
        """
        if not self.routes:
            return float("inf")
        return min(distance_to_route((lat, lng), r.coords) for r in self.routes)

    def contains_point(self, lat: float, lng: float) -> bool:
        """True if the point is within buffer_m of any feasible route."""
        return self.distance_to_corridor(lat, lng) <= self.buffer_m


# ---------------------------------------------------------------------------
# Route fetchers
# ---------------------------------------------------------------------------

def _fetch_google_routes(origin: LatLng, dest: LatLng) -> List[RouteOption]:
    """
    Call Google Directions with alternatives=true.
    Returns up to 3 RouteOptions, or [] on failure.
    Returns [] immediately if GOOGLE_MAPS_API_KEY is not configured.
    """
    api_key = maps_config.get_api_key()
    if not api_key:
        return []
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{dest[0]},{dest[1]}",
        "mode": "driving",
        "alternatives": "true",
        "key": api_key,
    }
    try:
        r = SESSION.get(GOOGLE_ROUTE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK" or not data.get("routes"):
            print(f"[WARN] Google Directions: {data.get('status')} | {data.get('error_message', '')}")
            return []
        options: List[RouteOption] = []
        for idx, route in enumerate(data["routes"]):
            coords = polyline_codec.decode(route["overview_polyline"]["points"])
            dist_m = float(sum(leg["distance"]["value"] for leg in route["legs"]))
            options.append(RouteOption(source=f"google_{idx}", coords=coords, distance_m=dist_m))
        return options
    except Exception as e:
        print(f"[WARN] Google routes fetch lỗi: {e}")
        return []


def _fetch_osrm_routes(origin: LatLng, dest: LatLng) -> List[RouteOption]:
    """
    Call OSRM /route endpoint with alternatives=3.
    Note: OSRM uses lon,lat coordinate order.
    Returns up to 3 RouteOptions, or [] on failure.
    """
    coord_str = f"{origin[1]},{origin[0]};{dest[1]},{dest[0]}"
    url = f"{OSRM_ROUTE}{coord_str}"
    params = {
        "alternatives": "3",
        "overview": "full",
        "geometries": "polyline",  # precision-5, decoded by polyline library
    }
    try:
        r = SESSION.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            print(f"[WARN] OSRM route: code={data.get('code')}")
            return []
        options: List[RouteOption] = []
        for idx, route in enumerate(data["routes"]):
            coords = polyline_codec.decode(route["geometry"])
            dist_m = float(route["distance"])
            options.append(RouteOption(source=f"osrm_{idx}", coords=coords, distance_m=dist_m))
        return options
    except Exception as e:
        print(f"[WARN] OSRM routes fetch lỗi: {e}")
        return []


# ---------------------------------------------------------------------------
# Google Routes API v2 fetcher (new — requires ROUTE_PROVIDER=google_routes)
# ---------------------------------------------------------------------------

def _fetch_google_routes_v2(origin: LatLng, dest: LatLng) -> List[RouteOption]:
    """
    Call Google Routes API v2 computeRoutes.

    Only active when ROUTE_PROVIDER=google_routes AND GOOGLE_MAPS_API_KEY is set.
    Returns [] immediately (without error) when either condition is not met.
    Returns [] and logs a warning on any API failure so the caller can fall
    through to the next source.

    One base call with computeAlternativeRoutes=true is always made.
    Additional calls are made for enabled route modifiers:
      - ENABLE_AVOID_TOLLS_ROUTE=true  → extra toll-free alternative
      - ENABLE_AVOID_HIGHWAYS_ROUTE=true → extra no-highway alternative
      - ENABLE_AVOID_FERRIES_ROUTE=true  → extra no-ferry alternative
    """
    if maps_config.get_route_provider() != "google_routes":
        return []
    api_key = maps_config.get_api_key()
    if not api_key:
        return []

    options: List[RouteOption] = []

    def _call(avoid_tolls: bool, avoid_highways: bool, avoid_ferries: bool,
              source_prefix: str) -> None:
        try:
            results = google_routes_service.fetch_routes(
                origin, dest, api_key,
                compute_alternatives=True,
                avoid_tolls=avoid_tolls,
                avoid_highways=avoid_highways,
                avoid_ferries=avoid_ferries,
                source_prefix=source_prefix,
            )
            options.extend(
                RouteOption(source=r.source, coords=r.coords, distance_m=r.distance_m)
                for r in results
            )
        except google_routes_service.GoogleRoutesError as exc:
            print(f"[WARN] Google Routes API v2 ({source_prefix}): {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Google Routes API v2 ({source_prefix}) unexpected error: {exc}")

    # Base call — always attempted when provider is configured.
    _call(False, False, False, "google_routes_v2")

    # Optional modifier calls — each adds a distinct route alternative.
    if maps_config.is_avoid_tolls_enabled():
        _call(True, False, False, "google_routes_v2_notolls")
    if maps_config.is_avoid_highways_enabled():
        _call(False, True, False, "google_routes_v2_nohwy")
    if maps_config.is_avoid_ferries_enabled():
        _call(False, False, True, "google_routes_v2_noferry")

    return options


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_leg_corridor(
    leg_idx: int,
    origin: LatLng,
    dest: LatLng,
    buffer_m: float = CORRIDOR_BUFFER_M,
) -> CorridorLeg:
    """
    Fetch all feasible routes for one leg and return a CorridorLeg.

    Sources tried (union of all available results):
      1. Google Routes API v2 computeRoutes  (when ROUTE_PROVIDER=google_routes)
      2. Google Directions API (alternatives=true, up to 3 routes)
      3. OSRM /route (alternatives=3, up to 3 routes)

    Falls back to a straight-line "corridor" only if all three fail.
    """
    routes: List[RouteOption] = []
    routes.extend(_fetch_google_routes_v2(origin, dest))
    routes.extend(_fetch_google_routes(origin, dest))
    routes.extend(_fetch_osrm_routes(origin, dest))

    if not routes:
        straight_m = haversine(origin[0], origin[1], dest[0], dest[1])
        routes.append(RouteOption(
            source="fallback_straight",
            coords=[origin, dest],
            distance_m=straight_m,
        ))
        print(f"[WARN] Leg {leg_idx}: cả Google và OSRM đều lỗi — dùng đường thẳng fallback")

    leg = CorridorLeg(leg_idx=leg_idx, origin=origin, dest=dest, routes=routes, buffer_m=buffer_m)
    sources = ", ".join(r.source for r in routes)
    print(
        f"  Leg {leg_idx}: {len(routes)} route(s) [{sources}] | "
        f"{leg.min_distance_m / 1000:.1f}–{leg.max_distance_m / 1000:.1f} km corridor"
    )
    print(f"\n--- DEBUG Leg {leg_idx} coordinates ---")

    for r in routes:
        start = r.coords[0]
        end = r.coords[-1]

        print(
            f"Leg {leg_idx} | {r.source} | "
            f"origin={origin} -> route_start={start} | "
            f"route_end={end} -> dest={dest}"
        )
    return leg


def build_trip_corridors(
    stops: Sequence[dict],
    origin: Optional[LatLng] = None,
    buffer_m: float = CORRIDOR_BUFFER_M,
) -> List[CorridorLeg]:
    """
    Build corridors for every consecutive stop-pair in a trip.

    Tuyến luôn là vòng kín:
      effective_origin -> stop1 -> stop2 -> ... -> stopN -> effective_origin

    effective_origin:
      - origin param  nếu được truyền vào (xe xuất phát từ chi nhánh khác)
      - FIXED_ORIGIN  (global depot) nếu origin=None
    """
    geo_stops = [
        s for s in stops
        if not s.get("route_excluded")
        and s.get("lat") is not None
        and s.get("lng") is not None
    ]
    print(
        f"[DEBUG] total stops={len(stops)} | "
        f"geo_stops={len(geo_stops)} | "
        f"excluded={sum(1 for s in stops if s.get('route_excluded'))}"
    )
    if not geo_stops:
        return []

    effective_origin: LatLng = origin if origin is not None else FIXED_ORIGIN

    waypoints: List[LatLng] = [effective_origin]
    waypoints.extend((float(s["lat"]), float(s["lng"])) for s in geo_stops)

    # Luôn khép vòng: leg cuối quay về cùng điểm xuất phát
    waypoints.append(effective_origin)

    if len(waypoints) < 2:
        return []

    corridors: List[CorridorLeg] = []
    for i in range(len(waypoints) - 1):
        leg = build_leg_corridor(
            leg_idx=i,
            origin=waypoints[i],
            dest=waypoints[i + 1],
            buffer_m=buffer_m,
        )
        corridors.append(leg)

    return corridors


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # --- Unit tests (no network) ---
    print("=== Unit tests (offline) ===")

    p1: LatLng = (10.4549, 106.3416)
    p2: LatLng = (10.4000, 106.3600)
    p3: LatLng = (10.3634, 106.3700)

    # Dense-enough stub: 3 waypoints so every midpoint is within ~4 km
    # of the nearest vertex, keeping the test predictable.
    stub = RouteOption(source="test", coords=[p1, p2, p3], distance_m=12000)
    leg = CorridorLeg(leg_idx=0, origin=p1, dest=p3, routes=[stub], buffer_m=200)

    assert leg.contains_point(*p1),  "p1 (origin vertex) must be inside"
    assert leg.contains_point(*p2),  "p2 (middle vertex) must be inside"
    assert leg.contains_point(*p3),  "p3 (dest vertex) must be inside"
    assert not leg.contains_point(9.0, 105.0), "far point must be outside"
    print("  All offline assertions passed.")

    # --- Live smoke test (network) ---
    if "--live" in sys.argv:
        test_origin: LatLng = (10.4549, 106.3416)   # Mỹ Tho approximate
        test_dest: LatLng   = (10.3634, 106.3700)   # Gò Công approximate

        print("\n=== Live smoke test: build_leg_corridor ===")
        live_leg = build_leg_corridor(leg_idx=0, origin=test_origin, dest=test_dest)
        print(f"Routes collected : {len(live_leg.routes)}")
        print(f"Min distance     : {live_leg.min_distance_m / 1000:.2f} km")
        print(f"Max distance     : {live_leg.max_distance_m / 1000:.2f} km")
        # Origin is always a dense-polyline vertex → inside
        assert live_leg.contains_point(*test_origin), "origin must be inside live corridor"
        assert not live_leg.contains_point(9.0, 105.0), "far point must be outside live corridor"
        print("  Live assertions passed.")
