from __future__ import annotations

import math
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polyline
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import maps_config

OSRM_ROUTE = "https://router.project-osrm.org/route/v1/driving/"
OSRM_MATCH = "https://router.project-osrm.org/match/v1/driving/"
GOOGLE_ROUTE = "https://maps.googleapis.com/maps/api/directions/json"
GOOGLE_MAPS_API_KEY_ENV = "GOOGLE_MAPS_API_KEY"


LatLng = Tuple[float, float]


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        read=4,
        connect=4,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "truck-route-monitor/2.0"})
    return session


SESSION = build_session()


def parse_time_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_coord(coord_str: str) -> LatLng:
    lat, lon = str(coord_str).split(",")
    return float(lat), float(lon)


def thin_gps_points(
    points: Sequence[LatLng],
    timestamps: Optional[Sequence[Any]] = None,
    min_move_m: float = 25,
    max_points: int = 120,
) -> Tuple[List[LatLng], Optional[List[Any]]]:
    if not points:
        return [], [] if timestamps is not None else None

    kept_points: List[LatLng] = [points[0]]
    kept_ts: Optional[List[Any]] = [timestamps[0]] if timestamps is not None else None

    last = points[0]
    for i in range(1, len(points) - 1):
        p = points[i]
        d = haversine(last[0], last[1], p[0], p[1])
        if d >= min_move_m:
            kept_points.append(p)
            if kept_ts is not None:
                kept_ts.append(timestamps[i])
            last = p

    if len(points) > 1:
        kept_points.append(points[-1])
        if kept_ts is not None:
            kept_ts.append(timestamps[-1])

    if len(kept_points) > max_points:
        idx = np.linspace(0, len(kept_points) - 1, max_points).astype(int)
        kept_points = [kept_points[i] for i in idx]
        if kept_ts is not None:
            kept_ts = [kept_ts[i] for i in idx]

    return kept_points, kept_ts


def get_route(origin: LatLng, dest: LatLng) -> Tuple[List[LatLng], float]:
    api_key = maps_config.get_api_key()
    if not api_key:
        raise RuntimeError(f"Thiếu biến môi trường {GOOGLE_MAPS_API_KEY_ENV}")

    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{dest[0]},{dest[1]}",
        "mode": "driving",
        "key": api_key,
    }
    r = requests.get(GOOGLE_ROUTE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if data.get("status") != "OK":
        raise RuntimeError(f"Google Directions lỗi: {data.get('status')} | {data.get('error_message')}")
    if not data.get("routes"):
        raise RuntimeError("Google Directions không trả route")

    route = data["routes"][0]
    start_addr = route["legs"][0]["start_address"]
    end_addr = route["legs"][-1]["end_address"]
    poly = route["overview_polyline"]["points"]
    coords = polyline.decode(poly)
    expected_m = float(sum(leg["distance"]["value"] for leg in route["legs"]))
    expected_km = expected_m / 1000

    print("\n===== ROUTE DEBUG =====")
    print(f"Origin coord : {origin}")
    print(f"Origin name  : {start_addr}")

    print(f"Dest coord   : {dest}")
    print(f"Dest name    : {end_addr}")

    print(f"Expected km  : {expected_km:.2f} km")
    print("======================\n")
    return coords, expected_m


def map_match(
    points: Sequence[LatLng],
    timestamps: Optional[Sequence[Any]] = None,
    chunk_size: int = 80,
    radius_m: int = 30,
) -> List[LatLng]:
    if len(points) < 2:
        return list(points)

    if timestamps is None:
        base = int(time.time())
        timestamps = [base + i for i in range(len(points))]

    fixed_ts: List[int] = []
    last_ts: Optional[int] = None
    for t in timestamps:
        if pd.isna(t):
            t = None
        if isinstance(t, pd.Timestamp):
            t = int(t.timestamp())
        elif isinstance(t, datetime):
            t = int(t.timestamp())
        elif t is None:
            t = int(time.time()) if last_ts is None else last_ts + 1
        else:
            t = int(t)
        if last_ts is not None and t <= last_ts:
            t = last_ts + 1
        fixed_ts.append(t)
        last_ts = t

    all_paths: List[LatLng] = []
    start = 0
    overlap = 5
    while start < len(points):
        end = min(start + chunk_size, len(points))
        chunk_points = list(points[start:end])
        chunk_ts = fixed_ts[start:end]
        if len(chunk_points) < 2:
            break

        encoded = polyline.encode(chunk_points, precision=5)
        coord_part = f"polyline({encoded})"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "timestamps": ";".join(map(str, chunk_ts)),
            "radiuses": ";".join([str(radius_m)] * len(chunk_points)),
            "gaps": "split",
            "tidy": "true",
        }
        url = f"{OSRM_MATCH}{coord_part}"
        try:
            r = SESSION.get(url, params=params, timeout=(10, 60))
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "Ok" or not data.get("matchings"):
                raise RuntimeError(f"OSRM match lỗi: {data}")

            chunk_path: List[LatLng] = []
            for matching in data["matchings"]:
                coords = matching["geometry"]["coordinates"]
                chunk_path.extend([(c[1], c[0]) for c in coords])

            if all_paths and chunk_path and haversine(all_paths[-1][0], all_paths[-1][1], chunk_path[0][0], chunk_path[0][1]) < 10:
                chunk_path = chunk_path[1:]
            all_paths.extend(chunk_path)
        except Exception as e:
            # print(f"[WARN] map_match chunk {start}:{end} lỗi -> fallback raw GPS | {e}")
            raw_path = chunk_points[:]
            if all_paths and raw_path and haversine(all_paths[-1][0], all_paths[-1][1], raw_path[0][0], raw_path[0][1]) < 10:
                raw_path = raw_path[1:]
            all_paths.extend(raw_path)

        if end == len(points):
            break
        start = end - overlap
    return all_paths


def distance_to_route(point: LatLng, route: Sequence[LatLng]) -> float:
    lat, lon = point
    dmin = float("inf")
    for r in route:
        d = haversine(lat, lon, r[0], r[1])
        if d < dmin:
            dmin = d
    return dmin


def detect_uturn(points: Sequence[LatLng]) -> List[int]:
    headings: List[float] = []
    for i in range(len(points) - 1):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i + 1]
        y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
        brng = math.degrees(math.atan2(y, x))
        headings.append(brng)

    turns: List[int] = []
    for i in range(len(headings) - 1):
        d = abs(headings[i + 1] - headings[i])
        if d > 150:
            turns.append(i)
    return turns


def min_distance_point_to_path(stop: Dict[str, Any], path: Sequence[LatLng]) -> float:
    lat = float(stop["lat"])
    lng = float(stop["lng"])
    dmin = float("inf")
    for p in path:
        d = haversine(lat, lng, p[0], p[1])
        if d < dmin:
            dmin = d
    return dmin


def detect_visited_stops(path: Sequence[LatLng], stops: Sequence[Dict[str, Any]], threshold_m: float = 200) -> Tuple[List[dict], List[dict]]:
    visited: List[dict] = []
    missed: List[dict] = []
    for stop in stops:
        if stop.get("lat") is None or stop.get("lng") is None:
            missed.append({**stop, "min_distance_to_path_m": None, "reason": "Không có tọa độ geocode"})
            continue
        dmin = min_distance_point_to_path(stop, path)
        row = {**stop, "min_distance_to_path_m": round(float(dmin), 1)}
        if dmin <= threshold_m:
            visited.append(row)
        else:
            missed.append(row)
    return visited, missed


def _segment_distance(path: Sequence[LatLng]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        total += haversine(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
    return total


def build_expected_route_multi_stop(origin: LatLng, stops: Sequence[Dict[str, Any]]) -> Tuple[List[LatLng], float]:
    expected_route: List[LatLng] = []
    expected_m = 0.0
    current = origin

    geo_stops = [s for s in stops if s.get("lat") is not None and s.get("lng") is not None]
    if not geo_stops:
        return [], 0.0

    for idx, stop in enumerate(geo_stops):
        dest = (float(stop["lat"]), float(stop["lng"]))
        route, meters = get_route(current, dest)
        if expected_route and route and haversine(expected_route[-1][0], expected_route[-1][1], route[0][0], route[0][1]) < 10:
            route = route[1:]
        expected_route.extend(route)
        expected_m += meters
        current = dest
    return expected_route, expected_m


def analyze_trip_corridor(
    df: pd.DataFrame,
    stops: Sequence[Dict[str, Any]],
    origin: Optional[LatLng] = None,
    min_move_m: float = 25,
    max_points: int = 300,
    corridor_buffer_m: float = 200,
) -> Dict[str, Any]:
    """
    Corridor-based replacement for analyze_trip_multi_stop.

    Uses the three new modules (corridor_builder, trace_reconstructor,
    deviation_scorer) to score the trip against a multi-route feasible
    corridor rather than a single Google reference line.

    Return value has the same shape as analyze_trip_multi_stop (all legacy
    fields present) plus new corridor fields:
      corridor_compliance_pct, max_deviation_m, worst_leg_idx, leg_scores

    GPS thinning (max_points=300 by default) is tuned for full-day multi-stop
    routes to preserve stop evidence and enable reliable dwell detection.
    For short single-leg analysis, caller can pass max_points=120 or lower.

    Lazy imports are used to avoid circular dependencies: corridor_builder
    and trace_reconstructor both import from this module at the top level.
    """
    # Lazy imports — resolved at call time, not at module load.
    from corridor_builder import build_trip_corridors       # noqa: PLC0415
    from trace_reconstructor import reconstruct_trace       # noqa: PLC0415
    from deviation_scorer import score_trip                 # noqa: PLC0415

    # Extract raw GPS points and check availability.
    if "Tọa độ" not in df.columns:
        return {
            "actual_distance_km": 0.0,
            "expected_distance_km": None,
            "detour_ratio": None,
            "off_route_points": 0,
            "detour_flag": False,
            "off_route_flag": False,
            "wrong_turn_u_turn_flag": False,
            "u_turn_count": 0,
            "visited_stops": [],
            "missed_stops": list(stops),
            "path_points": 0,
            "corridor_compliance_pct": None,
            "max_deviation_m": None,
            "message": "Không đủ điểm GPS để phân tích",
        }

    raw_gps = [parse_coord(c) for c in df["Tọa độ"] if pd.notna(c)]
    if len(raw_gps) < 2:
        return {
            "actual_distance_km": 0.0,
            "expected_distance_km": None,
            "detour_ratio": None,
            "off_route_points": 0,
            "detour_flag": False,
            "off_route_flag": False,
            "wrong_turn_u_turn_flag": False,
            "u_turn_count": 0,
            "visited_stops": [],
            "missed_stops": list(stops),
            "path_points": 0,
            "corridor_compliance_pct": None,
            "max_deviation_m": None,
            "message": "Không đủ điểm GPS để phân tích",
        }

    # Determine the effective origin for both corridor and trace.
    # If origin is not provided, use the first raw GPS point.
    # This ensures both build_trip_corridors and reconstruct_trace use
    # the same origin, preventing leg index misalignment.
    if origin is None:
        effective_origin = raw_gps[0]
    else:
        effective_origin = origin
    print(f"[DEBUG] analyze_trip_corridor stops={len(stops)}")
    print(f"[DEBUG] raw_gps={len(raw_gps)}")
    geo_stops = [s for s in stops if s.get("lat") is not None and s.get("lng") is not None]
    if not geo_stops:
        return {
            "actual_distance_km": 0.0,
            "expected_distance_km": None,
            "detour_ratio": None,
            "off_route_points": 0,
            "detour_flag": False,
            "off_route_flag": False,
            "wrong_turn_u_turn_flag": False,
            "u_turn_count": 0,
            "visited_stops": [],
            "missed_stops": list(stops),
            "path_points": 0,
            "corridor_compliance_pct": None,
            "max_deviation_m": None,
            "message": "Không có stop nào geocode thành công nên không tính được km kỳ vọng",
        }
    corridors = build_trip_corridors(stops, origin=effective_origin, buffer_m=corridor_buffer_m)
    geo_stops = [s for s in stops if s.get("lat") is not None and s.get("lng") is not None]
    print(f"[DEBUG] total stops={len(stops)} | geo_stops={len(geo_stops)}")
    print(f"[DEBUG] built corridors={len(corridors)}")
    trace = reconstruct_trace(
        df,
        stops=stops,
        origin=effective_origin,
        min_move_m=min_move_m,
        max_points=max_points,
        map_match_radius_m=30,
        dwell_radius_m=corridor_buffer_m,
    )

    uturns = detect_uturn(trace.matched_path)

    trip_score = score_trip(trace, corridors, stops, u_turn_indices=uturns)
    print(
        f"[DEBUG] expected_distance_km={trip_score.expected_distance_km} "
        f"| actual_distance_km={trip_score.actual_distance_km}"
    )
    return trip_score.to_dict()


if __name__ == "__main__":
    sample_file = "62C16099_0609.xlsx"
    if os.path.exists(sample_file):
        df = pd.read_excel(sample_file)
        origin = (10.8708, 106.4252)
        dest = (10.736662, 106.670769)
    else:
        print(f"Không tìm thấy file mẫu {sample_file}")
