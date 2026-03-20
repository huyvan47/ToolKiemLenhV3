"""
trace_reconstructor.py
----------------------
Turns raw GPS breadcrumbs into a structured, leg-segmented trace.

Responsibilities
----------------
1. Thin + map-match GPS points (delegates to existing helpers in vtracking_tool).
2. Assign each matched point to a trip leg (which stop-pair it belongs to).
3. Detect dwell events — time windows where the truck was near a delivery stop.

Output
------
A ReconstructedTrace dataclass containing:
  - raw_path    : thinned GPS before OSRM (kept for audit / debugging)
  - matched_path: OSRM map-matched path (used for all scoring)
  - leg_index   : parallel list of leg assignments per matched point
  - dwell_events: list of DwellEvent (one per stop that was visited)

This module is additive — no existing file is modified.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

import google_roads_service
import maps_config
from vtracking_tool import (
    haversine,
    map_match,
    parse_coord,
    parse_time_column,
    thin_gps_points,
)

logger = logging.getLogger(__name__)

LatLng = Tuple[float, float]

# How close a matched point must be to a stop to count as "at the stop".
DWELL_RADIUS_M: float = 200.0

# Minimum consecutive dwell points to register a dwell event
# (avoids false positives from a single stray point near a stop).
DWELL_MIN_POINTS: int = 2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DwellEvent:
    """Truck was near a stop for a contiguous run of matched-path points."""
    stop_idx: int                     # index into the stops list
    stop_label: str                   # normalized_text of the stop
    path_indices: List[int]           # matched_path indices that belong to this dwell
    start_time: Optional[Any] = None  # timestamp of first dwell point
    end_time: Optional[Any] = None    # timestamp of last dwell point
    min_distance_m: float = 0.0       # closest approach to stop during dwell


@dataclass
class ReconstructedTrace:
    """
    Full structured output of the GPS reconstruction step.

    matched_path and leg_index are parallel lists of equal length.
    matched_times is parallel too when timestamps were available, else None.
    """
    raw_path: List[LatLng]
    matched_path: List[LatLng]
    matched_times: Optional[List[Any]]   # pd.Timestamp or None per point
    leg_index: List[int]                 # which leg each matched point belongs to
    dwell_events: List[DwellEvent]
    n_legs: int                          # total number of legs (= len(stops) or +1 if origin given)


# ---------------------------------------------------------------------------
# Leg assignment
# ---------------------------------------------------------------------------

def assign_legs(
    path: Sequence[LatLng],
    waypoints: Sequence[LatLng],
) -> List[int]:
    """
    Assign each point in `path` to a leg index in [0, len(waypoints)-2].

    Strategy: walk the path in order; switch from leg k to leg k+1 once
    the path point is closer to waypoints[k+1] than to waypoints[k], AND
    we have passed the midpoint distance between them.

    This greedy approach handles out-of-order GPS without backtracking.

    Parameters
    ----------
    path      : sequence of (lat, lng) matched points
    waypoints : ordered list of (lat, lng) trip waypoints
                (first item is origin, remaining are stop coords)

    Returns
    -------
    List of integer leg indices, same length as path.
    All zeros if fewer than 2 waypoints.
    """
    if len(waypoints) < 2:
        return [0] * len(path)

    n_legs = len(waypoints) - 1
    assignments: List[int] = []
    current_leg = 0

    for pt in path:
        # Try to advance the leg pointer as far as possible from the current
        # position, but only commit to leg k+1 when we're clearly past the
        # midpoint between waypoints[k] and waypoints[k+1].
        while current_leg < n_legs - 1:
            wp_cur  = waypoints[current_leg]
            wp_next = waypoints[current_leg + 1]
            d_cur   = haversine(pt[0], pt[1], wp_cur[0],  wp_cur[1])
            d_next  = haversine(pt[0], pt[1], wp_next[0], wp_next[1])
            if d_next < d_cur:
                current_leg += 1
            else:
                break
        assignments.append(current_leg)

    return assignments


# ---------------------------------------------------------------------------
# Dwell detection
# ---------------------------------------------------------------------------

def detect_dwell_events(
    path: Sequence[LatLng],
    times: Optional[Sequence[Any]],
    stops: Sequence[dict],
    radius_m: float = DWELL_RADIUS_M,
    min_points: int = DWELL_MIN_POINTS,
) -> List[DwellEvent]:
    """
    Find contiguous runs of path points that lie within `radius_m` of a stop.

    Parameters
    ----------
    path      : matched GPS path
    times     : parallel timestamps (pd.Timestamp or None); None = no time info
    stops     : geocoded stop dicts with 'lat', 'lng', 'normalized_text'
    radius_m  : proximity threshold
    min_points: minimum run length to register an event

    Returns
    -------
    List of DwellEvent, one per (stop, contiguous run) pair.
    """
    events: List[DwellEvent] = []

    for stop_idx, stop in enumerate(stops):
        if stop.get("lat") is None or stop.get("lng") is None:
            continue
        slat = float(stop["lat"])
        slng = float(stop["lng"])
        label = str(stop.get("normalized_text") or f"stop_{stop_idx}")

        # Build boolean mask: is path[i] within radius of this stop?
        near = [haversine(p[0], p[1], slat, slng) <= radius_m for p in path]

        # Find contiguous True runs
        i = 0
        while i < len(near):
            if not near[i]:
                i += 1
                continue
            # Start of a run
            run_start = i
            while i < len(near) and near[i]:
                i += 1
            run_end = i  # exclusive

            if (run_end - run_start) < min_points:
                continue

            run_indices = list(range(run_start, run_end))
            run_points  = [path[j] for j in run_indices]
            dists = [haversine(p[0], p[1], slat, slng) for p in run_points]

            t_start = times[run_start] if times is not None else None
            t_end   = times[run_end - 1] if times is not None else None

            events.append(DwellEvent(
                stop_idx=stop_idx,
                stop_label=label,
                path_indices=run_indices,
                start_time=t_start,
                end_time=t_end,
                min_distance_m=round(min(dists), 1),
            ))

    return events


# ---------------------------------------------------------------------------
# Map-match dispatch (Google Roads → OSRM fallback)
# ---------------------------------------------------------------------------

def _do_map_match(
    raw_path: List[LatLng],
    raw_times: Optional[List[Any]],
    chunk_size: int,
    radius_m: int,
) -> List[LatLng]:
    """
    Dispatch map-matching to Google Roads API or OSRM based on
    MAP_MATCH_PROVIDER env var.

    Precedence
    ----------
    1. Google Roads (snap_path_to_roads) — when MAP_MATCH_PROVIDER=google_roads
       AND GOOGLE_MAPS_API_KEY is set.
    2. OSRM map_match — default, and automatic fallback when Google Roads is
       unavailable or returns an error.

    Output format is identical in both cases: List[LatLng].
    Callers must not assume output length equals input length.
    """
    if maps_config.get_map_match_provider() == "google_roads":
        api_key = maps_config.get_api_key()
        if api_key:
            try:
                result = google_roads_service.snap_path_to_roads(
                    raw_path,
                    api_key=api_key,
                    interpolate=True,
                    # Roads API hard limit is 100; honour caller's chunk_size
                    # if it is smaller, otherwise cap at 100.
                    chunk_size=min(chunk_size, google_roads_service.ROADS_MAX_POINTS),
                )
                logger.info(
                    "Google Roads map-match: %d raw → %d snapped points",
                    len(raw_path), len(result),
                )
                return result
            except google_roads_service.GoogleRoadsError as exc:
                logger.warning(
                    "Google Roads map-match failed (%s) — falling back to OSRM", exc
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Google Roads map-match unexpected error (%s) — falling back to OSRM",
                    exc,
                )
        else:
            logger.warning(
                "MAP_MATCH_PROVIDER=google_roads but GOOGLE_MAPS_API_KEY not set "
                "— falling back to OSRM"
            )

    # OSRM path (default or fallback).
    return map_match(raw_path, timestamps=raw_times, chunk_size=chunk_size, radius_m=radius_m)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconstruct_trace(
    df: pd.DataFrame,
    stops: Sequence[dict],
    origin: Optional[LatLng] = None,
    min_move_m: float = 25.0,
    max_points: int = 300,
    map_match_radius_m: int = 30,
    dwell_radius_m: float = DWELL_RADIUS_M,
) -> ReconstructedTrace:
    """
    Full pipeline: raw GPS → thinned → map-matched → leg-assigned + dwells.

    Parameters
    ----------
    df               : raw GPS DataFrame (must have 'Tọa độ' column;
                       optionally 'Thời gian')
    stops            : geocoded stops with 'lat', 'lng', 'normalized_text'
    origin           : optional explicit start coordinate
    min_move_m       : thin_gps_points minimum movement threshold
    max_points       : thin_gps_points maximum point count (default: 300)
                       For full-day trips (8–24 hours), preserve more points
                       to reliably detect short dwell periods at stops.
                       Typical: 300 pts → 900–1200 matched pts after OSRM.
    map_match_radius_m: OSRM map-match snap radius
    dwell_radius_m   : proximity radius for dwell detection

    Returns
    -------
    ReconstructedTrace
    """
    # 1. Parse raw GPS
    gps_points: List[LatLng] = [
        parse_coord(c) for c in df["Tọa độ"] if pd.notna(c)
    ]

    time_col = "Thời gian" if "Thời gian" in df.columns else None
    gps_times = parse_time_column(df[time_col]).tolist() if time_col else None
    if gps_times is not None:
        gps_times = gps_times[: len(gps_points)]

    # 2. Thin
    raw_path, raw_times = thin_gps_points(
        gps_points,
        timestamps=gps_times,
        min_move_m=min_move_m,
        max_points=max_points,
    )

    if len(raw_path) < 2:
        return ReconstructedTrace(
            raw_path=raw_path,
            matched_path=raw_path,
            matched_times=raw_times,
            leg_index=[0] * len(raw_path),
            dwell_events=[],
            n_legs=0,
        )

    # 3. Map-match (Google Roads or OSRM depending on MAP_MATCH_PROVIDER).
    matched_path = _do_map_match(raw_path, raw_times, chunk_size=80, radius_m=map_match_radius_m)
    # map_match does not return timestamps; keep raw_times aligned by index
    # (lengths may differ after OSRM interpolation — store as None in that case)
    matched_times: Optional[List[Any]]
    if raw_times is not None and len(matched_path) == len(raw_path):
        matched_times = raw_times
    else:
        matched_times = None

    # 4. Build ordered waypoints for leg assignment
    geo_stops = [s for s in stops if s.get("lat") is not None and s.get("lng") is not None]
    waypoints: List[LatLng] = []
    if origin is not None:
        waypoints.append(origin)
    elif matched_path:
        waypoints.append(matched_path[0])  # use first matched point as origin
    waypoints.extend((float(s["lat"]), float(s["lng"])) for s in geo_stops)

    # Always close the delivery loop by appending the depot as the final waypoint.
    # Mirrors corridor_builder.FIXED_ORIGIN which is always the trip start and end.
    # This ensures GPS points from the return trip are assigned to the return leg
    # (leg N), not crowded onto the last delivery leg (N-1).
    _depot = maps_config.get_depot_origin()
    if _depot is not None and geo_stops:
        waypoints.append(_depot)

    # 5. Assign legs
    leg_index = assign_legs(matched_path, waypoints)
    n_legs = max(leg_index) + 1 if leg_index else 0

    # 6. Detect dwell events
    dwell_events = detect_dwell_events(
        matched_path,
        matched_times,
        geo_stops,
        radius_m=dwell_radius_m,
    )

    return ReconstructedTrace(
        raw_path=raw_path,
        matched_path=matched_path,
        matched_times=matched_times,
        leg_index=leg_index,
        dwell_events=dwell_events,
        n_legs=n_legs,
    )


# ---------------------------------------------------------------------------
# Unit tests / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Unit tests: assign_legs ===")

    # 3 waypoints → 2 legs
    wp = [(10.0, 106.0), (10.1, 106.1), (10.2, 106.2)]

    # Points clearly on leg 0 (near wp[0])
    p0a: LatLng = (10.01, 106.01)
    p0b: LatLng = (10.04, 106.04)
    # Points clearly on leg 1 (past wp[1], heading toward wp[2])
    p1a: LatLng = (10.11, 106.11)
    p1b: LatLng = (10.18, 106.18)

    path = [p0a, p0b, p1a, p1b]
    legs = assign_legs(path, wp)
    assert legs[0] == 0, f"Expected leg 0, got {legs[0]}"
    assert legs[1] == 0, f"Expected leg 0, got {legs[1]}"
    assert legs[2] == 1, f"Expected leg 1, got {legs[2]}"
    assert legs[3] == 1, f"Expected leg 1, got {legs[3]}"
    print(f"  assign_legs: {legs}  OK")

    # Edge: single waypoint pair
    legs_single = assign_legs([(10.05, 106.05)], [(10.0, 106.0), (10.1, 106.1)])
    assert legs_single == [0], f"Expected [0], got {legs_single}"
    print(f"  single-leg: {legs_single}  OK")

    print("\n=== Unit tests: detect_dwell_events ===")

    stops = [
        {"lat": 10.1, "lng": 106.1, "normalized_text": "Stop A"},
        {"lat": 10.5, "lng": 106.5, "normalized_text": "Stop B"},
    ]
    # Path that passes near Stop A (indices 1,2,3) but not near Stop B
    path_dwell = [
        (10.0,  106.0),   # far from both
        (10.1,  106.1),   # at Stop A
        (10.1,  106.101), # still near Stop A
        (10.1,  106.102), # still near Stop A
        (10.3,  106.3),   # far again
    ]
    events = detect_dwell_events(path_dwell, None, stops, radius_m=500, min_points=2)
    assert len(events) == 1, f"Expected 1 dwell event, got {len(events)}"
    assert events[0].stop_idx == 0
    assert events[0].stop_label == "Stop A"
    assert len(events[0].path_indices) == 3
    print(f"  dwell events: {[(e.stop_label, e.path_indices) for e in events]}  OK")

    print("\nAll unit tests passed.")
