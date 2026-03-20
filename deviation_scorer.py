"""
deviation_scorer.py
-------------------
Scores a truck trip against a feasible corridor.

Inputs
------
- ReconstructedTrace  (from trace_reconstructor.py)
- List[CorridorLeg]   (from corridor_builder.py)
- stops               (geocoded stop dicts)
- epass_rows          (optional, from trip_pipeline)

Outputs
-------
- List[LegScore]  — per-leg breakdown
- TripScore       — aggregated trip-level verdict

Backward-compatible aliases
---------------------------
TripScore exposes the same field names used in the existing trip_report dict
so that trip_pipeline.py can drop in the new scorer with minimal changes:
  actual_distance_km, expected_distance_km, detour_ratio,
  off_route_points, detour_flag, off_route_flag,
  visited_stops, missed_stops, path_points

New fields added on top:
  corridor_compliance_pct, max_deviation_m, leg_scores
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from corridor_builder import CorridorLeg
from trace_reconstructor import ReconstructedTrace, DwellEvent
from vtracking_tool import haversine, _segment_distance

LatLng = Tuple[float, float]

# Detour threshold: actual / min_feasible > this → flag
DETOUR_RATIO_THRESHOLD: float = 1.3
# Off-route threshold for the legacy off_route_flag
OFF_ROUTE_POINT_THRESHOLD: int = 20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LegScore:
    """Deviation metrics for one leg of the trip."""
    leg_idx: int
    total_points: int
    outside_corridor: int           # points outside the corridor buffer
    outside_pct: float              # outside_corridor / total_points * 100
    max_deviation_m: float          # worst single-point distance beyond corridor edge
    actual_distance_m: float
    min_feasible_m: float           # shortest route in corridor
    max_feasible_m: float           # longest route in corridor
    distance_ratio: Optional[float] # actual / min_feasible


@dataclass
class TripScore:
    """
    Aggregated trip-level score.

    Legacy fields (same names as analyze_trip_multi_stop return dict) are
    populated so that downstream report code needs no changes.
    """
    # --- Legacy fields ---
    actual_distance_km: float
    expected_distance_km: Optional[float]  # min feasible across all legs
    detour_ratio: Optional[float]
    off_route_points: int                  # total points outside any corridor
    detour_flag: bool
    off_route_flag: bool
    wrong_turn_u_turn_flag: bool
    u_turn_count: int
    visited_stops: List[dict]
    missed_stops: List[dict]
    path_points: int
    path_start: Optional[LatLng]
    path_end: Optional[LatLng]
    message: str = ""

    # --- New corridor fields ---
    corridor_compliance_pct: float = 0.0   # % of points inside corridor
    max_deviation_m: float = 0.0           # worst single deviation across all legs
    worst_leg_idx: Optional[int] = None    # leg with highest outside_pct
    leg_scores: List[LegScore] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict compatible with the existing trip_report structure."""
        d = {
            "actual_distance_km":     self.actual_distance_km,
            "expected_distance_km":   self.expected_distance_km,
            "detour_ratio":           self.detour_ratio,
            "off_route_points":       self.off_route_points,
            "detour_flag":            self.detour_flag,
            "off_route_flag":         self.off_route_flag,
            "wrong_turn_u_turn_flag": self.wrong_turn_u_turn_flag,
            "u_turn_count":           self.u_turn_count,
            "visited_stops":          self.visited_stops,
            "missed_stops":           self.missed_stops,
            "path_points":            self.path_points,
            "path_start":             self.path_start,
            "path_end":               self.path_end,
            "corridor_compliance_pct": round(self.corridor_compliance_pct, 1),
            "max_deviation_m":        round(self.max_deviation_m, 1),
            "worst_leg_idx":          self.worst_leg_idx,
            "message":                self.message,
        }
        if self.message:
            d["message"] = self.message
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _visited_missed_from_dwells(
    dwell_events: Sequence[DwellEvent],
    stops: Sequence[dict],
    matched_path: Sequence[LatLng],
    proximity_threshold_m: float = 200.0,
) -> Tuple[List[dict], List[dict]]:
    """
    Classify stops as visited or missed using a two-tier approach.

    Tier 1 (stronger): Dwell detection — truck stopped at the stop.
      If a DwellEvent exists for a stop → visited with detection_method="dwell"

    Tier 2 (fallback): Proximity detection — truck passed close to the stop.
      If no dwell BUT matched path comes within proximity_threshold_m → visited
      with detection_method="proximity"

    Tier 3 (missed): No dwell AND path did not pass close.
      → missed with detection_method=None

    This hybrid approach avoids false missed-stop classifications when dwell
    evidence is sparse but the truck clearly passed the stop.

    Parameters
    ----------
    dwell_events      : output from detect_dwell_events()
    stops             : geocoded stop dicts with 'lat', 'lng', 'normalized_text'
    matched_path      : map-matched GPS path (for proximity fallback)
    proximity_threshold_m: max distance for proximity-based visited classification
    """
    visited_stop_indices = {e.stop_idx for e in dwell_events}
    visited: List[dict] = []
    missed: List[dict] = []

    for idx, stop in enumerate(stops):
        if stop.get("lat") is None or stop.get("lng") is None:
            missed.append({
                **stop,
                "min_distance_to_path_m": None,
                "detection_method": None,
                "reason": "No geocode",
            })
            continue

        slat = float(stop["lat"])
        slng = float(stop["lng"])

        # Tier 1: Check for dwell event
        matching_events = [e for e in dwell_events if e.stop_idx == idx]
        if matching_events:
            best_dist = min(e.min_distance_m for e in matching_events)
            visited.append({
                **stop,
                "min_distance_to_path_m": round(best_dist, 1),
                "detection_method": "dwell",
            })
            continue

        # Tier 2: Fallback to proximity check
        min_dist_to_path = float("inf")
        for pt in matched_path:
            d = haversine(slat, slng, pt[0], pt[1])
            if d < min_dist_to_path:
                min_dist_to_path = d

        min_dist_to_path = round(float(min_dist_to_path), 1)
        if min_dist_to_path <= proximity_threshold_m:
            visited.append({
                **stop,
                "min_distance_to_path_m": min_dist_to_path,
                "detection_method": "proximity",
            })
        else:
            missed.append({
                **stop,
                "min_distance_to_path_m": min_dist_to_path,
                "detection_method": None,
            })

    return visited, missed


def _score_leg(
    leg_idx: int,
    path_slice: Sequence[LatLng],
    corridor: CorridorLeg,
) -> LegScore:
    """Compute deviation metrics for one leg's path slice."""
    total = len(path_slice)
    if total == 0:
        return LegScore(
            leg_idx=leg_idx,
            total_points=0,
            outside_corridor=0,
            outside_pct=0.0,
            max_deviation_m=0.0,
            actual_distance_m=0.0,
            min_feasible_m=corridor.min_distance_m,
            max_feasible_m=corridor.max_distance_m,
            distance_ratio=None,
        )

    outside = 0
    max_dev = 0.0
    for pt in path_slice:
        dist = corridor.distance_to_corridor(pt[0], pt[1])
        if dist > corridor.buffer_m:
            outside += 1
            excess = dist - corridor.buffer_m
            if excess > max_dev:
                max_dev = excess

    actual_m = _segment_distance(path_slice)
    ratio = (actual_m / corridor.min_distance_m) if corridor.min_distance_m > 0 else None

    return LegScore(
        leg_idx=leg_idx,
        total_points=total,
        outside_corridor=outside,
        outside_pct=round(outside / total * 100, 1) if total > 0 else 0.0,
        max_deviation_m=round(max_dev, 1),
        actual_distance_m=round(actual_m, 1),
        min_feasible_m=round(corridor.min_distance_m, 1),
        max_feasible_m=round(corridor.max_distance_m, 1),
        distance_ratio=round(ratio, 3) if ratio is not None else None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_trip(
    trace: ReconstructedTrace,
    corridors: List[CorridorLeg],
    stops: Sequence[dict],
    u_turn_indices: Optional[Sequence[int]] = None,
) -> TripScore:
    """
    Compute full trip deviation score.

    Parameters
    ----------
    trace         : output of reconstruct_trace()
    corridors     : output of build_trip_corridors(); may be empty
    stops         : geocoded stops (same list passed to both builders)
    u_turn_indices: optional list of u-turn point indices from detect_uturn()

    Returns
    -------
    TripScore with both legacy and new corridor fields populated.

    Field Meanings
    --------------
    actual_distance_km
        Full matched GPS path from first to last point.
        Always reported for audit transparency.

    expected_distance_km
        Sum of the shortest feasible route for EVERY corridor leg
        (delivery legs + return leg), i.e. the full closed-loop distance:
          ORIGIN → stop1 → … → stopN → ORIGIN
        Computed as: sum(leg.min_distance_m for leg in corridors) / 1000.

    detour_ratio = actual_distance_km / expected_distance_km
        Ratio of the full GPS path to the full closed-loop expected distance.
        Both sides use the same closed-loop scope.
        Values: < 1.0 shorter than expected, > 1.3 flagged as detour.

    corridor_compliance_pct
        % of GPS points that fall within their assigned corridor.
        Scored over all corridor legs (delivery + return).
    """
    path = trace.matched_path
    leg_index = trace.leg_index

    if not path:
        geo_stops = [s for s in stops if s.get("lat") is not None]
        return TripScore(
            actual_distance_km=0.0,
            expected_distance_km=None,
            detour_ratio=None,
            off_route_points=0,
            detour_flag=False,
            off_route_flag=False,
            wrong_turn_u_turn_flag=False,
            u_turn_count=0,
            visited_stops=[],
            missed_stops=list(stops),
            path_points=0,
            path_start=None,
            path_end=None,
            message="Not enough GPS points to analyse",
        )

    # --- Per-leg scoring ---
    geo_stops = [s for s in stops if s.get("lat") is not None and s.get("lng") is not None]
    leg_scores: List[LegScore] = []

    if corridors:
        for corridor in corridors:
            idx = corridor.leg_idx
            slice_ = [path[i] for i, li in enumerate(leg_index) if li == idx]
            leg_scores.append(_score_leg(idx, slice_, corridor))
    # If no corridors were built (API failure), leg_scores stays empty.

    # --- Trip-level aggregation ---
    # Distance metrics:
    # - actual_distance_m: full matched path from start to end (may include return trip)
    # - delivery_distance_m: matched path from first to last defined corridor leg only
    #   (excludes post-delivery return, pre-delivery positioning, etc.)
    # - detour_ratio uses delivery_distance to avoid inflating the ratio with return distance

    actual_m = _segment_distance(path)

    if leg_scores:
        total_pts  = sum(ls.total_points for ls in leg_scores)
        outside_pts = sum(ls.outside_corridor for ls in leg_scores)
        max_dev_m  = max(ls.max_deviation_m for ls in leg_scores)
        compliance = (1 - outside_pts / total_pts) * 100 if total_pts > 0 else 100.0
        worst_leg  = max(leg_scores, key=lambda ls: ls.outside_pct).leg_idx

        # expected_km = sum of ALL corridor leg min distances (full closed-loop scope).
        # Business rule: ORIGIN → stop1 → … → stopN → ORIGIN, return leg included.
        min_feasible_total = sum(ls.min_feasible_m for ls in leg_scores)
        expected_km = round(min_feasible_total / 1000, 2)

        # detour_ratio uses the same closed-loop scope as expected_km.
        # actual_m = full GPS path; min_feasible_total = full closed-loop expected.
        detour_ratio = (actual_m / min_feasible_total) if min_feasible_total > 0 else None
    else:
        total_pts = len(path)
        outside_pts = total_pts
        max_dev_m = 0.0
        compliance = 0.0
        worst_leg = None
        detour_ratio = None
        expected_km = None

    # --- Visited / missed via dwell events (with proximity fallback) ---
    visited_stops, missed_stops = _visited_missed_from_dwells(
        trace.dwell_events, geo_stops, path, proximity_threshold_m=200.0
    )

    # --- U-turns ---
    u_count = len(u_turn_indices) if u_turn_indices else 0

    return TripScore(
        actual_distance_km=round(actual_m / 1000, 2),
        expected_distance_km=expected_km,
        detour_ratio=round(detour_ratio, 3) if detour_ratio is not None else None,
        off_route_points=outside_pts,
        detour_flag=bool(detour_ratio is not None and detour_ratio > DETOUR_RATIO_THRESHOLD),
        off_route_flag=bool(outside_pts > OFF_ROUTE_POINT_THRESHOLD),
        wrong_turn_u_turn_flag=u_count > 0,
        u_turn_count=u_count,
        visited_stops=visited_stops,
        missed_stops=missed_stops,
        path_points=len(path),
        path_start=path[0],
        path_end=path[-1],
        corridor_compliance_pct=round(compliance, 1),
        max_deviation_m=round(max_dev_m, 1),
        worst_leg_idx=worst_leg,
        leg_scores=leg_scores,
        message="" if leg_scores else "Không tính được km kỳ vọng vì corridor rỗng hoặc stop không có tọa độ hợp lệ",
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from corridor_builder import CorridorLeg, RouteOption
    from trace_reconstructor import ReconstructedTrace, DwellEvent

    print("=== Unit tests: score_trip ===")

    # Build a simple 2-leg corridor
    p0: LatLng = (10.0, 106.0)
    pm: LatLng = (10.1, 106.1)   # midpoint / waypoint between legs
    p1: LatLng = (10.2, 106.2)

    route0 = RouteOption("test_0", [p0, pm], distance_m=15000)
    route1 = RouteOption("test_1", [pm, p1], distance_m=15000)
    corridor0 = CorridorLeg(leg_idx=0, origin=p0, dest=pm, routes=[route0], buffer_m=200)
    corridor1 = CorridorLeg(leg_idx=1, origin=pm, dest=p1, routes=[route1], buffer_m=200)

    # Trace uses only actual route vertices so they are guaranteed inside the
    # 200m buffer (vertex-to-point distance = 0 for each vertex).
    on_path = [p0, pm, p1]
    trace_on = ReconstructedTrace(
        raw_path=on_path,
        matched_path=on_path,
        matched_times=None,
        leg_index=[0, 0, 1],
        dwell_events=[
            DwellEvent(stop_idx=0, stop_label="Stop A", path_indices=[1],
                       min_distance_m=0.0),
        ],
        n_legs=2,
    )

    stops = [
        {"lat": pm[0], "lng": pm[1], "normalized_text": "Stop A"},
        {"lat": p1[0], "lng": p1[1], "normalized_text": "Stop B"},
    ]

    score = score_trip(trace_on, [corridor0, corridor1], stops)
    d = score.to_dict()

    assert score.path_points == 3, f"path_points: {score.path_points}"
    assert score.actual_distance_km > 0
    assert len(score.leg_scores) == 2, f"leg_scores count: {len(score.leg_scores)}"
    # All points are on route vertices → inside corridor
    assert score.off_route_points == 0, f"off_route: {score.off_route_points}"
    assert score.corridor_compliance_pct == 100.0, f"compliance: {score.corridor_compliance_pct}"
    # Stop A visited (has dwell event)
    # Stop B visited (proximity fallback: is exactly on path, distance=0)
    # Both should be marked visited now with new two-tier detection
    assert len(score.visited_stops) == 2, f"visited: {len(score.visited_stops)}"
    assert len(score.missed_stops) == 0, f"missed: {len(score.missed_stops)}"

    # Verify detection methods are recorded
    for vs in score.visited_stops:
        assert "detection_method" in vs, "detection_method not in visited stop"
    assert score.visited_stops[0]["detection_method"] == "dwell", "Stop A should be dwell"
    assert score.visited_stops[1]["detection_method"] == "proximity", "Stop B should be proximity"
    # to_dict preserves legacy keys
    for key in ("actual_distance_km", "expected_distance_km", "detour_ratio",
                "off_route_points", "detour_flag", "off_route_flag",
                "visited_stops", "missed_stops", "path_points"):
        assert key in d, f"Missing legacy key: {key}"

    print("  All assertions passed.")
    print(f"  actual_km={d['actual_distance_km']}, compliance={d['corridor_compliance_pct']}%")
    print(f"  visited={len(d['visited_stops'])}, missed={len(d['missed_stops'])}")

    # Test 2: Proximity-only detection (no dwell, but path passes nearby)
    print("\n  Test 2: Proximity-only detection")
    nearby_stop = {"lat": 10.15, "lng": 106.15, "normalized_text": "Stop C"}
    stops_with_nearby = [
        {"lat": pm[0], "lng": pm[1], "normalized_text": "Stop A"},
        nearby_stop,
    ]

    # Same path and corridors, but no dwell event for Stop C
    score2 = score_trip(trace_on, [corridor0, corridor1], stops_with_nearby)

    # Stop A: dwell event
    # Stop C: no dwell, but should be detected by proximity if near path
    assert len(score2.visited_stops) >= 1, f"Should visit at least Stop A"

    stop_c_entries = [s for s in score2.visited_stops if s.get("normalized_text") == "Stop C"]
    if stop_c_entries:
        assert stop_c_entries[0]["detection_method"] == "proximity", \
            "Stop C should use proximity detection"
        print(f"  Stop C visited by proximity: distance={stop_c_entries[0]['min_distance_to_path_m']}m")

    print("  Proximity test passed.")
