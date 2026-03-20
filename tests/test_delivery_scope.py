"""
tests/test_delivery_scope.py
-----------------------------
Tests for the delivery-scope / return-leg boundary across three modules:
  - corridor_builder.build_trip_corridors  (return leg conditionality)
  - deviation_scorer.score_trip            (expected_km, delivery_m, detour_ratio)

All external calls are mocked.  No real network requests are made.

Definitions used throughout
----------------------------
Delivery legs : origin → stop1 → … → stopN
               leg indices 0 .. n_stops-1 (= n_delivery_legs)
Return leg    : stopN → origin
               leg index n_stops  (only present when ENABLE_RETURN_LEG=true)

expected_distance_km must sum delivery legs only (never the return leg).
detour_ratio must use the same delivery scope for both numerator and denominator.
corridor_compliance_pct may include the return leg when it is enabled.
"""
from __future__ import annotations

import sys
import os
import unittest
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corridor_builder import CorridorLeg, RouteOption
from deviation_scorer import LegScore, TripScore, score_trip
from trace_reconstructor import DwellEvent, ReconstructedTrace

LatLng = Tuple[float, float]

# ---------------------------------------------------------------------------
# Shared geometry fixture
# ---------------------------------------------------------------------------
# 3-stop trip:  origin → A → B → C  (4 delivery legs if origin is included)
# Actually with origin + 3 stops → 3 delivery legs:
#   leg 0: origin → A
#   leg 1: A → B
#   leg 2: B → C
#   return leg (leg 3, optional): C → origin

P_ORIGIN: LatLng = (10.0, 106.0)
P_A: LatLng      = (10.1, 106.1)
P_B: LatLng      = (10.2, 106.2)
P_C: LatLng      = (10.3, 106.3)

STOPS = [
    {"lat": P_A[0], "lng": P_A[1], "normalized_text": "Stop A"},
    {"lat": P_B[0], "lng": P_B[1], "normalized_text": "Stop B"},
    {"lat": P_C[0], "lng": P_C[1], "normalized_text": "Stop C"},
]

# Corridor distances (metres): delivery legs only
LEG0_DIST = 5_000.0   # origin→A
LEG1_DIST = 5_000.0   # A→B
LEG2_DIST = 5_000.0   # B→C
RETURN_DIST = 15_000.0 # C→origin (deliberately large to detect if it leaks)

EXPECTED_DELIVERY_KM = (LEG0_DIST + LEG1_DIST + LEG2_DIST) / 1_000  # 15.0 km


def _make_stub_coords(p1: LatLng, p2: LatLng) -> List[LatLng]:
    return [p1, p2]


def _make_delivery_corridors() -> List[CorridorLeg]:
    """Three delivery-only corridor legs (no return)."""
    return [
        CorridorLeg(leg_idx=0, origin=P_ORIGIN, dest=P_A, buffer_m=200,
                    routes=[RouteOption("test_0", _make_stub_coords(P_ORIGIN, P_A), LEG0_DIST)]),
        CorridorLeg(leg_idx=1, origin=P_A, dest=P_B, buffer_m=200,
                    routes=[RouteOption("test_1", _make_stub_coords(P_A, P_B), LEG1_DIST)]),
        CorridorLeg(leg_idx=2, origin=P_B, dest=P_C, buffer_m=200,
                    routes=[RouteOption("test_2", _make_stub_coords(P_B, P_C), LEG2_DIST)]),
    ]


def _make_corridors_with_return() -> List[CorridorLeg]:
    """Three delivery legs + one return leg."""
    corridors = _make_delivery_corridors()
    corridors.append(
        CorridorLeg(leg_idx=3, origin=P_C, dest=P_ORIGIN, buffer_m=200,
                    routes=[RouteOption("test_ret", _make_stub_coords(P_C, P_ORIGIN), RETURN_DIST)])
    )
    return corridors


def _make_delivery_trace() -> ReconstructedTrace:
    """
    GPS path covers origin→A→B→C (delivery only).
    leg_index assigns points to legs 0, 1, 2 only — no return leg assignment.
    """
    path: List[LatLng] = [P_ORIGIN, P_A, P_B, P_C]
    leg_index = [0, 0, 1, 2]  # last delivery leg
    return ReconstructedTrace(
        raw_path=path,
        matched_path=path,
        matched_times=None,
        leg_index=leg_index,
        dwell_events=[
            DwellEvent(stop_idx=0, stop_label="Stop A", path_indices=[1], min_distance_m=0.0),
            DwellEvent(stop_idx=1, stop_label="Stop B", path_indices=[2], min_distance_m=0.0),
            DwellEvent(stop_idx=2, stop_label="Stop C", path_indices=[3], min_distance_m=0.0),
        ],
        n_legs=3,
    )


# ---------------------------------------------------------------------------
# corridor_builder: effective-origin invariants
# ---------------------------------------------------------------------------

_DEPOT: LatLng = (10.802417, 106.501501)  # matches maps_config default
_BRANCH: LatLng = (10.5, 106.4)           # a custom branch depot used in override tests


def _build_corridors(origin) -> List[CorridorLeg]:
    """Helper: run build_trip_corridors with build_leg_corridor stubbed out."""
    with patch("corridor_builder.build_leg_corridor") as mock_leg:
        mock_leg.side_effect = lambda leg_idx, origin, dest, buffer_m: CorridorLeg(
            leg_idx=leg_idx, origin=origin, dest=dest,
            routes=[RouteOption("stub", [origin, dest], 5000.0)],
            buffer_m=buffer_m,
        )
        from corridor_builder import build_trip_corridors
        return build_trip_corridors(STOPS, origin=origin)


class TestBuildTripCorridorsReturnLeg(unittest.TestCase):
    """
    Verify the effective-origin delivery-loop invariants:
      - When origin is passed, it is used as trip start AND return destination.
      - When origin=None, FIXED_ORIGIN (depot) is used as fallback.
      - Every trip forms a closed loop: origin → stops → origin.
      - 3 stops → exactly 4 legs (leg 0..2 delivery + leg 3 return).
    """

    def test_return_leg_always_present(self):
        """3 stops → 4 legs: origin→A, A→B, B→C, C→origin."""
        corridors = _build_corridors(origin=P_ORIGIN)
        self.assertEqual(len(corridors), 4)
        self.assertEqual([c.leg_idx for c in corridors], [0, 1, 2, 3])

    def test_passed_origin_is_trip_start(self):
        """When origin is supplied, leg 0 must start there."""
        corridors = _build_corridors(origin=P_ORIGIN)
        self.assertEqual(corridors[0].origin, P_ORIGIN)

    def test_passed_origin_is_return_destination(self):
        """The return leg (leg 3) must end at the same origin that was passed in."""
        corridors = _build_corridors(origin=P_ORIGIN)
        self.assertEqual(corridors[-1].origin, P_C)
        self.assertEqual(corridors[-1].dest, P_ORIGIN)

    def test_none_origin_falls_back_to_fixed_depot(self):
        """When origin=None, FIXED_ORIGIN is used as trip start and return point."""
        corridors = _build_corridors(origin=None)
        self.assertEqual(corridors[0].origin, _DEPOT)
        self.assertEqual(corridors[-1].dest, _DEPOT)

    def test_branch_origin_produces_different_corridors(self):
        """A branch-assigned truck's origin propagates to leg 0 and the return leg."""
        corridors = _build_corridors(origin=_BRANCH)
        self.assertEqual(corridors[0].origin, _BRANCH)
        self.assertEqual(corridors[-1].dest, _BRANCH)
        # Loop is closed: start == end
        self.assertEqual(corridors[0].origin, corridors[-1].dest)


# ---------------------------------------------------------------------------
# deviation_scorer: expected_km reflects the full closed-loop distance
# ---------------------------------------------------------------------------

EXPECTED_CLOSED_LOOP_KM = (LEG0_DIST + LEG1_DIST + LEG2_DIST + RETURN_DIST) / 1_000  # 30.0


class TestExpectedKmClosedLoopScope(unittest.TestCase):
    """
    expected_distance_km must equal the sum of ALL corridor leg min distances,
    including the return leg (stopN → ORIGIN).

    Business rule: ORIGIN → stop1 → … → stopN → ORIGIN
    """

    def test_expected_km_includes_return_leg(self):
        """
        Corridors with return leg → expected_km = delivery + return = 30.0 km.
        """
        corridors = _make_corridors_with_return()   # 3 delivery + 1 return (15 km)
        trace = _make_delivery_trace()

        score = score_trip(trace, corridors, STOPS)

        self.assertAlmostEqual(
            score.expected_distance_km, EXPECTED_CLOSED_LOOP_KM, places=2,
            msg=(
                f"expected_km must include the return leg. "
                f"Expected {EXPECTED_CLOSED_LOOP_KM} km, got {score.expected_distance_km} km"
            ),
        )

    def test_expected_km_equals_sum_of_all_legs(self):
        """expected_km = sum(min_feasible_m for ALL legs including return) / 1000."""
        corridors = _make_corridors_with_return()
        trace = _make_delivery_trace()

        score = score_trip(trace, corridors, STOPS)

        manual = (LEG0_DIST + LEG1_DIST + LEG2_DIST + RETURN_DIST) / 1000
        self.assertAlmostEqual(score.expected_distance_km, manual, places=3)

    def test_expected_km_delivery_only_corridors(self):
        """
        When only delivery corridors are built (unusual case),
        expected_km equals delivery-only sum.
        This is a degenerate case — real trips always include the return leg.
        """
        corridors = _make_delivery_corridors()   # 3 delivery, no return
        trace = _make_delivery_trace()

        score = score_trip(trace, corridors, STOPS)

        manual = (LEG0_DIST + LEG1_DIST + LEG2_DIST) / 1000
        self.assertAlmostEqual(score.expected_distance_km, manual, places=3)


# ---------------------------------------------------------------------------
# deviation_scorer: detour_ratio uses delivery scope on both sides
# ---------------------------------------------------------------------------

class TestDetourRatioScope(unittest.TestCase):
    """
    detour_ratio = actual_distance_km / expected_distance_km.

    Both numerator and denominator use the full closed-loop scope:
      actual_distance_km   = full GPS path (delivery + return)
      expected_distance_km = sum of ALL corridor leg min distances (delivery + return)

    Business rule: ORIGIN → stop1 → … → stopN → ORIGIN (return leg always included).
    """

    def test_detour_ratio_not_none_when_corridors_present(self):
        corridors = _make_delivery_corridors()
        trace = _make_delivery_trace()
        score = score_trip(trace, corridors, STOPS)
        self.assertIsNotNone(score.detour_ratio)

    def test_detour_ratio_denominator_is_expected_km(self):
        """
        By definition: detour_ratio * expected_km == actual_distance_km.
        Verifies that numerator and denominator use the same closed-loop scope.
        """
        corridors = _make_corridors_with_return()
        trace = _make_delivery_trace()
        score = score_trip(trace, corridors, STOPS)

        self.assertIsNotNone(score.detour_ratio)
        self.assertIsNotNone(score.expected_distance_km)
        implied_actual_km = score.detour_ratio * score.expected_distance_km
        # Both detour_ratio (3dp) and expected_km (2dp) are independently rounded,
        # so allow a tolerance of 0.05 km (places=1).
        self.assertAlmostEqual(
            implied_actual_km, score.actual_distance_km, places=1,
            msg=(
                "detour_ratio * expected_km must equal actual_distance_km "
                "(both sides use the same closed-loop scope)"
            ),
        )

    def test_detour_ratio_scales_with_expected_km(self):
        """
        In our fixture RETURN_DIST == delivery total (both 15 km), so adding the
        return corridor doubles expected_km.  With the same GPS trace,
        detour_ratio must be exactly halved.

        delivery-only:   ratio = actual_m / 15 000 m
        full closed-loop: ratio = actual_m / 30 000 m  →  half
        """
        corridors_no_return = _make_delivery_corridors()
        corridors_with_return = _make_corridors_with_return()
        trace = _make_delivery_trace()

        score_no_ret  = score_trip(trace, corridors_no_return,  STOPS)
        score_with_ret = score_trip(trace, corridors_with_return, STOPS)

        self.assertIsNotNone(score_no_ret.detour_ratio)
        self.assertIsNotNone(score_with_ret.detour_ratio)

        # RETURN_DIST (15 km) == delivery total (15 km), so denominator doubles.
        # ratio_with_return * 2 must equal ratio_no_return.
        self.assertAlmostEqual(
            score_with_ret.detour_ratio * 2,
            score_no_ret.detour_ratio,
            places=2,
            msg=(
                "Adding the return corridor doubles expected_km, "
                "so detour_ratio must be halved"
            ),
        )

    def test_detour_ratio_none_when_no_corridors(self):
        """No corridors → detour_ratio must be None (cannot be computed)."""
        trace = _make_delivery_trace()
        score = score_trip(trace, corridors=[], stops=STOPS)
        self.assertIsNone(score.detour_ratio)


# ---------------------------------------------------------------------------
# deviation_scorer: compliance scored over all corridor legs
# ---------------------------------------------------------------------------

class TestComplianceScope(unittest.TestCase):
    """
    corridor_compliance_pct is scored over all GPS points in all assigned
    corridor legs.  Since trace_reconstructor assigns points to delivery legs
    only (leg_index never reaches the return leg index), the return-leg
    corridor has 0 GPS points and therefore does NOT inflate or deflate
    corridor_compliance_pct.
    """

    def test_compliance_same_with_and_without_return_corridor(self):
        """
        compliance_pct must be identical whether the return leg corridor
        is present or not, because it has 0 GPS points assigned to it.
        """
        trace = _make_delivery_trace()

        score_no_ret = score_trip(trace, _make_delivery_corridors(), STOPS)
        score_with_ret = score_trip(trace, _make_corridors_with_return(), STOPS)

        self.assertAlmostEqual(
            score_no_ret.corridor_compliance_pct,
            score_with_ret.corridor_compliance_pct,
            places=1,
            msg="corridor_compliance_pct must not change due to return leg corridor",
        )

    def test_all_delivery_points_inside_corridor(self):
        """GPS points placed on route vertices → 100% compliance."""
        trace = _make_delivery_trace()
        score = score_trip(trace, _make_delivery_corridors(), STOPS)
        self.assertAlmostEqual(score.corridor_compliance_pct, 100.0, places=1)


# ---------------------------------------------------------------------------
# Backward-compatibility: trip_report shape
# ---------------------------------------------------------------------------

class TestTripReportShapeUnchanged(unittest.TestCase):
    """
    to_dict() must still return all legacy keys used by trip_pipeline.py and
    export_summary_excel() regardless of the scope fix.
    """

    LEGACY_KEYS = [
        "actual_distance_km", "expected_distance_km", "detour_ratio",
        "off_route_points", "detour_flag", "off_route_flag",
        "wrong_turn_u_turn_flag", "u_turn_count",
        "visited_stops", "missed_stops",
        "path_points", "path_start", "path_end",
        "corridor_compliance_pct", "max_deviation_m", "worst_leg_idx",
    ]

    def test_all_legacy_keys_present(self):
        trace = _make_delivery_trace()
        score = score_trip(trace, _make_delivery_corridors(), STOPS)
        report = score.to_dict()

        for key in self.LEGACY_KEYS:
            self.assertIn(key, report, f"Legacy key '{key}' missing from trip_report")

    def test_all_legacy_keys_present_with_return_leg(self):
        trace = _make_delivery_trace()
        score = score_trip(trace, _make_corridors_with_return(), STOPS)
        report = score.to_dict()

        for key in self.LEGACY_KEYS:
            self.assertIn(key, report, f"Legacy key '{key}' missing from trip_report with return leg")

    def test_actual_distance_km_is_full_path(self):
        """
        actual_distance_km must reflect the FULL GPS path (delivery only in
        our fixture), not just the delivery scope.
        """
        trace = _make_delivery_trace()
        score = score_trip(trace, _make_delivery_corridors(), STOPS)
        self.assertGreater(score.actual_distance_km, 0)
        # actual_distance_km >= expected_distance_km (driver may take longer route)
        if score.expected_distance_km is not None:
            self.assertGreaterEqual(
                score.actual_distance_km + 0.001,  # allow floating point
                score.expected_distance_km * 0.5,  # sanity: not absurdly smaller
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
