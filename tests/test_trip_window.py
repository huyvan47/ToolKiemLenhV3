"""
tests/test_trip_window.py
--------------------------
Unit tests for detect_trip_window_from_origin().

All five required scenarios are covered:
  1. Vehicle starts inside origin, departs, and returns.
  2. Vehicle starts inside, departs, but never returns.
  3. Vehicle never leaves the origin zone.
  4. File starts with vehicle already outside origin (mid-trip start).
  5. Noisy GPS near boundary does not trigger false departure/return.

No network calls. All GPS data is synthetic DataFrames.
"""
from __future__ import annotations

import os
import sys
import unittest
from typing import List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Stub genuinely unimportable modules before importing trip_pipeline.
# tearDownModule restores sys.modules so sibling tests are unaffected.
# ---------------------------------------------------------------------------
from typing import Any, Dict
from unittest.mock import MagicMock

_NEEDS_STUB = ["gpt_data", "phat_hien_quay_dau_data"]
_saved: Dict[str, Any] = {}
for _name in _NEEDS_STUB:
    _saved[_name] = sys.modules.get(_name)
    sys.modules[_name] = MagicMock()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trip_pipeline import detect_trip_window_from_origin  # noqa: E402


def tearDownModule():
    for name, original in _saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ORIGIN = (10.802417, 106.501501)   # depot
_RADIUS = 700.0                     # metres

# A coordinate that is clearly inside the origin radius (~0 m away)
_INSIDE_PT  = (10.802417, 106.501501)   # exactly on depot
# A coordinate that is well outside (~5 km away)
_OUTSIDE_PT = (10.850000, 106.550000)


def _make_df(rows: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Build a minimal GPS DataFrame from (timestamp_str, coord_str) pairs.
    Invalid entries can be passed as ("", "") and will produce NaT / NaN.
    """
    times  = [r[0] if r[0] else None for r in rows]
    coords = [r[1] if r[1] else None for r in rows]
    return pd.DataFrame({"Thời gian": times, "Tọa độ": coords})


def _ts(hhmm: str) -> str:
    """Return a full datetime string for a given HH:MM."""
    return f"01/01/2024 {hhmm}:00"


def _inside(hhmm: str) -> Tuple[str, str]:
    return _ts(hhmm), f"{_INSIDE_PT[0]},{_INSIDE_PT[1]}"


def _outside(hhmm: str) -> Tuple[str, str]:
    return _ts(hhmm), f"{_OUTSIDE_PT[0]},{_OUTSIDE_PT[1]}"


def _invalid_row() -> Tuple[str, str]:
    return "", ""


# ---------------------------------------------------------------------------
# Scenario 1 — starts inside, departs, returns
# ---------------------------------------------------------------------------

class TestDepartedAndReturned(unittest.TestCase):
    """Happy path: truck leaves depot, delivers, comes back."""

    def setUp(self):
        rows = (
            [_inside("06:00")] * 3            # parked at depot
            + [_outside("07:00")] * 4         # N+ consecutive outside → departure
            + [_outside("10:00")] * 10        # on the road
            + [_inside("14:00")] * 4          # N+ consecutive inside → return
        )
        self.df = _make_df(rows)
        self.result = detect_trip_window_from_origin(
            self.df, _ORIGIN, origin_radius_m=_RADIUS, min_consecutive_points=3
        )

    def test_status(self):
        self.assertEqual(self.result["status"], "departed_and_returned")

    def test_departure_time_is_first_outside_point(self):
        # departure_time = timestamp of first point in the outside run
        self.assertEqual(self.result["start"], "07:00")

    def test_return_time_is_first_inside_point_after_departure(self):
        self.assertEqual(self.result["end"], "14:00")

    def test_detection_method(self):
        self.assertEqual(self.result["detection_method"], "origin_radius")

    def test_fallback_not_used(self):
        self.assertFalse(self.result["fallback_used"])

    def test_started_inside(self):
        self.assertTrue(self.result["started_inside"])

    def test_departure_index_is_set(self):
        self.assertIsNotNone(self.result["departure_index"])

    def test_return_index_is_set(self):
        self.assertIsNotNone(self.result["return_index"])


# ---------------------------------------------------------------------------
# Scenario 2 — departs but never returns
# ---------------------------------------------------------------------------

class TestDepartedNoReturn(unittest.TestCase):
    """Truck leaves but file ends before it comes back."""

    def setUp(self):
        rows = (
            [_inside("06:00")] * 3
            + [_outside("07:30")] * 20        # departs, never returns
        )
        self.df = _make_df(rows)
        self.result = detect_trip_window_from_origin(
            self.df, _ORIGIN, origin_radius_m=_RADIUS, min_consecutive_points=3
        )

    def test_status(self):
        self.assertEqual(self.result["status"], "departed_no_return")

    def test_departure_time_detected(self):
        self.assertEqual(self.result["start"], "07:30")

    def test_return_time_is_none(self):
        self.assertIsNone(self.result["end"])

    def test_fallback_not_used(self):
        self.assertFalse(self.result["fallback_used"])


# ---------------------------------------------------------------------------
# Scenario 3 — vehicle never leaves origin zone
# ---------------------------------------------------------------------------

class TestNeverDeparted(unittest.TestCase):
    """Truck stays parked at depot the whole file."""

    def setUp(self):
        rows = [_inside("06:00"), _inside("07:00"), _inside("08:00"), _inside("09:00")]
        self.df = _make_df(rows)
        self.result = detect_trip_window_from_origin(
            self.df, _ORIGIN, origin_radius_m=_RADIUS, min_consecutive_points=3
        )

    def test_status(self):
        self.assertEqual(self.result["status"], "never_departed")

    def test_departure_time_is_none(self):
        self.assertIsNone(self.result["start"])

    def test_return_time_is_none(self):
        self.assertIsNone(self.result["end"])

    def test_departure_index_none(self):
        self.assertIsNone(self.result["departure_index"])

    def test_fallback_not_used(self):
        self.assertFalse(self.result["fallback_used"])


# ---------------------------------------------------------------------------
# Scenario 4 — file starts outside origin (mid-trip start)
# ---------------------------------------------------------------------------

class TestStartedOutside(unittest.TestCase):
    """GPS file begins when truck is already away from depot."""

    def setUp(self):
        rows = (
            [_outside("09:00")] * 5          # already outside; departure = 09:00
            + [_inside("13:00")] * 4         # returns to depot
        )
        self.df = _make_df(rows)
        self.result = detect_trip_window_from_origin(
            self.df, _ORIGIN, origin_radius_m=_RADIUS, min_consecutive_points=3
        )

    def test_started_outside(self):
        self.assertFalse(self.result["started_inside"])

    def test_departure_time_is_first_row(self):
        # departure_candidate is set to first row; confirmed after 3 outside points
        self.assertEqual(self.result["start"], "09:00")

    def test_return_detected(self):
        self.assertEqual(self.result["end"], "13:00")

    def test_status(self):
        self.assertEqual(self.result["status"], "started_outside_returned")

    def test_fallback_not_used(self):
        self.assertFalse(self.result["fallback_used"])


class TestStartedOutsideNoReturn(unittest.TestCase):
    """File starts outside and truck never returns."""

    def setUp(self):
        rows = [_outside("08:00")] * 10
        self.df = _make_df(rows)
        self.result = detect_trip_window_from_origin(
            self.df, _ORIGIN, origin_radius_m=_RADIUS, min_consecutive_points=3
        )

    def test_status(self):
        self.assertEqual(self.result["status"], "started_outside_no_return")

    def test_departure_detected(self):
        self.assertEqual(self.result["start"], "08:00")

    def test_return_is_none(self):
        self.assertIsNone(self.result["end"])


# ---------------------------------------------------------------------------
# Scenario 5 — noisy points near boundary do not trigger false events
# ---------------------------------------------------------------------------

class TestNoisyBoundary(unittest.TestCase):
    """
    1 or 2 outside points sandwiched between inside points must NOT
    trigger a departure. min_consecutive_points=3 is the guard.
    """

    def setUp(self):
        rows = (
            [_inside("06:00")] * 5
            + [_outside("06:10")]             # single blip out
            + [_inside("06:20")] * 5          # back inside → no departure
            + [_outside("06:30")] * 2         # two blips out
            + [_inside("06:40")] * 5          # back inside → still no departure
            + [_outside("07:00")] * 4         # genuine departure (>= 3 consecutive)
        )
        self.df = _make_df(rows)
        self.result = detect_trip_window_from_origin(
            self.df, _ORIGIN, origin_radius_m=_RADIUS, min_consecutive_points=3
        )

    def test_departure_not_triggered_by_single_blip(self):
        # departure must be at 07:00, not at the 06:10 blip
        self.assertEqual(self.result["start"], "07:00")

    def test_departure_not_triggered_by_two_blips(self):
        # same check — the 06:30 pair (only 2 consecutive) must not fire
        self.assertEqual(self.result["start"], "07:00")

    def test_status(self):
        # truck departed but file ends before return
        self.assertEqual(self.result["status"], "departed_no_return")


class TestNoisyReturn(unittest.TestCase):
    """
    1 or 2 inside points near origin during the delivery route must NOT
    trigger a false return.
    """

    def setUp(self):
        rows = (
            [_inside("06:00")] * 3
            + [_outside("07:00")] * 5         # genuine departure
            + [_inside("09:00")]              # single inside blip during route
            + [_outside("09:10")] * 5         # back outside → no return yet
            + [_inside("12:00")] * 4          # genuine return
        )
        self.df = _make_df(rows)
        self.result = detect_trip_window_from_origin(
            self.df, _ORIGIN, origin_radius_m=_RADIUS, min_consecutive_points=3
        )

    def test_return_not_triggered_by_single_inside_blip(self):
        # return must be at 12:00, not at the 09:00 blip
        self.assertEqual(self.result["end"], "12:00")

    def test_status(self):
        self.assertEqual(self.result["status"], "departed_and_returned")


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestFallbackCases(unittest.TestCase):
    """Fallback to min/max timestamp when origin-based detection cannot run."""

    def test_fallback_when_origin_is_none(self):
        df = _make_df([_outside("08:00"), _outside("09:00")])
        result = detect_trip_window_from_origin(df, origin=None)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["detection_method"], "min_max_timestamp")

    def test_fallback_when_no_valid_gps(self):
        df = _make_df([_invalid_row(), _invalid_row()])
        result = detect_trip_window_from_origin(df, _ORIGIN)
        self.assertTrue(result["fallback_used"])

    def test_fallback_when_missing_columns(self):
        df = pd.DataFrame({"col_a": [1, 2]})
        result = detect_trip_window_from_origin(df, _ORIGIN)
        self.assertTrue(result["fallback_used"])

    def test_fallback_preserves_start_end_keys(self):
        """Fallback dict must still have 'start' and 'end' for downstream code."""
        df = _make_df([_invalid_row()])
        result = detect_trip_window_from_origin(df, _ORIGIN)
        self.assertIn("start", result)
        self.assertIn("end", result)


# ---------------------------------------------------------------------------
# Counter fields
# ---------------------------------------------------------------------------

class TestCounterFields(unittest.TestCase):
    """n_valid_points, n_inside_points, n_outside_points are accurate."""

    def test_counts(self):
        rows = (
            [_inside("06:00")] * 3      # 3 inside
            + [_invalid_row()] * 2      # 2 invalid — not counted
            + [_outside("08:00")] * 5   # 5 outside
        )
        result = detect_trip_window_from_origin(
            _make_df(rows), _ORIGIN, origin_radius_m=_RADIUS
        )
        self.assertEqual(result["n_valid_points"],   8)
        self.assertEqual(result["n_inside_points"],  3)
        self.assertEqual(result["n_outside_points"], 5)


# ---------------------------------------------------------------------------
# Movement-aware departure: parked rows must not trigger false departure
# ---------------------------------------------------------------------------

def _make_df_with_movement(
    rows: List[Tuple[str, str]],
    speeds: List[int],
    statuses: List[str],
) -> pd.DataFrame:
    """Build a GPS DataFrame with speed and status columns."""
    times  = [r[0] if r[0] else None for r in rows]
    coords = [r[1] if r[1] else None for r in rows]
    return pd.DataFrame({
        "Thời gian":  times,
        "Tọa độ":     coords,
        "Tốc độ":     speeds,
        "Trạng thái": statuses,
    })


class TestParkedOutsideOriginNoDepartureBeforeMoving(unittest.TestCase):
    """
    Mirrors the 62C19132 scenario: truck is parked outside the origin radius
    before the trip starts.  The parked row must NOT trigger departure.
    Departure must equal the first moving outside row.
    """

    def setUp(self):
        # Row 0: parked outside origin (speed=0, stop) → must be skipped
        # Rows 1-3: moving outside → departure confirmed at row 1
        # Rows 4-6: moving inside return zone (200 m) → return confirmed at row 4
        rows = (
            [_outside("00:00")]          # pre-trip parked
            + [_outside("04:47")] * 4   # moving → departure
            + [_inside("20:09")] * 4    # return (inside 200 m zone)
        )
        speeds   = [0,  8,  9, 10, 11,   0,  5, 15, 18]
        statuses = ["stop", "run", "run", "run", "run",
                    "run", "run", "run", "run"]
        self.df = _make_df_with_movement(rows, speeds, statuses)
        self.result = detect_trip_window_from_origin(
            self.df, _ORIGIN, origin_radius_m=_RADIUS,
            return_radius_m=200.0, min_consecutive_points=3,
        )

    def test_departure_not_at_parked_row(self):
        """start must be 04:47 (first moving row), not 00:00 (parked row)."""
        self.assertNotEqual(self.result["start"], "00:00")
        self.assertEqual(self.result["start"], "04:47")

    def test_return_detected(self):
        self.assertEqual(self.result["end"], "20:09")

    def test_status(self):
        self.assertIn(
            self.result["status"],
            ("started_outside_returned", "departed_and_returned"),
        )

    def test_fallback_not_used(self):
        self.assertFalse(self.result["fallback_used"])


class TestOutboundPassNearOriginDoesNotFalseReturn(unittest.TestCase):
    """
    Mirrors the 62C19132_0809/0909 scenario: truck departs from outside origin
    and its OUTBOUND route clips through the return zone (within 200 m of
    origin) very shortly after departure, before the truck has travelled far.

    With max_dist_since_dep still < 2 × origin_radius_m (return_arm_dist),
    the return detector must NOT fire on that outbound clip.
    Only after the truck has been far away (>= return_arm_dist) is the
    detector armed.  If the truck never re-enters the return zone after that,
    end must be None.
    """

    # Geometry used in this test
    # _OUTSIDE_PT  ≈ 7 600 m from _ORIGIN  → satisfies return_arm_dist (1 400 m)
    # _INSIDE_PT   is exactly _ORIGIN       → satisfies return_radius_m (200 m)
    # The "near-origin clip" coord is 150 m from _ORIGIN — inside 200 m zone.
    _CLIP = (10.803767, 106.501501)   # ~150 m north of depot, inside 200 m zone

    def setUp(self):
        # Sequence:
        #   Rows 0-2  : truck at ~_OUTSIDE_PT (far from origin) — just departed,
        #               max_dist ~7 600 m; departure confirmed here.
        #   Rows 3-5  : outbound clip — truck passes within 150 m of origin.
        #               At this point max_dist is still ~7 600 m? No — in this
        #               synthetic test the truck goes from outside DIRECTLY to
        #               the clip, so max_dist never gets much past 7 600 m before
        #               the clip.  But we want to test the case where max_dist
        #               is SMALL (truck has barely left home).
        #
        # To simulate the real scenario (truck only slightly outside origin
        # radius before clipping), we use a near-origin outside point that
        # is only ~800 m away, then immediately clip inside 200 m:
        _NEAR_OUTSIDE = (10.809, 106.501501)   # ~740 m from _ORIGIN, outside 700 m
        rows = (
            [(_ts("04:43"), f"{_NEAR_OUTSIDE[0]},{_NEAR_OUTSIDE[1]}")] * 4   # just departed, ~740 m
            + [(_ts("04:47"), f"{self._CLIP[0]},{self._CLIP[1]}")] * 4       # outbound clip <200 m
            + [(_ts("08:00"), f"{_OUTSIDE_PT[0]},{_OUTSIDE_PT[1]}")] * 4    # far delivery
        )
        # No speed/status columns → _is_moving always True
        times  = [r[0] for r in rows]
        coords = [r[1] for r in rows]
        self.df = pd.DataFrame({"Thời gian": times, "Tọa độ": coords})
        self.result = detect_trip_window_from_origin(
            self.df, _ORIGIN, origin_radius_m=_RADIUS,
            return_radius_m=200.0, min_consecutive_points=3,
        )

    def test_departure_confirmed(self):
        self.assertIsNotNone(self.result["start"])

    def test_outbound_clip_does_not_become_return(self):
        """The 04:47 inside-200m rows are outbound — must NOT be the return."""
        self.assertNotEqual(self.result["end"], "04:47")

    def test_return_is_none_when_no_post_armed_inside_rows(self):
        """After arming (truck far away), no further inside-200m rows → null."""
        self.assertIsNone(self.result["end"])


class TestSlowSpeedRowsSkippedForDeparture(unittest.TestCase):
    """
    Rows with speed ≤ min_moving_speed_kmh (default 5 km/h) must be skipped
    for departure detection — they do not count AND do not reset the counter.
    """

    def setUp(self):
        # speed=4 rows interspersed between the real departure rows
        rows = (
            [_outside("04:47")] * 6   # alternating fast/slow
        )
        speeds   = [0, 4, 5, 8, 10, 12]   # rows 0-2 are idle (≤5); rows 3-5 are moving
        statuses = ["stop", "run", "run", "run", "run", "run"]
        self.df = _make_df_with_movement(rows, speeds, statuses)
        self.result = detect_trip_window_from_origin(
            self.df, _ORIGIN, origin_radius_m=_RADIUS,
            min_consecutive_points=3, min_moving_speed_kmh=5.0,
        )

    def test_departure_anchors_to_first_qualifying_moving_row(self):
        """Departure candidate should be first row with speed > 5 (row 3 = 04:47)."""
        self.assertEqual(self.result["start"], "04:47")

    def test_departure_confirmed(self):
        self.assertIsNotNone(self.result["start"])


# ---------------------------------------------------------------------------
# Real-file regression: 62C19132_1403.xlsx
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402


class TestRealFileRegression(unittest.TestCase):
    """
    Regression test against the real GPS file 62C19132_1403.xlsx.

    Expected behaviour
    ------------------
    - Truck is parked at its branch lot (~844 m from configured origin) from
      00:00:40 until 04:47:21.  That single stopped row must NOT become the
      departure time.
    - First real movement: 14/03/2026 04:47:21, speed=8 → departure = 04:47.
    - Truck returns within 200 m of origin at ~20:09:14 → return = 20:09.
    - Current (wrong) output was start=00:00, end=04:44 — this test proves that
      the fixed detector no longer produces those values.
    """

    _DATA_FILE = (
        Path(__file__).resolve().parent.parent
        / "data" / "raw" / "vtracking" / "62C19132_1403.xlsx"
    )
    _ORIGIN = (10.802417, 106.501501)

    @classmethod
    def setUpClass(cls):
        if not cls._DATA_FILE.exists():
            raise unittest.SkipTest(
                f"Real GPS file not found — skipping regression: {cls._DATA_FILE}"
            )
        cls.df = pd.read_excel(str(cls._DATA_FILE))
        cls.result = detect_trip_window_from_origin(
            cls.df,
            cls._ORIGIN,
            origin_radius_m=700.0,
            return_radius_m=200.0,
            min_consecutive_points=3,
        )

    def test_departure_is_0447_not_0000(self):
        """Trip start must be 04:47 (first moving row), never 00:00 (parked row)."""
        self.assertEqual(self.result["start"], "04:47",
                         f"Expected 04:47, got {self.result['start']}. "
                         f"Full result: {self.result}")

    def test_return_is_2009(self):
        """Trip end must be 20:09 (first qualifying return point near origin)."""
        self.assertEqual(self.result["end"], "20:09",
                         f"Expected 20:09, got {self.result['end']}. "
                         f"Full result: {self.result}")

    def test_start_is_not_wrong_0000(self):
        """Regression guard: old wrong value must not reappear."""
        self.assertNotEqual(self.result["start"], "00:00")

    def test_end_is_not_wrong_0444(self):
        """Regression guard: old wrong end value must not reappear."""
        self.assertNotEqual(self.result["end"], "04:44")

    def test_detection_method_is_origin_radius(self):
        self.assertEqual(self.result["detection_method"], "origin_radius")

    def test_fallback_not_used(self):
        self.assertFalse(self.result["fallback_used"])

    def test_status_indicates_real_trip(self):
        self.assertIn(
            self.result["status"],
            ("started_outside_returned", "departed_and_returned"),
        )
