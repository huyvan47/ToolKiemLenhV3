"""
tests/test_google_roads_service.py
-----------------------------------
Unit tests for google_roads_service and _do_map_match dispatch in
trace_reconstructor.

All HTTP calls are mocked — no real network requests are made.
No API key is required to run these tests.

Run:
    python -m pytest tests/test_google_roads_service.py -v
    # or
    python -m unittest tests.test_google_roads_service
"""
from __future__ import annotations

import sys
import os
import unittest
from typing import List, Tuple
from unittest.mock import MagicMock, patch, call

# Ensure project root is on sys.path when running from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google_roads_service
from google_roads_service import (
    GoogleRoadsError,
    ROADS_MAX_POINTS,
    _snap_chunk,
    _haversine_m,
    snap_path_to_roads,
)

LatLng = Tuple[float, float]

# ---------------------------------------------------------------------------
# Helpers shared by tests
# ---------------------------------------------------------------------------

def _make_mock_response(data: dict, status_code: int = 200) -> MagicMock:
    """Return a mock requests.Response with .json() and .raise_for_status()."""
    mock = MagicMock()
    mock.json.return_value = data
    mock.status_code = status_code
    if status_code >= 400:
        from requests.exceptions import HTTPError
        mock.raise_for_status.side_effect = HTTPError(f"HTTP {status_code}")
    else:
        mock.raise_for_status.return_value = None
    return mock


def _snapped_points_response(*coords: LatLng) -> dict:
    """Build a minimal valid snapToRoads response payload."""
    return {
        "snappedPoints": [
            {"location": {"latitude": lat, "longitude": lng}, "originalIndex": i}
            for i, (lat, lng) in enumerate(coords)
        ]
    }


# ---------------------------------------------------------------------------
# _haversine_m
# ---------------------------------------------------------------------------

class TestHaversineM(unittest.TestCase):

    def test_same_point_is_zero(self):
        p = (10.0, 106.0)
        self.assertAlmostEqual(_haversine_m(p, p), 0.0, places=3)

    def test_known_distance(self):
        # ~111 km per degree latitude at equator
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        d = _haversine_m(p1, p2)
        self.assertAlmostEqual(d, 111_195, delta=100)

    def test_symmetry(self):
        p1 = (10.5, 106.3)
        p2 = (10.6, 106.4)
        self.assertAlmostEqual(_haversine_m(p1, p2), _haversine_m(p2, p1), places=3)


# ---------------------------------------------------------------------------
# _snap_chunk
# ---------------------------------------------------------------------------

class TestSnapChunk(unittest.TestCase):

    def setUp(self):
        self.chunk: List[LatLng] = [(10.1, 106.1), (10.2, 106.2), (10.3, 106.3)]
        self.key = "test_api_key"

    @patch("google_roads_service._SESSION")
    def test_success_returns_coords(self, mock_session):
        mock_session.get.return_value = _make_mock_response(
            _snapped_points_response((10.101, 106.101), (10.201, 106.201), (10.301, 106.301))
        )
        result = _snap_chunk(self.chunk, self.key, True, 30)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0][0], 10.101)
        self.assertAlmostEqual(result[1][1], 106.201)

    @patch("google_roads_service._SESSION")
    def test_interpolated_points_included(self, mock_session):
        """When interpolate=True the API may return more points than sent."""
        payload = {
            "snappedPoints": [
                {"location": {"latitude": 10.1, "longitude": 106.1}, "originalIndex": 0},
                {"location": {"latitude": 10.15, "longitude": 106.15}},  # interpolated, no originalIndex
                {"location": {"latitude": 10.2, "longitude": 106.2}, "originalIndex": 1},
            ]
        }
        mock_session.get.return_value = _make_mock_response(payload)
        result = _snap_chunk([(10.1, 106.1), (10.2, 106.2)], self.key, True, 30)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[1][0], 10.15)

    @patch("google_roads_service._SESSION")
    def test_api_error_in_response_raises(self, mock_session):
        """API returns HTTP 200 but body contains an 'error' field."""
        mock_session.get.return_value = _make_mock_response({
            "error": {"code": 400, "message": "Invalid key", "status": "INVALID_ARGUMENT"}
        })
        with self.assertRaises(GoogleRoadsError) as ctx:
            _snap_chunk(self.chunk, self.key, True, 30)
        self.assertIn("400", str(ctx.exception))
        self.assertIn("Invalid key", str(ctx.exception))

    @patch("google_roads_service._SESSION")
    def test_http_error_raises(self, mock_session):
        """HTTP 4xx/5xx response raises GoogleRoadsError."""
        mock_session.get.return_value = _make_mock_response({}, status_code=403)
        with self.assertRaises(GoogleRoadsError) as ctx:
            _snap_chunk(self.chunk, self.key, True, 30)
        self.assertIn("HTTP error", str(ctx.exception))

    @patch("google_roads_service._SESSION")
    def test_request_exception_raises(self, mock_session):
        """Network-level exception wrapped in GoogleRoadsError."""
        import requests as _req
        mock_session.get.side_effect = _req.ConnectionError("connection refused")
        with self.assertRaises(GoogleRoadsError) as ctx:
            _snap_chunk(self.chunk, self.key, True, 30)
        self.assertIn("HTTP error", str(ctx.exception))

    @patch("google_roads_service._SESSION")
    def test_empty_snapped_points_raises(self, mock_session):
        """API returns empty snappedPoints list."""
        mock_session.get.return_value = _make_mock_response({"snappedPoints": []})
        with self.assertRaises(GoogleRoadsError) as ctx:
            _snap_chunk(self.chunk, self.key, True, 30)
        self.assertIn("no snappedPoints", str(ctx.exception))

    @patch("google_roads_service._SESSION")
    def test_missing_snapped_points_key_raises(self, mock_session):
        """API returns body without 'snappedPoints' key."""
        mock_session.get.return_value = _make_mock_response({"warningMessage": "sparse"})
        with self.assertRaises(GoogleRoadsError) as ctx:
            _snap_chunk(self.chunk, self.key, True, 30)
        self.assertIn("no snappedPoints", str(ctx.exception))

    @patch("google_roads_service._SESSION")
    def test_correct_path_format_sent(self, mock_session):
        """Verifies the path parameter is formatted as lat,lng|lat,lng."""
        mock_session.get.return_value = _make_mock_response(
            _snapped_points_response((10.1, 106.1))
        )
        _snap_chunk([(10.1, 106.1)], self.key, False, 30)
        _, kwargs = mock_session.get.call_args
        params = kwargs.get("params", mock_session.get.call_args[0][1] if len(mock_session.get.call_args[0]) > 1 else {})
        # params may be positional arg or keyword arg depending on call
        actual_params = mock_session.get.call_args[1].get("params", {})
        self.assertIn("10.1,106.1", actual_params.get("path", ""))
        self.assertEqual(actual_params.get("interpolate"), "false")
        self.assertEqual(actual_params.get("key"), self.key)


# ---------------------------------------------------------------------------
# snap_path_to_roads — chunking and deduplication
# ---------------------------------------------------------------------------

class TestSnapPathToRoads(unittest.TestCase):

    def test_empty_input_returns_empty(self):
        result = snap_path_to_roads([], api_key="k")
        self.assertEqual(result, [])

    def test_chunk_size_exceeds_limit_raises(self):
        with self.assertRaises(ValueError) as ctx:
            snap_path_to_roads([(0.0, 0.0)], api_key="k", chunk_size=101)
        self.assertIn("101", str(ctx.exception))
        self.assertIn("100", str(ctx.exception))

    @patch("google_roads_service._snap_chunk")
    def test_small_input_single_chunk(self, mock_snap):
        """≤ chunk_size points → exactly one _snap_chunk call."""
        pts: List[LatLng] = [(10.0 + i * 0.001, 106.0) for i in range(50)]
        mock_snap.return_value = pts  # passthrough for simplicity
        result = snap_path_to_roads(pts, api_key="k", chunk_size=100)
        mock_snap.assert_called_once()
        self.assertEqual(result, pts)

    @patch("google_roads_service._snap_chunk")
    def test_large_input_multiple_chunks(self, mock_snap):
        """
        110 points with chunk_size=100, overlap=5 →
        chunk 0: points[0:100]   (100 pts)
        chunk 1: points[95:110]  (15 pts)
        = exactly 2 _snap_chunk calls.
        """
        pts: List[LatLng] = [(10.0 + i * 0.001, 106.0) for i in range(110)]

        def fake_snap(chunk, api_key, interpolate, timeout):
            # Return same coords so dedup logic is exercised.
            return list(chunk)

        mock_snap.side_effect = fake_snap
        result = snap_path_to_roads(pts, api_key="k", chunk_size=100)
        self.assertEqual(mock_snap.call_count, 2)
        # First call should have 100 points.
        first_call_chunk = mock_snap.call_args_list[0][0][0]
        self.assertEqual(len(first_call_chunk), 100)
        # Second call should start from index 95 (100 - overlap=5).
        second_call_chunk = mock_snap.call_args_list[1][0][0]
        self.assertEqual(second_call_chunk[0], pts[95])

    @patch("google_roads_service._snap_chunk")
    def test_boundary_deduplication(self, mock_snap):
        """
        When the first point of chunk N+1 is within 10 m of the last point
        of chunk N's output, the duplicate is dropped.
        """
        p_shared: LatLng = (10.1, 106.1)  # last of chunk 0, first of chunk 1 output
        p_far: LatLng = (10.2, 106.2)

        call_count = 0

        def fake_snap(chunk, api_key, interpolate, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [(10.0, 106.0), p_shared]
            else:
                # first point is identical to p_shared → should be deduped
                return [p_shared, p_far]

        mock_snap.side_effect = fake_snap

        pts: List[LatLng] = [(10.0 + i * 0.001, 106.0) for i in range(110)]
        result = snap_path_to_roads(pts, api_key="k", chunk_size=100)

        # p_shared must appear exactly once in result.
        shared_count = sum(1 for p in result if p == p_shared)
        self.assertEqual(shared_count, 1, f"Expected p_shared once, got {shared_count}. result={result}")

    @patch("google_roads_service._snap_chunk")
    def test_no_deduplication_when_far(self, mock_snap):
        """
        When consecutive chunk boundaries are >10 m apart (gap in coverage),
        both points are kept — no silent data loss.
        """
        p1: LatLng = (10.1, 106.1)
        p2: LatLng = (11.0, 107.0)  # clearly >10 m away

        call_count = 0

        def fake_snap(chunk, api_key, interpolate, timeout):
            nonlocal call_count
            call_count += 1
            return [p1] if call_count == 1 else [p2]

        mock_snap.side_effect = fake_snap
        pts: List[LatLng] = [(10.0 + i * 0.001, 106.0) for i in range(110)]
        result = snap_path_to_roads(pts, api_key="k", chunk_size=100)
        self.assertIn(p1, result)
        self.assertIn(p2, result)

    @patch("google_roads_service._snap_chunk")
    def test_chunk_size_exactly_100_accepted(self, mock_snap):
        """chunk_size=100 (the limit itself) must not raise."""
        mock_snap.return_value = [(10.0, 106.0)]
        snap_path_to_roads([(10.0, 106.0)], api_key="k", chunk_size=100)

    @patch("google_roads_service._snap_chunk")
    def test_exactly_chunk_size_points_single_call(self, mock_snap):
        """Exactly chunk_size points → one call, no overlap logic needed."""
        pts: List[LatLng] = [(10.0 + i * 0.001, 106.0) for i in range(100)]
        mock_snap.return_value = pts
        result = snap_path_to_roads(pts, api_key="k", chunk_size=100)
        mock_snap.assert_called_once()
        self.assertEqual(result, pts)


# ---------------------------------------------------------------------------
# _do_map_match dispatch in trace_reconstructor
# ---------------------------------------------------------------------------

class TestDoMapMatch(unittest.TestCase):
    """
    Tests for trace_reconstructor._do_map_match().
    No real HTTP requests; all external calls are mocked.
    """

    SAMPLE_PATH: List[LatLng] = [
        (10.0, 106.0),
        (10.1, 106.1),
        (10.2, 106.2),
    ]
    OSRM_RESULT: List[LatLng] = [
        (10.01, 106.01),
        (10.11, 106.11),
        (10.21, 106.21),
    ]
    ROADS_RESULT: List[LatLng] = [
        (10.001, 106.001),
        (10.101, 106.101),
        (10.201, 106.201),
    ]

    def _call(self, raw_path=None, raw_times=None, chunk_size=80, radius_m=30):
        from trace_reconstructor import _do_map_match
        return _do_map_match(
            raw_path or self.SAMPLE_PATH,
            raw_times,
            chunk_size=chunk_size,
            radius_m=radius_m,
        )

    @patch("trace_reconstructor.map_match")
    @patch("trace_reconstructor.maps_config")
    def test_default_provider_uses_osrm(self, mock_cfg, mock_osrm):
        """Default provider (osrm) → OSRM map_match called, Google Roads not called."""
        mock_cfg.get_map_match_provider.return_value = "osrm"
        mock_osrm.return_value = self.OSRM_RESULT

        result = self._call()

        mock_osrm.assert_called_once()
        self.assertEqual(result, self.OSRM_RESULT)

    @patch("trace_reconstructor.google_roads_service")
    @patch("trace_reconstructor.map_match")
    @patch("trace_reconstructor.maps_config")
    def test_google_roads_provider_with_key_calls_snap(
        self, mock_cfg, mock_osrm, mock_roads_svc
    ):
        """MAP_MATCH_PROVIDER=google_roads + key set → Google Roads called, OSRM not called."""
        mock_cfg.get_map_match_provider.return_value = "google_roads"
        mock_cfg.get_api_key.return_value = "fake_key"
        mock_roads_svc.snap_path_to_roads.return_value = self.ROADS_RESULT
        mock_roads_svc.ROADS_MAX_POINTS = 100
        # GoogleRoadsError must be a real exception class so except clauses work.
        mock_roads_svc.GoogleRoadsError = GoogleRoadsError

        result = self._call()

        mock_roads_svc.snap_path_to_roads.assert_called_once()
        mock_osrm.assert_not_called()
        self.assertEqual(result, self.ROADS_RESULT)

    @patch("trace_reconstructor.google_roads_service")
    @patch("trace_reconstructor.map_match")
    @patch("trace_reconstructor.maps_config")
    def test_google_roads_api_error_falls_back_to_osrm(
        self, mock_cfg, mock_osrm, mock_roads_svc
    ):
        """Google Roads raises GoogleRoadsError → silent fallback to OSRM."""
        mock_cfg.get_map_match_provider.return_value = "google_roads"
        mock_cfg.get_api_key.return_value = "fake_key"
        mock_roads_svc.GoogleRoadsError = GoogleRoadsError
        mock_roads_svc.ROADS_MAX_POINTS = 100
        mock_roads_svc.snap_path_to_roads.side_effect = GoogleRoadsError("quota exceeded")
        mock_osrm.return_value = self.OSRM_RESULT

        result = self._call()

        mock_roads_svc.snap_path_to_roads.assert_called_once()
        mock_osrm.assert_called_once()
        self.assertEqual(result, self.OSRM_RESULT)

    @patch("trace_reconstructor.google_roads_service")
    @patch("trace_reconstructor.map_match")
    @patch("trace_reconstructor.maps_config")
    def test_google_roads_unexpected_error_falls_back_to_osrm(
        self, mock_cfg, mock_osrm, mock_roads_svc
    ):
        """Unexpected exception from Google Roads service → fallback to OSRM."""
        mock_cfg.get_map_match_provider.return_value = "google_roads"
        mock_cfg.get_api_key.return_value = "fake_key"
        mock_roads_svc.GoogleRoadsError = GoogleRoadsError
        mock_roads_svc.ROADS_MAX_POINTS = 100
        mock_roads_svc.snap_path_to_roads.side_effect = RuntimeError("unexpected")
        mock_osrm.return_value = self.OSRM_RESULT

        result = self._call()

        mock_osrm.assert_called_once()
        self.assertEqual(result, self.OSRM_RESULT)

    @patch("trace_reconstructor.google_roads_service")
    @patch("trace_reconstructor.map_match")
    @patch("trace_reconstructor.maps_config")
    def test_google_roads_no_key_falls_back_to_osrm(
        self, mock_cfg, mock_osrm, mock_roads_svc
    ):
        """MAP_MATCH_PROVIDER=google_roads but no API key → skip Roads, use OSRM."""
        mock_cfg.get_map_match_provider.return_value = "google_roads"
        mock_cfg.get_api_key.return_value = None  # key absent
        mock_osrm.return_value = self.OSRM_RESULT

        result = self._call()

        mock_roads_svc.snap_path_to_roads.assert_not_called()
        mock_osrm.assert_called_once()
        self.assertEqual(result, self.OSRM_RESULT)

    @patch("trace_reconstructor.google_roads_service")
    @patch("trace_reconstructor.map_match")
    @patch("trace_reconstructor.maps_config")
    def test_chunk_size_capped_at_roads_max(
        self, mock_cfg, mock_osrm, mock_roads_svc
    ):
        """chunk_size > ROADS_MAX_POINTS is capped when passed to snap_path_to_roads."""
        mock_cfg.get_map_match_provider.return_value = "google_roads"
        mock_cfg.get_api_key.return_value = "fake_key"
        mock_roads_svc.ROADS_MAX_POINTS = 100
        mock_roads_svc.GoogleRoadsError = GoogleRoadsError
        mock_roads_svc.snap_path_to_roads.return_value = self.ROADS_RESULT

        # Pass chunk_size=200 — should be capped to 100 before forwarding.
        from trace_reconstructor import _do_map_match
        _do_map_match(self.SAMPLE_PATH, None, chunk_size=200, radius_m=30)

        _, kwargs = mock_roads_svc.snap_path_to_roads.call_args
        self.assertLessEqual(kwargs.get("chunk_size", 999), 100)

    @patch("trace_reconstructor.map_match")
    @patch("trace_reconstructor.maps_config")
    def test_osrm_receives_original_chunk_size(self, mock_cfg, mock_osrm):
        """When using OSRM the original chunk_size is forwarded unchanged."""
        mock_cfg.get_map_match_provider.return_value = "osrm"
        mock_osrm.return_value = self.OSRM_RESULT

        from trace_reconstructor import _do_map_match
        _do_map_match(self.SAMPLE_PATH, None, chunk_size=80, radius_m=30)

        _, kwargs = mock_osrm.call_args
        self.assertEqual(kwargs.get("chunk_size"), 80)
        self.assertEqual(kwargs.get("radius_m"), 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
