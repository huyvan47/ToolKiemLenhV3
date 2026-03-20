"""
tests/test_trip_pipeline.py
----------------------------
Unit tests for trip_pipeline.py.

Key tests:
  - process_all_plates() processes ALL plates, not just the first one
  - process_all_plates() continues after a per-plate error
  - export_summary_excel() produces the correct column set
  - export_reports_json() writes valid JSON
  - match_turnaround_to_stops() classifies u-turns correctly

Only modules that genuinely cannot be imported in this test environment
are stubbed in sys.modules (gpt_data needs openai; phat_hien_quay_dau_data
needs geopy).  Everything else is imported normally to avoid polluting
the sys.modules cache used by @patch string lookups in sibling test files.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal sys.modules stubs — ONLY for genuinely unimportable dependencies.
# tearDownModule restores sys.modules so sibling test files are not affected.
# ---------------------------------------------------------------------------
_NEEDS_STUB = ["gpt_data", "phat_hien_quay_dau_data"]
_saved: Dict[str, Any] = {}

for _name in _NEEDS_STUB:
    _saved[_name] = sys.modules.get(_name)
    sys.modules[_name] = MagicMock()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trip_pipeline  # noqa: E402


def tearDownModule():
    """Restore stubbed entries so sibling test files' @patch lookups work."""
    for name, original in _saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ok_report(plate: str) -> Dict[str, Any]:
    return {
        "plate": plate,
        "trip_window": {"start": "07:00", "end": "17:00"},
        "stops": [],
        "trip_report": {
            "actual_distance_km": 100.0,
            "expected_distance_km": 80.0,
            "detour_ratio": 1.25,
            "off_route_points": 5,
            "detour_flag": False,
            "off_route_flag": False,
            "wrong_turn_u_turn_flag": False,
            "u_turn_count": 0,
            "visited_stops": [],
            "missed_stops": [],
            "path_points": 200,
            "path_start": (10.0, 106.0),
            "path_end": (10.5, 106.5),
            "corridor_compliance_pct": 95.0,
            "max_deviation_m": 150.0,
            "worst_leg_idx": 1,
        },
        "turnaround_valid": [],
        "turnaround_suspicious": [],
        "epass_plate": plate,
        "epass_rows": [],
        "epass_count": 0,
    }


def _make_error_report(plate: str, msg: str = "File not found") -> Dict[str, Any]:
    return {"plate": plate, "error": msg}


# ---------------------------------------------------------------------------
# Tests: process_all_plates
# ---------------------------------------------------------------------------

class TestProcessAllPlates(unittest.TestCase):
    """
    Critical regression tests for the return-inside-loop bug fixed in PHA 5.
    Before the fix, process_all_plates() returned after the FIRST plate.
    """

    def _run(self, plate_addrs: Dict[str, List[str]], side_effects=None):
        with patch.object(trip_pipeline, "LayDuLieuFileLenh") as mock_orders, \
             patch.object(trip_pipeline, "process_one_plate") as mock_one:

            mock_orders.return_value = plate_addrs
            if side_effects is not None:
                mock_one.side_effect = side_effects
            else:
                mock_one.side_effect = lambda plate, addrs, day_code=None: _make_ok_report(plate)

            return trip_pipeline.process_all_plates(), mock_one

    def test_processes_all_plates_not_just_first(self):
        """process_all_plates must process all 3 plates, returning 3 reports."""
        plates = {"001": ["addr1"], "002": ["addr2"], "003": ["addr3"]}
        reports, mock_one = self._run(plates)
        self.assertEqual(len(reports), 3)
        self.assertEqual(mock_one.call_count, 3)

    def test_returns_plate_in_each_report(self):
        plates = {"ABC": ["addr"], "XYZ": ["addr"]}
        reports, _ = self._run(plates)
        returned_plates = [r["plate"] for r in reports]
        self.assertIn("ABC", returned_plates)
        self.assertIn("XYZ", returned_plates)

    def test_empty_orders_returns_empty_list(self):
        reports, mock_one = self._run({})
        self.assertEqual(reports, [])
        mock_one.assert_not_called()

    def test_error_in_one_plate_continues_processing_others(self):
        """
        If process_one_plate raises for plate '002', the pipeline must
        still process '001' and '003' and include an error dict for '002'.
        """
        plates = {"001": ["a"], "002": ["b"], "003": ["c"]}

        def side_effect(plate, addrs, day_code=None):
            if plate == "002":
                raise FileNotFoundError("No vtracking file")
            return _make_ok_report(plate)

        reports, mock_one = self._run(plates, side_effects=side_effect)

        self.assertEqual(len(reports), 3)
        self.assertEqual(mock_one.call_count, 3)
        ok_plates  = [r["plate"] for r in reports if "error" not in r]
        err_plates = [r["plate"] for r in reports if "error" in r]
        self.assertIn("001", ok_plates)
        self.assertIn("003", ok_plates)
        self.assertIn("002", err_plates)

    def test_all_errors_still_returns_all_plates(self):
        plates = {"001": ["a"], "002": ["b"], "003": ["c"]}

        def always_fail(plate, addrs, day_code=None):
            raise RuntimeError("some failure")

        reports, mock_one = self._run(plates, side_effects=always_fail)
        self.assertEqual(len(reports), 3)
        self.assertTrue(all("error" in r for r in reports))

    def test_day_code_forwarded_to_process_one_plate(self):
        plates = {"001": ["a"]}
        with patch.object(trip_pipeline, "LayDuLieuFileLenh") as mock_orders, \
             patch.object(trip_pipeline, "process_one_plate") as mock_one:
            mock_orders.return_value = plates
            mock_one.return_value = _make_ok_report("001")

            trip_pipeline.process_all_plates(day_code="20240315")
            mock_one.assert_called_once_with("001", ["a"], day_code="20240315")


# ---------------------------------------------------------------------------
# Tests: export_summary_excel columns
# ---------------------------------------------------------------------------

class TestExportSummaryExcel(unittest.TestCase):
    EXPECTED_COLUMNS = [
        "Biển số", "Giờ đi", "Giờ về",
        "Km thực tế", "Km kỳ vọng", "Tỷ lệ vòng",
        "Lệch tuyến", "Có quay đầu",
        "Điểm giao đã ghé", "Điểm giao bỏ sót",
        "Quay đầu hợp lệ", "Quay đầu nghi vấn",
        "Số vé ePass",
        "Tuân thủ hành lang (%)", "Lệch tối đa (m)", "Chân hàng tệ nhất",
    ]

    def _capture_df(self, reports):
        import pandas as pd
        captured = {}

        def fake_to_excel(self_df, path, **kwargs):
            captured["df"] = self_df

        with patch("pandas.DataFrame.to_excel", fake_to_excel):
            trip_pipeline.export_summary_excel(reports, filename="test_out.xlsx")
        return captured.get("df")

    def test_all_expected_columns_present_for_ok_report(self):
        df = self._capture_df([_make_ok_report("001")])
        self.assertIsNotNone(df)
        for col in self.EXPECTED_COLUMNS:
            self.assertIn(col, df.columns, f"Column '{col}' missing from summary Excel")

    def test_error_report_produces_bien_so_and_loi(self):
        df = self._capture_df([_make_error_report("002", "Something went wrong")])
        self.assertIsNotNone(df)
        self.assertIn("Biển số", df.columns)
        self.assertIn("Lỗi", df.columns)
        self.assertEqual(df.iloc[0]["Biển số"], "002")
        self.assertEqual(df.iloc[0]["Lỗi"], "Something went wrong")

    def test_mixed_ok_and_error_rows(self):
        reports = [_make_ok_report("001"), _make_error_report("002"), _make_ok_report("003")]
        df = self._capture_df(reports)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)


# ---------------------------------------------------------------------------
# Tests: export_reports_json
# ---------------------------------------------------------------------------

class TestExportReportsJson(unittest.TestCase):
    def test_writes_valid_json(self):
        reports = [_make_ok_report("001"), _make_error_report("002")]
        written: Dict = {}

        def fake_open(path, mode="r", **kwargs):
            import io
            buf = io.StringIO()

            class FakeFile:
                def __enter__(self): return buf
                def __exit__(self, *a): written["text"] = buf.getvalue()

            return FakeFile()

        with patch("builtins.open", fake_open):
            trip_pipeline.export_reports_json(reports, filename="test.json")

        self.assertIn("text", written)
        parsed = json.loads(written["text"])
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["plate"], "001")
        self.assertEqual(parsed[1]["plate"], "002")


# ---------------------------------------------------------------------------
# Tests: match_turnaround_to_stops
# ---------------------------------------------------------------------------
# vtracking_tool is importable, so trip_pipeline.haversine and parse_coord
# are real implementations — no patching needed here.

class TestMatchTurnaroundToStops(unittest.TestCase):
    def test_within_threshold_is_valid(self):
        """U-turn at exact stop location → valid."""
        stop = {"lat": 10.0, "lng": 106.0, "normalized_text": "Stop A"}
        turn_row = {"Tọa độ": "10.0,106.0"}
        valid, suspicious = trip_pipeline.match_turnaround_to_stops([turn_row], [stop])
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(suspicious), 0)
        self.assertAlmostEqual(valid[0]["distance_to_nearest_stop_m"], 0.0, places=0)

    def test_far_from_all_stops_is_suspicious(self):
        """U-turn ~150 km away → suspicious."""
        stop = {"lat": 10.0, "lng": 106.0, "normalized_text": "Stop A"}
        turn_row = {"Tọa độ": "11.0,107.0"}
        valid, suspicious = trip_pipeline.match_turnaround_to_stops([turn_row], [stop])
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(suspicious), 1)

    def test_unparseable_coord_is_suspicious(self):
        """Bad coordinate string → suspicious with reason field."""
        stop = {"lat": 10.0, "lng": 106.0, "normalized_text": "Stop A"}
        turn_row = {"Tọa độ": "not_a_coordinate"}
        valid, suspicious = trip_pipeline.match_turnaround_to_stops([turn_row], [stop])
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(suspicious), 1)
        self.assertIn("reason", suspicious[0])

    def test_empty_turn_rows(self):
        stop = {"lat": 10.0, "lng": 106.0}
        valid, suspicious = trip_pipeline.match_turnaround_to_stops([], [stop])
        self.assertEqual(valid, [])
        self.assertEqual(suspicious, [])

    def test_no_geocoded_stops_all_suspicious(self):
        """Stops without lat/lng → all u-turns suspicious."""
        stop = {"lat": None, "lng": None}
        turn_row = {"Tọa độ": "10.0,106.0"}
        valid, suspicious = trip_pipeline.match_turnaround_to_stops([turn_row], [stop])
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(suspicious), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
