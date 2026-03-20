"""
tests/test_maps_config.py
--------------------------
Unit tests for maps_config.py.

All tests override environment variables via os.environ patching —
no real API keys or network calls involved.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import maps_config


class TestGetApiKey(unittest.TestCase):
    def test_returns_none_when_not_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_MAPS_API_KEY", None)
            self.assertIsNone(maps_config.get_api_key())

    def test_returns_none_when_empty_string(self):
        with patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": ""}):
            self.assertIsNone(maps_config.get_api_key())

    def test_returns_none_when_whitespace_only(self):
        with patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "   "}):
            self.assertIsNone(maps_config.get_api_key())

    def test_returns_key_when_set(self):
        with patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "AIzaSyTEST"}):
            self.assertEqual(maps_config.get_api_key(), "AIzaSyTEST")

    def test_strips_whitespace(self):
        with patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "  AIzaSyTEST  "}):
            self.assertEqual(maps_config.get_api_key(), "AIzaSyTEST")

    def test_has_api_key_false_when_not_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_MAPS_API_KEY", None)
            self.assertFalse(maps_config.has_api_key())

    def test_has_api_key_true_when_set(self):
        with patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "AIzaSyTEST"}):
            self.assertTrue(maps_config.has_api_key())


class TestGetMapMatchProvider(unittest.TestCase):
    def test_default_is_osrm(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAP_MATCH_PROVIDER", None)
            self.assertEqual(maps_config.get_map_match_provider(), "osrm")

    def test_osrm_explicit(self):
        with patch.dict(os.environ, {"MAP_MATCH_PROVIDER": "osrm"}):
            self.assertEqual(maps_config.get_map_match_provider(), "osrm")

    def test_google_roads_accepted(self):
        with patch.dict(os.environ, {"MAP_MATCH_PROVIDER": "google_roads"}):
            self.assertEqual(maps_config.get_map_match_provider(), "google_roads")

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"MAP_MATCH_PROVIDER": "OSRM"}):
            self.assertEqual(maps_config.get_map_match_provider(), "osrm")

    def test_invalid_value_falls_back_to_osrm(self):
        with patch.dict(os.environ, {"MAP_MATCH_PROVIDER": "invalid_provider"}):
            self.assertEqual(maps_config.get_map_match_provider(), "osrm")


class TestGetRouteProvider(unittest.TestCase):
    def test_default_is_google_directions(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROUTE_PROVIDER", None)
            self.assertEqual(maps_config.get_route_provider(), "google_directions")

    def test_google_directions_explicit(self):
        with patch.dict(os.environ, {"ROUTE_PROVIDER": "google_directions"}):
            self.assertEqual(maps_config.get_route_provider(), "google_directions")

    def test_google_routes_accepted(self):
        with patch.dict(os.environ, {"ROUTE_PROVIDER": "google_routes"}):
            self.assertEqual(maps_config.get_route_provider(), "google_routes")

    def test_osrm_accepted(self):
        with patch.dict(os.environ, {"ROUTE_PROVIDER": "osrm"}):
            self.assertEqual(maps_config.get_route_provider(), "osrm")

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"ROUTE_PROVIDER": "GOOGLE_ROUTES"}):
            self.assertEqual(maps_config.get_route_provider(), "google_routes")

    def test_invalid_value_falls_back_to_google_directions(self):
        with patch.dict(os.environ, {"ROUTE_PROVIDER": "unknown"}):
            self.assertEqual(maps_config.get_route_provider(), "google_directions")


class TestGetCorridorBufferM(unittest.TestCase):
    def test_default_is_200(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORRIDOR_BUFFER_M", None)
            self.assertEqual(maps_config.get_corridor_buffer_m(), 200.0)

    def test_custom_positive_value(self):
        with patch.dict(os.environ, {"CORRIDOR_BUFFER_M": "500"}):
            self.assertEqual(maps_config.get_corridor_buffer_m(), 500.0)

    def test_float_value(self):
        with patch.dict(os.environ, {"CORRIDOR_BUFFER_M": "150.5"}):
            self.assertAlmostEqual(maps_config.get_corridor_buffer_m(), 150.5)

    def test_zero_falls_back_to_default(self):
        with patch.dict(os.environ, {"CORRIDOR_BUFFER_M": "0"}):
            self.assertEqual(maps_config.get_corridor_buffer_m(), 200.0)

    def test_negative_falls_back_to_default(self):
        with patch.dict(os.environ, {"CORRIDOR_BUFFER_M": "-100"}):
            self.assertEqual(maps_config.get_corridor_buffer_m(), 200.0)

    def test_invalid_string_falls_back_to_default(self):
        with patch.dict(os.environ, {"CORRIDOR_BUFFER_M": "not_a_number"}):
            self.assertEqual(maps_config.get_corridor_buffer_m(), 200.0)

    def test_empty_string_falls_back_to_default(self):
        with patch.dict(os.environ, {"CORRIDOR_BUFFER_M": ""}):
            self.assertEqual(maps_config.get_corridor_buffer_m(), 200.0)


class TestGetDepotOrigin(unittest.TestCase):
    _DEFAULT_LAT = 10.802417
    _DEFAULT_LNG = 106.501501

    def test_default_values(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEPOT_LAT", None)
            os.environ.pop("DEPOT_LNG", None)
            lat, lng = maps_config.get_depot_origin()
            self.assertAlmostEqual(lat, self._DEFAULT_LAT, places=5)
            self.assertAlmostEqual(lng, self._DEFAULT_LNG, places=5)

    def test_custom_values(self):
        with patch.dict(os.environ, {"DEPOT_LAT": "10.5", "DEPOT_LNG": "106.9"}):
            lat, lng = maps_config.get_depot_origin()
            self.assertAlmostEqual(lat, 10.5, places=5)
            self.assertAlmostEqual(lng, 106.9, places=5)

    def test_missing_lat_falls_back_to_default(self):
        with patch.dict(os.environ, {"DEPOT_LNG": "106.9"}, clear=False):
            os.environ.pop("DEPOT_LAT", None)
            lat, lng = maps_config.get_depot_origin()
            self.assertAlmostEqual(lat, self._DEFAULT_LAT, places=5)

    def test_missing_lng_falls_back_to_default(self):
        with patch.dict(os.environ, {"DEPOT_LAT": "10.5"}, clear=False):
            os.environ.pop("DEPOT_LNG", None)
            lat, lng = maps_config.get_depot_origin()
            self.assertAlmostEqual(lat, self._DEFAULT_LAT, places=5)

    def test_invalid_lat_falls_back_to_default(self):
        with patch.dict(os.environ, {"DEPOT_LAT": "not_a_number", "DEPOT_LNG": "106.9"}):
            lat, lng = maps_config.get_depot_origin()
            self.assertAlmostEqual(lat, self._DEFAULT_LAT, places=5)

    def test_returns_tuple(self):
        result = maps_config.get_depot_origin()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestBooleanFlags(unittest.TestCase):
    """Test is_return_leg_enabled / is_avoid_* flags."""

    def _env_test(self, env_var: str, func, default_val: bool):
        # Default (not set)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(env_var, None)
            self.assertEqual(func(), default_val, f"Default for {env_var} should be {default_val}")

        # Explicit true
        with patch.dict(os.environ, {env_var: "true"}):
            self.assertTrue(func(), f"{env_var}=true should return True")

        # Explicit false
        with patch.dict(os.environ, {env_var: "false"}):
            self.assertFalse(func(), f"{env_var}=false should return False")

        # Case insensitive
        with patch.dict(os.environ, {env_var: "TRUE"}):
            self.assertTrue(func(), f"{env_var}=TRUE should return True")

        # Invalid value → default
        with patch.dict(os.environ, {env_var: "yes"}):
            self.assertEqual(func(), default_val, f"Invalid {env_var} value should fall back to default")

    def test_return_leg_default_true(self):
        self._env_test("ENABLE_RETURN_LEG", maps_config.is_return_leg_enabled, True)

    def test_avoid_tolls_default_false(self):
        self._env_test("ENABLE_AVOID_TOLLS_ROUTE", maps_config.is_avoid_tolls_enabled, False)

    def test_avoid_highways_default_false(self):
        self._env_test("ENABLE_AVOID_HIGHWAYS_ROUTE", maps_config.is_avoid_highways_enabled, False)

    def test_avoid_ferries_default_false(self):
        self._env_test("ENABLE_AVOID_FERRIES_ROUTE", maps_config.is_avoid_ferries_enabled, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
