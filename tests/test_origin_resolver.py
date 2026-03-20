"""
tests/test_origin_resolver.py
------------------------------
Unit tests for origin_resolver.resolve_trip_origin().

Tests cover:
  Case 1 — plate NOT in registry → returns global_default depot coordinate
  Case 2 — plate IN registry     → returns vehicle_registry coordinate
  Case 3 — registry file absent  → behaves as if empty (global_default)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import origin_resolver

_DEPOT = (10.802417, 106.501501)
_BRANCH = (10.555, 106.444)


class TestResolveGlobalDefault(unittest.TestCase):
    """Case 1: plate has no custom entry — must return global depot."""

    def setUp(self):
        origin_resolver.reload_registry()

    def test_unknown_plate_uses_depot_coord(self):
        with unittest.mock.patch.object(origin_resolver, "_REGISTRY_PATH",
                                        _nonexistent_path()):
            origin_resolver.reload_registry()
            res = origin_resolver.resolve_trip_origin("62C99999")
        self.assertAlmostEqual(res.lat, _DEPOT[0], places=5)
        self.assertAlmostEqual(res.lng, _DEPOT[1], places=5)

    def test_unknown_plate_source_is_global_default(self):
        with unittest.mock.patch.object(origin_resolver, "_REGISTRY_PATH",
                                        _nonexistent_path()):
            origin_resolver.reload_registry()
            res = origin_resolver.resolve_trip_origin("62C99999")
        self.assertEqual(res.source, "global_default")

    def test_as_latlng_returns_tuple(self):
        with unittest.mock.patch.object(origin_resolver, "_REGISTRY_PATH",
                                        _nonexistent_path()):
            origin_resolver.reload_registry()
            res = origin_resolver.resolve_trip_origin("62C99999")
        self.assertIsInstance(res.as_latlng(), tuple)
        self.assertEqual(len(res.as_latlng()), 2)


class TestResolveVehicleRegistry(unittest.TestCase):
    """Case 2: plate exists in vehicle_origins.json."""

    def setUp(self):
        origin_resolver.reload_registry()

    def _with_registry(self, data: dict):
        """Write data to a temp JSON file and point the resolver at it."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(data, tmp)
        tmp.close()
        return tmp.name

    def test_registered_plate_returns_custom_coord(self):
        path = self._with_registry({"62C12345": [_BRANCH[0], _BRANCH[1]]})
        try:
            with unittest.mock.patch.object(origin_resolver, "_REGISTRY_PATH",
                                            _path_obj(path)):
                origin_resolver.reload_registry()
                res = origin_resolver.resolve_trip_origin("62C12345")
            self.assertAlmostEqual(res.lat, _BRANCH[0], places=5)
            self.assertAlmostEqual(res.lng, _BRANCH[1], places=5)
        finally:
            os.unlink(path)

    def test_registered_plate_source_is_vehicle_registry(self):
        path = self._with_registry({"62C12345": [_BRANCH[0], _BRANCH[1]]})
        try:
            with unittest.mock.patch.object(origin_resolver, "_REGISTRY_PATH",
                                            _path_obj(path)):
                origin_resolver.reload_registry()
                res = origin_resolver.resolve_trip_origin("62C12345")
            self.assertEqual(res.source, "vehicle_registry")
        finally:
            os.unlink(path)

    def test_unregistered_plate_in_populated_registry_falls_back(self):
        """Registry has other plates but not this one — must still fall back."""
        path = self._with_registry({"62C00001": [_BRANCH[0], _BRANCH[1]]})
        try:
            with unittest.mock.patch.object(origin_resolver, "_REGISTRY_PATH",
                                            _path_obj(path)):
                origin_resolver.reload_registry()
                res = origin_resolver.resolve_trip_origin("62C99999")
            self.assertEqual(res.source, "global_default")
        finally:
            os.unlink(path)

    def test_malformed_coord_entry_is_skipped(self):
        """A malformed entry must not crash — the plate falls back to global_default."""
        path = self._with_registry({"62C12345": "bad_value"})
        try:
            with unittest.mock.patch.object(origin_resolver, "_REGISTRY_PATH",
                                            _path_obj(path)):
                origin_resolver.reload_registry()
                res = origin_resolver.resolve_trip_origin("62C12345")
            self.assertEqual(res.source, "global_default")
        finally:
            os.unlink(path)


class TestRegistryFileMissing(unittest.TestCase):
    """Case 3: vehicle_origins.json does not exist."""

    def setUp(self):
        origin_resolver.reload_registry()

    def test_missing_file_does_not_crash(self):
        with unittest.mock.patch.object(origin_resolver, "_REGISTRY_PATH",
                                        _nonexistent_path()):
            origin_resolver.reload_registry()
            res = origin_resolver.resolve_trip_origin("62C12345")
        self.assertEqual(res.source, "global_default")

    def test_missing_file_returns_depot_coord(self):
        with unittest.mock.patch.object(origin_resolver, "_REGISTRY_PATH",
                                        _nonexistent_path()):
            origin_resolver.reload_registry()
            res = origin_resolver.resolve_trip_origin("62C12345")
        self.assertAlmostEqual(res.lat, _DEPOT[0], places=5)
        self.assertAlmostEqual(res.lng, _DEPOT[1], places=5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import unittest.mock
from pathlib import Path


def _nonexistent_path():
    """A Path that is guaranteed not to exist."""
    return Path("/nonexistent/vehicle_origins.json")


def _path_obj(s: str):
    return Path(s)
