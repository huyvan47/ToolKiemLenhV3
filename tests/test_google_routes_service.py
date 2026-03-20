"""
tests/test_google_routes_service.py
------------------------------------
Unit tests for google_routes_service and _fetch_google_routes_v2 integration
in corridor_builder.

All HTTP calls are mocked — no real network requests are made.
No API key is required to run these tests.

Run:
    python -m pytest tests/test_google_routes_service.py -v
    # or
    python -m unittest tests.test_google_routes_service
"""
from __future__ import annotations

import json
import sys
import os
import unittest
from typing import List, Tuple
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google_routes_service
from google_routes_service import (
    GoogleRoutesError,
    RoutesApiRoute,
    _build_request_body,
    _FIELD_MASK,
    fetch_routes,
)

LatLng = Tuple[float, float]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ORIGIN: LatLng = (10.1, 106.1)
DEST: LatLng   = (10.2, 106.2)
FAKE_KEY = "test_api_key_abc123"

# Minimal valid polyline for two points (generated offline, precision-5).
# We will use a real encoded polyline so _polyline_codec.decode works correctly.
# polyline.encode([(10.1, 106.1), (10.15, 106.15), (10.2, 106.2)]) = '_dpbBm~ueS...'
# To avoid importing polyline in the test, we mock the decoded result separately.
_ENCODED_POLY_STUB = "encodedStub123"


def _make_mock_response(data: dict, status_code: int = 200) -> MagicMock:
    """Return a mock requests.Response."""
    mock = MagicMock()
    mock.json.return_value = data
    mock.status_code = status_code
    if status_code >= 400:
        from requests.exceptions import HTTPError
        mock.raise_for_status.side_effect = HTTPError(f"HTTP {status_code}")
    else:
        mock.raise_for_status.return_value = None
    return mock


def _routes_response(*distances: float, poly: str = _ENCODED_POLY_STUB) -> dict:
    """Build a minimal valid computeRoutes response."""
    return {
        "routes": [
            {
                "distanceMeters": int(d),
                "polyline": {"encodedPolyline": poly},
            }
            for d in distances
        ]
    }


# ---------------------------------------------------------------------------
# _build_request_body
# ---------------------------------------------------------------------------

class TestBuildRequestBody(unittest.TestCase):

    def _body(self, **kwargs) -> dict:
        defaults = dict(
            compute_alternatives=True,
            avoid_tolls=False,
            avoid_highways=False,
            avoid_ferries=False,
            routing_preference="TRAFFIC_UNAWARE",
        )
        defaults.update(kwargs)
        return _build_request_body(ORIGIN, DEST, **defaults)

    def test_origin_coords(self):
        body = self._body()
        ll = body["origin"]["location"]["latLng"]
        self.assertAlmostEqual(ll["latitude"], ORIGIN[0])
        self.assertAlmostEqual(ll["longitude"], ORIGIN[1])

    def test_destination_coords(self):
        body = self._body()
        ll = body["destination"]["location"]["latLng"]
        self.assertAlmostEqual(ll["latitude"], DEST[0])
        self.assertAlmostEqual(ll["longitude"], DEST[1])

    def test_travel_mode_is_drive(self):
        self.assertEqual(self._body()["travelMode"], "DRIVE")

    def test_compute_alternatives_true(self):
        self.assertTrue(self._body(compute_alternatives=True)["computeAlternativeRoutes"])

    def test_compute_alternatives_false(self):
        self.assertFalse(self._body(compute_alternatives=False)["computeAlternativeRoutes"])

    def test_avoid_tolls(self):
        body = self._body(avoid_tolls=True)
        self.assertTrue(body["routeModifiers"]["avoidTolls"])
        self.assertFalse(body["routeModifiers"]["avoidHighways"])
        self.assertFalse(body["routeModifiers"]["avoidFerries"])

    def test_avoid_highways(self):
        body = self._body(avoid_highways=True)
        self.assertFalse(body["routeModifiers"]["avoidTolls"])
        self.assertTrue(body["routeModifiers"]["avoidHighways"])

    def test_avoid_ferries(self):
        body = self._body(avoid_ferries=True)
        self.assertTrue(body["routeModifiers"]["avoidFerries"])

    def test_routing_preference_forwarded(self):
        body = self._body(routing_preference="TRAFFIC_AWARE")
        self.assertEqual(body["routingPreference"], "TRAFFIC_AWARE")

    def test_polyline_quality_overview(self):
        self.assertEqual(self._body()["polylineQuality"], "OVERVIEW")


# ---------------------------------------------------------------------------
# fetch_routes — HTTP + parsing
# ---------------------------------------------------------------------------

class TestFetchRoutes(unittest.TestCase):

    def _decoded_coords(self) -> List[LatLng]:
        return [(10.1, 106.1), (10.15, 106.15), (10.2, 106.2)]

    @patch("google_routes_service._polyline_codec")
    @patch("google_routes_service._SESSION")
    def test_success_single_route(self, mock_session, mock_poly):
        mock_session.post.return_value = _make_mock_response(
            _routes_response(5000.0)
        )
        mock_poly.decode.return_value = self._decoded_coords()

        results = fetch_routes(ORIGIN, DEST, FAKE_KEY)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RoutesApiRoute)
        self.assertAlmostEqual(results[0].distance_m, 5000.0)
        self.assertEqual(results[0].coords, self._decoded_coords())
        self.assertEqual(results[0].source, "google_routes_v2_0")

    @patch("google_routes_service._polyline_codec")
    @patch("google_routes_service._SESSION")
    def test_success_three_alternatives(self, mock_session, mock_poly):
        mock_session.post.return_value = _make_mock_response(
            _routes_response(5000.0, 6000.0, 7000.0)
        )
        mock_poly.decode.return_value = self._decoded_coords()

        results = fetch_routes(ORIGIN, DEST, FAKE_KEY)

        self.assertEqual(len(results), 3)
        sources = [r.source for r in results]
        self.assertEqual(sources, [
            "google_routes_v2_0",
            "google_routes_v2_1",
            "google_routes_v2_2",
        ])
        self.assertAlmostEqual(results[1].distance_m, 6000.0)

    @patch("google_routes_service._polyline_codec")
    @patch("google_routes_service._SESSION")
    def test_custom_source_prefix(self, mock_session, mock_poly):
        mock_session.post.return_value = _make_mock_response(
            _routes_response(5000.0)
        )
        mock_poly.decode.return_value = self._decoded_coords()

        results = fetch_routes(ORIGIN, DEST, FAKE_KEY, source_prefix="google_routes_v2_notolls")

        self.assertEqual(results[0].source, "google_routes_v2_notolls_0")

    @patch("google_routes_service._SESSION")
    def test_api_error_in_response_raises(self, mock_session):
        mock_session.post.return_value = _make_mock_response({
            "error": {"code": 400, "message": "Missing field mask", "status": "INVALID_ARGUMENT"}
        })
        with self.assertRaises(GoogleRoutesError) as ctx:
            fetch_routes(ORIGIN, DEST, FAKE_KEY)
        self.assertIn("400", str(ctx.exception))
        self.assertIn("Missing field mask", str(ctx.exception))

    @patch("google_routes_service._SESSION")
    def test_http_error_raises(self, mock_session):
        mock_session.post.return_value = _make_mock_response({}, status_code=403)
        with self.assertRaises(GoogleRoutesError) as ctx:
            fetch_routes(ORIGIN, DEST, FAKE_KEY)
        self.assertIn("HTTP error", str(ctx.exception))

    @patch("google_routes_service._SESSION")
    def test_network_exception_raises(self, mock_session):
        import requests as _req
        mock_session.post.side_effect = _req.ConnectionError("connection refused")
        with self.assertRaises(GoogleRoutesError) as ctx:
            fetch_routes(ORIGIN, DEST, FAKE_KEY)
        self.assertIn("HTTP error", str(ctx.exception))

    @patch("google_routes_service._SESSION")
    def test_empty_routes_list_raises(self, mock_session):
        mock_session.post.return_value = _make_mock_response({"routes": []})
        with self.assertRaises(GoogleRoutesError) as ctx:
            fetch_routes(ORIGIN, DEST, FAKE_KEY)
        self.assertIn("no routes", str(ctx.exception))

    @patch("google_routes_service._SESSION")
    def test_missing_routes_key_raises(self, mock_session):
        mock_session.post.return_value = _make_mock_response({})
        with self.assertRaises(GoogleRoutesError) as ctx:
            fetch_routes(ORIGIN, DEST, FAKE_KEY)
        self.assertIn("no routes", str(ctx.exception))

    @patch("google_routes_service._polyline_codec")
    @patch("google_routes_service._SESSION")
    def test_route_missing_polyline_skipped_others_kept(self, mock_session, mock_poly):
        """Route without encodedPolyline is skipped; parseable routes are returned."""
        payload = {
            "routes": [
                {"distanceMeters": 5000, "polyline": {"encodedPolyline": _ENCODED_POLY_STUB}},
                {"distanceMeters": 6000},   # missing polyline
                {"distanceMeters": 7000, "polyline": {"encodedPolyline": _ENCODED_POLY_STUB}},
            ]
        }
        mock_session.post.return_value = _make_mock_response(payload)
        mock_poly.decode.return_value = self._decoded_coords()

        results = fetch_routes(ORIGIN, DEST, FAKE_KEY)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].source, "google_routes_v2_0")
        self.assertEqual(results[1].source, "google_routes_v2_2")

    @patch("google_routes_service._polyline_codec")
    @patch("google_routes_service._SESSION")
    def test_all_routes_unparseable_raises(self, mock_session, mock_poly):
        """If every route is malformed, GoogleRoutesError is raised."""
        payload = {"routes": [{"distanceMeters": 5000}]}  # no polyline
        mock_session.post.return_value = _make_mock_response(payload)

        with self.assertRaises(GoogleRoutesError) as ctx:
            fetch_routes(ORIGIN, DEST, FAKE_KEY)
        self.assertIn("parseable", str(ctx.exception))

    @patch("google_routes_service._polyline_codec")
    @patch("google_routes_service._SESSION")
    def test_request_headers_correct(self, mock_session, mock_poly):
        """X-Goog-Api-Key and X-Goog-FieldMask headers must be set."""
        mock_session.post.return_value = _make_mock_response(_routes_response(5000.0))
        mock_poly.decode.return_value = self._decoded_coords()

        fetch_routes(ORIGIN, DEST, FAKE_KEY)

        _, kwargs = mock_session.post.call_args
        headers = kwargs.get("headers", {})
        self.assertEqual(headers.get("X-Goog-Api-Key"), FAKE_KEY)
        self.assertEqual(headers.get("X-Goog-FieldMask"), _FIELD_MASK)
        self.assertEqual(headers.get("Content-Type"), "application/json")

    @patch("google_routes_service._polyline_codec")
    @patch("google_routes_service._SESSION")
    def test_avoid_tolls_in_request_body(self, mock_session, mock_poly):
        """avoid_tolls=True must appear in routeModifiers of the POST body."""
        mock_session.post.return_value = _make_mock_response(_routes_response(5000.0))
        mock_poly.decode.return_value = self._decoded_coords()

        fetch_routes(ORIGIN, DEST, FAKE_KEY, avoid_tolls=True)

        _, kwargs = mock_session.post.call_args
        body = kwargs.get("json", {})
        self.assertTrue(body["routeModifiers"]["avoidTolls"])
        self.assertFalse(body["routeModifiers"]["avoidHighways"])

    @patch("google_routes_service._polyline_codec")
    @patch("google_routes_service._SESSION")
    def test_compute_alternatives_false_in_body(self, mock_session, mock_poly):
        mock_session.post.return_value = _make_mock_response(_routes_response(5000.0))
        mock_poly.decode.return_value = self._decoded_coords()

        fetch_routes(ORIGIN, DEST, FAKE_KEY, compute_alternatives=False)

        _, kwargs = mock_session.post.call_args
        body = kwargs.get("json", {})
        self.assertFalse(body["computeAlternativeRoutes"])


# ---------------------------------------------------------------------------
# _fetch_google_routes_v2 in corridor_builder
# ---------------------------------------------------------------------------

class TestFetchGoogleRoutesV2(unittest.TestCase):
    """
    Tests for corridor_builder._fetch_google_routes_v2().
    Patches maps_config and google_routes_service at corridor_builder scope.
    """

    def _call(self):
        from corridor_builder import _fetch_google_routes_v2
        return _fetch_google_routes_v2(ORIGIN, DEST)

    @patch("corridor_builder.maps_config")
    def test_skipped_when_provider_not_google_routes(self, mock_cfg):
        mock_cfg.get_route_provider.return_value = "google_directions"
        result = self._call()
        self.assertEqual(result, [])

    @patch("corridor_builder.maps_config")
    def test_skipped_when_provider_is_osrm(self, mock_cfg):
        mock_cfg.get_route_provider.return_value = "osrm"
        result = self._call()
        self.assertEqual(result, [])

    @patch("corridor_builder.maps_config")
    def test_skipped_when_no_api_key(self, mock_cfg):
        mock_cfg.get_route_provider.return_value = "google_routes"
        mock_cfg.get_api_key.return_value = None
        result = self._call()
        self.assertEqual(result, [])

    @patch("corridor_builder.google_routes_service")
    @patch("corridor_builder.maps_config")
    def test_success_returns_route_options(self, mock_cfg, mock_svc):
        from corridor_builder import RouteOption
        mock_cfg.get_route_provider.return_value = "google_routes"
        mock_cfg.get_api_key.return_value = FAKE_KEY
        mock_cfg.is_avoid_tolls_enabled.return_value = False
        mock_cfg.is_avoid_highways_enabled.return_value = False
        mock_cfg.is_avoid_ferries_enabled.return_value = False

        mock_svc.fetch_routes.return_value = [
            RoutesApiRoute(source="google_routes_v2_0",
                           coords=[(10.1, 106.1), (10.2, 106.2)],
                           distance_m=5000.0),
            RoutesApiRoute(source="google_routes_v2_1",
                           coords=[(10.1, 106.1), (10.15, 106.15), (10.2, 106.2)],
                           distance_m=6000.0),
        ]
        mock_svc.GoogleRoutesError = GoogleRoutesError

        result = self._call()

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], RouteOption)
        self.assertEqual(result[0].source, "google_routes_v2_0")
        self.assertAlmostEqual(result[0].distance_m, 5000.0)
        self.assertEqual(result[1].source, "google_routes_v2_1")

    @patch("corridor_builder.google_routes_service")
    @patch("corridor_builder.maps_config")
    def test_api_error_returns_empty_not_raises(self, mock_cfg, mock_svc):
        """GoogleRoutesError must be swallowed — returns [] so caller falls through."""
        mock_cfg.get_route_provider.return_value = "google_routes"
        mock_cfg.get_api_key.return_value = FAKE_KEY
        mock_cfg.is_avoid_tolls_enabled.return_value = False
        mock_cfg.is_avoid_highways_enabled.return_value = False
        mock_cfg.is_avoid_ferries_enabled.return_value = False
        mock_svc.GoogleRoutesError = GoogleRoutesError
        mock_svc.fetch_routes.side_effect = GoogleRoutesError("quota exceeded")

        result = self._call()

        self.assertEqual(result, [])

    @patch("corridor_builder.google_routes_service")
    @patch("corridor_builder.maps_config")
    def test_unexpected_exception_returns_empty(self, mock_cfg, mock_svc):
        mock_cfg.get_route_provider.return_value = "google_routes"
        mock_cfg.get_api_key.return_value = FAKE_KEY
        mock_cfg.is_avoid_tolls_enabled.return_value = False
        mock_cfg.is_avoid_highways_enabled.return_value = False
        mock_cfg.is_avoid_ferries_enabled.return_value = False
        mock_svc.GoogleRoutesError = GoogleRoutesError
        mock_svc.fetch_routes.side_effect = RuntimeError("unexpected")

        result = self._call()

        self.assertEqual(result, [])

    @patch("corridor_builder.google_routes_service")
    @patch("corridor_builder.maps_config")
    def test_avoid_tolls_flag_triggers_extra_call(self, mock_cfg, mock_svc):
        """ENABLE_AVOID_TOLLS_ROUTE=true must generate a second fetch_routes call."""
        from corridor_builder import RouteOption
        mock_cfg.get_route_provider.return_value = "google_routes"
        mock_cfg.get_api_key.return_value = FAKE_KEY
        mock_cfg.is_avoid_tolls_enabled.return_value = True
        mock_cfg.is_avoid_highways_enabled.return_value = False
        mock_cfg.is_avoid_ferries_enabled.return_value = False
        mock_svc.GoogleRoutesError = GoogleRoutesError

        base_route = RoutesApiRoute("google_routes_v2_0",
                                    [(10.1, 106.1), (10.2, 106.2)], 5000.0)
        tolls_route = RoutesApiRoute("google_routes_v2_notolls_0",
                                     [(10.1, 106.1), (10.18, 106.2)], 5500.0)
        mock_svc.fetch_routes.side_effect = [[base_route], [tolls_route]]

        result = self._call()

        self.assertEqual(mock_svc.fetch_routes.call_count, 2)
        self.assertEqual(len(result), 2)
        sources = [r.source for r in result]
        self.assertIn("google_routes_v2_0", sources)
        self.assertIn("google_routes_v2_notolls_0", sources)

        # Verify second call used avoid_tolls=True.
        second_call_kwargs = mock_svc.fetch_routes.call_args_list[1][1]
        self.assertTrue(second_call_kwargs.get("avoid_tolls"))
        self.assertEqual(second_call_kwargs.get("source_prefix"), "google_routes_v2_notolls")

    @patch("corridor_builder.google_routes_service")
    @patch("corridor_builder.maps_config")
    def test_no_modifier_flags_single_call(self, mock_cfg, mock_svc):
        """When all avoid_* flags are False, only one fetch_routes call is made."""
        mock_cfg.get_route_provider.return_value = "google_routes"
        mock_cfg.get_api_key.return_value = FAKE_KEY
        mock_cfg.is_avoid_tolls_enabled.return_value = False
        mock_cfg.is_avoid_highways_enabled.return_value = False
        mock_cfg.is_avoid_ferries_enabled.return_value = False
        mock_svc.GoogleRoutesError = GoogleRoutesError
        mock_svc.fetch_routes.return_value = [
            RoutesApiRoute("google_routes_v2_0", [(10.1, 106.1)], 5000.0)
        ]

        self._call()

        self.assertEqual(mock_svc.fetch_routes.call_count, 1)


# ---------------------------------------------------------------------------
# build_leg_corridor — integration: v2 routes added to corridor
# ---------------------------------------------------------------------------

class TestBuildLegCorridorWithV2(unittest.TestCase):

    STUB_COORDS = [(10.1, 106.1), (10.15, 106.15), (10.2, 106.2)]

    @patch("corridor_builder._fetch_osrm_routes")
    @patch("corridor_builder._fetch_google_routes")
    @patch("corridor_builder._fetch_google_routes_v2")
    def test_v2_routes_included_in_corridor(
        self, mock_v2, mock_directions, mock_osrm
    ):
        """Routes from all three sources must all appear in the corridor."""
        from corridor_builder import RouteOption, build_leg_corridor

        mock_v2.return_value = [
            RouteOption("google_routes_v2_0", self.STUB_COORDS, 5000.0)
        ]
        mock_directions.return_value = [
            RouteOption("google_0", self.STUB_COORDS, 5100.0)
        ]
        mock_osrm.return_value = [
            RouteOption("osrm_0", self.STUB_COORDS, 5200.0)
        ]

        leg = build_leg_corridor(leg_idx=0, origin=ORIGIN, dest=DEST)

        sources = [r.source for r in leg.routes]
        self.assertIn("google_routes_v2_0", sources)
        self.assertIn("google_0", sources)
        self.assertIn("osrm_0", sources)
        self.assertEqual(len(leg.routes), 3)

    @patch("corridor_builder._fetch_osrm_routes")
    @patch("corridor_builder._fetch_google_routes")
    @patch("corridor_builder._fetch_google_routes_v2")
    def test_v2_empty_fallback_to_directions_and_osrm(
        self, mock_v2, mock_directions, mock_osrm
    ):
        """When v2 returns [], corridor is built from Directions + OSRM."""
        from corridor_builder import RouteOption, build_leg_corridor

        mock_v2.return_value = []
        mock_directions.return_value = [
            RouteOption("google_0", self.STUB_COORDS, 5100.0)
        ]
        mock_osrm.return_value = [
            RouteOption("osrm_0", self.STUB_COORDS, 5200.0)
        ]

        leg = build_leg_corridor(leg_idx=0, origin=ORIGIN, dest=DEST)

        sources = [r.source for r in leg.routes]
        self.assertNotIn("google_routes_v2_0", sources)
        self.assertIn("google_0", sources)
        self.assertIn("osrm_0", sources)

    @patch("builtins.print")   # suppress Vietnamese print() on Windows cp1252 consoles
    @patch("corridor_builder._fetch_osrm_routes")
    @patch("corridor_builder._fetch_google_routes")
    @patch("corridor_builder._fetch_google_routes_v2")
    def test_all_sources_empty_uses_straight_line_fallback(
        self, mock_v2, mock_directions, mock_osrm, _mock_print
    ):
        """When all three sources fail, corridor must still be built (straight-line)."""
        from corridor_builder import build_leg_corridor

        mock_v2.return_value = []
        mock_directions.return_value = []
        mock_osrm.return_value = []

        leg = build_leg_corridor(leg_idx=0, origin=ORIGIN, dest=DEST)

        self.assertEqual(len(leg.routes), 1)
        self.assertEqual(leg.routes[0].source, "fallback_straight")

    @patch("corridor_builder._fetch_osrm_routes")
    @patch("corridor_builder._fetch_google_routes")
    @patch("corridor_builder._fetch_google_routes_v2")
    def test_v2_called_before_directions(self, mock_v2, mock_directions, mock_osrm):
        """Call order: v2 first, then Directions, then OSRM."""
        from corridor_builder import RouteOption, build_leg_corridor
        call_order = []

        def track_v2(*a, **kw):
            call_order.append("v2")
            return [RouteOption("v2_0", self.STUB_COORDS, 5000.0)]

        def track_dir(*a, **kw):
            call_order.append("directions")
            return [RouteOption("dir_0", self.STUB_COORDS, 5100.0)]

        def track_osrm(*a, **kw):
            call_order.append("osrm")
            return [RouteOption("osrm_0", self.STUB_COORDS, 5200.0)]

        mock_v2.side_effect = track_v2
        mock_directions.side_effect = track_dir
        mock_osrm.side_effect = track_osrm

        build_leg_corridor(leg_idx=0, origin=ORIGIN, dest=DEST)

        self.assertEqual(call_order, ["v2", "directions", "osrm"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
