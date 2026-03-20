# vtracking_trip_system_bundle — Developer Guide

## Setup

```bash
# 1. Install runtime dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and set GOOGLE_MAPS_API_KEY (optional — falls back to OSRM)

# 3. Place raw data files
#    data/raw/vtracking/62C<PLATE>_<YYYYMMDD>.xlsx
#    (one file per plate per day)
```

## Environment Variables

See `.env.example` for the full list with explanations.  Key variables:

| Variable | Default | Notes |
|---|---|---|
| `GOOGLE_MAPS_API_KEY` | _(empty)_ | Required for Google APIs; falls back to OSRM if absent |
| `MAP_MATCH_PROVIDER` | `osrm` | `osrm` or `google_roads` |
| `ROUTE_PROVIDER` | `google_directions` | `google_directions`, `google_routes`, or `osrm` |
| `CORRIDOR_BUFFER_M` | `200.0` | Lateral tolerance in metres |
| `ENABLE_RETURN_LEG` | `false` | Set `true` to score the return trip in compliance% |

## Run

```bash
# Process all plates for today's dispatch orders and write reports/
python trip_pipeline.py

# Process a specific day
python -c "from trip_pipeline import main; main(day_code='20240315')"
```

## Tests

```bash
# Run all unit tests (no network, no real files)
python -m unittest discover -s tests -v

# Run a specific suite
python -m unittest tests.test_trip_pipeline -v
python -m unittest tests.test_maps_config -v
python -m unittest tests.test_delivery_scope -v
python -m unittest tests.test_google_roads_service -v
python -m unittest tests.test_google_routes_service -v
```

## Offline Smoke Tests (no API key needed)

Each service module has a built-in `__main__` block for quick sanity checks:

```bash
python corridor_builder.py        # offline unit assertions
python deviation_scorer.py        # 2-leg scoring assertions
python google_roads_service.py    # offline haversine + chunking tests
python google_routes_service.py   # offline request body tests
```

## Module Overview

```
trip_pipeline.py          — orchestrator: load → geocode → analyze → export
vtracking_tool.py         — GPS helpers, OSRM map-matching, route fetching
corridor_builder.py       — multi-route feasible corridor per trip leg
trace_reconstructor.py    — leg segmentation + dwell detection
deviation_scorer.py       — corridor-based compliance scoring
maps_config.py            — single source of truth for all env config
geocode_service.py        — Google Geocoding with MD5-keyed JSON cache
google_roads_service.py   — Google Roads API (snap GPS to road)
google_routes_service.py  — Google Routes API v2 (compute route alternatives)
```

## Key Design Decisions

- **Delivery scope**: `expected_distance_km` and `detour_ratio` only cover
  origin → stop1 → … → stopN.  The return leg never inflates these metrics.
- **Fallback chain** (routes): Google Routes v2 → Google Directions → OSRM → straight-line
- **Fallback chain** (map-matching): Google Roads → OSRM
- **No real API calls in tests**: all network calls are mocked.
