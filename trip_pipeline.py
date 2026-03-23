from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

import origin_resolver
from geocode_service import geocode_address_obj
from gpt_data import ChuanHoaDiaChiTrongFileLenh
from lenh_data import LayDuLieuFileLenh
from phat_hien_quay_dau_data import DiaChiNghiVanQuayDau
from utils import BienSoXeChoFileEpass
from VeEpassCuaChuyen import LayIndexVe, df as epass_df
from vtracking_tool import analyze_trip_corridor, analyze_trip_multi_stop, haversine, parse_coord
from stop_fallback_resolver import enrich_stops_with_vtracking_fallback


DATA_DIR = Path(__file__).resolve().parent / "data" / "raw" / "vtracking"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_trip_df(plate: str, day_code: Optional[str] = None) -> pd.DataFrame:
    if day_code is None:
        files = sorted(DATA_DIR.glob(f"62C{plate}_*.xlsx"))
        if not files:
            raise FileNotFoundError(f"Không tìm thấy file vtracking cho xe {plate}")
        file_path = files[-1]
    else:
        file_path = DATA_DIR / f"62C{plate}_{day_code}.xlsx"
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file {file_path.name}")
    return pd.read_excel(file_path)


def normalize_and_geocode_stops(
    addr_list,
    trip_df=None,
    apply_vtracking_fallback=True,
    fallback_match_level="medium",
):
    normalized = ChuanHoaDiaChiTrongFileLenh(list(addr_list)) or []
    results = []

    for item in normalized:
        try:
            geo = geocode_address_obj(item)
            row = {**item, **geo}
            results.append(row)
            print(
                f"[GEOCODE] status={geo.get('status')} "
                f"query={geo.get('query')} "
                f"lat={geo.get('lat')} lng={geo.get('lng')} "
                f"location_type={geo.get('location_type')} "
                f"partial_match={geo.get('partial_match')}"
            )
        except Exception as e:
            results.append({
                **item,
                "status": "ERROR",
                "lat": None,
                "lng": None,
                "geocode_error": str(e),
            })

    if apply_vtracking_fallback and trip_df is not None and not trip_df.empty:
        try:
            results = enrich_stops_with_vtracking_fallback(
                stops=results,
                df=trip_df,
                api_key=None,
                min_match_level=fallback_match_level,   # "medium" = khớp huyện+tỉnh là đủ
            )
            print("[VT-FALLBACK] applied to geocoded stops")

            usable = [
                s for s in results
                if not s.get("route_excluded")
                and s.get("lat") is not None
                and s.get("lng") is not None
            ]
            excluded = [s for s in results if s.get("route_excluded")]

            print(
                f"[VT-FALLBACK] total={len(results)} "
                f"| route_usable={len(usable)} "
                f"| excluded={len(excluded)}"
            )

            for s in excluded:
                label = s.get("raw_text") or s.get("normalized_text") or s.get("query")
                print(
                    f"[EXCLUDED] {label} | "
                    f"reason={s.get('exclude_reason')} | "
                    f"note={s.get('coord_resolution_note')}"
                )

        except Exception as e:
            print(f"[WARN] vtracking fallback failed: {e}")

    return results


def build_trip_window_from_df(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if "Thời gian" not in df.columns or df.empty:
        return None, None
    ts = pd.to_datetime(df["Thời gian"], errors="coerce", dayfirst=True).dropna()
    if ts.empty:
        return None, None
    return ts.min().strftime("%H:%M"), ts.max().strftime("%H:%M")


def detect_trip_window_from_origin(
    df: pd.DataFrame,
    origin: Optional[Tuple[float, float]],
    origin_radius_m: float = 700.0,
    return_radius_m: Optional[float] = 200.0,
    min_consecutive_points: int = 3,
    min_moving_speed_kmh: float = 5.0,
) -> dict:
    """
    Detect trip departure and return times using origin-zone crossing with
    movement-context awareness.

    Problem with naive zone-crossing
    ---------------------------------
    A truck may be parked overnight at a spot that is OUTSIDE the origin radius
    (e.g. a branch depot 844 m from the configured origin).  Without movement
    awareness, the very first GPS row (parked at 00:00) would be classified as
    "outside origin" and immediately confirm a false departure.

    Solution
    ---------
    Phase 1 — seek departure (movement-aware):
      Only MOVING rows (speed > min_moving_speed_kmh AND status not "stop")
      participate in departure confirmation.  Stopped/idle rows are silently
      skipped — they neither advance nor reset the departure counter.  This
      ensures pre-trip parked rows (even if outside the radius) cannot trigger
      a false departure.

      Departure is confirmed when min_consecutive_points consecutive MOVING rows
      are found outside origin_radius_m.  Departure time = timestamp of the FIRST
      moving outside row in that run (i.e. when the truck first started moving
      away, not the Nth confirmation point).

    Phase 2 — seek return (tighter inner zone):
      After departure is confirmed, return is detected using return_radius_m
      (default 200 m), which is deliberately tighter than origin_radius_m.
      This avoids confirming a return when the truck merely skirts the outer
      edge of the origin zone.

      Return is confirmed after min_consecutive_points consecutive GPS points
      inside return_radius_m.  Return time = timestamp of the FIRST qualifying
      inside point in that run.

    Movement classification
    -----------------------
    A row is classified as IDLE (not moving) if any of:
      - "Tốc độ" / speed column value ≤ min_moving_speed_kmh
      - "Trạng thái" / status column contains "stop", "dừng", or "dung"
    If neither column is present, every row is treated as moving (safe default
    — existing tests have no speed/status columns).

    Parameters
    ----------
    origin_radius_m
        Departure zone radius (metres).  Truck must exit this zone for departure.
        Default 700 m.
    return_radius_m
        Return confirmation zone radius (metres).  Must be ≤ origin_radius_m.
        Default 200 m.  Pass None to use origin_radius_m (backward-compatible
        but may confirm return too early for trucks with parking outside origin).
    min_consecutive_points
        Number of consecutive qualifying rows to confirm departure or return.
        Default 3.
    min_moving_speed_kmh
        Speed below which a row is treated as idle (skipped for departure
        detection).  Default 5.0 km/h.

    Returns
    -------
    dict with keys:
      start               HH:MM string (departure time) or None
      end                 HH:MM string (return time) or None
      departure_index     row index of first moving-outside point in confirmed
                          departure run, or None
      return_index        row index of first inside-return-zone point in
                          confirmed return run, or None
      detection_method    "origin_radius"
      origin_radius_m     float
      return_radius_m     float  (actual radius used for return detection)
      status              short outcome code (see below)
      n_valid_points      int  — rows with parseable coord + timestamp
      n_inside_points     int  — rows inside origin_radius_m
      n_outside_points    int
      started_inside      bool or None
      fallback_used       bool
      fallback_reason     str  (only present when fallback_used=True)

    Status codes
    ------------
      departed_and_returned       — normal closed trip
      started_outside_returned    — file starts outside origin; truck came back
      departed_no_return          — left depot, file ends before return
      started_outside_no_return   — file starts outside, no return seen
      never_departed              — truck stayed in origin zone throughout
    """

    _return_r: float = return_radius_m if return_radius_m is not None else origin_radius_m

    # --- Internal fallback (min/max timestamp) --------------------------------
    def _fallback(reason: str) -> dict:
        start, end = build_trip_window_from_df(df)
        return {
            "start": start,
            "end": end,
            "departure_index": None,
            "return_index": None,
            "detection_method": "min_max_timestamp",
            "origin_radius_m": origin_radius_m,
            "return_radius_m": _return_r,
            "status": f"fallback:{reason}",
            "n_valid_points": 0,
            "n_inside_points": 0,
            "n_outside_points": 0,
            "started_inside": None,
            "fallback_used": True,
            "fallback_reason": reason,
        }

    if origin is None:
        return _fallback("origin_not_provided")

    if "Tọa độ" not in df.columns or "Thời gian" not in df.columns or df.empty:
        return _fallback("missing_required_columns")

    origin_lat = float(origin[0])
    origin_lng = float(origin[1])

    # Parse all timestamps up-front (vectorised, fast)
    timestamps: pd.Series = pd.to_datetime(  # type: ignore[assignment]
        df["Thời gian"], errors="coerce", dayfirst=True
    )

    # --- Detect speed / status columns (robust to column-name variants) -------
    _speed_col: Optional[str] = next(
        (c for c in df.columns if "tốc" in c.lower() or "speed" in c.lower()), None
    )
    _status_col: Optional[str] = next(
        (c for c in df.columns
         if "trạng" in c.lower() or "trang" in c.lower() or "status" in c.lower()),
        None,
    )
    _IDLE_STATUS_KEYS = ("stop", "dừng", "dung")

    def _is_moving(row_i: int) -> bool:
        """True if this GPS row represents a moving (non-parked) truck.

        Falls back to True (moving) when speed and status info are unavailable,
        preserving backward compatibility with DataFrames that have no such cols.
        """
        if _speed_col is not None:
            spd = df[_speed_col].iloc[row_i]
            if pd.notna(spd):
                try:
                    if float(spd) <= min_moving_speed_kmh:
                        return False
                except (TypeError, ValueError):
                    pass
        if _status_col is not None:
            st = str(df[_status_col].iloc[row_i]).strip().lower()
            if any(k in st for k in _IDLE_STATUS_KEYS):
                return False
        return True

    # --- State machine --------------------------------------------------------
    consecutive_outside: int = 0
    consecutive_inside_post_dep: int = 0

    departure_confirmed: bool = False
    departure_candidate: Optional[Tuple[Any, int]] = None  # (ts, row_idx)
    departure_ts = None
    departure_index: Optional[int] = None

    return_confirmed: bool = False
    return_candidate: Optional[Tuple[Any, int]] = None
    return_ts = None
    return_index: Optional[int] = None

    # Return-arming: only activate return detection after the truck has been
    # sufficiently far from origin.  This prevents an outbound pass near the
    # origin (within return_radius_m) from being mistaken for a return when
    # the truck's parking spot is outside origin_radius_m and its outbound
    # route clips the origin area on the way out.
    # Threshold: 2 × origin_radius_m.  A truck that parks at 844 m and
    # departs toward origin (max_dist stays ~844 m < 1400 m) will NOT arm
    # the detector until it has driven to actual delivery distances.
    _return_arm_dist: float = origin_radius_m * 2.0
    max_dist_since_dep: float = 0.0
    return_armed: bool = False

    n_valid: int = 0
    n_inside: int = 0
    n_outside: int = 0
    started_inside: Optional[bool] = None

    coord_col = df["Tọa độ"]

    for row_idx in range(len(df)):
        ts = timestamps.iloc[row_idx]
        coord_raw = coord_col.iloc[row_idx]

        if pd.isna(ts) or pd.isna(coord_raw):
            continue
        try:
            lat, lng = parse_coord(str(coord_raw))
        except Exception:
            continue

        n_valid += 1
        dist = haversine(lat, lng, origin_lat, origin_lng)
        is_inside = dist <= origin_radius_m

        if started_inside is None:
            started_inside = is_inside

        if is_inside:
            n_inside += 1
        else:
            n_outside += 1

        # ---- Phase 1: seek departure -----------------------------------------
        # Only MOVING rows participate.  Idle/stopped rows are silently ignored
        # so that a truck parked outside origin before the trip does not trigger
        # a premature departure confirmation.
        if not departure_confirmed:
            if _is_moving(row_idx):
                if is_inside:
                    # Moving inside origin zone — reset departure run
                    consecutive_outside = 0
                    departure_candidate = None
                else:
                    # Moving outside origin zone — advance departure run
                    consecutive_outside += 1
                    if departure_candidate is None:
                        departure_candidate = (ts, row_idx)
                    if consecutive_outside >= min_consecutive_points:
                        departure_confirmed = True
                        departure_ts = departure_candidate[0]
                        departure_index = departure_candidate[1]
            # else: idle row — skip silently (don't count, don't reset)

        # ---- Phase 2: seek return (only after confirmed departure) -----------
        # Uses return_radius_m (tighter inner zone) to avoid false positives
        # from trucks that merely pass near the origin edge while on route.
        #
        # Return-arming guard: tracks the maximum distance reached since
        # departure.  The return zone is only armed once the truck has been at
        # least _return_arm_dist (2 × origin_radius_m) away from origin.
        # This prevents a short outbound detour near the origin (e.g. a truck
        # whose parking lot is ~844 m from the depot and whose outbound route
        # clips within 200 m of the depot) from being mis-classified as a
        # return just minutes after departure.
        else:
            max_dist_since_dep = max(max_dist_since_dep, dist)
            if not return_armed and max_dist_since_dep >= _return_arm_dist:
                return_armed = True

            if return_armed:
                is_inside_return_zone = dist <= _return_r
                if is_inside_return_zone:
                    consecutive_inside_post_dep += 1
                    if return_candidate is None:
                        return_candidate = (ts, row_idx)
                    if consecutive_inside_post_dep >= min_consecutive_points:
                        return_confirmed = True
                        return_ts = return_candidate[0]
                        return_index = return_candidate[1]
                        break  # early exit — both events found
                else:
                    # Outside return zone — reset inside run
                    consecutive_inside_post_dep = 0
                    return_candidate = None

    if n_valid == 0:
        return _fallback("no_valid_gps_points")

    # --- Determine status -----------------------------------------------------
    if not departure_confirmed:
        status = "never_departed"
    elif started_inside and return_confirmed:
        status = "departed_and_returned"
    elif not started_inside and return_confirmed:
        status = "started_outside_returned"
    elif started_inside and not return_confirmed:
        status = "departed_no_return"
    else:
        status = "started_outside_no_return"

    def _fmt(ts_val) -> Optional[str]:
        if ts_val is None or pd.isna(ts_val):
            return None
        try:
            return pd.Timestamp(ts_val).strftime("%H:%M")
        except Exception:
            return None

    return {
        "start": _fmt(departure_ts),
        "end": _fmt(return_ts),
        "departure_index": departure_index,
        "return_index": return_index,
        "detection_method": "origin_radius",
        "origin_radius_m": origin_radius_m,
        "return_radius_m": _return_r,
        "status": status,
        "n_valid_points": n_valid,
        "n_inside_points": n_inside,
        "n_outside_points": n_outside,
        "started_inside": started_inside,
        "fallback_used": False,
    }


def get_epass_rows_for_trip(start_hhmm: Optional[str], end_hhmm: Optional[str], bien_so_xe: str) -> List[dict]:
    if not start_hhmm or not end_hhmm:
        return []

    idxs = LayIndexVe(bien_so_xe)
    rows: List[dict] = []
    start_dt = pd.to_datetime(start_hhmm, format="%H:%M", errors="coerce")
    end_dt = pd.to_datetime(end_hhmm, format="%H:%M", errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        return []

    for i in idxs:
        value = str(epass_df["Unnamed: 3"].iloc[i])
        if " " not in value:
            continue
        gio_vao_tram = value.split(" ")[1][:5]
        t = pd.to_datetime(gio_vao_tram, format="%H:%M", errors="coerce")
        if pd.isna(t):
            continue
        if start_dt <= t <= end_dt:
            rows.append({
                "tram": epass_df["Unnamed: 2"].iloc[i],
                "thoi_gian": value,
                "bien_so": epass_df["Unnamed: 7"].iloc[i],
            })
    return rows


def match_turnaround_to_stops(turn_rows: Sequence[dict], stops: Sequence[dict], threshold_m: float = 250) -> Tuple[List[dict], List[dict]]:
    suspicious: List[dict] = []
    valid_turns: List[dict] = []

    geo_stops = [s for s in stops if s.get("lat") is not None and s.get("lng") is not None]

    for row in turn_rows:
        try:
            lat, lng = parse_coord(row["Tọa độ"])
        except Exception:
            suspicious.append({**row, "reason": "Không parse được tọa độ"})
            continue

        best = None
        best_d = float("inf")
        for stop in geo_stops:
            d = haversine(lat, lng, float(stop["lat"]), float(stop["lng"]))
            if d < best_d:
                best_d = d
                best = stop

        item = {
            **row,
            "nearest_stop": best.get("normalized_text") if best else None,
            "distance_to_nearest_stop_m": round(best_d, 1) if best_d != float("inf") else None,
        }
        if best and best_d <= threshold_m:
            valid_turns.append(item)
        else:
            suspicious.append(item)
    return valid_turns, suspicious


def process_one_plate(plate: str, addr_list: Sequence[str], day_code: Optional[str] = None) -> Dict[str, Any]:
    df = load_trip_df(plate, day_code=day_code)
    stops = normalize_and_geocode_stops(
        addr_list,
        trip_df=df,
        apply_vtracking_fallback=True,
        fallback_match_level="medium",
    )
    route_usable_stops = [
        s for s in stops
        if not s.get("route_excluded")
        and s.get("lat") is not None
        and s.get("lng") is not None
    ]
    print(f"[DEBUG] plate={plate}")
    print(f"[DEBUG] raw addr_list count={len(addr_list)}")
    print(f"[DEBUG] geocoded stops count={len(stops)}")
    print(f"[DEBUG] route usable stops count={len(route_usable_stops)}")
    print(f"[DEBUG] geocoded valid lat/lng count={sum(1 for s in stops if s.get('lat') is not None and s.get('lng') is not None)}")
    print(f"[DEBUG] df columns={list(df.columns)}")
    print(f"[DEBUG] valid GPS rows={df['Tọa độ'].notna().sum() if 'Tọa độ' in df.columns else 0}")
    origin_res = origin_resolver.resolve_trip_origin(plate)
    print(f"[DEBUG] origin=({origin_res.lat}, {origin_res.lng}) source={origin_res.source}")

    trip_window = detect_trip_window_from_origin(df, origin_res.as_latlng())
    print(f"[DEBUG] trip_window status={trip_window['status']} "
          f"start={trip_window['start']} end={trip_window['end']}")

    trip_report = analyze_trip_corridor(df, stops=stops, origin=origin_res.as_latlng())
    trip_report["_debug_total_stops"] = len(stops)
    trip_report["_debug_valid_geo_stops"] = sum(
        1 for s in stops if s.get("lat") is not None and s.get("lng") is not None
)
    turn_rows = DiaChiNghiVanQuayDau(plate) or []
    valid_turns, suspicious_turns = match_turnaround_to_stops(turn_rows, stops, threshold_m=250)

    epass_plate = BienSoXeChoFileEpass(plate)
    epass_rows = get_epass_rows_for_trip(trip_window["start"], trip_window["end"], epass_plate)

    return {
        "plate": plate,
        "trip_window": trip_window,
        "origin_lat": origin_res.lat,
        "origin_lng": origin_res.lng,
        "origin_source": origin_res.source,
        "stops": stops,
        "trip_report": trip_report,
        "turnaround_valid": valid_turns,
        "turnaround_suspicious": suspicious_turns,
        "epass_plate": epass_plate,
        "epass_rows": epass_rows,
        "epass_count": len(epass_rows),
    }


def process_all_plates(day_code: Optional[str] = None) -> List[Dict[str, Any]]:
    orders = LayDuLieuFileLenh()
    reports: List[Dict[str, Any]] = []
    for plate, addr_list in orders.items():
        try:
            report = process_one_plate(plate, addr_list, day_code=day_code)
            reports.append(report)
            print(f"OK Xong xe {plate}")
        except Exception as e:
            reports.append({"plate": plate, "error": str(e)})
            print(f"LOI xe {plate}: {e}")
    return reports


def export_reports_json(reports: Sequence[Dict[str, Any]], filename: str = "trip_report.json") -> Path:
    path = REPORT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(reports), f, ensure_ascii=False, indent=2, default=str)
    return path


def export_summary_excel(reports: Sequence[Dict[str, Any]], filename: str = "trip_summary.xlsx") -> Path:
    summary_rows: List[dict] = []
    for report in reports:
        if report.get("error"):
            summary_rows.append({
                "Biển số": report.get("plate"),
                "Lỗi": report.get("error"),
            })
            continue

        trip = report.get("trip_report", {})
        summary_rows.append({
            "Biển số": report.get("plate"),
            "Giờ đi": report.get("trip_window", {}).get("start"),
            "Giờ về": report.get("trip_window", {}).get("end"),
            "Km thực tế": trip.get("actual_distance_km"),
            "Km kỳ vọng": trip.get("expected_distance_km"),
            "Tỷ lệ vòng": trip.get("detour_ratio"),
            "Lệch tuyến": trip.get("off_route_points"),
            "Có quay đầu": trip.get("wrong_turn_u_turn_flag"),
            "Điểm giao đã ghé": len(trip.get("visited_stops", [])),
            "Điểm giao bỏ sót": len(trip.get("missed_stops", [])),
            "Quay đầu hợp lệ": len(report.get("turnaround_valid", [])),
            "Quay đầu nghi vấn": len(report.get("turnaround_suspicious", [])),
            "Số vé ePass": report.get("epass_count"),
            # new corridor columns
            "Tuân thủ hành lang (%)": trip.get("corridor_compliance_pct"),
            "Lệch tối đa (m)": trip.get("max_deviation_m"),
            "Chân hàng tệ nhất": trip.get("worst_leg_idx"),
        })

    path = REPORT_DIR / filename
    pd.DataFrame(summary_rows).to_excel(path, index=False)
    return path


def _day_code_from_config() -> Optional[str]:
    """Derive DDMM day_code from config.DATE (format DD/MM/YYYY).

    Returns e.g. "1403" for "14/03/2026", or None if config is unavailable.
    """
    try:
        from config import DATE as _DATE  # type: ignore[import]
        parts = str(_DATE).strip().split("/")
        if len(parts) >= 2:
            return parts[0].zfill(2) + parts[1].zfill(2)
    except Exception:
        pass
    return None


def main(day_code: Optional[str] = None) -> Tuple[Path, Path]:
    if day_code is None:
        day_code = _day_code_from_config()
    if day_code:
        print(f"[INFO] Xử lý ngày: {day_code}")
    reports = process_all_plates(day_code=day_code)
    json_path = export_reports_json(reports)
    xlsx_path = export_summary_excel(reports)
    print(f"✅ Đã xuất JSON: {json_path}")
    print(f"✅ Đã xuất Excel: {xlsx_path}")
    return json_path, xlsx_path


if __name__ == "__main__":
    main()
