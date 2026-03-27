from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests

try:
    import maps_config
except Exception:
    maps_config = None  # type: ignore

from vtracking_tool import haversine, parse_coord

LatLng = Tuple[float, float]

GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_MAPS_API_KEY_ENV = "GOOGLE_MAPS_API_KEY"


# ----------------------------
# text utils
# ----------------------------

def _norm_text(s: Any) -> str:
    text = str(s or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "d")
    for ch in [",", ".", ";", ":", "-", "/", "\\", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())

def _norm_admin_name(s: Any) -> str:
    t = _norm_text(s)

    prefixes = [
        "xa ", "phuong ", "thi tran ",
        "huyen ", "quan ", "thi xa ", "thanh pho ",
        "tinh ", "tp "
    ]

    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if t.startswith(p):
                t = t[len(p):].strip()
                changed = True

    if t in {"viet nam", "vietnam", "vn"}:
        return ""

    return t

def _contains_token(haystack: str, needle: str) -> bool:
    h = _norm_text(haystack)
    n = _norm_text(needle)
    return bool(n) and n in h


def _norm_set(values: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for v in values:
        nv = _norm_text(v)
        if nv:
            out.add(nv)
    return out


def _get_api_key() -> Optional[str]:
    if maps_config is not None:
        try:
            key = maps_config.get_api_key()
            if key:
                return key
        except Exception:
            pass
    return os.getenv(GOOGLE_MAPS_API_KEY_ENV)


# ----------------------------
# geocode confidence
# ----------------------------

def geocode_confidence(stop: Dict[str, Any]) -> str:
    """
    High:
      - status OK
      - có lat/lng
      - not partial_match
      - location_type mạnh

    Medium:
      - status OK
      - có lat/lng
      - admin formatted khớp >= 2 cấp

    Low:
      - còn lại
    """
    if stop.get("status") != "OK":
        return "low"
    if stop.get("lat") is None or stop.get("lng") is None:
        return "low"

    loc_type = str(stop.get("location_type") or "").upper()
    partial = bool(stop.get("partial_match"))
    formatted = str(
        stop.get("geocoded_formatted_address") or stop.get("formatted_address") or ""
    )
    # Compare CANONICAL admin (stop["ward/district/province"] after OLD-admin
    # canonical override) against the geocoder's formatted_address.
    # This correctly gives low confidence when the geocoder returned a different
    # province (e.g. "Đồng Tháp") from the canonical OLD-admin ("Tiền Giang").
    # geocoded_province is now the province parsed FROM formatted_address and
    # must NOT be used here — it would cause a false-positive match with itself.
    ward     = str(stop.get("ward")     or "")
    district = str(stop.get("district") or "")
    province = str(stop.get("province") or "")

    admin_hits = 0
    if ward and _contains_token(formatted, ward):
        admin_hits += 1
    if district and _contains_token(formatted, district):
        admin_hits += 1
    if province and _contains_token(formatted, province):
        admin_hits += 1

    if not partial and loc_type in {"ROOFTOP", "RANGE_INTERPOLATED"}:
        return "high"

    if admin_hits >= 2:
        return "medium"

    if loc_type in {"GEOMETRIC_CENTER", "APPROXIMATE"} and admin_hits >= 1:
        return "medium"

    return "low"


# ----------------------------
# reverse geocode helpers
# ----------------------------

import re

def sanitize_and_validate_address(addr_obj: dict, expected_province: str = None) -> dict:
    def clean_text(s: str) -> str:
        if not s:
            return ""
        s = str(s)
        s = re.sub(r"thu\s*\d[\d\.,]*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"[-–—]+", " ", s)
        s = re.sub(r",\s*,+", ", ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip(" ,.")

    normalized_text = clean_text(addr_obj.get("normalized_text"))
    ward = clean_text(addr_obj.get("ward"))
    district = clean_text(addr_obj.get("district"))
    province = clean_text(addr_obj.get("province"))

    if expected_province:
        province_mismatch = (
            province and expected_province.lower() not in province.lower()
        )
    else:
        province_mismatch = False

    return {
        "normalized_text": normalized_text,
        "ward": ward,
        "district": district,
        "province": province,
        "_province_mismatch": province_mismatch,
    }


def score_geocode_candidate(candidate: dict, addr_obj: dict, expected_province: str = None) -> float:
    score = 0.0

    formatted = (candidate.get("formatted_address") or "").lower()
    location_type = (candidate.get("location_type") or "").upper()
    partial = candidate.get("partial_match", False)

    ward = (addr_obj.get("ward") or "").lower()
    district = (addr_obj.get("district") or "").lower()
    province = (addr_obj.get("province") or "").lower()

    if province and province in formatted:
        score += 50
    else:
        score -= 100

    if district and district in formatted:
        score += 25

    if ward and ward in formatted:
        score += 15

    if location_type == "ROOFTOP":
        score += 15
    elif location_type == "GEOMETRIC_CENTER":
        score += 5
    elif location_type == "APPROXIMATE":
        score -= 10

    if partial:
        score -= 10

    if expected_province:
        if expected_province.lower() in formatted:
            score += 50
        else:
            score -= 200

    return score

def _extract_admin_sets_from_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Set[str]]:
    wards: Set[str] = set()
    districts: Set[str] = set()
    provinces: Set[str] = set()
    formatted_pool: Set[str] = set()
    location_types: Set[str] = set()

    for res in results:
        formatted = str(res.get("formatted_address") or "")
        if formatted:
            formatted_pool.add(_norm_text(formatted))

        geometry = res.get("geometry") or {}
        loc_type = str(geometry.get("location_type") or "").upper()
        if loc_type:
            location_types.add(loc_type)

        for comp in res.get("address_components", []) or []:
            long_name = str(comp.get("long_name") or "").strip()
            if not long_name:
                continue
            types = comp.get("types", []) or []
            n = _norm_admin_name(long_name)

            if not n:
                continue

            if "administrative_area_level_1" in types:
                provinces.add(n)

            # district thật
            if "administrative_area_level_2" in types:
                districts.add(n)

            # locality ở VN thường tương đương thành phố/thị xã/huyện trong bài toán của bạn
            if "locality" in types:
                districts.add(n)

            # ward/xã/phường
            if (
                "administrative_area_level_3" in types
                or "sublocality_level_1" in types
                or "sublocality" in types
            ):
                wards.add(n)

    return {
        "wards": wards,
        "districts": districts,
        "provinces": provinces,
        "formatted_pool": formatted_pool,
        "location_types": location_types,
    }


def reverse_geocode(lat: float, lng: float, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Giữ cả top result để debug, và giữ admin_sets gom từ nhiều result để đối chiếu.
    """
    key = api_key or _get_api_key()
    if not key:
        return {
            "status": "NO_API_KEY",
            "formatted_address": None,
            "location_type": None,
            "address_components": {},
            "results": [],
            "results_count": 0,
            "admin_sets": {
                "wards": set(),
                "districts": set(),
                "provinces": set(),
                "formatted_pool": set(),
                "location_types": set(),
            },
        }

    params = {
        "latlng": f"{lat},{lng}",
        "language": "vi",
        "key": key,
    }
    r = requests.get(GOOGLE_GEOCODE_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    results = data.get("results") or []
    if not results:
        return {
            "status": data.get("status") or "ZERO_RESULTS",
            "formatted_address": None,
            "location_type": None,
            "address_components": {},
            "results": [],
            "results_count": 0,
            "admin_sets": {
                "wards": set(),
                "districts": set(),
                "provinces": set(),
                "formatted_pool": set(),
                "location_types": set(),
            },
        }

    cleaned = sanitize_and_validate_address(
        item,
        expected_province=item.get("province")
    )

    scored = []
    for c in results:
        s = score_geocode_candidate(
            c,
            cleaned,
            expected_province=cleaned.get("province")
        )
        scored.append((s, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_score, top = scored[0]

    comp_map: Dict[str, str] = {}
    for comp in top.get("address_components", []) or []:
        long_name = comp.get("long_name")
        for t in comp.get("types", []) or []:
            if t not in comp_map and long_name:
                comp_map[t] = long_name

    admin_sets = _extract_admin_sets_from_results(results[:8])

    return {
        "status": data.get("status") or "OK",
        "formatted_address": top.get("formatted_address"),
        "location_type": (top.get("geometry") or {}).get("location_type"),
        "address_components": comp_map,
        "results": results[:8],
        "results_count": len(results),
        "admin_sets": admin_sets,
    }


# ----------------------------
# dwell clusters
# ----------------------------

@dataclass
class DwellCluster:
    cluster_idx: int
    start_row: int
    end_row: int
    start_time: Optional[pd.Timestamp]
    end_time: Optional[pd.Timestamp]
    duration_min: float
    lat: float
    lng: float
    n_points: int
    reverse_address: Optional[str]
    reverse_location_type: Optional[str]
    reverse_components: Dict[str, str]
    reverse_results_count: int
    reverse_admin_sets: Dict[str, Set[str]]
    point_rows: List[int]
    point_coords: List[Tuple[float, float]]


def _find_speed_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = c.lower()
        if "tốc" in cl or "speed" in cl:
            return c
    return None


def _find_status_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = c.lower()
        if "trạng" in cl or "trang" in cl or "status" in cl:
            return c
    return None


def _is_idle_row(
    df: pd.DataFrame,
    row_idx: int,
    speed_col: Optional[str],
    status_col: Optional[str],
    min_moving_speed_kmh: float,
) -> bool:
    if speed_col is not None:
        spd = df[speed_col].iloc[row_idx]
        if pd.notna(spd):
            try:
                spd_f = float(spd)
                if spd_f <= min_moving_speed_kmh:
                    return True
            except Exception:
                pass

    if status_col is not None:
        st = str(df[status_col].iloc[row_idx] or "").strip().lower()
        if any(k in st for k in ("stop", "dừng", "dung")):
            return True

    return False

def _nearest_point_in_cluster(
    cluster: DwellCluster,
    target_lat: float,
    target_lng: float,
) -> Optional[Dict[str, Any]]:
    if not cluster.point_coords:
        return None

    best = None
    best_dist = float("inf")

    for idx, (lat, lng) in enumerate(cluster.point_coords):
        d = haversine(target_lat, target_lng, lat, lng)
        if d < best_dist:
            best_dist = d
            best = {
                "row_idx": cluster.point_rows[idx],
                "lat": lat,
                "lng": lng,
                "distance_m": d,
            }

    return best

def _admin_match_detail_from_admin_sets(
    stop: Dict[str, Any],
    admin_sets: Dict[str, Set[str]],
) -> Dict[str, Any]:
    s = _stop_admin(stop)
    inferred = _infer_missing_stop_admin_from_text(stop)

    wards = set(admin_sets.get("wards") or set())
    districts = set(admin_sets.get("districts") or set())
    provinces = set(admin_sets.get("provinces") or set())

    ward = s["ward"]
    district = s["district"]
    province = s["province"]

    ward_ok = bool(ward) and ward in wards
    district_ok = bool(district) and district in districts
    province_ok = bool(province) and province in provinces

    matched_levels = int(ward_ok) + int(district_ok) + int(province_ok)

    if ward_ok and district_ok and province_ok:
        level = "strong"
    elif district_ok and province_ok:
        level = "medium"
    elif province_ok and not district and not ward:
        level = "province_only"
    else:
        level = "none"

    return {
        "level": level,
        "matched_levels": matched_levels,
        "ward_ok": ward_ok,
        "district_ok": district_ok,
        "province_ok": province_ok,
        "stop_admin_used": s,
        "stop_admin_inferred_from_text": {
            "ward": inferred["ward"],
            "district": inferred["district"],
            "province": inferred["province"],
            "infer_parts": inferred["infer_parts"],
            "raw_infer_text": inferred["raw_infer_text"],
        },
        "cluster_wards": sorted(wards),
        "cluster_districts": sorted(districts),
        "cluster_provinces": sorted(provinces),
    }

def _score_point_candidate(
    stop: Dict[str, Any],
    point_distance_m: float,
    cluster: DwellCluster,
    match_detail: Dict[str, Any],
) -> float:
    if match_detail["level"] == "strong":
        score = 100.0
    elif match_detail["level"] == "medium":
        score = 75.0
    else:
        return -1.0

    # point càng gần geocode stop càng tốt
    if point_distance_m <= 100:
        score += 20
    elif point_distance_m <= 300:
        score += 15
    elif point_distance_m <= 1000:
        score += 10
    elif point_distance_m <= 3000:
        score += 5

    # cluster dwell lâu hơn đáng tin hơn
    score += min(cluster.duration_min, 15.0)

    return score

def _debug_print_point_candidates(
    stop: Dict[str, Any],
    point_candidates: List[Dict[str, Any]],
    top_n_preview: int = 10,
) -> None:
    label = stop.get("raw_text") or stop.get("normalized_text") or stop.get("query") or "<unknown>"
    print(f"[DEBUG-CANDIDATES] stop={label}")
    print(f"  stop_admin={_stop_admin(stop)}")
    print(f"  total_point_candidates={len(point_candidates)}")

    for idx, cand in enumerate(point_candidates[:top_n_preview], 1):
        cl = cand["cluster"]
        print(
            f"  cand#{idx} "
            f"cluster#{cl.cluster_idx} "
            f"row={cand['row_idx']} "
            f"lat={cand['lat']:.6f} lng={cand['lng']:.6f} "
            f"dist_to_stop_geocode={cand['distance_m']:.1f}m "
            f"cluster_duration={cl.duration_min}m "
            f"cluster_addr={cl.reverse_address}"
        )

def _debug_print_reverse_match(
    stop: Dict[str, Any],
    cand: Dict[str, Any],
    rev: Dict[str, Any],
    detail: Dict[str, Any],
) -> None:
    cl = cand["cluster"]
    label = stop.get("raw_text") or stop.get("normalized_text") or stop.get("query") or "<unknown>"

    print(f"[DEBUG-REVERSE] stop={label}")
    print(
        f"  cluster#{cl.cluster_idx} row={cand['row_idx']} "
        f"lat={cand['lat']:.6f} lng={cand['lng']:.6f} "
        f"dist_to_stop_geocode={cand['distance_m']:.1f}m"
    )
    print(f"  reverse_address={rev.get('formatted_address')}")
    print(f"  reverse_results_count={rev.get('results_count')}")
    print(f"  wards={sorted((rev.get('admin_sets') or {}).get('wards') or [])}")
    print(f"  districts={sorted((rev.get('admin_sets') or {}).get('districts') or [])}")
    print(f"  provinces={sorted((rev.get('admin_sets') or {}).get('provinces') or [])}")
    print(
        f"  match_level={detail['level']} "
        f"| ward_ok={detail['ward_ok']} "
        f"| district_ok={detail['district_ok']} "
        f"| province_ok={detail['province_ok']}"
    )

def _scan_weak_points_near_anchor(
    df: pd.DataFrame,
    anchor_lat: float,
    anchor_lng: float,
    max_anchor_dist_m: float = 12000.0,
    max_speed_kmh: float = 20.0,
    max_points: int = 30,
) -> List[Dict[str, Any]]:
    """
    Quét trực tiếp các row GPS gần điểm neo, không cần phải thành cluster mạnh.
    Ưu tiên các điểm:
    - gần anchor
    - tốc độ thấp
    - hoặc status có dấu hiệu dừng
    """
    if "Tọa độ" not in df.columns or df.empty:
        return []

    speed_col = _find_speed_col(df)
    status_col = _find_status_col(df)

    candidates: List[Dict[str, Any]] = []

    for i in range(len(df)):
        raw = df["Tọa độ"].iloc[i]
        if pd.isna(raw):
            continue

        try:
            lat, lng = parse_coord(str(raw))
        except Exception:
            continue

        d = haversine(anchor_lat, anchor_lng, float(lat), float(lng))
        if d > max_anchor_dist_m:
            continue

        speed_ok = True
        speed_val = None
        if speed_col is not None:
            spd = df[speed_col].iloc[i]
            if pd.notna(spd):
                try:
                    speed_val = float(spd)
                    speed_ok = speed_val <= max_speed_kmh
                except Exception:
                    speed_ok = True

        status_hint = False
        if status_col is not None:
            st = str(df[status_col].iloc[i] or "").strip().lower()
            if any(k in st for k in ("stop", "dừng", "dung", "park", "idle")):
                status_hint = True

        # chỉ lấy điểm gần anchor và có dấu hiệu chậm/dừng
        if not speed_ok and not status_hint:
            continue

        candidates.append({
            "row_idx": i,
            "lat": float(lat),
            "lng": float(lng),
            "distance_m": float(d),
            "speed": speed_val,
            "status_hint": status_hint,
        })

    candidates.sort(key=lambda x: (x["distance_m"], 999 if x["speed"] is None else x["speed"]))
    return candidates[:max_points]

def extract_dwell_clusters_from_vtracking(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    min_points: int = 2,
    min_duration_min: float = 1.0,
    max_cluster_radius_m: float = 150.0,
    min_moving_speed_kmh: float = 12.0,
    max_clusters_for_reverse: int = 80,
) -> List[DwellCluster]:
    if "Tọa độ" not in df.columns or "Thời gian" not in df.columns or df.empty:
        return []

    ts_series = pd.to_datetime(df["Thời gian"], errors="coerce", dayfirst=True)
    coord_series = df["Tọa độ"]
    speed_col = _find_speed_col(df)
    status_col = _find_status_col(df)

    rows: List[Dict[str, Any]] = []
    for i in range(len(df)):
        ts = ts_series.iloc[i]
        raw = coord_series.iloc[i]
        if pd.isna(ts) or pd.isna(raw):
            continue
        try:
            lat, lng = parse_coord(str(raw))
        except Exception:
            continue
        rows.append(
            {
                "df_idx": i,
                "ts": pd.Timestamp(ts),
                "lat": float(lat),
                "lng": float(lng),
                "idle": _is_idle_row(df, i, speed_col, status_col, min_moving_speed_kmh),
            }
        )

    clusters_raw: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for row in rows:
        if not row["idle"]:
            if current:
                clusters_raw.append(current)
                current = []
            continue

        if not current:
            current = [row]
            continue

        center_lat = sum(x["lat"] for x in current) / len(current)
        center_lng = sum(x["lng"] for x in current) / len(current)
        dist = haversine(center_lat, center_lng, row["lat"], row["lng"])

        if dist <= max_cluster_radius_m:
            current.append(row)
        else:
            clusters_raw.append(current)
            current = [row]

    if current:
        clusters_raw.append(current)

    clusters: List[DwellCluster] = []
    reverse_budget = 0

    for grp in clusters_raw:
        if len(grp) < min_points:
            continue

        start_time = grp[0]["ts"]
        end_time = grp[-1]["ts"]
        duration_min = (end_time - start_time).total_seconds() / 60.0
        if duration_min < min_duration_min:
            continue

        lat = sum(x["lat"] for x in grp) / len(grp)
        lng = sum(x["lng"] for x in grp) / len(grp)

        rev = {
            "formatted_address": None,
            "location_type": None,
            "address_components": {},
            "results_count": 0,
            "admin_sets": {
                "wards": set(),
                "districts": set(),
                "provinces": set(),
                "formatted_pool": set(),
                "location_types": set(),
            },
        }

        if reverse_budget < max_clusters_for_reverse:
            try:
                rev = reverse_geocode(lat, lng, api_key=api_key)
            except Exception:
                pass
            reverse_budget += 1

        clusters.append(
            DwellCluster(
                cluster_idx=len(clusters),
                start_row=grp[0]["df_idx"],
                end_row=grp[-1]["df_idx"],
                start_time=start_time,
                end_time=end_time,
                duration_min=round(duration_min, 1),
                lat=lat,
                lng=lng,
                n_points=len(grp),
                reverse_address=rev.get("formatted_address"),
                reverse_location_type=rev.get("location_type"),
                reverse_components=rev.get("address_components") or {},
                reverse_results_count=int(rev.get("results_count") or 0),
                reverse_admin_sets=rev.get("admin_sets")
                or {
                    "wards": set(),
                    "districts": set(),
                    "provinces": set(),
                    "formatted_pool": set(),
                    "location_types": set(),
                },
                point_rows=[x["df_idx"] for x in grp],
                point_coords=[(x["lat"], x["lng"]) for x in grp],
            )
        )

    return clusters


# ----------------------------
# stop <-> cluster match
# ----------------------------

def _infer_missing_stop_admin_from_text(stop: Dict[str, Any]) -> Dict[str, str]:
    """
    Suy diễn thêm admin bị thiếu từ raw_text / normalized_text / query / formatted_address.

    Mục tiêu chính:
    - cứu case kiểu: "Phước Long, tỉnh Bình Phước"
    - khi upstream chưa parse ra district nhưng text vẫn có tên district/thị xã/thành phố

    Nguyên tắc an toàn:
    - chỉ infer district khi stop chưa có district
    - chỉ infer ward khi stop chưa có ward và text có >= 3 mảnh admin rõ hơn
    - tránh gán lại province nếu đã có sẵn
    """
    text_candidates = [
        stop.get("raw_text"),
        stop.get("normalized_text"),
        stop.get("query"),
        stop.get("formatted_address"),
        stop.get("address"),
        stop.get("full_address"),
    ]

    raw = ""
    for cand in text_candidates:
        if cand:
            raw = str(cand)
            break

    raw_parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    parts = [_norm_admin_name(x) for x in raw_parts]
    parts = [x for x in parts if x and x not in {"viet nam", "vietnam", "vn"}]

    ward = _norm_admin_name(stop.get("ward") or "")
    district = _norm_admin_name(stop.get("district") or "")
    province = _norm_admin_name(stop.get("province") or "")

    if not province and parts:
        province = parts[-1]

    if not district:
        if len(parts) >= 2:
            district = parts[-2]

    if not ward:
        if len(parts) >= 3:
            ward = parts[-3]

    return {
        "ward": ward,
        "district": district,
        "province": province,
        "raw_infer_text": raw,
        "infer_parts": parts,
    }

def _stop_admin(stop: Dict[str, Any]) -> Dict[str, str]:
    base = {
        "ward": _norm_admin_name(stop.get("ward") or ""),
        "district": _norm_admin_name(stop.get("district") or ""),
        "province": _norm_admin_name(stop.get("province") or ""),
    }

    inferred = _infer_missing_stop_admin_from_text(stop)

    return {
        "ward": base["ward"] or inferred["ward"],
        "district": base["district"] or inferred["district"],
        "province": base["province"] or inferred["province"],
    }

def _address_part_count(stop: Dict[str, Any]) -> int:
    text = (
        stop.get("raw_text")
        or stop.get("normalized_text")
        or stop.get("query")
        or stop.get("formatted_address")
        or ""
    )
    parts = [p.strip() for p in str(text).split(",") if str(p).strip()]
    return len(parts)


def _candidate_distance_threshold_m(
    stop: Dict[str, Any],
    geocode_conf: Optional[str] = None,
    base_threshold_m: float = 8000.0,
) -> float:
    """
    Xác định ngưỡng khoảng cách chấp nhận giữa anchor geocode của lệnh
    và candidate GPS.

    Gợi ý:
    - địa chỉ chi tiết (>= 3 mảnh): siết hơn
    - địa chỉ ngắn / mơ hồ: nới hơn
    - geocode low: nới nhẹ vì anchor có thể lệch
    """
    parts = _address_part_count(stop)
    conf = str(geocode_conf or "").lower()

    if parts >= 3:
        th = 5000.0
    elif parts == 2:
        th = 8000.0
    else:
        th = 12000.0

    if conf == "low":
        th += 2000.0
    elif conf == "high":
        th = min(th, 5000.0)

    return max(th, base_threshold_m if parts == 0 else th)


def _score_cluster_distance_candidate(
    point_distance_m: float,
    cluster: DwellCluster,
) -> float:
    """
    Score cho candidate lấy từ dwell cluster.
    Trọng tâm là khoảng cách tới anchor.
    """
    # Điểm nền theo khoảng cách
    if point_distance_m <= 200:
        score = 100.0
    elif point_distance_m <= 500:
        score = 94.0
    elif point_distance_m <= 1000:
        score = 88.0
    elif point_distance_m <= 2000:
        score = 80.0
    elif point_distance_m <= 3000:
        score = 72.0
    elif point_distance_m <= 5000:
        score = 60.0
    elif point_distance_m <= 8000:
        score = 48.0
    elif point_distance_m <= 12000:
        score = 36.0
    else:
        score = 0.0

    # dwell càng lâu càng đáng tin hơn
    score += min(cluster.duration_min, 20.0)

    # cluster nhiều điểm thì cộng nhẹ
    score += min(cluster.n_points * 0.8, 8.0)

    return score


def _score_weak_distance_candidate(
    point_distance_m: float,
    speed: Optional[float],
    status_hint: bool,
) -> float:
    """
    Score cho weak point.
    Vẫn ưu tiên khoảng cách, nhưng bonus nhỏ cho tốc độ thấp / status dừng.
    """
    if point_distance_m <= 200:
        score = 92.0
    elif point_distance_m <= 500:
        score = 86.0
    elif point_distance_m <= 1000:
        score = 80.0
    elif point_distance_m <= 2000:
        score = 72.0
    elif point_distance_m <= 3000:
        score = 64.0
    elif point_distance_m <= 5000:
        score = 54.0
    elif point_distance_m <= 8000:
        score = 44.0
    elif point_distance_m <= 12000:
        score = 34.0
    else:
        score = 0.0

    if speed is not None:
        if speed <= 5:
            score += 8.0
        elif speed <= 10:
            score += 5.0
        elif speed <= 20:
            score += 2.0

    if status_hint:
        score += 5.0

    return score


def _debug_print_distance_choice(
    stop: Dict[str, Any],
    candidate_type: str,
    best: Dict[str, Any],
    accept_threshold_m: float,
) -> None:
    label = stop.get("raw_text") or stop.get("normalized_text") or stop.get("query") or "<unknown>"
    print(f"[DEBUG-DISTANCE-CHOICE] stop={label}")
    print(f"  candidate_type={candidate_type}")
    print(f"  accept_threshold_m={accept_threshold_m:.1f}")
    print(
        f"  row={best.get('row_idx')} "
        f"lat={best.get('lat'):.6f} lng={best.get('lng'):.6f} "
        f"dist_to_stop_geocode={best.get('distance_m'):.1f}m "
        f"score={best.get('score'):.1f}"
    )


def enrich_stops_with_vtracking_distance_fallback(
    stops: Sequence[Dict[str, Any]],
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    default_accept_radius_km: float = 20.0,
    max_cluster_candidate_dist_m: float = 20000.0,
    weak_scan_radius_m: float = 12000.0,
) -> List[Dict[str, Any]]:
    """
    Resolver mới:
    - KHÔNG match theo cấp hành chính
    - KHÔNG reverse geocode candidate point để xét admin
    - Chỉ dùng anchor geocode của stop + khoảng cách tới candidate GPS

    Ý tưởng:
    1) stop geocode = anchor
    2) lấy cluster candidates gần anchor
    3) nếu có candidate trong ngưỡng chấp nhận -> chọn best theo score
    4) nếu chưa có -> quét weak GPS points gần anchor
    5) nếu vẫn không có -> exclude
    """

    # Không cần reverse cluster nữa vì không còn dùng admin match
    clusters = extract_dwell_clusters_from_vtracking(
        df,
        api_key=None,
        max_clusters_for_reverse=0,
    )

    resolved: List[Dict[str, Any]] = []

    for stop in stops:
        row = dict(stop)
        row["geocode_confidence"] = geocode_confidence(row)
        row["coord_source"] = "google_geocode"
        row["coord_resolution_note"] = ""
        row["route_excluded"] = False
        row["exclude_reason"] = None
        row["fallback_candidate_score"] = None
        row["candidate_distance_m"] = None
        row["distance_accept_threshold_m"] = None
        row["candidate_type"] = None

        # Nếu geocode stop đã rất tốt thì giữ nguyên như cũ
        if row["geocode_confidence"] == "high":
            resolved.append(row)
            continue

        stop_lat = row.get("lat")
        stop_lng = row.get("lng")

        if stop_lat is None or stop_lng is None:
            row["orig_lat"] = row.get("lat")
            row["orig_lng"] = row.get("lng")
            row["lat"] = None
            row["lng"] = None
            row["route_excluded"] = True
            row["exclude_reason"] = "no_stop_geocode_anchor"
            row["coord_source"] = "excluded_no_anchor"
            row["coord_resolution_note"] = (
                "không có tọa độ geocode đại diện của địa chỉ lệnh để làm anchor; "
                "không thể fallback theo khoảng cách"
            )
            resolved.append(row)
            continue

        accept_threshold_m = max(
            default_accept_radius_km * 1000.0,
            _candidate_distance_threshold_m(row, row["geocode_confidence"]),
        )
        row["distance_accept_threshold_m"] = round(accept_threshold_m, 1)

        # -------------------------
        # 1) cluster candidates
        # -------------------------
        cluster_candidates: List[Dict[str, Any]] = []

        for cl in clusters:
            p = _nearest_point_in_cluster(cl, float(stop_lat), float(stop_lng))
            if p is None:
                continue

            effective_cluster_scan_m = max(max_cluster_candidate_dist_m, accept_threshold_m)

            if p["distance_m"] > effective_cluster_scan_m:
                continue

            score = _score_cluster_distance_candidate(
                point_distance_m=float(p["distance_m"]),
                cluster=cl,
            )

            cluster_candidates.append({
                "candidate_type": "cluster",
                "cluster": cl,
                "row_idx": p["row_idx"],
                "lat": p["lat"],
                "lng": p["lng"],
                "distance_m": float(p["distance_m"]),
                "score": float(score),
            })

        cluster_candidates.sort(key=lambda x: (-x["score"], x["distance_m"]))

        best_cluster = None
        for cand in cluster_candidates:
            if cand["distance_m"] <= accept_threshold_m:
                best_cluster = cand
                break

        if best_cluster is not None:
            row["orig_lat"] = row.get("lat")
            row["orig_lng"] = row.get("lng")
            row["lat"] = round(best_cluster["lat"], 6)
            row["lng"] = round(best_cluster["lng"], 6)
            row["coord_source"] = "vtracking_cluster_distance_matched"
            row["candidate_type"] = "cluster"
            row["candidate_distance_m"] = round(best_cluster["distance_m"], 1)
            row["fallback_candidate_score"] = round(best_cluster["score"], 1)
            row["cluster_idx"] = best_cluster["cluster"].cluster_idx
            row["cluster_start_time"] = (
                str(best_cluster["cluster"].start_time)
                if best_cluster["cluster"].start_time is not None else None
            )
            row["cluster_end_time"] = (
                str(best_cluster["cluster"].end_time)
                if best_cluster["cluster"].end_time is not None else None
            )
            row["coord_resolution_note"] = (
                f"matched theo khoảng cách từ dwell cluster | "
                f"row={best_cluster['row_idx']} | "
                f"dist_to_stop_geocode={best_cluster['distance_m']:.1f}m | "
                f"threshold={accept_threshold_m:.1f}m | "
                f"score={best_cluster['score']:.1f}"
            )

            _debug_print_distance_choice(row, "cluster", best_cluster, accept_threshold_m)
            resolved.append(row)
            continue

        # -------------------------
        # 2) weak candidates
        # -------------------------

        effective_weak_scan_m = max(weak_scan_radius_m, accept_threshold_m)

        weak_points = _scan_weak_points_near_anchor(
            df=df,
            anchor_lat=float(stop_lat),
            anchor_lng=float(stop_lng),
            max_anchor_dist_m=effective_weak_scan_m,
            max_speed_kmh=20.0,
            max_points=30,
        )

        weak_candidates: List[Dict[str, Any]] = []
        for wp in weak_points:
            score = _score_weak_distance_candidate(
                point_distance_m=float(wp["distance_m"]),
                speed=wp.get("speed"),
                status_hint=bool(wp.get("status_hint")),
            )
            weak_candidates.append({
                "candidate_type": "weak",
                "row_idx": wp["row_idx"],
                "lat": wp["lat"],
                "lng": wp["lng"],
                "distance_m": float(wp["distance_m"]),
                "speed": wp.get("speed"),
                "status_hint": bool(wp.get("status_hint")),
                "score": float(score),
            })

        weak_candidates.sort(key=lambda x: (-x["score"], x["distance_m"]))

        best_weak = None
        for cand in weak_candidates:
            if cand["distance_m"] <= accept_threshold_m:
                best_weak = cand
                break

        if best_weak is not None:
            row["orig_lat"] = row.get("lat")
            row["orig_lng"] = row.get("lng")
            row["lat"] = round(best_weak["lat"], 6)
            row["lng"] = round(best_weak["lng"], 6)
            row["coord_source"] = "vtracking_weak_distance_matched"
            row["candidate_type"] = "weak"
            row["candidate_distance_m"] = round(best_weak["distance_m"], 1)
            row["fallback_candidate_score"] = round(best_weak["score"], 1)
            row["coord_resolution_note"] = (
                f"matched theo khoảng cách từ weak GPS point | "
                f"row={best_weak['row_idx']} | "
                f"dist_to_stop_geocode={best_weak['distance_m']:.1f}m | "
                f"threshold={accept_threshold_m:.1f}m | "
                f"score={best_weak['score']:.1f}"
            )

            _debug_print_distance_choice(row, "weak", best_weak, accept_threshold_m)
            resolved.append(row)
            continue

        # -------------------------
        # 3) không có candidate đủ gần
        # -------------------------
        row["orig_lat"] = row.get("lat")
        row["orig_lng"] = row.get("lng")
        row["lat"] = None
        row["lng"] = None
        row["route_excluded"] = True
        row["exclude_reason"] = "no_candidate_within_distance_threshold"
        row["coord_source"] = "excluded_distance_threshold"
        row["coord_resolution_note"] = (
            "không có cluster candidate hay weak point nào nằm trong ngưỡng khoảng cách chấp nhận "
            f"({accept_threshold_m:.1f}m) so với tọa độ đại diện của địa chỉ lệnh"
        )

        resolved.append(row)

    return resolved

# ----------------------------
# main resolver
# ----------------------------

def enrich_stops_with_vtracking_fallback(
    stops: Sequence[Dict[str, Any]],
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    min_match_level: str = "medium",
) -> List[Dict[str, Any]]:
    # min_match_level giữ lại cho tương thích chữ ký hàm cũ, nhưng không còn dùng
    return enrich_stops_with_vtracking_distance_fallback(
        stops=stops,
        df=df,
        api_key=api_key,
        default_accept_radius_km=20.0,
        max_cluster_candidate_dist_m=12000.0,
        weak_scan_radius_m=20000.0,
    )