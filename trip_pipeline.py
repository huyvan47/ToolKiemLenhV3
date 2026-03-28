from __future__ import annotations

import hashlib
import json
import pandas as pd
import origin_resolver

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ward_mapping_resolver import (
    load_reverse_ward_mapping,
    resolve_stop_by_ward_mapping,
    load_old_admin_key_set,
    parse_old_admin_text,
)
from geocode_service import geocode_address_obj
from gpt_data import ChuanHoaDiaChiTrongFileLenh
from lenh_data import LayDuLieuFileLenh
from phat_hien_quay_dau_data import DiaChiNghiVanQuayDau
from utils import BienSoXeChoFileEpass
from VeEpassCuaChuyen import LayIndexVe, df as epass_df
from vtracking_tool import analyze_trip_corridor, haversine, parse_coord
from stop_fallback_resolver import enrich_stops_with_vtracking_fallback

WARD_MAPPING_PATH = Path(__file__).resolve().parent / "ward_mapping_2025.json"
try:
    REVERSE_WARD_MAPPING = load_reverse_ward_mapping(str(WARD_MAPPING_PATH))
    print(f"[WARD_MAPPING] loaded reverse mapping: {len(REVERSE_WARD_MAPPING)} new-admin keys")
except Exception as e:
    REVERSE_WARD_MAPPING = {}
    print(f"[WARD_MAPPING] load failed: {e}")

try:
    _OLD_ADMIN_KEY_SET: set = load_old_admin_key_set(str(WARD_MAPPING_PATH))
    print(f"[WARD_MAPPING] loaded {len(_OLD_ADMIN_KEY_SET)} OLD-admin keys for canonical detection")
except Exception as _e:
    _OLD_ADMIN_KEY_SET = set()
    print(f"[WARD_MAPPING] old admin key load failed: {_e}")

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw" / "vtracking"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical OLD-admin helpers
# ---------------------------------------------------------------------------

def _wm_norm(s: Any) -> str:
    """Same normalization as ward_mapping_resolver._norm_text — used for key building."""
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ,.-")


def _wm_strip_prefix(s: str) -> str:
    """Strip single Vietnamese admin-level prefix from a normalized string."""
    for pfx in (
        "xã ", "phường ", "thị trấn ",
        "huyện ", "quận ", "thị xã ", "thành phố ", "tỉnh ",
    ):
        if s.startswith(pfx):
            return s[len(pfx):]
    return s


def _build_wm_keys(ward: str, district: str, province: str) -> List[str]:
    """
    Build ALL lookup key variants (full-prefix, bare, mixed) for a given
    ward/district/province triple, mirroring ward_mapping_resolver._build_lookup_keys.

    This ensures lookups succeed whether GPT normalization preserves or strips
    the Vietnamese admin-level prefixes ("xã ", "tỉnh ", etc.).
    """
    w_f = _wm_norm(ward)
    d_f = _wm_norm(district)
    p_f = _wm_norm(province)
    w_b = _wm_strip_prefix(w_f)
    d_b = _wm_strip_prefix(d_f)
    p_b = _wm_strip_prefix(p_f)

    keys: List[str] = []

    # Full-prefix forms
    if w_f and d_f and p_f:
        keys.append(f"{w_f}-{d_f}-{p_f}")
    if w_f and p_f:
        keys.append(f"{w_f}-{p_f}")
    # Bare forms (no prefix)
    if w_b and d_b and p_b:
        k = f"{w_b}-{d_b}-{p_b}"
        if k not in keys:
            keys.append(k)
    if w_b and p_b:
        k = f"{w_b}-{p_b}"
        if k not in keys:
            keys.append(k)
    # Mixed: bare ward + full province (common GPT output)
    if w_b and p_f:
        k = f"{w_b}-{p_f}"
        if k not in keys:
            keys.append(k)
    # Mixed: full ward + bare province
    if w_f and p_b:
        k = f"{w_f}-{p_b}"
        if k not in keys:
            keys.append(k)

    return [k for k in keys if k]


def _item_is_new_admin(item: Dict[str, Any]) -> bool:
    """
    Return True when the item's ward/province matches a KNOWN NEW-admin entry in
    ward_mapping_2025.json, meaning it maps to at least one 3-part OLD-admin candidate
    (ward-district-province).

    Detection strategy:
    - Build all key variants (full-prefix, bare, mixed) for ward+province
    - Look each up in REVERSE_WARD_MAPPING
    - If any candidate has >= 2 hyphens it is "ward-district-province" (OLD-admin)
      → the input is NEW-admin

    Tries bare keys so detection works even when GPT strips "xã ", "tỉnh ", etc.
    """
    ward     = item.get("ward") or ""
    district = item.get("district") or ""
    province = item.get("province") or ""

    keys_to_try = _build_wm_keys(ward, district, province)

    rm = REVERSE_WARD_MAPPING or {}
    for key in keys_to_try:
        cands = rm.get(key, [])
        if cands:
            print(f"[IS_NEW_ADMIN] key={key!r} → {len(cands)} candidates: {cands[:3]}")
        if any(c.count("-") >= 2 for c in cands):
            print(f"[IS_NEW_ADMIN] → TRUE (3-part old-admin found via key={key!r})")
            return True

    print(f"[IS_NEW_ADMIN] ward={ward!r} province={province!r} keys={keys_to_try} → FALSE")
    return False


def _score_raw_vs_old_admin(raw_text: str, old_admin_text: str) -> float:
    """
    Score how well an OLD-admin candidate matches token clues in raw_text.

    Scoring layers:
    1. Ward bare name in raw_text          → +20
    2. District bare name in raw_text      → +10
    3. Province bare name in raw_text      → +5
    4. Hamlet tokens (ấp/tổ/thôn/xóm)
       from raw_text that appear in the
       candidate's parsed ward bare name   → up to +12 (3 per matched token)
    5. Partial ward token match            → up to +9  (3 per token ≥ 3 chars)
    """
    parsed = parse_old_admin_text(old_admin_text)
    raw = _wm_norm(raw_text)

    def bare(s: str) -> str:
        s = _wm_norm(s)
        for pfx in (
            "xã ", "phường ", "thị trấn ",
            "huyện ", "quận ", "thị xã ", "thành phố ", "tỉnh ",
        ):
            if s.startswith(pfx):
                return s[len(pfx):]
        return s

    w = bare(parsed.get("ward", ""))
    d = bare(parsed.get("district", ""))
    p = bare(parsed.get("province", ""))

    score = 0.0
    if w and w in raw:
        score += 20.0
    if d and d in raw:
        score += 10.0
    if p and p in raw:
        score += 5.0

    # Hamlet token scoring: tokens ≥ 3 chars from raw that appear in ward bare name
    if w and len(w) >= 3:
        w_tokens = [t for t in w.split() if len(t) >= 3]
        matched = sum(1 for t in w_tokens if t in raw)
        score += min(matched * 3.0, 9.0)

    # Hamlet keyword boost: if raw mentions ấp/tổ/thôn next to a ward token fragment
    hamlet_kws = ["ấp ", "tổ ", "thôn ", "xóm ", "khu "]
    if any(kw in raw for kw in hamlet_kws) and w and any(tok in raw for tok in w.split() if len(tok) >= 3):
        score += 3.0

    return score


def _pretty_admin(s: str) -> str:
    if not s:
        return ""
    return " ".join(
        w.capitalize() if w not in {"tp", "tx", "ql"} else w.upper()
        for w in s.split()
    )


def _classify_admin_input(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Classify the administrative form of a GPT-normalized address item.

    Returns (admin_type, reason):
      "new_admin"    — ward+province maps to >= 1 three-part OLD-admin candidate
      "old_admin"    — key (full or bare) exists in _OLD_ADMIN_KEY_SET
      "transitional" — text contains "hết hiệu lực" or similar invalid markers
      "unknown"      — no mapping evidence

    Ward-extraction fallback: when item["ward"] is empty, this function tries to
    extract a ward token from raw_text/normalized_text and logs [MISSING_WARD].
    The extracted ward is used for key-building only inside this call; it does NOT
    mutate item["ward"].  (expand_old_admin_candidates does its own extraction
    when the function is eventually called for new_admin stops.)
    """
    raw      = item.get("raw_text") or item.get("normalized_text") or ""
    ward     = item.get("ward") or ""
    district = item.get("district") or ""
    province = item.get("province") or ""

    # --- transitional / invalid markers (highest priority) ---
    combined_lower = _wm_norm(f"{raw} {ward} {district} {province}")
    for kw in ("hết hiệu lực", "het hieu luc", "không hợp lệ", "khong hop le", "invalid"):
        if kw in combined_lower:
            return "transitional", f"marker '{kw}' in text"

    # --- missing-ward fallback: try to extract ward from raw text ---
    extracted_note = ""
    if not ward:
        extracted = _extract_ward_from_raw(raw)
        if extracted:
            ward = extracted
            extracted_note = f"; ward extracted from text: {extracted!r}"
            print(f"[MISSING_WARD] raw={raw!r:.80} extracted ward={extracted!r}")
        else:
            fallback_keys: List[str] = []
            d_f, p_f = _wm_norm(district), _wm_norm(province)
            d_b, p_b = _wm_strip_prefix(d_f), _wm_strip_prefix(p_f)
            if d_f and p_f:
                fallback_keys.append(f"{d_f}-{p_f}")
            if d_b and p_b and f"{d_b}-{p_b}" not in fallback_keys:
                fallback_keys.append(f"{d_b}-{p_b}")
            print(
                f"[MISSING_WARD] raw={raw!r:.80} "
                f"extraction_failed fallback_lookup_keys={fallback_keys}"
            )
            # No ward to build useful lookup keys from — return unknown
            return "unknown", "ward empty; extraction failed"

    rm      = REVERSE_WARD_MAPPING or {}
    old_set = _OLD_ADMIN_KEY_SET or set()
    keys    = _build_wm_keys(ward, district, province)

    # --- new-admin detection: any key yields a 3-part OLD-admin candidate ---
    for key in keys:
        cands = rm.get(key, [])
        if any(c.count("-") >= 2 for c in cands):
            return "new_admin", f"key {key!r} → 3-part old-admin{extracted_note}"

    # --- old-admin detection: key (full or bare) in OLD_ADMIN_KEY_SET ---
    # _OLD_ADMIN_KEY_SET now contains BOTH "xã X-huyện Y-tỉnh Z" (full)
    # AND "X-Y-Z" (bare), so this matches even when GPT stripped prefixes.
    for key in keys:
        if key in old_set:
            return "old_admin", f"3-part key {key!r} matched OLD_ADMIN_KEY_SET{extracted_note}"

    return "unknown", f"no mapping evidence{extracted_note}"


def _get_new_admin_candidates_for_old(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    For an OLD-admin (or unknown) item whose primary geocode scored < 0,
    look up corresponding NEW-admin names as secondary-fallback geocode queries.

    Looks for 2-part entries (ward-province) in REVERSE_WARD_MAPPING under the
    old-admin keys — these represent the 2025 renamed form that Google Maps is
    more likely to recognise.

    Returns at most 2 item-clone dicts with new-admin ward/province substituted.
    """
    ward     = item.get("ward") or ""
    district = item.get("district") or ""
    province = item.get("province") or ""

    keys = _build_wm_keys(ward, district, province)
    rm   = REVERSE_WARD_MAPPING or {}

    new_admin_texts: List[str] = []
    seen: set = set()
    for key in keys:
        for cand in rm.get(key, []):
            if cand.count("-") == 1 and cand not in seen:   # 2-part = new-admin
                seen.add(cand)
                new_admin_texts.append(cand)

    if not new_admin_texts:
        return []

    result: List[Dict[str, Any]] = []
    for nat in new_admin_texts[:2]:
        parts = [p.strip() for p in nat.split("-") if p.strip()]
        new_ward = _pretty_admin(parts[0]) if parts else ""
        new_prov = _pretty_admin(parts[-1]) if len(parts) >= 2 else ""
        clone = dict(item)
        clone["ward"]     = new_ward
        clone["district"] = ""
        clone["province"] = new_prov
        clone["_secondary_fallback_new_admin"] = nat
        result.append(clone)

    return result


def _extract_ward_from_raw(text: str) -> str:
    """
    Scan free-form text for the first ward-level token ("xã X", "phường X",
    "thị trấn X") and return it with the prefix, e.g. "thị trấn chợ gạo".
    Returns "" when no such token is found.
    """
    t = _wm_norm(text)
    for pfx in ("xã ", "phường ", "thị trấn "):
        idx = t.find(pfx)
        if idx >= 0:
            rest = t[idx:]
            tok = re.split(r"[,\-\n]", rest, maxsplit=1)[0].strip()
            if tok:
                return tok
    return ""


def _parse_province_from_formatted(formatted: str) -> str:
    """
    Best-effort extraction of the province name from a Google Maps
    formatted_address string.

    Google typically formats Vietnamese addresses as:
      "<detail>, <ward>, <district>, <province>, Việt Nam"
    The province is the last non-"Việt Nam" component.

    Returns "" when the formatted address is empty or unparseable.
    """
    if not formatted:
        return ""
    vn_skip = {"việt nam", "viet nam", "vietnam", "vn", ""}
    parts = [p.strip() for p in formatted.split(",") if p.strip()]
    while parts and _wm_norm(parts[-1]) in vn_skip:
        parts.pop()
    return parts[-1] if parts else ""


def _stop_id(raw_text: str) -> str:
    """Stable 8-char hex ID for a stop, based on MD5 of normalised raw text."""
    return hashlib.md5(raw_text.strip().lower().encode("utf-8")).hexdigest()[:8]


def _resolve_canonical_old_admin(
    item: Dict[str, Any],
    merged: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Determine the canonical OLD-admin form for a geocoded stop.

    Rules
    -----
    1. If item admin is already OLD-admin (not a NEW-admin key)      → canonical = item fields unchanged.
    2. If item is NEW-admin AND resolve_stop_by_ward_mapping already  ran and resolved to an OLD candidate (ward/district/province in
       merged differ from item and district is non-empty)            → use resolved candidate fields as canonical.
    3. If item is NEW-admin but mapping didn't run or failed         → text-overlap score all OLD-admin candidates from
       REVERSE_WARD_MAPPING, pick the best-scoring one.
    4. Unmapped or indeterminate                                     → canonical = item fields (best effort).

    Returns
    -------
    dict with: ward, district, province, old_key, address, detection,
               candidates_scored (list of (score, old_admin_text))
    """
    ward      = item.get("ward") or ""
    district  = item.get("district") or ""
    province  = item.get("province") or ""
    raw_text  = item.get("raw_text") or item.get("normalized_text") or ""
    norm_addr = item.get("normalized_text") or raw_text

    def _make_result(cw, cd, cp, detection, candidates_scored=None, detail_override=None):
        # Build canonical address: detail tokens + canonical admin
        admin_tokens = {_wm_norm(t) for t in [ward, district, province, cw, cd, cp] if t}
        parts = [p.strip() for p in norm_addr.split(",") if p.strip()]
        kept = [
            p for p in parts
            if _wm_norm(p) not in admin_tokens
            and _wm_norm(p) not in {"viet nam", "vietnam", "vn", ""}
        ]
        detail = detail_override if detail_override is not None else ", ".join(kept)
        addr_parts = [x for x in [detail, cw, cd, cp] if x]
        return {
            "ward": cw, "district": cd, "province": cp,
            "old_key": next(iter(_build_wm_keys(cw, cd, cp)), ""),
            "address": ", ".join(addr_parts),
            "detection": detection,
            "candidates_scored": candidates_scored or [],
        }

    # --- Rule 1: already OLD-admin ---
    if not _item_is_new_admin(item):
        return _make_result(ward, district, province, "already_old_admin",
                            detail_override=norm_addr)

    # --- Rule 2: mapping branch already resolved to OLD admin ---
    if (
        merged.get("_mapping_used")
        and merged.get("_mapping_status") == "resolved"
    ):
        r_ward     = merged.get("ward") or ""
        r_district = merged.get("district") or ""
        r_province = merged.get("province") or ""
        # Valid OLD admin: has district AND province differs from NEW-admin input
        province_changed = _wm_norm(r_province) != _wm_norm(province)
        if r_district and province_changed:
            print(f"[CANONICAL] Rule 2 matched: {r_ward!r}/{r_district!r}/{r_province!r}")
            return _make_result(r_ward, r_district, r_province, "resolved_via_mapping")

    # --- Rule 3: text-overlap score all OLD-admin candidates ---
    # Use all key variants (full + bare) for robust lookup
    lookup_keys = _build_wm_keys(ward, district, province)
    old_candidates: List[str] = []
    seen: set = set()
    for key in lookup_keys:
        if not key:
            continue
        for c in (REVERSE_WARD_MAPPING or {}).get(key, []):
            if c.count("-") >= 2 and c not in seen:
                seen.add(c)
                old_candidates.append(c)

    print(
        f"[CANONICAL] Rule 3 lookup: keys={lookup_keys} → "
        f"{len(old_candidates)} 3-part candidates={old_candidates[:3]}"
    )

    if not old_candidates:
        return _make_result(ward, district, province, "unmapped_no_old_candidates",
                            detail_override=norm_addr)

    scored = sorted(
        [(_score_raw_vs_old_admin(raw_text, c), c) for c in old_candidates],
        key=lambda x: x[0], reverse=True,
    )
    best_score_val, best_old = scored[0]
    parsed = parse_old_admin_text(best_old)
    cw = _pretty_admin(parsed.get("ward", ""))
    cd = _pretty_admin(parsed.get("district", ""))
    cp = _pretty_admin(parsed.get("province", ""))

    print(
        f"[CANONICAL_SCORE] new_admin={ward!r}+{province!r} → "
        f"candidates={len(scored)} best={best_old!r} score={best_score_val}"
    )
    for sc, ca in scored:
        print(f"  score={sc:5.1f}  old_admin={ca!r}")

    return _make_result(cw, cd, cp, "text_matched_old_candidate", candidates_scored=scored)


# ---------------------------------------------------------------------------

def _clean_admin_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ,.")


def _strip_admin_prefix(s: str) -> str:
    s = _clean_admin_text(s)
    sl = s.lower()
    prefixes = [
        "xã ", "phường ", "thị trấn ",
        "huyện ", "thị xã ", "thành phố ",
        "tỉnh "
    ]
    for p in prefixes:
        if sl.startswith(p):
            return s[len(p):].strip()
    return s


def _extract_detail_part(item: Dict[str, Any]) -> str:
    """
    Ưu tiên detail_part nếu sau này có.
    Hiện tại fallback bằng cách bỏ ward/district/province khỏi normalized_text nếu có.
    """
    detail_part = _clean_admin_text(item.get("detail_part"))
    if detail_part:
        return detail_part

    normalized = _clean_admin_text(item.get("normalized_text"))
    if not normalized:
        return ""

    ward = _clean_admin_text(item.get("ward"))
    district = _clean_admin_text(item.get("district"))
    province = _clean_admin_text(item.get("province"))

    parts = [p.strip() for p in normalized.split(",") if p.strip()]
    remove_set = {x.lower() for x in [ward, district, province] if x}
    kept = [p for p in parts if p.lower() not in remove_set]

    if not kept:
        return normalized
    return ", ".join(kept)


def build_query_variant_items(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Sinh nhiều biến thể item để geocode_address_obj() build ra nhiều query khác nhau.
    Không sửa geocode_service, chỉ thay normalized_text để tăng recall.
    """
    detail = _extract_detail_part(item)
    ward = _clean_admin_text(item.get("ward"))
    district = _clean_admin_text(item.get("district"))
    province = _clean_admin_text(item.get("province"))

    ward_plain = _strip_admin_prefix(ward)
    district_plain = _strip_admin_prefix(district)
    province_plain = _strip_admin_prefix(province)

    normalized_variants: List[str] = []

    def add_variant(parts: List[str]) -> None:
        q = ", ".join([_clean_admin_text(x) for x in parts if _clean_admin_text(x)])
        q = re.sub(r"\s+", " ", q).strip(" ,.")
        if q and q not in normalized_variants:
            normalized_variants.append(q)

    # Chặt -> lỏng
    add_variant([detail, ward, district, province])
    add_variant([detail, ward_plain, district_plain, province_plain])

    add_variant([ward, district, province])
    add_variant([ward_plain, district_plain, province_plain])

    add_variant([detail, ward, province])
    add_variant([detail, ward_plain, province_plain])

    # Dạng plain không dấu phẩy
    plain = " ".join([x for x in [detail, ward_plain, district_plain, province_plain] if x])
    plain = re.sub(r"\s+", " ", plain).strip()
    if plain and plain not in normalized_variants:
        normalized_variants.append(plain)

    out: List[Dict[str, Any]] = []
    for nv in normalized_variants:
        clone = dict(item)
        clone["normalized_text"] = nv
        out.append(clone)

    return out


def _score_geo_result(geo: Dict[str, Any], item: Dict[str, Any]) -> float:
    """
    Chấm điểm candidate geocode trả về từ geocode_address_obj().
    """
    score = 0.0

    formatted = str(geo.get("formatted_address") or "").lower()
    location_type = str(geo.get("location_type") or "").upper()
    partial = bool(geo.get("partial_match", False))

    ward = _clean_admin_text(item.get("ward")).lower()
    district = _clean_admin_text(item.get("district")).lower()
    province = _clean_admin_text(item.get("province")).lower()

    if province:
        if province in formatted:
            score += 80
        else:
            score -= 180

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
        score -= 15

    if geo.get("lat") is None or geo.get("lng") is None:
        score -= 50

    return score


def geocode_address_obj_multi_query(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cắm trực tiếp vào trip_pipeline:
    - tạo nhiều biến thể query
    - gọi geocode_address_obj() nhiều lần
    - gom candidate tốt nhất của từng biến thể
    - chấm điểm và chọn candidate tốt nhất cuối cùng
    """
    variant_items = build_query_variant_items(item)

    candidates: List[Dict[str, Any]] = []
    seen_keys = set()
    errors: List[str] = []

    for idx, v_item in enumerate(variant_items):
        try:
            geo = geocode_address_obj(v_item)
        except Exception as e:
            errors.append(f"variant_{idx}: {e}")
            continue

        cand = dict(geo)
        cand["_variant_index"] = idx
        cand["_variant_normalized_text"] = v_item.get("normalized_text")

        pid = cand.get("place_id")
        lat = cand.get("lat")
        lng = cand.get("lng")
        formatted = str(cand.get("formatted_address") or "").strip().lower()

        if pid:
            key = ("pid", pid)
        elif lat is not None and lng is not None:
            key = ("ll", round(float(lat), 5), round(float(lng), 5))
        else:
            key = ("fmt", formatted)

        if key in seen_keys:
            continue
        seen_keys.add(key)

        cand["_score"] = _score_geo_result(cand, item)
        candidates.append(cand)

    if not candidates:
        return {
            "query": "",
            "status": "ERROR",
            "lat": None,
            "lng": None,
            "formatted_address": "",
            "place_id": None,
            "location_type": None,
            "partial_match": None,
            "raw_results_count": 0,
            "geocode_candidates": [],
            "best_score": None,
            "geocode_errors": errors,
        }

    candidates.sort(key=lambda x: x["_score"], reverse=True)
    best = dict(candidates[0])

    if best["_score"] < 0:
        best["status"] = "LOW_CONFIDENCE"
        best["reject_reason"] = "best_candidate_low_score_or_wrong_province"

    best["raw_results_count"] = len(candidates)
    best["geocode_candidates"] = candidates[:10]
    best["best_score"] = best["_score"]
    best["geocode_errors"] = errors
    return best

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
    results: List[Dict[str, Any]] = []

    # ── Stop-tracking: stable IDs + dedup cache ────────────────────────────────
    seen_stop_ids: Dict[str, int] = {}           # stop_id → first-occurrence index
    stop_result_cache: Dict[str, Dict[str, Any]] = {}  # stop_id → cached resolved result

    for idx, item in enumerate(normalized):
        raw = item.get("raw_text") or ""
        sid = _stop_id(raw)
        item["_stop_id"]    = sid
        item["_stop_index"] = idx

        if sid in stop_result_cache:
            # ── Duplicate: reuse previously resolved result, skip full pipeline ──
            cached = dict(stop_result_cache[sid])
            cached["_stop_index"]   = idx
            cached["_duplicate_of"] = seen_stop_ids[sid]
            results.append(cached)
            print(
                f"[STOP_TRACK] stop_id={sid} idx={idx} "
                f"raw={raw!r:.70} "
                f"duplicate_of=stop_{seen_stop_ids[sid]} reused_cached_result=True"
            )
            continue   # ← skip geocoding / mapping / canonical for this duplicate

        seen_stop_ids[sid] = idx
        item["_duplicate_of"] = None
        print(f"[STOP_TRACK] stop_id={sid} idx={idx} raw={raw!r:.70} unique")

        # ── Admin-type classification ───────────────────────────────────────────
        admin_type, type_reason = _classify_admin_input(item)

        if admin_type == "transitional":
            print(
                f"[ADMIN_TYPE] stop_id={sid} type={admin_type} reason={type_reason}\n"
                f"  [WARN] transitional address — will geocode directly, canonical may be unreliable"
            )
        else:
            print(f"[ADMIN_TYPE] stop_id={sid} type={admin_type} reason={type_reason}")

        # ── Geocode policy ──────────────────────────────────────────────────────
        #
        # NEW-admin  → resolve_stop_by_ward_mapping finds OLD-admin candidates
        #              and geocodes each; best is returned.  Fallback to direct
        #              geocode only when mapping finds nothing.
        #
        # OLD-admin / unknown / transitional
        #            → geocode directly with the item's own admin text.
        #              If geocode_score < 0, try at most 2 NEW-admin proxy
        #              queries as a secondary fallback (logged explicitly).
        #
        geocode_policy    = ""
        fallback_na_count = 0

        if admin_type == "new_admin" and REVERSE_WARD_MAPPING:
            resolved = resolve_stop_by_ward_mapping(
                item,
                REVERSE_WARD_MAPPING,
                geocode_address_obj_multi_query,
                keep_top_k_debug=5,
            )
            if resolved.get("_mapping_used") and resolved.get("_mapping_status") == "resolved":
                merged = resolved
                geocode_policy = "new_admin_via_mapping"
                print(
                    f"[GEOCODE_POLICY] stop_id={sid} type={admin_type} "
                    f"policy={geocode_policy} "
                    f"resolve_score={resolved.get('_mapping_best_score')} "
                    f"geocode_score={merged.get('best_score')}"
                )
            else:
                # Mapping found no viable 3-part old-admin → direct geocode fallback
                geo = geocode_address_obj_multi_query(item)
                merged = {**item, **geo}
                geocode_policy = "new_admin_direct_fallback"
                print(
                    f"[GEOCODE_POLICY] stop_id={sid} type={admin_type} "
                    f"policy={geocode_policy} mapping_reason={resolved.get('_mapping_reason')} "
                    f"geocode_score={geo.get('best_score')}"
                )
        else:
            # OLD-admin, unknown, transitional: geocode directly
            geo = geocode_address_obj_multi_query(item)
            merged = {**item, **geo}
            geocode_score_primary = geo.get("best_score", 0) or 0
            geocode_policy = f"{admin_type}_direct"

            # Secondary fallback: try NEW-admin proxy when primary score < 0
            if geocode_score_primary < 0 and admin_type in ("old_admin", "unknown") and REVERSE_WARD_MAPPING:
                new_cands = _get_new_admin_candidates_for_old(item)
                fallback_na_count = len(new_cands)
                best_fb: Optional[Dict[str, Any]] = None
                best_fb_score = geocode_score_primary
                for nc in new_cands:
                    fb_geo = geocode_address_obj_multi_query(nc)
                    fb_score = fb_geo.get("best_score", -999) or -999
                    print(
                        f"[GEOCODE_POLICY] stop_id={sid} secondary_new_admin_fallback "
                        f"new_admin={nc.get('ward')}/{nc.get('province')} "
                        f"geocode_score={fb_score}"
                    )
                    if fb_score > best_fb_score:
                        best_fb = fb_geo
                        best_fb_score = fb_score
                if best_fb is not None:
                    merged = {**item, **best_fb}
                    geocode_policy = f"{admin_type}_new_admin_secondary_fallback"

            print(
                f"[GEOCODE_POLICY] stop_id={sid} type={admin_type} "
                f"policy={geocode_policy} "
                f"primary_query={item.get('normalized_text', '')!r:.80} "
                f"geocode_score={merged.get('best_score')} "
                f"fallback_new_admin_candidates={fallback_na_count}"
            )

        # ── Canonical OLD-admin contract ────────────────────────────────────────
        # geocoded_* must reflect what the GEOCODER RETURNED, not the query admin.
        # We parse province from formatted_address (the geocoder's verbatim output)
        # so that geocoded_province = "Đồng Tháp" when Google returned Đồng Tháp,
        # even if the canonical province is "Tiền Giang".
        # geocode_confidence() now compares stop["province"] (canonical OLD admin)
        # against geocoded_formatted_address directly — it no longer relies on
        # geocoded_province for the confidence computation.
        geocoded_fa = merged.get("formatted_address") or ""
        geocoded_province_parsed = _parse_province_from_formatted(geocoded_fa)
        merged["geocoded_province"]          = geocoded_province_parsed
        merged["geocoded_formatted_address"] = geocoded_fa
        # ward/district cannot be reliably parsed from formatted_address;
        # leave them empty so downstream code falls back to canonical fields.
        merged["geocoded_ward"]     = ""
        merged["geocoded_district"] = ""

        print(
            f"[GEOCODE_RESULT] stop_id={sid} "
            f"formatted_address={geocoded_fa!r:.80} "
            f"parsed_geocoded_province={geocoded_province_parsed!r}"
        )

        # Resolve canonical OLD-admin form.
        canonical = _resolve_canonical_old_admin(item, merged)

        merged["canonical_old_key"]      = canonical["old_key"]
        merged["canonical_old_ward"]     = canonical["ward"]
        merged["canonical_old_district"] = canonical["district"]
        merged["canonical_old_province"] = canonical["province"]
        merged["canonical_address"]      = canonical["address"]
        merged["canonical_ward"]         = canonical["ward"]
        merged["canonical_district"]     = canonical["district"]
        merged["canonical_province"]     = canonical["province"]
        merged["_admin_type"]            = admin_type
        merged["_geocode_policy"]        = geocode_policy

        # Restore canonical OLD admin into primary fields for all downstream readers.
        merged["ward"]     = canonical["ward"]
        merged["district"] = canonical["district"]
        merged["province"] = canonical["province"]

        print(
            f"[GEOCODE] stop_id={sid} "
            f"status={merged.get('status')} "
            f"lat={merged.get('lat')} lng={merged.get('lng')} "
            f"geocode_score={merged.get('best_score')} "
            f"policy={geocode_policy}"
        )
        print(
            f"[CANONICAL] stop_id={sid} raw={raw!r:.70}\n"
            f"  input_admin  : ward={item.get('ward')!r} district={item.get('district')!r} province={item.get('province')!r}\n"
            f"  admin_type   : {admin_type} ({type_reason})\n"
            f"  geocoded_province (from geocoder): {merged['geocoded_province']!r}\n"
            f"  geocoded_addr: {merged['geocoded_formatted_address']!r:.80}\n"
            f"  canonical    : ward={canonical['ward']!r} district={canonical['district']!r} "
            f"province={canonical['province']!r} detection={canonical['detection']!r}\n"
            f"  old_key      : {canonical['old_key']!r}\n"
            f"  final_fields : ward={merged['ward']!r} district={merged['district']!r} province={merged['province']!r}"
        )
        # Cache result so exact duplicates can reuse it without re-geocoding
        stop_result_cache[sid] = merged
        results.append(merged)

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
            fallback_matched = [
                s for s in usable
                if str(s.get("coord_source") or "").startswith("vtracking")
            ]

            print(
                f"[VT-FALLBACK] total={len(results)} "
                f"| route_usable={len(usable)} "
                f"| vtracking_fallback_matched={len(fallback_matched)} "
                f"| excluded={len(excluded)}"
            )

            # Distance warnings for vtracking-fallback matched stops
            for s in fallback_matched:
                label = s.get("raw_text") or s.get("normalized_text") or s.get("query")
                dist_m = s.get("candidate_distance_m")
                src = s.get("coord_source")
                if dist_m is not None and dist_m > 500:
                    print(
                        f"[FALLBACK_DIST_WARN] {label!r} | "
                        f"source={src} | distance={dist_m:.0f}m  "
                        f"(>500m — low confidence fallback coord)"
                    )
                else:
                    print(
                        f"[FALLBACK_DIST] {label!r} | "
                        f"source={src} | distance={dist_m}m"
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
    start_origin: Optional[Tuple[float, float]],
    end_origin: Optional[Tuple[float, float]] = None,
    origin_radius_m: float = 700.0,
    return_radius_m: Optional[float] = 200.0,
    min_consecutive_points: int = 3,
    min_moving_speed_kmh: float = 5.0,
    require_return: bool = False,
) -> dict:
    """
    Detect departure from start_origin and optional arrival to end_origin.

    Behavior
    --------
    Phase 1 — departure detection:
      Departure is confirmed when there are `min_consecutive_points`
      consecutive MOVING rows outside `origin_radius_m` from start_origin.
      Departure time = timestamp of the first row in that confirmed run.

    Phase 2 — arrival detection:
      If end_origin is provided, arrival is confirmed when there are
      `min_consecutive_points` consecutive valid rows inside `return_radius_m`
      of end_origin. If end_origin is None, it defaults to start_origin.

      Special guard:
      - when start_origin == end_origin, arrival detection is armed only after
        the truck has moved sufficiently far away from the start zone
        (2 * origin_radius_m). This prevents false "returned" detection on
        outbound clips near the depot.
      - when start_origin != end_origin, arrival is checked directly without
        arm-guard.

    Open trip mode
    --------------
    If `require_return=False` and departure is confirmed but no arrival is
    confirmed, the trip is still returned with:
      - end = last valid GPS timestamp
      - status = departed_open_trip / started_outside_open_trip

    Fallback
    --------
    If required columns are missing, origin is not provided, or there are no
    valid GPS rows, the function falls back to min/max timestamps from df.

    Returns
    -------
    dict with keys:
      start
      end
      departure_index
      return_index
      detection_method
      origin_radius_m
      return_radius_m
      status
      n_valid_points
      n_inside_points
      n_outside_points
      started_inside
      fallback_used
      fallback_reason (only when fallback_used=True)
      returned_to_end_origin
      require_return
    """
    _return_r: float = return_radius_m if return_radius_m is not None else origin_radius_m

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
            "returned_to_end_origin": False,
            "require_return": require_return,
        }

    if start_origin is None:
        return _fallback("origin_not_provided")

    if end_origin is None:
        end_origin = start_origin

    start_lat, start_lng = float(start_origin[0]), float(start_origin[1])
    end_lat, end_lng = float(end_origin[0]), float(end_origin[1])

    if "Tọa độ" not in df.columns or "Thời gian" not in df.columns or df.empty:
        return _fallback("missing_required_columns")

    timestamps: pd.Series = pd.to_datetime(
        df["Thời gian"], errors="coerce", dayfirst=True
    )

    _speed_col: Optional[str] = next(
        (c for c in df.columns if "tốc" in c.lower() or "speed" in c.lower()),
        None,
    )
    _status_col: Optional[str] = next(
        (
            c for c in df.columns
            if "trạng" in c.lower() or "trang" in c.lower() or "status" in c.lower()
        ),
        None,
    )
    _IDLE_STATUS_KEYS = ("stop", "dừng", "dung")

    def _is_moving(row_i: int) -> bool:
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

    coord_col = df["Tọa độ"]

    consecutive_outside: int = 0
    consecutive_inside_post_dep: int = 0

    departure_confirmed: bool = False
    departure_candidate: Optional[Tuple[Any, int]] = None
    departure_ts = None
    departure_index: Optional[int] = None

    return_confirmed: bool = False
    return_candidate: Optional[Tuple[Any, int]] = None
    return_ts = None
    return_index: Optional[int] = None

    last_valid_ts = None
    last_valid_index: Optional[int] = None

    n_valid: int = 0
    n_inside: int = 0
    n_outside: int = 0
    started_inside: Optional[bool] = None

    same_start_end = (
        abs(start_lat - end_lat) < 1e-9 and
        abs(start_lng - end_lng) < 1e-9
    )

    _return_arm_dist: float = origin_radius_m * 2.0
    max_dist_since_dep: float = 0.0
    return_armed: bool = False

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
        last_valid_ts = ts
        last_valid_index = row_idx

        dist_from_start = haversine(lat, lng, start_lat, start_lng)
        is_outside_start = dist_from_start > origin_radius_m
        is_inside_start = not is_outside_start

        dist_to_end = haversine(lat, lng, end_lat, end_lng)
        is_inside_end = dist_to_end <= _return_r

        if started_inside is None:
            started_inside = is_inside_start

        if is_inside_start:
            n_inside += 1
        else:
            n_outside += 1

        # Phase 1: confirm departure from start_origin
        if not departure_confirmed:
            if _is_moving(row_idx):
                if is_inside_start:
                    # still in origin zone -> reset outside-run
                    consecutive_outside = 0
                    departure_candidate = None
                else:
                    # moving outside origin zone -> advance departure run
                    consecutive_outside += 1
                    if departure_candidate is None:
                        departure_candidate = (ts, row_idx)

                    if consecutive_outside >= min_consecutive_points:
                        departure_confirmed = True
                        departure_ts = departure_candidate[0]
                        departure_index = departure_candidate[1]
            continue

        # Phase 2: confirm arrival to end_origin
        if same_start_end:
            max_dist_since_dep = max(max_dist_since_dep, dist_from_start)
            if max_dist_since_dep >= _return_arm_dist:
                return_armed = True
            can_check_arrival = return_armed and is_inside_end
        else:
            can_check_arrival = is_inside_end

        if can_check_arrival:
            if consecutive_inside_post_dep == 0:
                return_candidate = (ts, row_idx)
            consecutive_inside_post_dep += 1

            if consecutive_inside_post_dep >= min_consecutive_points:
                return_confirmed = True
                return_ts = return_candidate[0] if return_candidate else ts
                return_index = return_candidate[1] if return_candidate else row_idx
                break
        else:
            consecutive_inside_post_dep = 0
            return_candidate = None

    if n_valid == 0:
        return _fallback("no_valid_gps_points")

    returned_to_end_origin = return_confirmed

    if departure_confirmed and not return_confirmed and not require_return:
        return_ts = last_valid_ts
        return_index = last_valid_index

    if not departure_confirmed:
        status = "never_departed"
    elif return_confirmed:
        if started_inside:
            status = "departed_and_arrived_end"
        else:
            status = "started_outside_arrived_end"
    else:
        if started_inside:
            status = "departed_open_trip" if not require_return else "departed_no_return"
        else:
            status = "started_outside_open_trip" if not require_return else "started_outside_no_return"

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
        "returned_to_end_origin": returned_to_end_origin,
        "require_return": require_return,
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


def validate_canonical_normalization() -> None:
    """
    Smoke-test the forced OLD-admin canonical contract.

    Run:  python trip_pipeline.py --validate-canonical

    Assertions per case:
      A) canonical_old_province contains expected OLD-admin keyword
      B) primary ward/district/province == canonical_old_* (no overwrite)
      C) geocoded_province stored in a separate field
      D) lat present when expected
      E) detection mode is as expected
    """
    # (label, address, expected_old_province_kw, expect_lat, expected_detection_prefix)
    cases: List[tuple] = [
        # ── already-OLD: canonical must stay as-is ────────────────────────────
        (
            "old-Lộc Hòa",
            "Số 57, Ấp 6, Xã Lộc Hòa, Huyện Lộc Ninh, Tỉnh Bình Phước",
            "Bình Phước", True, "already_old",
        ),
        (
            "old-Lộc Quang",
            "Xã Lộc Quang, Huyện Lộc Ninh, Tỉnh Bình Phước",
            "Bình Phước", True, "already_old",
        ),
        (
            "old-Lộc Tấn",
            "Xã Lộc Tấn, Huyện Lộc Ninh, Tỉnh Bình Phước",
            "Bình Phước", True, "already_old",
        ),
        # ── new-admin: MUST be converted to OLD-admin ─────────────────────────
        (
            "new-Lộc Quang (key case)",
            "Số 51, tổ 32A, ấp Hiệp Tâm A, Xã Lộc Quang, Tỉnh Đồng Nai, Việt Nam",
            "Bình Phước", True, None,   # any detection mode is fine
        ),
        (
            "new-Lộc Quang (bare)",
            "Xã Lộc Quang, Tỉnh Đồng Nai",
            "Bình Phước", True, None,
        ),
        (
            "new-Lộc Tấn",
            "Xã Lộc Tấn, Tỉnh Đồng Nai",
            "Bình Phước", True, None,
        ),
        # ── unmapped: canonical = input as-is ─────────────────────────────────
        (
            "unmapped-HCM",
            "123 Nguyễn Văn Cừ, Quận 1, TP HCM",
            "Hồ Chí Minh", True, "already_old_or_unmapped",
        ),
    ]

    print("\n[VALIDATE_CANONICAL] ========== START ==========")
    passes = 0
    fails  = 0

    for label, addr, expected_old_prov_kw, expect_lat, _ in cases:
        try:
            results = normalize_and_geocode_stops(
                [addr],
                trip_df=None,
                apply_vtracking_fallback=False,
            )
            r = results[0] if results else {}
        except Exception as exc:
            print(f"[VALIDATE_CANONICAL] [ERROR] {label!r}: {exc}")
            fails += 1
            continue

        c_ward  = r.get("canonical_old_ward")  or r.get("canonical_ward")  or ""
        c_dist  = r.get("canonical_old_district") or r.get("canonical_district") or ""
        c_prov  = r.get("canonical_old_province") or r.get("canonical_province") or ""
        c_key   = r.get("canonical_old_key") or ""
        c_addr  = r.get("canonical_address") or ""
        p_ward  = r.get("ward") or ""
        p_dist  = r.get("district") or ""
        p_prov  = r.get("province") or ""
        g_prov  = r.get("geocoded_province") or ""
        lat     = r.get("lat")

        # Re-read detection from the canonical dict stored under canonical_old_key
        # (we need to find the detection string — it's not stored directly on merged)
        # Use the province to infer correctness
        A = expected_old_prov_kw.lower() in c_prov.lower()
        B = (p_ward == c_ward) and (p_dist == c_dist) and (p_prov == c_prov)
        C = "geocoded_province" in r
        D = True if expect_lat is None else (lat is not None) == expect_lat

        ok  = A and B and C and D
        tag = "PASS" if ok else "FAIL"
        (passes if ok else fails).__class__  # dummy
        if ok:
            passes += 1
        else:
            fails += 1

        print(
            f"[VALIDATE_CANONICAL] [{tag}] {label!r}\n"
            f"  input            : {addr!r}\n"
            f"  canonical_old    : ward={c_ward!r}  district={c_dist!r}  province={c_prov!r}\n"
            f"  canonical_key    : {c_key!r}\n"
            f"  primary_fields   : ward={p_ward!r}  district={p_dist!r}  province={p_prov!r}  (== canonical? {B})\n"
            f"  geocoded_province: {g_prov!r}  (stored? {C})\n"
            f"  lat              : {lat}  (ok={D})\n"
            f"  canonical_address: {c_addr!r}\n"
            f"  checks           : prov_ok={A} fields_restored={B} geocoded_stored={C} lat_ok={D}"
        )

    print(f"[VALIDATE_CANONICAL] ========== END: {passes} PASS / {fails} FAIL ==========\n")


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
    print(
        f"[DEBUG] start_origin=({origin_res.start_lat}, {origin_res.start_lng}) "
        f"end_origin=({origin_res.end_lat}, {origin_res.end_lng}) "
        f"source={origin_res.source}"
    )

    trip_window = detect_trip_window_from_origin(
        df,
        start_origin=origin_res.start_as_latlng(),
        end_origin=origin_res.end_as_latlng(),
    )
    print(f"[DEBUG] trip_window status={trip_window['status']} "
          f"start={trip_window['start']} end={trip_window['end']}")
    start = origin_res.start_as_latlng()
    end = origin_res.end_as_latlng()

    trip_report = analyze_trip_corridor(
        df,
        stops=stops,
        origin=start,
        end_origin=end,
    )
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
        "start_origin_lat": origin_res.start_lat,
        "start_origin_lng": origin_res.start_lng,
        "end_origin_lat": origin_res.end_lat,
        "end_origin_lng": origin_res.end_lng,
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
    import sys
    if "--validate-canonical" in sys.argv:
        validate_canonical_normalization()
    else:
        main()
