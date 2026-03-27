import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ,.-")


_ADMIN_PREFIXES = (
    "xã ",
    "phường ",
    "thị trấn ",
    "huyện ",
    "quận ",
    "thị xã ",
    "thành phố ",
    "tỉnh ",
)


def _strip_prefix_keep_name(s: str) -> str:
    """Strip admin-level prefix from a single normalized segment."""
    s = _norm_text(s)
    for p in _ADMIN_PREFIXES:
        if s.startswith(p):
            return s[len(p):].strip()
    return s


def _bare_key(combined: str) -> str:
    """
    Strip admin-level prefixes from every '-'-separated segment and rejoin.
    E.g. "xã lộc quang-tỉnh đồng nai" → "lộc quang-đồng nai"
    """
    parts = [p.strip() for p in combined.split("-") if p.strip()]
    return "-".join(_strip_prefix_keep_name(p) for p in parts)


def _admin_key_from_parts(ward: str = "", district: str = "", province: str = "") -> str:
    """
    Key chuẩn cho stop hiện tại, ưu tiên:
    ward-district-province
    fallback:
    ward-province
    """
    ward = _norm_text(ward)
    district = _norm_text(district)
    province = _norm_text(province)

    if ward and district and province:
        return f"{ward}-{district}-{province}"
    if ward and province:
        return f"{ward}-{province}"
    if ward:
        return ward
    return ""


def _admin_key_from_new_admin_text(new_admin_text: str) -> str:
    """
    Key cho phía 'new admin' trong file mapping.
    Ví dụ:
    'Xã Lộc Quang-Tỉnh Đồng Nai' -> 'xã lộc quang-tỉnh đồng nai'
    """
    return _norm_text(new_admin_text)


def _parse_old_admin_text(old_admin_text: str) -> Dict[str, str]:
    """
    Parse old_admin_text dạng:
    'xã lộc quang-huyện lộc ninh-tỉnh bình phước'
    """
    raw = _norm_text(old_admin_text)
    parts = [p.strip() for p in raw.split("-") if p.strip()]

    out = {
        "ward": "",
        "district": "",
        "province": "",
        "raw_admin_text": raw,
    }

    for p in parts:
        if p.startswith(("xã ", "phường ", "thị trấn ")):
            out["ward"] = p
        elif p.startswith(("huyện ", "quận ", "thị xã ", "thành phố ")):
            out["district"] = p
        elif p.startswith("tỉnh "):
            out["province"] = p

    return out


def load_old_admin_key_set(path: str) -> set:
    """
    Return the set of normalized OLD-admin key strings from ward_mapping_2025.json.

    Stores BOTH:
      • full-prefix form: "xã thanh bình-huyện chợ gạo-tỉnh tiền giang"
      • bare form:        "thanh bình-chợ gạo-tiền giang"

    The bare form is essential because GPT often strips admin-level prefixes
    ("xã ", "huyện ", "tỉnh ") from ward/district/province fields, making the
    full-prefix key un-matchable.  Storing both ensures _classify_admin_input()
    correctly identifies these as OLD-admin even without prefixes.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result: set = set()
    for k in raw.keys():
        full_key = _norm_text(k)
        result.add(full_key)
        # Also add the bare form (no admin-level prefixes) for prefix-agnostic lookup
        bare = _bare_key(full_key)
        if bare:
            result.add(bare)
    return result


def parse_old_admin_text(s: str) -> Dict[str, str]:
    """Public wrapper for _parse_old_admin_text."""
    return _parse_old_admin_text(s)


def load_reverse_ward_mapping(path: str) -> Dict[str, List[str]]:
    """
    Build a bidirectional lookup dict from ward_mapping_2025.json.

    JSON format:
      key   = OLD admin (with prefixes, lowercase) e.g. "xã lộc quang-huyện lộc ninh-tỉnh bình phước"
      value = NEW admin (Title Case)               e.g. "Xã Lộc Quang-Tỉnh Đồng Nai"

    The returned dict contains FOUR types of entries per JSON row
    so that lookups succeed regardless of whether the caller preserves or
    strips admin-level prefixes ("xã ", "tỉnh ", etc.):

      1.  full new_key  → [old_key]   "xã lộc quang-tỉnh đồng nai"               → [old]
      2.  bare new_key  → [old_key]   "lộc quang-đồng nai"                         → [old]
      3.  full old_key  → [new_key]   "xã lộc quang-huyện lộc ninh-tỉnh bình phước"→ [new]  (bidirectional)
      4.  bare old_key  → [new_key]   "lộc quang-lộc ninh-bình phước"              → [new]  (bidirectional)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rev: Dict[str, List[str]] = defaultdict(list)

    for old_admin, new_admin in raw.items():
        old_key = _norm_text(old_admin)   # already lowercase in JSON, just normalize
        new_key = _norm_text(new_admin)    # Title Case → lowercase

        if not old_key or not new_key:
            continue

        # 1. full new → old
        rev[new_key].append(old_key)
        # 2. bare new → old (GPT may strip prefixes from new-admin ward/province)
        bare_new = _bare_key(new_key)
        if bare_new and bare_new != new_key:
            rev[bare_new].append(old_key)
        # 3. full old → new (bidirectional: old-admin stop, geocode via new-admin proxy)
        rev[old_key].append(new_key)
        # 4. bare old → new
        bare_old = _bare_key(old_key)
        if bare_old and bare_old != old_key:
            rev[bare_old].append(new_key)

    # Sort for stable output
    result = {k: sorted(set(v)) for k, v in rev.items()}

    print(
        f"[REVERSE_MAPPING] loaded {len(raw)} JSON entries → "
        f"{len(result)} reverse-mapping keys "
        f"(including bare-key aliases)"
    )
    return result


def _extract_ward_from_text(text: str) -> str:
    """
    Extract a ward-like token from free-form address text.
    Looks for "xã X", "phường X", "thị trấn X" patterns.
    Returns the full prefixed form e.g. "thị trấn chợ gạo", or "" if not found.
    """
    t = _norm_text(text)
    for pfx in ("xã ", "phường ", "thị trấn "):
        idx = t.find(pfx)
        if idx >= 0:
            rest = t[idx:]
            # Read until next comma, hyphen, or end
            m = re.split(r"[,\-\n]", rest, maxsplit=1)
            ward_tok = m[0].strip() if m else ""
            if ward_tok:
                return ward_tok
    return ""


def _build_lookup_keys(ward: str, district: str, province: str) -> List[str]:
    """
    Build all lookup key variants for a given ward/district/province.
    Covers: full-prefix, bare (no prefix), and mixed combinations.
    Returns [] when ward is empty (caller should handle missing-ward fallback).
    Deduplicates while preserving priority order.
    """
    w_full = _norm_text(ward)
    d_full = _norm_text(district)
    p_full = _norm_text(province)
    w_bare = _strip_prefix_keep_name(w_full)
    d_bare = _strip_prefix_keep_name(d_full)
    p_bare = _strip_prefix_keep_name(p_full)

    candidates = []

    if not w_full:
        # No ward: no useful keys for ward-based mapping
        return []

    # Full-prefix combos (3-part and 2-part)
    if w_full and d_full and p_full:
        candidates.append(f"{w_full}-{d_full}-{p_full}")
    if w_full and p_full:
        candidates.append(f"{w_full}-{p_full}")

    # Bare combos
    if w_bare and d_bare and p_bare and (w_bare != w_full or d_bare != d_full or p_bare != p_full):
        candidates.append(f"{w_bare}-{d_bare}-{p_bare}")
    if w_bare and p_bare and f"{w_bare}-{p_bare}" not in candidates:
        candidates.append(f"{w_bare}-{p_bare}")

    # Mixed: bare ward + full province (common when GPT omits "xã " but keeps "Tỉnh ")
    if w_bare and p_full and f"{w_bare}-{p_full}" not in candidates:
        candidates.append(f"{w_bare}-{p_full}")

    # Mixed: full ward + bare province
    if w_full and p_bare and f"{w_full}-{p_bare}" not in candidates:
        candidates.append(f"{w_full}-{p_bare}")

    # Deduplicate preserving order
    seen = set()
    out = []
    for k in candidates:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def expand_old_admin_candidates(stop: Dict[str, Any], reverse_mapping: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Từ 1 stop thuộc địa chỉ mới, expand thành nhiều stop ứng viên địa chỉ cũ.

    Input stop kỳ vọng có:
    - raw_text
    - normalized_text
    - ward
    - district
    - province

    Output: list stop candidates (OLD-admin form), sorted by mapping rank.
    """
    ward = stop.get("ward") or ""
    district = stop.get("district") or ""
    province = stop.get("province") or ""

    # When ward is missing, try to extract it from the raw/normalized text
    if not ward:
        raw_for_extract = str(stop.get("raw_text") or stop.get("normalized_text") or "")
        ward = _extract_ward_from_text(raw_for_extract)
        if ward:
            print(
                f"[MISSING_WARD] raw={stop.get('raw_text', '')!r:.80} "
                f"→ extracted ward={ward!r} from text"
            )
        else:
            # No ward available — try district+province fallback key
            d_f = _norm_text(district)
            p_f = _norm_text(province)
            d_b = _strip_prefix_keep_name(d_f)
            p_b = _strip_prefix_keep_name(p_f)
            fallback_keys = []
            if d_f and p_f:
                fallback_keys.append(f"{d_f}-{p_f}")
            if d_b and p_b and f"{d_b}-{p_b}" not in fallback_keys:
                fallback_keys.append(f"{d_b}-{p_b}")
            print(
                f"[MISSING_WARD] raw={stop.get('raw_text', '')!r:.80} "
                f"ward still empty — using district+province fallback_keys={fallback_keys}"
            )
            lookup_keys = fallback_keys
            old_admin_candidates: List[str] = []
            seen_cands: set = set()
            for key in lookup_keys:
                for cand in reverse_mapping.get(key, []):
                    # CRITICAL: only accept 3-part OLD-admin (ward-district-province)
                    if cand.count("-") >= 2 and cand not in seen_cands:
                        seen_cands.add(cand)
                        old_admin_candidates.append(cand)
            print(
                f"[EXPAND] ward='' (missing) district={district!r} province={province!r}\n"
                f"  fallback_lookup_keys={lookup_keys}\n"
                f"  found={len(old_admin_candidates)} 3-part candidates={old_admin_candidates[:3]}"
            )
            if not old_admin_candidates:
                return []
            # Skip the normal key-building below and jump to expand
            _do_normal_expand = False
    else:
        _do_normal_expand = True

    if _do_normal_expand:
        lookup_keys = _build_lookup_keys(ward, district, province)
        old_admin_candidates = []
        seen_cands = set()
        for key in lookup_keys:
            for cand in reverse_mapping.get(key, []):
                # CRITICAL: only collect 3-part OLD-admin entries (ward-district-province).
                # The bidirectional mapping also stores old→new (2-part), which must
                # NOT be used as geocode candidates — they are the wrong direction.
                if cand.count("-") >= 2 and cand not in seen_cands:
                    seen_cands.add(cand)
                    old_admin_candidates.append(cand)

        print(
            f"[EXPAND] ward={ward!r} district={district!r} province={province!r}\n"
            f"  lookup_keys={lookup_keys}\n"
            f"  found={len(old_admin_candidates)} 3-part old-admin candidates={old_admin_candidates[:3]}"
        )

    if not old_admin_candidates:
        return []

    expanded: List[Dict[str, Any]] = []

    raw_text = str(stop.get("raw_text") or "")
    normalized_text = str(stop.get("normalized_text") or "")

    # Keep detail text by stripping current admin tokens from the tail of normalized_text
    detail_text = normalized_text
    for token in [province, district, ward]:
        token_norm = _norm_text(token)
        if token_norm and _norm_text(detail_text).endswith(token_norm):
            idx = _norm_text(detail_text).rfind(token_norm)
            if idx > 0:
                detail_text = detail_text[:idx].rstrip(" ,-")
    detail_text = detail_text.strip(" ,")

    for i, old_admin in enumerate(old_admin_candidates):
        parsed = _parse_old_admin_text(old_admin)

        candidate = dict(stop)
        candidate["_mapping_source"] = "ward_mapping_2025"
        candidate["_mapping_rank"] = i
        candidate["_mapping_old_admin"] = old_admin
        candidate["_mapping_lookup_keys"] = lookup_keys

        old_ward = parsed.get("ward", "")
        old_district = parsed.get("district", "")
        old_province = parsed.get("province", "")

        def pretty(s: str) -> str:
            if not s:
                return ""
            return " ".join(
                w.capitalize() if w not in {"tp", "tx", "ql"} else w.upper()
                for w in s.split()
            )

        candidate["ward"] = pretty(old_ward)
        candidate["district"] = pretty(old_district)
        candidate["province"] = pretty(old_province)

        normalized_parts = [detail_text, candidate["ward"], candidate["district"], candidate["province"]]
        candidate["normalized_text"] = ", ".join([x for x in normalized_parts if x]).strip(" ,")

        candidate["_mapping_detail_text"] = detail_text
        candidate["_mapping_from_raw_text"] = raw_text

        expanded.append(candidate)

    return expanded


def _score_hamlet_tokens(raw_text: str, candidate: Dict[str, Any]) -> float:
    """
    Bonus score for hamlet-level tokens (ấp, tổ, xóm, thôn) in raw_text
    overlapping with the candidate's detail text.
    """
    raw = _norm_text(raw_text)
    detail = _norm_text(candidate.get("_mapping_detail_text", ""))
    if not detail:
        return 0.0
    # Extract tokens >= 3 chars from detail that appear in raw
    detail_tokens = [t for t in re.split(r"[,\s]+", detail) if len(t) >= 3]
    matched = sum(1 for t in detail_tokens if t in raw)
    return min(matched * 3.0, 15.0)


def _score_text_overlap(raw_text: str, candidate: Dict[str, Any]) -> float:
    raw = _norm_text(raw_text)
    score = 0.0

    ward = _strip_prefix_keep_name(candidate.get("ward", ""))
    district = _strip_prefix_keep_name(candidate.get("district", ""))
    province = _strip_prefix_keep_name(candidate.get("province", ""))

    if ward and ward in raw:
        score += 20.0
    if district and district in raw:
        score += 8.0
    if province and province in raw:
        score += 5.0

    score += _score_hamlet_tokens(raw_text, candidate)

    return score


def _score_geo_result(geo: Dict[str, Any], stop_candidate: Dict[str, Any]) -> float:
    """
    Score a geocode result for one candidate stop.
    """
    score = 0.0

    status = str(geo.get("status") or "")
    if status == "REJECTED":
        return -999.0
    if status not in {"OK", "LOW_CONFIDENCE", "ZERO_RESULTS"}:
        score -= 30.0

    formatted = _norm_text(geo.get("formatted_address") or "")
    location_type = str(geo.get("location_type") or "").upper()
    partial = bool(geo.get("partial_match", False))

    ward = _strip_prefix_keep_name(stop_candidate.get("ward", ""))
    district = _strip_prefix_keep_name(stop_candidate.get("district", ""))
    province = _strip_prefix_keep_name(stop_candidate.get("province", ""))

    if province:
        if province in formatted:
            score += 60.0
        else:
            score -= 120.0

    if district and district in formatted:
        score += 20.0

    if ward and ward in formatted:
        score += 12.0

    if location_type == "ROOFTOP":
        score += 12.0
    elif location_type == "GEOMETRIC_CENTER":
        score += 5.0
    elif location_type == "APPROXIMATE":
        score -= 8.0

    if partial:
        score -= 10.0

    if geo.get("lat") is None or geo.get("lng") is None:
        score -= 40.0

    score += _score_text_overlap(stop_candidate.get("raw_text", ""), stop_candidate)

    return score


def resolve_stop_by_ward_mapping(
    stop: Dict[str, Any],
    reverse_mapping: Dict[str, List[str]],
    geocode_func,
    keep_top_k_debug: int = 5,
) -> Dict[str, Any]:
    """
    geocode_func(candidate_stop) phải trả về dict geo kiểu geocode_address_obj(...)
    hoặc geocode_address_obj_multi_query(...)

    Trả về dict stop đã resolve, gồm:
    - dữ liệu stop tốt nhất
    - geocode tốt nhất
    - debug candidates
    """
    print(
        f"[RESOLVE] raw={stop.get('raw_text')!r} "
        f"ward={stop.get('ward')!r} district={stop.get('district')!r} province={stop.get('province')!r}"
    )
    candidates = expand_old_admin_candidates(stop, reverse_mapping)
    print(f"[RESOLVE] old_admin_candidates={len(candidates)}")

    if not candidates:
        return {
            **stop,
            "_mapping_used": False,
            "_mapping_reason": "no_reverse_mapping_match",
        }

    debug_rows: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None

    for cand in candidates:
        geo = geocode_func(cand)
        s = _score_geo_result(geo, cand)

        row = {
            "candidate_stop": cand,
            "geo": geo,
            "_resolve_score": s,
        }
        debug_rows.append(row)

        if best_row is None or s > best_score:
            best_row = row
            best_score = s

    debug_rows.sort(key=lambda x: x["_resolve_score"], reverse=True)
    print(
        f"[RESOLVE] best_score={best_score} "
        f"best_candidate={best_row['candidate_stop'].get('normalized_text') if best_row else None}"
    )

    if not best_row:
        return {
            **stop,
            "_mapping_used": True,
            "_mapping_status": "no_best_candidate",
            "_mapping_debug": debug_rows[:keep_top_k_debug],
        }

    best_stop = dict(best_row["candidate_stop"])
    best_geo = dict(best_row["geo"])

    resolved = {
        **best_stop,
        **best_geo,
        "_mapping_used": True,
        "_mapping_status": "resolved",
        "_mapping_best_score": best_row["_resolve_score"],
        "_mapping_debug": debug_rows[:keep_top_k_debug],
    }

    return resolved
