from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

import maps_config

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
CACHE_PATH = Path(__file__).resolve().parent / "cache_geocode.json"


def _load_cache() -> Dict[str, dict]:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


_CACHE = _load_cache()


def _save_cache() -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_CACHE, f, ensure_ascii=False, indent=2)


def build_query(addr_obj: dict) -> str:
    """Ghép chuỗi geocode từ dữ liệu GPT chuẩn hóa."""
    parts: List[str] = []

    normalized_text = str(addr_obj.get("normalized_text") or "").strip()
    ward = str(addr_obj.get("ward") or "").strip()
    district = str(addr_obj.get("district") or "").strip()
    province = str(addr_obj.get("province") or "").strip()

    if normalized_text:
        parts.append(normalized_text)
    for p in [ward, district, province]:
        if p and p not in parts:
            parts.append(p)
    if not parts or parts[-1].lower() != "việt nam":
        parts.append("Việt Nam")
    return ", ".join(parts)


def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()


def geocode_query(query: str, api_key: Optional[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
    api_key = api_key or maps_config.get_api_key()

    key = _cache_key(query)
    if not force_refresh and key in _CACHE:
        return _CACHE[key]

    params = {
        "address": query,
        "key": api_key,
        "language": "vi",
        "region": "vn",
    }
    response = requests.get(GEOCODE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    result: Dict[str, Any]
    if data.get("status") != "OK" or not data.get("results"):
        result = {
            "query": query,
            "status": data.get("status"),
            "lat": None,
            "lng": None,
            "formatted_address": None,
            "place_id": None,
            "location_type": None,
            "partial_match": None,
            "raw_results_count": len(data.get("results") or []),
        }
    else:
        best = data["results"][0]
        loc = best.get("geometry", {}).get("location", {})
        result = {
            "query": query,
            "status": data.get("status"),
            "lat": loc.get("lat"),
            "lng": loc.get("lng"),
            "formatted_address": best.get("formatted_address"),
            "place_id": best.get("place_id"),
            "location_type": best.get("geometry", {}).get("location_type"),
            "partial_match": best.get("partial_match", False),
            "raw_results_count": len(data.get("results") or []),
        }

    _CACHE[key] = result
    _save_cache()
    return result


def geocode_address_obj(addr_obj: dict, api_key: Optional[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
    query = build_query(addr_obj)
    return geocode_query(query=query, api_key=api_key, force_refresh=force_refresh)


def geocode_many(addresses: Iterable[dict], api_key: Optional[str] = None, force_refresh: bool = False) -> List[dict]:
    rows: List[dict] = []
    for addr in addresses:
        geo = geocode_address_obj(addr, api_key=api_key, force_refresh=force_refresh)
        rows.append({**addr, **geo})
    return rows


if __name__ == "__main__":
    sample = {
        "normalized_text": "Ấp Bắc, Thị trấn Vĩnh Bình, Huyện Gò Công Tây, Tiền Giang",
        "province": "Tiền Giang",
        "district": "Huyện Gò Công Tây",
        "ward": "Thị trấn Vĩnh Bình",
    }
    print(geocode_address_obj(sample))
