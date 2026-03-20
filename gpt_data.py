from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

CACHE_PATH = Path(__file__).resolve().parent / "cache_gpt_address.json"

def _load_cache() -> Dict[str, Any]:
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


def _get_client() -> OpenAI:
    return OpenAI(api_key='...')


def _normalize_cache_key(addresses: List[str]) -> str:
    return "||".join([str(a).strip() for a in addresses])


def _extract_json_array(result_text: str) -> List[dict]:
    candidates = [
        re.search(r"```json\s*(.*?)\s*```", result_text, re.DOTALL),
        re.search(r"(\[.*\])", result_text, re.DOTALL),
    ]
    for match in candidates:
        if not match:
            continue
        raw = match.group(1).strip()
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    raise ValueError(f"Không tìm thấy JSON array hợp lệ trong phản hồi GPT: {result_text[:500]}")


def _build_prompt(addresses: List[str]) -> str:
    lines = [
        "Bạn là chuyên gia chuẩn hoá địa chỉ Việt Nam.",
        "Với mỗi địa chỉ, hãy trả về JSON array theo đúng thứ tự đầu vào.",
        "Mỗi phần tử gồm các khóa:",
        "- raw_text",
        "- normalized_text",
        "- province",
        "- district",
        "- ward",
        "Quy tắc:",
        "- normalized_text là địa chỉ được làm sạch, bỏ ký tự thừa nhưng không bịa thêm dữ liệu.",
        "- province/district/ward để null nếu không xác định chắc chắn.",
        "- Chỉ trả JSON array, không markdown, không giải thích.",
        "Danh sách địa chỉ:",
    ]
    for idx, addr in enumerate(addresses, 1):
        lines.append(f'{idx}) "{addr}"')
    return "\n".join(lines)


def ChuanHoaDiaChiTrongFileLenh(lstDiaChi: List[str], use_cache: bool = True) -> Optional[List[dict]]:
    addresses = [str(x).strip() for x in (lstDiaChi or []) if str(x).strip() and str(x).strip().lower() != "nan"]
    if not addresses:
        return []

    cache_key = _normalize_cache_key(addresses)
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    client = _get_client()
    prompt = _build_prompt(addresses)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    result_text = response.choices[0].message.content or "[]"
    batch_result = _extract_json_array(result_text)

    # ép dữ liệu ổn định theo schema mong muốn
    rows: List[dict] = []
    for raw_text, item in zip(addresses, batch_result):
        item = item or {}
        row = {
            "raw_text": item.get("raw_text") or raw_text,
            "normalized_text": item.get("normalized_text") or raw_text,
            "province": item.get("province"),
            "district": item.get("district"),
            "ward": item.get("ward"),
        }
        rows.append(row)

    _CACHE[cache_key] = rows
    _save_cache()
    return rows


if __name__ == "__main__":
    from lenh_data import LayDuLieuFileLenh

    addresses = LayDuLieuFileLenh()
    for plate, addr_list in addresses.items():
        print(f"=== {plate} ===")
        print(json.dumps(ChuanHoaDiaChiTrongFileLenh(addr_list), ensure_ascii=False, indent=2))
