import os, requests, sys

api_key = (os.getenv("GOOGLE_MAPS_API_KEY") or "").strip().strip('"').strip("'")
print("[PYTHON]", sys.executable)
print("[KEY REPR]", repr(api_key))
print("[KEY LEN]", len(api_key))

params = {
    "address": "3/6 QL22, Quận 12, Hồ Chí Minh, Việt Nam",
    "key": api_key,
}
r = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, timeout=20)
print("[URL]", r.url.replace(api_key, "***"))
print("[HTTP]", r.status_code)
print("[JSON]", r.json().get("status"), r.json().get("error_message"))