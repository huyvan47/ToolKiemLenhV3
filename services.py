import requests
import pandas as pd
import json
import sys
import os

from lenh_data import LayDuLieuFileLenh
from pathlib import Path
from datetime import datetime
from config import COOKIE_VTRACKING, FROMDATE, TODATE, DATE, EPASSUSERNAME, EPASSPASSWORD, VTRACKINGUSERNAME, VTRACKINGPASSWORD
from validate_plate import normalize_plate, load_vehicle_ids_from_yaml

vehicle_ids = load_vehicle_ids_from_yaml()

vehicle_dict = {}

def login_get_token_epass(EpassUsername: str, EpassPassword: str) -> str:
    url = "https://backend.epass-vdtc.com.vn/crm2/api/v1/login"
    data = {
        "grant_type": "password",
        "client_id": "portal-chu-pt",
        "username": EpassUsername,
        "password": EpassPassword,
    }

    response = requests.post(url, data=data)
    if response.status_code != 200:
        raise Exception(f"❌ Login thất bại: {response.status_code} - {response.text}")

    token = response.json().get("access_token")
    if not token:
        raise Exception("❌ Không lấy được token!")
    return token

def download_epass_excel_epass(token: str, _date= str) -> Path:
    url = "https://backend.epass-vdtc.com.vn/doisoat2/api/v1/transactions-vehicles/export-excel"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "contractId": "1597080",
        "timestampInFrom": _date,
        "timestampInTo": _date
    }

    response = requests.post(url, data=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"❌ Tải file thất bại: {response.status_code} - {response.text}")

    # Tạo đường dẫn file
    save_dir = Path(__file__).resolve().parent / "data" / "raw" / "epass"
    save_dir.mkdir(parents=True, exist_ok=True)
    parts = _date.split("/")  # ['08', '06', '2025']
    result = parts[0] + parts[1]
    filename = f"epass{result}.xlsx"
    file_path = save_dir / filename

    # Lưu file
    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"✅ File ePass đã lưu tại: {file_path}")
    return file_path

def download_vtracking_excel(
    raw_cookie_str: str,
    from_date: str,
    to_date: str,
    _date: str,
    vehicle_dict: dict
) -> None:

    url = "https://vtracking2.viettel.vn/getHistoryTracking"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

    cookie_dict = dict(
        pair.strip().split("=", 1)
        for pair in raw_cookie_str.split(";")
        if "=" in pair
    )

    save_dir = Path(__file__).resolve().parent / "data" / "raw" / "vtracking"
    save_dir.mkdir(parents=True, exist_ok=True)

    for plate, info in vehicle_dict.items():
        vehicle_id = info["id"]
        bks_vtracking = info["vtracking"]

        payload = {
            "fromDate": from_date,
            "toDate": to_date,
            "id": vehicle_id,
            "after": "",
            "before": "",
            "limit": 20000
        }

        response = requests.post(url, headers=headers, cookies=cookie_dict, json=payload)

        if "html" in response.text.lower():
            raise Exception(f"⚠️ Cookie hết hạn hoặc sai khi truy vấn xe {plate}.")

        response_json = response.json()
        logs = response_json.get("content", {}).get("logs", [])
        if not logs:
            print(f"⚠️ Không có dữ liệu logs cho xe {plate}.")
            continue

        rows = []
        for i, log in enumerate(logs):
            value = log.get("value", {})
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    continue

            ts = log.get("ts", 0)
            timestamp_str = datetime.fromtimestamp(ts / 1000).strftime("%d/%m/%Y %H:%M:%S") if ts else ""

            lat = value.get("latitude")
            lng = value.get("longitude")
            if lat is not None and lng is not None:
                toado = f"{float(lat):.6f},{float(lng):.6f}"
            else:
                toado = ""

            rows.append({
                "STT": i + 1,
                "Thời gian": timestamp_str,
                "Tốc độ": value.get("speed", 0),
                "Trạng thái": value.get("status", ""),
                "Tọa độ": toado,
                "Vị trí": value.get("geocoding", "")
            })

        df = pd.DataFrame(rows)
        parts = _date.split("/")  # ví dụ: ['11','06','2025']
        result = "".join(parts[:2])  # '1106'
        filename = f"{bks_vtracking}_{result}.xlsx"
        file_path = save_dir / filename
        df.to_excel(file_path, index=False)

        print(f"✅ Đã lưu file cho xe {plate} tại: {file_path}")

def InLenhChiNhanh():
    folder_path = Path(__file__).resolve().parent / "data" / "raw" / "lenh"
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            print(f"In file: {file}")
            os.startfile(full_path, "print")



if __name__ == "__main__":

    lstBKS = LayDuLieuFileLenh().keys()

    #Kiểm tra biển số, chỉ 1 biển số không đúng (sai hoặc không có trong hệ thống) thoát luôn
    for plate in lstBKS:

        #Chuyển đổi biến số cho phù hợp để lấy dữ liệu từ hệ thống vtracking và epass
        bks_epass, bks_vtracking = normalize_plate(plate)

        #lọc lấy ID dựa vào file tổng chứa thông tin các xe kèm theo ID, không thấy dữ liệu thoát luôn
        vehicle_id = vehicle_ids.get(bks_epass)
        if not vehicle_id:
            print(f"⚠️ Không tìm thấy ID cho {bks_epass}")
            sys.exit()

        vehicle_dict[plate] = {
            "vtracking": bks_vtracking,
            "epass": bks_epass,
            "id": vehicle_id
        }

    #Dữ liệu sửa hằng ngày khi kiểm lệnh trong config.py
    fromDate = FROMDATE
    toDate = TODATE
    _date = DATE

    #Dữ liệu sửa hằng ngày khi kiểm lệnh -> vào file config.py để sửa cookie vtracking
    cookie_string = COOKIE_VTRACKING

    #Chức năng tải file epass theo ngày chỉnh sửa bên trên. Khi có thay đổi mật khẩu thì chỉnh sửa thông tin bên dưới
    EpassUsername = EPASSUSERNAME
    EpassPassword = EPASSPASSWORD
    token = login_get_token_epass(EpassUsername, EpassPassword)
    download_epass_excel_epass(token, _date=_date)

    # Chức năng tải file epass theo ngày chỉnh sửa bên trên. Khi có thay đổi mật khẩu hoặc biển số xe thì chỉnh sửa thông tin bên dưới và VEHICLE_PLATES_LONGAN trong config.py
    VtrackingUsername = VTRACKINGUSERNAME
    VtrackingPassword = VTRACKINGPASSWORD

    download_vtracking_excel(raw_cookie_str=cookie_string, from_date=fromDate, to_date=toDate , _date=_date, vehicle_dict=vehicle_dict)

    # InLenhChiNhanh()
