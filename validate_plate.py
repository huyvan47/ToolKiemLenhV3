import sys
import yaml
from pathlib import Path
from typing import Set, Tuple

#Đường dẫn nơi lưu thông tin biển số xe
TRUCK_LIST_PATH = Path(__file__).resolve().parent / "company_trucks.txt"

#Lấy dữ liệu biển số có kèm theo thông tin ID của xe mà vtracking cấp
def load_vehicle_ids_from_yaml(path= Path(__file__).resolve().parent / "vehicles.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

#Lấy dữ liệu biển số xe đang sử dụng tại công ty
def load_company_trucks() -> Set[str]:
    """Load danh sách xe tải hợp lệ từ file."""
    if not TRUCK_LIST_PATH.exists():
        print(f"❌ File không tồn tại: {TRUCK_LIST_PATH}")
        sys.exit(1)

    with open(TRUCK_LIST_PATH, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())
    
#Kiểm tra biển số và chuyển đổi biển số phù hợp với định dạng trong file vtracking và epass
def normalize_plate(raw_plate: str) -> Tuple[str, str]:
    """
    Kiểm tra và chuyển đổi biển số dạng '17926' → ('62-17926', '62C17926').
    Dừng chương trình nếu không có trong danh sách hợp lệ.
    """
    truck_list = load_company_trucks()

    plate = raw_plate.strip()
    if plate not in truck_list:
        print(f"❌ Biển số không thuộc danh sách công ty: {plate}")
        sys.exit(1)

    bks_epass = f"62C-{plate}"
    bks_vtracking = f"62C{plate}"
    return bks_epass, bks_vtracking

if __name__ == "__main__":
    #Ví du nhập một chuổi các danh sách biển số xe: input_plate = ["17926", "16993", "19149"]
    plates = ["10191", "15068"]
    vehicles_data = load_vehicle_ids_from_yaml()
    result = {}

    for plate in plates:
        bks_epass, bks_vtracking = normalize_plate(plate)
        vehicle_id = vehicles_data.get(bks_epass)
        result[plate] = {
            "vtracking": bks_vtracking,
            "epass": bks_epass,
            "id": vehicle_id
        }

    print(result)