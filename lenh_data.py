import pandas as pd
import re
from utils import BienSoXeDeLayDuLieuVtracking
from pathlib import Path
from config import DATE

pattern1 = r"[\wÀ-Ỵà-ỹ.]+[_ ]\d{1,4}\.\d{1,2}"
pattern2 = r"^[A-Za-zÀ-Ỵà-ỹ\.]+\s\+\s[A-Za-zÀ-Ỵà-ỹ\.]+_[0-9]{1,3}\.[0-9]{2}$"

#Đường dẫn nơi lưu lệnh điều xe
VEHICLE_ORDER_PATH = Path(__file__).resolve().parent / "data" / "raw" / "lenh"
ngay = DATE.replace("/","")
fileLenh = f"lenh{ngay}.xlsx"
pathFileLenh = VEHICLE_ORDER_PATH / fileLenh
df = pd.read_excel(pathFileLenh)
dataDictBKS = {}
dongDuLieuDiaChi = 0

def LayDuLieuFileLenh():
    dataDictBKS = {}
    dongDuLieuDiaChi = 0
    #Dữ liệu trả về sẽ ở dạng từ điển gồm khóa biển số và giá trị địa chị dạng list
    for row_idx in range(len(df)):
        for col_idx in range(len(df.columns)):
            cell_value = str(df.iat[row_idx, col_idx]).strip()
            if re.match(pattern1, cell_value) or re.match(pattern2, cell_value):
                # #Bật khi cần debug nếu không export được dữ liệu từ file lệnh
                # print(f"Tìm thấy: '{cell_value}' tại dòng {row_idx + 1}, cột {col_idx + 1}")
                dataDictBKS[BienSoXeDeLayDuLieuVtracking(cell_value[-6:])] = df["Địa chỉ (Giao hàng)"].iloc[dongDuLieuDiaChi:row_idx].tolist()
                dongDuLieuDiaChi = row_idx + 1
    return dataDictBKS



if __name__ == '__main__':
    LayDuLieuFileLenh()
