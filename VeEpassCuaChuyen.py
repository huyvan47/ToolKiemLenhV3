import datetime as dt
import pandas as pd
import os
from pathlib import Path
from config import DATE

TENFILEEPASS = 'epass' + DATE.split("/")[0] + DATE.split("/")[1] + '.xlsx'
VEHICLE_ORDER_PATH = Path(__file__).resolve().parent / "data" / "raw" / "epass"
file_path = VEHICLE_ORDER_PATH / TENFILEEPASS
df = pd.read_excel(file_path)

def LayIndexVe(bienSoXe):
    dataFileEpassFiltered = df[df["Unnamed: 7"] == bienSoXe].index
    print('Tổng số trạm có trong file: ', len(dataFileEpassFiltered))
    return dataFileEpassFiltered

def DuLieuVeEpassTheoBienSoXeCuaChuyen(thoiGianDi, thoiGianVe, bienSoXe):
    dataFileEpassFiltered = LayIndexVe(bienSoXe)
    tongSoTramDuocGhiNhan = 0
    for i in dataFileEpassFiltered:
        gioVaoTram = df["Unnamed: 3"].iloc[i].split(" ")[1][:5]
        try:
            if(dt.datetime.strptime(gioVaoTram,"%H:%M") >= dt.datetime.strptime(thoiGianDi,"%H:%M") and dt.datetime.strptime(gioVaoTram,"%H:%M") <= dt.datetime.strptime(thoiGianVe,"%H:%M")):
                print("Tên trạm: ", df["Unnamed: 2"].iloc[i], df["Unnamed: 3"].iloc[i])
                tongSoTramDuocGhiNhan +=1
        except Exception as e:
            print (e)
    print("Tổng số trạm ghi nhận theo chuyến: ", tongSoTramDuocGhiNhan)
if __name__ == '__main__':
    thoiGianDi = '01:00'
    thoiGianVe = '23:00'
    bienSoXe = '62C18256'
    DuLieuVeEpassTheoBienSoXeCuaChuyen(thoiGianDi, thoiGianVe, bienSoXe)
