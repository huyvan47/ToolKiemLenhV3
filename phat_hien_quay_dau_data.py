# 1️⃣ Nhập các thư viện cần thiết
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from pathlib import Path
from config import DATE
from lenh_data import LayDuLieuFileLenh

bien_so_list = LayDuLieuFileLenh().keys()
ngay = DATE.replace("/","")[:4]

# Đường dẫn nơi lưu vtracking data
data_folder = Path(__file__).resolve().parent / "data" / "raw" / "vtracking"

# 📌 3) Kết quả gộp (tuỳ chọn)
all_turnarounds = []

def DiaChiNghiVanQuayDau(bienSoXe):
    file_name = f"62C{bienSoXe}_{ngay}.xlsx"
    file_path = data_folder / file_name
    print(f"👉 Đang xử lý file: {file_name}")

    if not file_path.exists():
        print(f"⚠️ File {file_name} không tồn tại, bỏ qua.")
        return

    # 5️⃣ Đọc Excel
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name='Sheet1')

    # 6️⃣ Tách toạ độ
    df[['Lat', 'Lon']] = df['Tọa độ'].str.split(',', expand=True).astype(float)

    # 7️⃣ Chuẩn hoá thời gian
    df['Thời gian'] = pd.to_datetime(df['Thời gian'], format="%d/%m/%Y %H:%M:%S")
    df = df.sort_values('Thời gian').reset_index(drop=True)

    # 8️⃣ Tính khoảng cách giữa các điểm liên tiếp
    distances = [0.0]
    for i in range(1, len(df)):
        prev_point = (df.loc[i-1, 'Lat'], df.loc[i-1, 'Lon'])
        curr_point = (df.loc[i, 'Lat'], df.loc[i, 'Lon'])
        dist = geodesic(prev_point, curr_point).meters
        distances.append(dist)
    df['Distance_m'] = distances

    # ---------- Các hàm phụ dùng chung ----------
    def _choose_axis(_df):
        lat_range = _df['Lat'].max() - _df['Lat'].min()
        lon_range = _df['Lon'].max() - _df['Lon'].min()
        return 'Lat' if abs(lat_range) >= abs(lon_range) else 'Lon'

    def _rolling_sign(series, window=3):
        """Làm mượt median rồi lấy dấu vi phân (tăng/giảm)."""
        s_smooth = series.rolling(window=window, center=True, min_periods=1).median()
        ds = s_smooth.diff()
        sign = np.sign(ds)
        sign = sign.replace(0, np.nan).ffill().bfill().fillna(0)  # giữ hướng liên tục
        return sign, s_smooth

    def _run_length(arr, idx, direction):
        """Đếm số điểm liên tiếp cùng dấu xung quanh idx."""
        n = len(arr)
        if direction == 'back':
            if idx == 0: return 0
            val, k = arr[idx-1], 0
            for j in range(idx-1, -1, -1):
                if arr[j] == val: k += 1
                else: break
            return k
        else:
            if idx >= n-1: return 0
            val, k = arr[idx], 0
            for j in range(idx, n):
                if arr[j] == val: k += 1
                else: break
            return k

    # ---------- 1) QUAY ĐẦU THEO TRỤC CHÍNH (Lat/Lon) ----------
    def detect_turnaround_by_axis(_df, axis=None, smooth_w=5, min_run=3, min_disp_m=20):
        """
        Phát hiện quay đầu dựa đổi chiều theo 1 trục (Lat/Lon), KHÔNG phụ thuộc góc/tốc độ.
        """
        if axis is None:
            axis = _choose_axis(_df)

        sign, _ = _rolling_sign(_df[axis], window=smooth_w)
        sign = sign.to_numpy()

        change_idx = np.where(np.diff(sign) != 0)[0] + 1
        if len(change_idx) == 0:
            return []

        cum = _df['Distance_m'].fillna(0).cumsum().to_numpy()
        turn_indices = []
        for idx in change_idx:
            back_run = _run_length(sign, idx, 'back')
            fwd_run  = _run_length(sign, idx, 'forward')
            if back_run < min_run or fwd_run < min_run:
                continue

            left_idx  = max(0, idx - back_run)
            right_idx = min(len(_df)-1, idx + fwd_run)
            disp_left  = cum[idx] - cum[left_idx]
            disp_right = cum[right_idx] - cum[idx]
            if disp_left >= min_disp_m and disp_right >= min_disp_m:
                turn_indices.append(idx)
        return turn_indices

    # ---------- 2) QUAY ĐẦU THEO LỆCH GÓC (BEARING) ----------
    # Công thức bearing giữa 2 điểm
    def _bearing(pointA, pointB):
        lat1, lat2 = np.radians(pointA[0]), np.radians(pointB[0])
        dlon = np.radians(pointB[1] - pointA[1])
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
        brg = (np.degrees(np.arctan2(x, y)) + 360) % 360
        return brg

    # góc chênh nhỏ nhất |a-b| trên vòng tròn
    def _ang_diff(a, b):
        d = (a - b + 180) % 360 - 180
        return abs(d)

    def _smooth_bearing_series(_df, win=5):
        """Làm mượt bearing bằng cách average trên vector đơn vị để tránh wrap-around."""
        # tính bearing từng bước
        bearings = [np.nan]
        for i in range(1, len(_df)):
            p1 = ( _df.loc[i-1,'Lat'], _df.loc[i-1,'Lon'] )
            p2 = ( _df.loc[i,  'Lat'], _df.loc[i,  'Lon'] )
            bearings.append(_bearing(p1, p2))
        b = pd.Series(bearings)

        # chuyển sang vector đơn vị
        theta = np.radians(b)
        x = np.cos(theta)
        y = np.sin(theta)

        x_s = x.rolling(win, center=True, min_periods=1).mean()
        y_s = y.rolling(win, center=True, min_periods=1).mean()

        smoothed = (np.degrees(np.arctan2(y_s, x_s)) + 360) % 360
        return smoothed

    def detect_turnaround_by_angle(_df,
                                   smooth_w=5,
                                   min_angle_change=140,   # ngưỡng quay đầu theo góc
                                   min_run=3,
                                   min_disp_m=20):
        """
        Phát hiện quay đầu dựa vào thay đổi góc hướng (bearing) đã làm mượt.
        Không xét tốc độ. Kiểm tra độ bền & quãng đường 2 phía để tránh nhiễu.
        """
        b_smooth = _smooth_bearing_series(_df, win=smooth_w)
        # diff theo cửa sổ 1 bước
        diff = [_ang_diff(b_smooth.iloc[i], b_smooth.iloc[i-1]) if i>0 else np.nan
                for i in range(len(_df))]
        diff = pd.Series(diff).fillna(0)

        # đánh dấu điểm có thay đổi góc lớn
        cand = np.where(diff >= min_angle_change)[0]
        if len(cand) == 0:
            return []

        # dùng sign-of-progress theo bearing (xấp xỉ hướng tiến/lùi trên vòng tròn)
        # dựa vào cos(bearing) & sin(bearing) để xác định "hướng tổng quát"
        theta = np.radians(b_smooth)
        vx = np.cos(theta); vy = np.sin(theta)
        # vector hướng chính theo rolling median để lấy dấu
        vx_sign = np.sign(pd.Series(vx).rolling(smooth_w, center=True, min_periods=1).median().diff()).replace(0, np.nan).ffill().bfill().fillna(0).to_numpy()
        vy_sign = np.sign(pd.Series(vy).rolling(smooth_w, center=True, min_periods=1).median().diff()).replace(0, np.nan).ffill().bfill().fillna(0).to_numpy()

        cum = _df['Distance_m'].fillna(0).cumsum().to_numpy()
        turn_idx = []
        for idx in cand:
            # yêu cầu có đổi dấu theo 1 trong 2 trục vectơ hướng (vx hoặc vy)
            back_run_x = _run_length(vx_sign, idx, 'back')
            fwd_run_x  = _run_length(vx_sign, idx, 'forward')
            back_run_y = _run_length(vy_sign, idx, 'back')
            fwd_run_y  = _run_length(vy_sign, idx, 'forward')

            ok_x = back_run_x >= min_run and fwd_run_x >= min_run
            ok_y = back_run_y >= min_run and fwd_run_y >= min_run
            if not (ok_x or ok_y):
                continue

            left_idx  = max(0, idx - max(back_run_x, back_run_y))
            right_idx = min(len(_df)-1, idx + max(fwd_run_x, fwd_run_y))
            if (cum[idx] - cum[left_idx] >= min_disp_m) and (cum[right_idx] - cum[idx] >= min_disp_m):
                turn_idx.append(idx)
        return sorted(set(turn_idx))

    # --------- Chạy 2 bộ phát hiện ----------
    # A) Theo trục chính
    axis_turn_idx = detect_turnaround_by_axis(
        df,
        axis=None,
        smooth_w=10,
        min_run=5,
        min_disp_m=50
    )

    # B) Theo lệch góc (bearing)
    angle_turn_idx = detect_turnaround_by_angle(
        df,
        smooth_w=5,
        min_angle_change=90,   # mềm hơn 160 để bắt cả quay đầu nhanh
        min_run=3,
        min_disp_m=20
    )

    # Hợp nhất (OR): bắt quay đầu "bất kể góc/tốc độ/khoảng cách"
    turn_flags = np.zeros(len(df), dtype=bool)
    turn_flags[axis_turn_idx] = True
    turn_flags[angle_turn_idx] = True

    df['AxisTurn']  = False; df.loc[axis_turn_idx,  'AxisTurn']  = True
    df['AngleTurn'] = False; df.loc[angle_turn_idx, 'AngleTurn'] = True
    df['TurnFinal'] = turn_flags

    turnarounds = df[df['TurnFinal']].copy()
    turnarounds['BienSo'] = bienSoXe
    turnarounds['Ngay'] = ngay

    turnarounds['Tốc độ'] = pd.to_numeric(turnarounds['Tốc độ'], errors='coerce')

    # Giữ các dòng có Tốc độ trong [5, 30]
    turnarounds = turnarounds[turnarounds['Tốc độ'].between(5, 30, inclusive='both')]

    # Loại trừ vị trí không cần
    turnarounds = turnarounds.query(
        '`Vị trí` != "Xã Đức Hòa Đông, Huyện Đức Hòa, Tỉnh Long An" and `Vị trí` != "Xã Đức Hòa Hạ, Huyện Đức Hòa, Tỉnh Long An"'
    ).copy()

    # Trả kết quả
    return turnarounds.to_dict(orient='records')


if __name__ == '__main__':
    DiaChiNghiVanQuayDau("18856")
