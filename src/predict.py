import pickle
import pandas as pd
import joblib
import numpy as np

def load_model(filepath):
    """Tải model từ file .pkl"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_single(model, input_dict, scaler_path="../models/scaler.pkl"):
    """
    Dự đoán cho 1 khách hàng từ dictionary đầu vào.
    Hàm này tự động load scaler để chuẩn hóa dữ liệu giống hệt lúc train.
    """
    # 1. Chuyển dictionary thành DataFrame
    df_in = pd.DataFrame([input_dict])
    
    # 2. Xử lý dữ liệu (Cần đảm bảo logic giống hệt preprocess lúc train)
    # Lưu ý: Để đơn giản cho demo, ta giả định input đã được encode số (như 'Gender_Male': 1)
    # Nếu input là text ('Male'), cần thêm bước map/encode ở đây.
    
    # 3. Load Scaler và chuẩn hóa
    try:
        scaler = joblib.load(scaler_path)
        # Đảm bảo thứ tự cột khớp với scaler. Nếu input_dict thiếu cột, code sẽ lỗi,
        # vì vậy cần đảm bảo input_dict đủ các feature như lúc train.
        data_scaled = scaler.transform(df_in)
    except FileNotFoundError:
        print("Cảnh báo: Không tìm thấy scaler.pkl, sử dụng dữ liệu thô.")
        data_scaled = df_in
    except Exception as e:
        print(f"Lỗi khi scale dữ liệu: {e}")
        data_scaled = df_in

    # 4. Dự đoán
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }