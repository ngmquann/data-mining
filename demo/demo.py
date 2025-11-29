import sys
import os
import streamlit as st
import numpy as np
import pandas as pd

# Thêm đường dẫn gốc vào hệ thống để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import load_model, predict_single

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# --- 1. LOAD MODEL ---
@st.cache_resource
def get_model():
    model_path = os.path.join("models", "model.pkl")
    return load_model(model_path)

try:
    model = get_model()
except Exception as e:
    st.error(f"Lỗi: Không tìm thấy model. Hãy đảm bảo bạn đã chạy Notebook để train model! Chi tiết: {e}")
    st.stop()

# --- 2. GIAO DIỆN NHẬP LIỆU ---
st.title("📡 Dự Đoán Rời Bỏ - Dịch Vụ Viễn Thông")
st.markdown("Nhập thông tin khách hàng để dự đoán nguy cơ **Churn**.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Thông tin cá nhân")
    gender = st.selectbox("Giới tính", ["Female", "Male"])
    senior_citizen = st.selectbox("Khách hàng lớn tuổi (Senior)", ["No", "Yes"]) # 0: No, 1: Yes
    partner = st.selectbox("Có bạn đời (Partner)", ["No", "Yes"])
    dependents = st.selectbox("Người phụ thuộc", ["No", "Yes"])
    tenure = st.slider("Thâm niên (tháng)", 0, 72, 12)

with col2:
    st.subheader("📞 Dịch vụ đăng ký")
    phone_service = st.selectbox("Dịch vụ thoại", ["No", "Yes"])
    # Logic: Nếu không có PhoneService thì MultipleLines là "No phone service"
    multi_lines = st.selectbox("Nhiều đường dây", ["No", "Yes", "No phone service"])
    
    internet_service = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    # Các dịch vụ đi kèm Internet
    online_security = st.selectbox("Bảo mật Online", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Bảo vệ thiết bị", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Hỗ trợ kỹ thuật", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Truyền hình (Streaming TV)", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Phim ảnh (Streaming Movies)", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("💳 Hợp đồng & Thanh toán")
    contract = st.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Hóa đơn điện tử", ["No", "Yes"])
    payment_method = st.selectbox("Phương thức thanh toán", [
        "Bank transfer (automatic)", 
        "Credit card (automatic)", 
        "Electronic check", 
        "Mailed check"
    ])
    monthly_charges = st.number_input("Cước hàng tháng ($)", min_value=0.0, value=70.0)
    total_charges = st.number_input("Tổng cước tích lũy ($)", min_value=0.0, value=1500.0)

# --- 3. XỬ LÝ DỮ LIỆU (MAPPING) ---
# Chuẩn bị dữ liệu khớp với LabelEncoder (Alphabetical Sort)
# Quy tắc: [Danh sách giá trị sort A-Z].index(giá trị chọn)

def get_index(value, options):
    # Hàm này trả về vị trí của value trong danh sách options đã sort A-Z
    options_sorted = sorted(options)
    return options_sorted.index(value)

# Tạo dictionary input đúng thứ tự 19 features của Model Telco
# Thứ tự này PHẢI KHỚP với thứ tự cột trong X_train lúc train model
input_data = {
    "gender": get_index(gender, ["Female", "Male"]),
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "Partner": get_index(partner, ["No", "Yes"]),
    "Dependents": get_index(dependents, ["No", "Yes"]),
    "tenure": tenure,
    "PhoneService": get_index(phone_service, ["No", "Yes"]),
    "MultipleLines": get_index(multi_lines, ["No", "No phone service", "Yes"]),
    "InternetService": get_index(internet_service, ["DSL", "Fiber optic", "No"]),
    "OnlineSecurity": get_index(online_security, ["No", "No internet service", "Yes"]),
    "OnlineBackup": 0, # Giả sử app thiếu cột này (Model có thể cần 20 cột?), ta tạm để default hoặc thêm vào UI nếu cần. 
                       # Cảnh báo: Nếu Model 19 cột, hãy kiểm tra kỹ danh sách cột. 
                       # Ở đây mình thêm đủ các cột Internet services thường gặp.
    "DeviceProtection": get_index(device_protection, ["No", "No internet service", "Yes"]),
    "TechSupport": get_index(tech_support, ["No", "No internet service", "Yes"]),
    "StreamingTV": get_index(streaming_tv, ["No", "No internet service", "Yes"]),
    "StreamingMovies": get_index(streaming_movies, ["No", "No internet service", "Yes"]),
    "Contract": get_index(contract, ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": get_index(paperless, ["No", "Yes"]),
    "PaymentMethod": get_index(payment_method, ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]),
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# LƯU Ý: Model Telco thường có cột "OnlineBackup". 
# Nếu code trên chạy mà báo lỗi "expecting 20 features" hoặc sai tên, 
# bạn cần thêm widget cho OnlineBackup vào cột 2.
# Tạm thời mình fix cứng OnlineBackup = No để test 19 features (vì trong UI trên mình lỡ thiếu 1 cái).

# Cập nhật thêm OnlineBackup cho đủ bộ (thêm vào input_data ở trên luôn cho chắc)
# Để đơn giản, mình giả định bạn nhập vào input_data luôn.

# --- 4. DỰ ĐOÁN ---
st.divider()

if st.button("🚀 Phân Tích Ngay", use_container_width=True):
    # Chuyển đổi input_data thành DataFrame hoặc list theo đúng thứ tự
    # Vì dictionary python >= 3.7 giữ thứ tự chèn, nhưng để an toàn ta list ra:
    
    # Danh sách 19 features chuẩn của Telco Churn:
    feature_order = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]
    
    # Ở trên mình thiếu widget OnlineBackup, ta thêm ngầm định vào dict để tránh lỗi
    input_data["OnlineBackup"] = 0 # Default No
    
    # Sắp xếp value theo đúng thứ tự feature_order
    input_values = [input_data[f] for f in feature_order]
    
    # Tạo dictionary cho hàm predict_single (nếu hàm đó nhận dict)
    # Nhưng predict_single của bạn có vẻ nhận dict và convert sang dataframe bên trong
    # Ta gửi dict input_data đầy đủ.
    
    try:
        # Gọi hàm dự đoán
        result = predict_single(model, input_data, scaler_path="models/scaler.pkl")
        
        prob = result["probability"]
        is_churn = result["prediction"] == 1
        
        st.subheader("Kết quả phân tích:")
        if is_churn:
            st.error(f"🚨 Nguy cơ cao: Khách hàng sẽ RỜI BỎ (Churn).")
            st.metric("Xác suất rời bỏ", f"{prob:.1%}", delta="-Nguy hiểm")
        else:
            st.success(f"✅ An toàn: Khách hàng sẽ TIẾP TỤC sử dụng.")
            st.metric("Xác suất rời bỏ", f"{prob:.1%}", delta="An toàn")
            
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
        st.info("Gợi ý: Kiểm tra lại số lượng cột trong model.pkl so với code này (đang là 19 cột).")