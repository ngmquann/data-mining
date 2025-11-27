import streamlit as st
import pandas as pd
import sys
sys.path.append('../src')
from predict import predict_churn

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction Demo")
st.markdown("Dự đoán khả năng khách hàng rời bỏ dịch vụ")

# Sidebar
with st.sidebar:
    st.header("⚙️ Thông tin khách hàng")
    
    gender = st.selectbox("Giới tính", ['Male', 'Female'])
    senior = st.selectbox("Người cao tuổi", [0, 1])
    partner = st.selectbox("Có bạn đời", ['Yes', 'No'])
    dependents = st.selectbox("Có người phụ thuộc", ['Yes', 'No'])
    tenure = st.slider("Thời gian sử dụng (tháng)", 0, 72, 12)
    
    st.divider()
    st.subheader("📞 Dịch vụ")
    
    phone_service = st.selectbox("Dịch vụ điện thoại", ['Yes', 'No'])
    multiple_lines = st.selectbox("Nhiều đường dây", ['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox("Internet", ['DSL', 'Fiber optic', 'No'])
    
    online_security = st.selectbox("Bảo mật online", ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox("Sao lưu online", ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox("Bảo vệ thiết bị", ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox("Hỗ trợ kỹ thuật", ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox("TV streaming", ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox("Movies streaming", ['No', 'Yes', 'No internet service'])
    
    st.divider()
    st.subheader("💳 Thanh toán")
    
    contract = st.selectbox("Loại hợp đồng", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Hóa đơn điện tử", ['Yes', 'No'])
    payment_method = st.selectbox("Phương thức thanh toán", 
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 
         'Credit card (automatic)'])
    
    monthly_charges = st.number_input("Chi phí tháng ($)", 0.0, 200.0, 70.0, 0.5)
    total_charges = st.number_input("Tổng chi phí ($)", 0.0, 10000.0, 840.0, 10.0)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Thông tin đầu vào")
    
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    st.json(input_data)

with col2:
    st.subheader("🎯 Kết quả dự đoán")
    
    if st.button("🔮 Dự đoán", type="primary", use_container_width=True):
        with st.spinner("Đang xử lý..."):
            result = predict_churn(input_data)
            
            if result['prediction'] == 1:
                st.error(f"⚠️ **Khách hàng có nguy cơ rời bỏ!**")
                st.metric("Xác suất Churn", f"{result['probability']:.1%}")
            else:
                st.success(f"✅ **Khách hàng trung thành**")
                st.metric("Xác suất Churn", f"{result['probability']:.1%}")
            
            # Biểu đồ
            st.progress(result['probability'])
            
            # Khuyến nghị
            st.divider()
            st.subheader("💡 Khuyến nghị")
            
            if result['prediction'] == 1:
                st.markdown("""
                - 🎁 Tặng ưu đãi đặc biệt
                - 📞 Liên hệ hỗ trợ khách hàng
                - 💰 Giảm giá dịch vụ
                - 📝 Chuyển sang hợp đồng dài hạn
                """)
            else:
                st.markdown("""
                - ⭐ Duy trì chất lượng dịch vụ
                - 🎯 Upsell thêm dịch vụ
                - 📧 Gửi chương trình khách hàng thân thiết
                """)

st.divider()
st.markdown("*Demo được xây dựng bởi Streamlit | Model: Random Forest*")