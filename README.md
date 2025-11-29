# 📡 Telco Customer Churn Prediction

## 📖 Giới Thiệu
Đây là ứng dụng Web giúp **dự đoán nguy cơ khách hàng rời bỏ dịch vụ viễn thông (Customer Churn)**. 

Dự án áp dụng quy trình chuẩn **CRISP-DM** và sử dụng thuật toán **Logistic Regression** để phân tích hành vi khách hàng. Ứng dụng giúp các công ty viễn thông:
* Phát hiện sớm khách hàng có nguy cơ rời mạng.
* Đưa ra các quyết định giữ chân khách hàng (Retention) dựa trên dữ liệu.
* Tối ưu hóa chi phí marketing và chăm sóc khách hàng.

## 🚀 Tính Năng Chính
* **🔮 Dự đoán thời gian thực:** Nhập thông tin khách hàng và nhận kết quả dự đoán (Churn/Non-Churn) ngay lập tức.
* **📊 Giao diện trực quan:** Nhập liệu dễ dàng với các thanh trượt và menu thả xuống (Dropdown) thông qua **Streamlit**.
* **⚙️ Xử lý dữ liệu tự động:** Tự động chuyển đổi dữ liệu thô (Text) sang dạng số (Encoding) khớp với mô hình đã huấn luyện.
* **⚠️ Cảnh báo rủi ro:** Hiển thị mức độ rủi ro (High/Low Risk) kèm theo xác suất cụ thể.

## 🛠️ Công Nghệ Sử Dụng
* **Ngôn ngữ:** Python 3.12
* **Giao diện:** Streamlit
* **Xử lý dữ liệu:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Logistic Regression)
* **Trực quan hóa:** Matplotlib, Seaborn

## 💻 Cài Đặt & Chạy Ứng Dụng

clone dự án

   git clone [https://github.com/username/telco-churn-prediction.git](https://github.com/username/telco-churn-prediction.git)
   
   cd telco-churn-prediction

Tạo môi trường ảo (Khuyên dùng Python 3.12):

   py -3.12 -m venv my_env
   
   .\my_env\Scripts\Activate

Cài đặt thư viện
   
   pip install pandas scikit-learn streamlit matplotlib seaborn

Chạy ứng dụng:

  streamlit run demo/demo.py

