import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_before_split(df: pd.DataFrame):  
    # Loại bỏ customerID nếu có (không có giá trị dự đoán)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Xử lý TotalCharges (chuyển từ object sang numeric)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # ===== 1) Xử lý Missing Values =====
    # Numerical columns → thay bằng median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Categorical columns → thay bằng mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    # ===== 2) Encoding (Chuyển dữ liệu chữ sang số) =====
    # Label Encoding cho các biến nhị phân (2 giá trị)
    binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling','Churn']
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    # One-hot Encoding cho các biến có nhiều giá trị
    df = pd.get_dummies(df, drop_first=True)
    
    # ===== 3) Feature Engineering (Tạo thuộc tính mới) =====
    if set(["MonthlyCharges", "tenure"]).issubset(df.columns):
        # Tổng doanh thu từ khách hàng
        df["Revenue"] = df["MonthlyCharges"] * df["tenure"]
        # Chi phí trung bình mỗi tháng
        df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    
    # Khách hàng lâu năm (> 12 tháng)
    df["LongTermCustomer"] = (df["tenure"] > 12).astype(int)
    # Khách hàng có chi phí cao
    df["HighCharge"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    
    return df


def prepare_data(df: pd.DataFrame, target_column="Churn"):
    """Tách dữ liệu thành X_train, X_test, y_train, y_test sau khi tiền xử lý đầy đủ."""
    
    # Tiền xử lý (chưa scaling)
    df = preprocess_before_split(df)
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Tách train/test với tỉ lệ 80-20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ===== 4) Scaling SAU KHI split (tránh data leakage) =====
    scale_cols = ['tenure','MonthlyCharges','TotalCharges','Revenue','AvgMonthlyCharge']
    scale_cols = [c for c in scale_cols if c in X_train.columns]  # tránh lỗi thiếu cột
    
    scaler = StandardScaler()
    # Fit trên train set, transform cả train và test
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])  # chỉ transform, không fit
    
    return X_train, X_test, y_train, y_test


# Chạy độc lập để kiểm tra module
if __name__ == "__main__":
    df = pd.read_csv("data/Customer_Churn.csv")
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(" Tiền xử lý hoàn tất!")
    print(f" Train size: {X_train.shape} | Test size: {X_test.shape}")
    print(f" Phân bố Churn - Train: {y_train.value_counts().to_dict()}")