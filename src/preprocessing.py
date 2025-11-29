import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_data(filepath):
    """Đọc dữ liệu từ file CSV"""
    df = pd.read_csv(filepath)
    return df

def preprocess(df, save_artifacts_path="../models/"):
    """
    Làm sạch dữ liệu, mã hóa và chia tập train/test.
    Lưu lại scaler và encoder để dùng cho lúc dự đoán sau này.
    """
    df_clean = df.copy()

    # 1. Xử lý cột customerID (nếu có)
    if "customerID" in df_clean.columns:
        df_clean.drop("customerID", axis=1, inplace=True)

    # 2. Xử lý cột TotalCharges (chuyển sang số và điền median)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)

    # 3. Mã hóa biến phân loại (Label Encoding)
    # Lưu ý: Cần lưu lại encoder/mapping để dùng cho lúc dự đoán
    cat_cols = df_clean.select_dtypes(include=["object"]).columns
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        encoders[col] = le
    
    # Lưu encoders (tùy chọn, để đơn giản ta bỏ qua bước save encoders trong demo này
    # nhưng thực tế cần lưu lại để map ngược)

    # 4. Tách X, y
    X = df_clean.drop("Churn", axis=1)
    y = df_clean["Churn"]

    # 5. Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Scaling (Chuẩn hóa)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Lưu scaler để dùng cho app dự đoán sau này
    joblib.dump(scaler, f"{save_artifacts_path}scaler.pkl")

    return X_train_s, X_test_s, y_train, y_test, X.columns