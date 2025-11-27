import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from preprocessing import DataPreprocessor

# Tạo thư mục models nếu chưa có
os.makedirs('models', exist_ok=True)

# 1) Load dữ liệu
df = pd.read_csv("../data/Customer_Churn.csv")

# 2) Preprocessing
preprocessor = DataPreprocessor()
X, y = preprocessor.fit_transform(df, target_column="Churn")

# Lưu preprocessor
preprocessor.save('models/preprocessor.pkl')

# 3) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Train models
print("🔄 Training Logistic Regression...")
log_model = LogisticRegression(max_iter=2000, random_state=42)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)

print("🔄 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# 5) Chọn model tốt nhất
best_model = rf_model if rf_acc > log_acc else log_model
best_model_name = "Random Forest" if rf_acc > log_acc else "Logistic Regression"

# 6) Lưu model
joblib.dump(best_model, "models/model.pkl")

# 7) In kết quả
print("\n" + "="*50)
print("📊 KẾT QUẢ TRAINING")
print("="*50)
print(f"Logistic Regression Accuracy: {log_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"\n✅ Model tốt nhất: {best_model_name}")
print(f"✅ Đã lưu vào models/model.pkl và models/preprocessor.pkl")
print("\n📋 Classification Report (Model tốt nhất):")
print(classification_report(y_test, rf_pred if rf_acc > log_acc else log_pred))