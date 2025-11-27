import pandas as pd
import joblib
import os
from preprocessing import DataPreprocessor

def load_artifacts():
    # Lấy thư mục chứa file predict.py
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    # Ghép đường dẫn tới models
    model_path = os.path.join(base_dir, "models", "model.pkl")
    preprocessor_path = os.path.join(base_dir, "models", "preprocessor.pkl")
    
    # Load model và preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor

def predict_churn(input_data: dict):
    """
    Dự đoán churn cho một khách hàng
    
    Args:
        input_data: dict chứa thông tin khách hàng
        
    Returns:
        dict: {'prediction': 0/1, 'probability': float}
    """
    # Load model và preprocessor
    model, preprocessor = load_artifacts()
    
    # Chuyển dict thành DataFrame
    df = pd.DataFrame([input_data])
    
    # Transform data
    X = preprocessor.transform(df)
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]  # Xác suất churn
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'churn_label': 'Yes' if prediction == 1 else 'No'
    }

# Test
if __name__ == "__main__":
    # Ví dụ khách hàng
    sample_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.35,
        'TotalCharges': 844.2
    }
    
    result = predict_churn(sample_customer)
    print("\n🎯 Kết quả dự đoán:")
    print(f"Churn: {result['churn_label']}")
    print(f"Xác suất churn: {result['probability']:.2%}")