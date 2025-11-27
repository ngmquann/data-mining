import pandas as pd
import joblib
from preprocessing import preprocess_before_split  # Import preprocessing function

def load_model(model_path='models/model.pkl'):
    """Load the trained model from the given path."""
    model = joblib.load(model_path)
    return model

def predict_churn(input_data):
    """Predict churn for the given input data."""
    # Input data preprocessing
    input_data_processed = preprocess_before_split(input_data)  # Preprocess the input data

    # Load the model
    model = load_model()

    # Dự đoán churn
    prediction = model.predict(input_data_processed)
    
    return prediction

# Example usage
if __name__ == "__main__":
    # Example input: replace with actual input data for prediction
    input_data = pd.DataFrame({
        'tenure': [1],  # Example feature (replace with actual features)
        'MonthlyCharges': [29.99],
        'TotalCharges': [100.0],
        'gender': ['Male'],
        # Add other features as per your dataset
    })

    prediction = predict_churn(input_data)
    print(f"Churn prediction: {prediction}")
