import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

class DataPreprocessor:
    """Class quản lý toàn bộ preprocessing pipeline"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def fit_transform(self, df: pd.DataFrame, target_column="Churn"):
        """Fit và transform dữ liệu training"""
        df = df.copy()
        
        # Loại bỏ customerID
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Xử lý TotalCharges
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Xử lý missing values
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        cat_cols = df.select_dtypes(include=["object"]).columns
        cat_cols = [c for c in cat_cols if c != target_column]
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
        
        # Feature Engineering
        if set(["MonthlyCharges", "tenure"]).issubset(df.columns):
            df["Revenue"] = df["MonthlyCharges"] * df["tenure"]
            df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
        
        df["LongTermCustomer"] = (df["tenure"] > 12).astype(int)
        df["HighCharge"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
        
        # Label Encoding cho binary columns
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                      'PaperlessBilling', target_column]
        
        for col in binary_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # One-hot encoding
        df = pd.get_dummies(df, drop_first=True)
        
        # Tách X, y
        y = df[target_column] if target_column in df.columns else None
        X = df.drop(target_column, axis=1) if target_column in df.columns else df
        
        # Lưu tên cột
        self.feature_columns = X.columns.tolist()
        
        # Scaling
        scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                     'Revenue', 'AvgMonthlyCharge']
        scale_cols = [c for c in scale_cols if c in X.columns]
        
        if scale_cols:
            X[scale_cols] = self.scaler.fit_transform(X[scale_cols])
        
        return X, y
    
    def transform(self, df: pd.DataFrame):
        """Transform dữ liệu mới (predict)"""
        df = df.copy()
        
        # Các bước tương tự fit_transform
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Missing values
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        cat_cols = df.select_dtypes(include=["object"]).columns
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
        
        # Feature Engineering
        if set(["MonthlyCharges", "tenure"]).issubset(df.columns):
            df["Revenue"] = df["MonthlyCharges"] * df["tenure"]
            df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
        
        df["LongTermCustomer"] = (df["tenure"] > 12).astype(int)
        df["HighCharge"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
        
        # Label Encoding
        for col, encoder in self.label_encoders.items():
            if col in df.columns and col != 'Churn':
                df[col] = encoder.transform(df[col])
        
        # One-hot encoding
        df = pd.get_dummies(df, drop_first=True)
        
        # Đảm bảo có đủ cột như training
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_columns]
        
        # Scaling
        scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                     'Revenue', 'AvgMonthlyCharge']
        scale_cols = [c for c in scale_cols if c in df.columns]
        
        if scale_cols:
            df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        return df
    
    def save(self, path='models/preprocessor.pkl'):
        """Lưu preprocessor"""
        joblib.dump(self, path)
    
    @staticmethod
    def load(path='models/preprocessor.pkl'):
        """Load preprocessor"""
        return joblib.load(path)