import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Huấn luyện danh sách các mô hình và trả về kết quả đánh giá.
    """
    # Khai báo các mô hình muốn thử nghiệm
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM RBF": SVC(probability=True)
    }

    results = {}
    best_model = None
    best_auc = 0
    
    print(f"{'Model':<20} | {'Accuracy':<10} | {'AUC':<10}")
    print("-" * 46)

    for name, model in models.items():
        # Huấn luyện
        model.fit(X_train, y_train)
        
        # Dự đoán
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:,1]

        # Đánh giá
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)
        
        results[name] = {"accuracy": acc, "auc": auc, "model": model}
        
        print(f"{name:<20} | {acc:.4f}     | {auc:.4f}")

        # Lưu lại model tốt nhất dựa trên AUC
        if auc > best_auc:
            best_auc = auc
            best_model = model

    print("-" * 46)
    print(f"Best Model: {best_model}")
    
    return best_model, results

def save_model(model, filepath):
    """Lưu model ra file .pkl"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Đã lưu model tại: {filepath}")