import os
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_ensemble_layer():
    print("="*60)
    print("RailGuard AI - Classical ML Ensemble Trainer")
    print("="*60)
    
    csv_path = "data/ml_features/railguard_features.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found! Run generate_ml_dataset.py first.")
        return
        
    print(f"\n[1/3] Loading Ground Truth Statistics from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract X (10-dimensional feature arrays)
    X = df.iloc[:, :-1].values
    
    # Extract Y (Continuous Risk Score 0.0 - 1.0)
    # We convert this into a Binary Classification target (0 = Safe, 1 = Threat)
    # so we can use .predict_proba() to get exact percentage confidences!
    y_raw = df.iloc[:, -1].values
    y = (y_raw > 0.4).astype(int) 
    
    # Exclude 20% of data to test the AI's true accuracy later
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[2/3] Dataset Vectorized: {len(X_train)} training arrays, {len(X_test)} validation arrays.")
    
    # Define the 4 Classical Machine Learning models specified in the Plan
    models = {
        "knn": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "logreg": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "dtree": DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        "rforest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    os.makedirs("models/ml", exist_ok=True)
    
    print("\n[3/3] Training Scikit-Learn Algorithms & Serializing Architectures...")
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    for name, model in models.items():
        # Perform 5-Fold Cross Validation across the ENTIRE dataset
        cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
        acc = cv_results['test_accuracy'].mean()
        prec = cv_results['test_precision'].mean()
        rec = cv_results['test_recall'].mean()
        f1 = cv_results['test_f1'].mean()
        
        # Fit the final model to the split training set to deploy its architecture, and get 
        # local predictions specifically so we can export the confusion_matrix graph!
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Save Model to Disk
        model_path = f"models/ml/{name}.pkl"
        joblib.dump(model, model_path)
        
        # ---------------------------------------------
        # ✨ GENERATE INTERPRETABILITY GRAPHS
        # ---------------------------------------------
        feature_names = ["Total Objects", "High Conf Anomalies", "Anomaly Ratio", 
                         "Max Confidence", "Avg Confidence", "Conf StdDev", 
                         "Max Area", "Avg Area", "Center Proximity", "Dominant Class"]
                         
        if name == "rforest":
            # 1. Save Feature Importance JSON for the React Frontend
            importances = model.feature_importances_.tolist()
            feat_dict = [{"name": feature_names[i], "value": round(importances[i] * 100, 1)} for i in range(len(feature_names))]
            # Sort by highest importance
            feat_dict = sorted(feat_dict, key=lambda x: x["value"], reverse=True)
            
            with open("models/ml/feature_importance.json", "w") as f:
                json.dump(feat_dict, f, indent=4)
                
            # 2. Confusion Matrix Heatmap (Brain Scan)
            try:
                cm = confusion_matrix(y_test, predictions)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Safe', 'Threat'], yticklabels=['Safe', 'Threat'])
                plt.title("Random Forest Confusion Matrix")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                plt.tight_layout()
                plt.savefig("models/ml/confusion_matrix.png", bg='transparent')
                plt.close()
            except Exception as e:
                print("Could not generate heatmap (matplotlib/seaborn missing)")
                
        if name == "dtree":
            # 3. Decision Tree Flowchart
            try:
                plt.figure(figsize=(15, 10))
                tree.plot_tree(model, max_depth=3, feature_names=feature_names, class_names=['Safe', 'Threat'], filled=True, rounded=True)
                plt.title("Decision Tree Logic Flow")
                plt.savefig("models/ml/decision_tree_flow.png", bg='transparent')
                plt.close()
            except Exception:
                pass
        
        print(f"      [{name.upper().ljust(7)}] Acc: {acc*100:.1f}% | Prec: {prec*100:.1f}% | Rec: {rec*100:.1f}% | F1: {f1*100:.1f}% -> Saved to {model_path}")
        
    print("\nTraining Complete! Layer 2 Intelligence is ready for the Inference Pipeline.")

if __name__ == "__main__":
    train_ensemble_layer()
