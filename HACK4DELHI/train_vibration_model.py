"""
Railway Vibration ML Model Training - Logistic Regression
==========================================================
Trains a Logistic Regression classifier to detect tampering
in railway track vibration data.

Features extracted:
- Statistical: mean, std, RMS, peak, variance
- Distribution: skewness, kurtosis, quartiles
- Frequency: energy, zero-crossing rate
- Signal characteristics: peak-to-peak, crest factor

Usage:
    python ml_models/train_vibration_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(csv_path):
    """
    Extract comprehensive features from vibration CSV
    
    Returns:
        dict: Dictionary of feature values
        None: If file cannot be processed
    """
    try:
        df = pd.read_csv(csv_path)
        
        if 'acceleration' not in df.columns:
            print(f"⚠️  Warning: 'acceleration' column not found in {csv_path}")
            return None
        
        acc = df['acceleration'].values
        
        # Basic checks
        if len(acc) < 10:
            print(f"⚠️  Warning: Too few samples in {csv_path}")
            return None
        
        # ====================================================================
        # STATISTICAL FEATURES
        # ====================================================================
        
        # Central tendency
        mean_val = np.mean(acc)
        median_val = np.median(acc)
        
        # Spread
        std_val = np.std(acc)
        variance = np.var(acc)
        
        # Root Mean Square (important for vibration analysis)
        rms = np.sqrt(np.mean(acc ** 2))
        
        # Peak values
        peak_val = np.max(np.abs(acc))
        peak_to_peak = np.ptp(acc)
        
        # Quartiles
        q25 = np.percentile(acc, 25)
        q75 = np.percentile(acc, 75)
        iqr = q75 - q25  # Interquartile range
        
        # ====================================================================
        # SHAPE FEATURES
        # ====================================================================
        
        # Skewness (asymmetry of distribution)
        skewness = pd.Series(acc).skew()
        
        # Kurtosis (tailedness of distribution)
        kurtosis = pd.Series(acc).kurtosis()
        
        # ====================================================================
        # ENERGY FEATURES
        # ====================================================================
        
        # Total energy
        energy = np.sum(acc ** 2)
        
        # Absolute energy
        abs_energy = np.sum(np.abs(acc))
        
        # ====================================================================
        # FREQUENCY-DOMAIN FEATURES
        # ====================================================================
        
        # Zero crossing rate (frequency indicator)
        zero_crossings = np.sum(np.diff(np.sign(acc)) != 0)
        zero_crossing_rate = zero_crossings / len(acc)
        
        # ====================================================================
        # AMPLITUDE FEATURES
        # ====================================================================
        
        # Crest factor (peak to RMS ratio)
        # High crest factor = impulsive (hammering)
        # Low crest factor = continuous (grinding)
        crest_factor = peak_val / (rms + 1e-10)
        
        # Shape factor
        shape_factor = rms / (np.mean(np.abs(acc)) + 1e-10)
        
        # Impulse factor
        impulse_factor = peak_val / (np.mean(np.abs(acc)) + 1e-10)
        
        # Clearance factor
        clearance_factor = peak_val / (np.mean(np.sqrt(np.abs(acc))) ** 2 + 1e-10)
        
        # ====================================================================
        # VARIABILITY FEATURES
        # ====================================================================
        
        # Coefficient of variation
        cv = std_val / (np.abs(mean_val) + 1e-10)
        
        # Range
        range_val = np.max(acc) - np.min(acc)
        
        # Mean absolute deviation
        mad = np.mean(np.abs(acc - mean_val))
        
        # ====================================================================
        # TEMPORAL FEATURES
        # ====================================================================
        
        # Count peaks (local maxima)
        # Useful for detecting periodic impacts
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(np.abs(acc), height=np.std(acc))
        num_peaks = len(peaks)
        peak_density = num_peaks / len(acc)
        
        # ====================================================================
        # ASSEMBLE FEATURE VECTOR
        # ====================================================================
        
        features = {
            # Statistical
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'variance': variance,
            'rms': rms,
            'peak': peak_val,
            'peak_to_peak': peak_to_peak,
            'q25': q25,
            'q75': q75,
            'iqr': iqr,
            
            # Shape
            'skewness': skewness,
            'kurtosis': kurtosis,
            
            # Energy
            'energy': energy,
            'abs_energy': abs_energy,
            
            # Frequency
            'zero_crossing_rate': zero_crossing_rate,
            
            # Amplitude factors
            'crest_factor': crest_factor,
            'shape_factor': shape_factor,
            'impulse_factor': impulse_factor,
            'clearance_factor': clearance_factor,
            
            # Variability
            'cv': cv,
            'range': range_val,
            'mad': mad,
            
            # Temporal
            'num_peaks': num_peaks,
            'peak_density': peak_density,
        }
        
        return features
        
    except Exception as e:
        print(f"❌ Error processing {csv_path}: {e}")
        return None

# ============================================================================
# LOAD TRAINING DATA
# ============================================================================

def load_training_data():
    """
    Load all CSV files and extract features
    
    Returns:
        X: Feature matrix (numpy array)
        y: Labels (0=normal, 1=abnormal)
        feature_names: List of feature names
    """
    
    normal_csvs = glob('training_data/normal/*.csv')
    abnormal_csvs = glob('training_data/abnormal/*.csv')
    
    print(f"📂 Found {len(normal_csvs)} normal samples")
    print(f"📂 Found {len(abnormal_csvs)} abnormal samples")
    
    if len(normal_csvs) == 0 or len(abnormal_csvs) == 0:
        raise ValueError("❌ No training data found! Run generate_vibration_data.py first.")
    
    X = []
    y = []
    feature_names = None
    
    # Process normal samples
    print("\n🔄 Processing normal samples...")
    for i, csv_file in enumerate(normal_csvs):
        features = extract_features(csv_file)
        if features:
            if feature_names is None:
                feature_names = list(features.keys())
            X.append(list(features.values()))
            y.append(0)  # 0 = normal
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(normal_csvs)} normal samples")
    
    # Process abnormal samples
    print("\n🔄 Processing abnormal samples...")
    for i, csv_file in enumerate(abnormal_csvs):
        features = extract_features(csv_file)
        if features:
            X.append(list(features.values()))
            y.append(1)  # 1 = abnormal
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(abnormal_csvs)} abnormal samples")
    
    return np.array(X), np.array(y), feature_names

# ============================================================================
# TRAIN MODEL
# ============================================================================

def train_model():
    """
    Train Logistic Regression model
    """
    
    print("\n" + "=" * 70)
    print("🚆 RAILWAY VIBRATION ML MODEL TRAINING - LOGISTIC REGRESSION")
    print("=" * 70)
    
    # Load data
    print("\n📊 Loading training data...")
    X, y, feature_names = load_training_data()
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Total samples: {len(X)}")
    print(f"   Normal samples: {np.sum(y == 0)}")
    print(f"   Abnormal samples: {np.sum(y == 1)}")
    print(f"   Features per sample: {len(feature_names)}")
    
    # Check for class balance
    class_ratio = np.sum(y == 1) / len(y)
    print(f"   Class balance: {class_ratio*100:.1f}% abnormal")
    
    # Split data
    print("\n🔀 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Maintain class balance
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Normalize features
    print("\n⚙️  Normalizing features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("   ✓ Features normalized")
    
    # Train Logistic Regression
    print("\n🧠 Training Logistic Regression model...")
    model = LogisticRegression(
        penalty='l2',              # L2 regularization
        C=1.0,                     # Regularization strength
        solver='lbfgs',            # Optimization algorithm
        max_iter=1000,             # Maximum iterations
        random_state=42,
        class_weight='balanced'    # Handle any class imbalance
    )
    
    model.fit(X_train_scaled, y_train)
    print("   ✓ Model trained successfully!")
    
    # Cross-validation
    print("\n📈 Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=5, 
        scoring='accuracy'
    )
    print(f"   Cross-validation scores: {cv_scores}")
    print(f"   Mean CV accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=['Normal', 'Abnormal'],
        digits=3
    ))
    
    # Confusion matrix
    print("🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"    [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
    print(f"   TN=True Negative, FP=False Positive")
    print(f"   FN=False Negative, TP=True Positive")
    
    # Feature importance (coefficients)
    print("\n⭐ Top 10 Most Important Features:")
    coef_abs = np.abs(model.coef_[0])
    feature_importance = sorted(
        zip(feature_names, coef_abs),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (feat, importance) in enumerate(feature_importance[:10], 1):
        print(f"   {i:2d}. {feat:20s}: {importance:.4f}")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'train_accuracy': model.score(X_train_scaled, y_train),
        'test_accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    os.makedirs('ml_models', exist_ok=True)
    model_path = 'ml_models/vibration_classifier.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save feature names separately (for reference)
    feature_path = 'ml_models/feature_names.txt'
    with open(feature_path, 'w') as f:
        for i, feat in enumerate(feature_names, 1):
            f.write(f"{i}. {feat}\n")
    
    print(f"📝 Feature names saved to: {feature_path}")
    
    print("\n" + "=" * 70)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print("\n🎯 Model Performance Summary:")
    print(f"   Training Accuracy: {model_data['train_accuracy']*100:.2f}%")
    print(f"   Testing Accuracy:  {model_data['test_accuracy']*100:.2f}%")
    print(f"   Cross-Val Accuracy: {model_data['cv_mean']*100:.2f}% ± {model_data['cv_std']*100:.2f}%")
    print(f"\n✅ Ready to use in your Streamlit app!")
    print(f"   The app will automatically load: {model_path}")
    print("=" * 70)
    
    return model_data

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Make sure you've run generate_vibration_data.py first!")
        import traceback
        traceback.print_exc()