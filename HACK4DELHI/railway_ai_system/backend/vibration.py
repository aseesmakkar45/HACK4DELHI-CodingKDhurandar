"""
Railway Vibration Analysis - ML-Powered (Enhanced)
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
from scipy.signal import find_peaks

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Use relative path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'ml_models', 'vibration_classifier.pkl')

CONFIDENCE_THRESHOLD = 0.6  # Require 60% confidence for "abnormal"
EXPECTED_SAMPLE_RATE = 100  # Hz

# ============================================================================
# LOAD ML MODEL
# ============================================================================

ML_ENABLED = False
ml_model = None
scaler = None
feature_names = None

if os.path.exists(MODEL_PATH):
    try:
        logger.info("Loading ML vibration model...")
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        ml_model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        logger.info(f"✅ Model loaded: {model_data['test_accuracy']*100:.2f}% accuracy")
        ML_ENABLED = True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        ML_ENABLED = False
else:
    logger.warning(f"Model not found at: {MODEL_PATH}")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(vibration_df, sample_rate=None):
    """
    Extract features with robust error handling
    """
    try:
        acc = vibration_df['acceleration'].values
        
        # Clean data
        acc = acc[np.isfinite(acc)]
        
        if len(acc) < 10:
            raise ValueError("Too few valid samples")
        
        # Estimate sample rate
        if sample_rate is None and 'time' in vibration_df.columns:
            time = vibration_df['time'].values
            dt = np.median(np.diff(time))
            sample_rate = 1.0 / dt if dt > 0 else EXPECTED_SAMPLE_RATE
        elif sample_rate is None:
            sample_rate = EXPECTED_SAMPLE_RATE
        
        if abs(sample_rate - EXPECTED_SAMPLE_RATE) > 10:
            logger.warning(f"Sample rate {sample_rate:.1f} Hz differs from expected {EXPECTED_SAMPLE_RATE} Hz")
        
        # Statistical features
        mean_val = np.mean(acc)
        median_val = np.median(acc)
        std_val = np.std(acc)
        variance = np.var(acc)
        rms = np.sqrt(np.mean(acc ** 2))
        peak_val = np.max(np.abs(acc))
        peak_to_peak = np.ptp(acc)
        q25, q75 = np.percentile(acc, [25, 75])
        iqr = q75 - q25
        
        # Shape features (with NaN protection)
        skewness = pd.Series(acc).skew()
        kurtosis = pd.Series(acc).kurtosis()
        skewness = 0.0 if not np.isfinite(skewness) else skewness
        kurtosis = 0.0 if not np.isfinite(kurtosis) else kurtosis
        
        # Energy features
        energy = np.sum(acc ** 2)
        abs_energy = np.sum(np.abs(acc))
        
        # Frequency features (rate-normalized)
        zero_crossings = np.sum(np.diff(np.sign(acc)) != 0)
        zero_crossing_rate = zero_crossings / len(acc)
        
        # Amplitude factors (with divide-by-zero protection)
        eps = 1e-10
        crest_factor = peak_val / (rms + eps)
        shape_factor = rms / (np.mean(np.abs(acc)) + eps)
        impulse_factor = peak_val / (np.mean(np.abs(acc)) + eps)
        clearance_factor = peak_val / (np.mean(np.sqrt(np.abs(acc))) ** 2 + eps)
        
        # Variability features
        cv = std_val / (np.abs(mean_val) + eps)
        range_val = np.max(acc) - np.min(acc)
        mad = np.mean(np.abs(acc - mean_val))
        
        # Temporal features
        peaks, _ = find_peaks(np.abs(acc), height=np.std(acc))
        num_peaks = len(peaks)
        peak_density = num_peaks / len(acc)
        
        features = {
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
            'skewness': skewness,
            'kurtosis': kurtosis,
            'energy': energy,
            'abs_energy': abs_energy,
            'zero_crossing_rate': zero_crossing_rate,
            'crest_factor': crest_factor,
            'shape_factor': shape_factor,
            'impulse_factor': impulse_factor,
            'clearance_factor': clearance_factor,
            'cv': cv,
            'range': range_val,
            'mad': mad,
            'num_peaks': num_peaks,
            'peak_density': peak_density,
        }
        
        # Final check for any remaining NaN/Inf
        for key in features:
            if not np.isfinite(features[key]):
                logger.warning(f"Feature {key} is not finite, setting to 0")
                features[key] = 0.0
        
        return features
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise

# ============================================================================
# ML ANALYSIS
# ============================================================================

def analyze_vibration_ml(vibration_df):
    """ML-based analysis with confidence threshold"""
    
    features = extract_features(vibration_df)
    
    # Ensure correct feature order
    if ML_ENABLED and feature_names:
        feature_vector = np.array([[features[name] for name in feature_names]])
    else:
        feature_vector = np.array([list(features.values())])
    
    # Scale and predict
    feature_vector_scaled = scaler.transform(feature_vector)
    probability = ml_model.predict_proba(feature_vector_scaled)[0]
    
    # Apply confidence threshold
    if probability[1] >= CONFIDENCE_THRESHOLD:
        prediction = 1
        confidence = probability[1] * 100
    else:
        prediction = 0
        confidence = probability[0] * 100
    
    result = "abnormal" if prediction == 1 else "normal"
    
    logger.info(f"Prediction: {result}, Confidence: {confidence:.1f}%")
    logger.info(f"Probabilities: Normal={probability[0]:.3f}, Abnormal={probability[1]:.3f}")
    
    # Log key features for abnormal cases
    if result == "abnormal":
        logger.info(f"Key features: RMS={features['rms']:.3f}, Peak={features['peak']:.3f}, "
                   f"Crest={features['crest_factor']:.2f}, ZCR={features['zero_crossing_rate']:.3f}")
    
    return result

# ============================================================================
# RULE-BASED FALLBACK
# ============================================================================

def analyze_vibration_rule_based(vibration_df):
    """Improved rule-based thresholds based on training data"""
    
    features = extract_features(vibration_df)
    rms = features['rms']
    peak = features['peak']
    crest = features['crest_factor']
    
    # Thresholds based on synthetic training data
    # Normal: RMS ~0.03-0.12, Peak ~0.03-0.15
    # Abnormal: RMS ~0.4-0.9, Peak ~0.4-2.2
    
    is_abnormal = (
        rms > 0.25 or          # Between normal and abnormal range
        peak > 0.4 or          # Abnormal starts here
        crest > 8.0            # High impulsiveness (hammering)
    )
    
    result = "abnormal" if is_abnormal else "normal"
    
    logger.info(f"Rule-based: {result} (RMS={rms:.3f}, Peak={peak:.3f}, Crest={crest:.2f})")
    
    return result

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def analyze_vibration(vibration_df):
    """
    Main analysis function with comprehensive error handling
    """
    
    # Validation
    if vibration_df is None or vibration_df.empty:
        logger.error("No data provided")
        return "no_data"
    
    if 'acceleration' not in vibration_df.columns:
        logger.error("Missing 'acceleration' column")
        return "invalid_format"
    
    if len(vibration_df) < 10:
        logger.error("Too few data points")
        return "invalid_format"
    
    # Analysis
    try:
        if ML_ENABLED:
            return analyze_vibration_ml(vibration_df)
        else:
            return analyze_vibration_rule_based(vibration_df)
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        try:
            return analyze_vibration_rule_based(vibration_df)
        except:
            return "invalid_format"