# backend/vibration.py
import pandas as pd
import numpy as np

def analyze_vibration(vibration_df):
    """
    Analyze vibration sensor data.
    Input: pandas DataFrame with 'acceleration' column
    Output: 'normal' or 'abnormal'
    """

    if vibration_df is None or vibration_df.empty:
        return "no_data"

    # Basic signal features
    rms = np.sqrt(np.mean(vibration_df["acceleration"] ** 2))
    peak = np.max(np.abs(vibration_df["acceleration"]))

    # Thresholds (tunable, explainable)
    if rms > 0.5 or peak > 1.0:
        return "abnormal"

    return "normal"