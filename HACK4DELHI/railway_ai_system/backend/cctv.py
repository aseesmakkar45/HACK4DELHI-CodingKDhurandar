import numpy as np
import tempfile
import os
import pickle
from PIL import Image

# =========================
# MODEL LOADING
# =========================

MODEL_PATH = 'HACK4DELHI/my_image_classifier.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")

print("🔄 Loading ML model...")

def load_model_safe(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return None

# 🔑 FIX: model_data is now properly loaded
model_data = load_model_safe(MODEL_PATH)

if model_data is None:
    raise RuntimeError("❌ Model loading failed!")

# Extract components (UNCHANGED LOGIC)
model = model_data["model"]
label_names = model_data["label_names"]   # dict: {0: class_name, ...}
img_size = model_data["img_size"]         # (64, 64)

print("✅ Model loaded successfully!")
print(f"📊 Trained classes: {list(label_names.values())}")
print(f"🖼️  Image size: {img_size}")

# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(image_path):
    """Extract features from image exactly like training"""
    img = Image.open(image_path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize(img_size)
    img_array = np.array(img).flatten() / 255.0

    return img_array.reshape(1, -1)

# =========================
# MAIN ANALYSIS LOGIC
# =========================

def analyze_visual(media_file):

    if media_file is None:
        return "no_feed"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(media_file.read())
        image_path = temp.name

    try:
        features = extract_features(image_path)
        prediction_idx = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        predicted_class = label_names[prediction_idx]
        confidence = probabilities[prediction_idx] * 100

        print(f"🔍 Prediction: {predicted_class} (Confidence: {confidence:.1f}%)")

    except Exception as e:
        print(f"❌ CCTV ML error: {e}")
        os.remove(image_path)
        return "normal"

    os.remove(image_path)

    label = predicted_class.lower()

    if label == "normal_track":
        return "normal"

    elif label == "human_on_track":
        return "suspicious(human detected)"

    elif label == "object_on_track":
        return "tampering(object detected)"

    elif label == "broken_track":
        return "tampering(broken track)"

    elif "normal" in label:
        return "normal"
    elif "human" in label:
        return "suspicious(human detected)"
    elif "object" in label:
        return "tampering(object detected)"
    elif "broken" in label:
        return "tampering(broken track)"

    else:
        print(f"⚠️  Unexpected class '{predicted_class}' - defaulting to normal")
        print("   Expected classes: Broken_track, human_on_track, normal_track, object_on_track")
        return "normal"

# =========================
# MODEL DIAGNOSTICS
# =========================

def test_model():
    """Test what classes your model actually outputs"""
    print("\n" + "=" * 60)
    print("🧪 MODEL DIAGNOSIS")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Trained classes: {label_names}")
    print(f"Number of classes: {len(label_names)}")
    print("=" * 60 + "\n")

# Streamlit-safe (no __main__ dependency)
test_model()

