import os
import pickle
import tempfile
import numpy as np
from PIL import Image

# =========================
# INTERNAL CACHED STATE
# =========================

_MODEL = None
_LABEL_NAMES = None
_IMG_SIZE = None
_MODEL_LOADED = False


# =========================
# SAFE MODEL LOADER
# =========================

def _load_model_once():
    """
    Loads the drone ML model only once.
    Never crashes the app if model is missing.
    """

    global _MODEL, _LABEL_NAMES, _IMG_SIZE, _MODEL_LOADED

    if _MODEL_LOADED:
        return _MODEL, _LABEL_NAMES, _IMG_SIZE

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "my_2class_model.pkl")

    if not os.path.exists(MODEL_PATH):
        print("⚠️ Drone ML model not found → running in fallback mode")
        _MODEL_LOADED = True
        return None, None, None

    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)

        _MODEL = model_data["model"]
        _LABEL_NAMES = model_data["label_names"]
        _IMG_SIZE = model_data["img_size"]

        print("✅ Drone ML model loaded successfully")
        print("📊 Classes:", list(_LABEL_NAMES.values()))
        print("🖼️ Image size:", _IMG_SIZE)

    except Exception as e:
        print(f"❌ Failed to load Drone ML model: {e}")
        _MODEL = _LABEL_NAMES = _IMG_SIZE = None

    _MODEL_LOADED = True
    return _MODEL, _LABEL_NAMES, _IMG_SIZE


# =========================
# FEATURE EXTRACTION
# =========================

def _extract_features(image_path, img_size):
    img = Image.open(image_path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize(img_size)
    arr = np.array(img).flatten() / 255.0

    return arr.reshape(1, -1)


# =========================
# PUBLIC API
# =========================

def analyze_drone_image(media_file):
    """
    Returns:
      - normal
      - anomaly
      - no_feed
    """

    if media_file is None:
        return "no_feed"

    model, label_names, img_size = _load_model_once()

    # ---------- FALLBACK MODE ----------
    if model is None:
        return "normal"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(media_file.read())
        image_path = tmp.name

    try:
        features = _extract_features(image_path, img_size)

        pred_idx = model.predict(features)[0]
        predicted_label = label_names[pred_idx].lower()

        confidence = max(model.predict_proba(features)[0])
        print(f"🔍 Drone Prediction: {predicted_label} ({confidence:.1f})")

    except Exception as e:
        print("❌ Drone ML inference error:", e)
        os.remove(image_path)
        return "normal"

    os.remove(image_path)

    # ---------- DECISION LOGIC (UNCHANGED) ----------
    if predicted_label in ["normal", "safe", "clear"]:
        return "normal"

    return "anomaly"
