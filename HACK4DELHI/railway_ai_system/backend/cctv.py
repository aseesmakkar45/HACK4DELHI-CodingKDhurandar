print("🔥🔥🔥 CCTV MODULE LOADED FROM:", __file__)
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
    Loads the CCTV ML model only once.
    Never raises import-time errors.
    """

    global _MODEL, _LABEL_NAMES, _IMG_SIZE, _MODEL_LOADED

    if _MODEL_LOADED:
        return _MODEL, _LABEL_NAMES, _IMG_SIZE

    # Path relative to project root
        # ✅ Path relative to railway_ai_system/
      SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
      MODEL_PATH = os.path.abspath(
       os.path.join(SCRIPT_DIR, "..", "my_image_classifier.pkl")
)

      print("🔥🔥🔥 CCTV MODULE LOADED FROM:", __file__)
      print("📍 CCTV model path:", MODEL_PATH)
      print("📦 Exists:", os.path.exists(MODEL_PATH))


    if not os.path.exists(MODEL_PATH):
        print("⚠️ CCTV ML model not found → running in fallback mode")
        _MODEL_LOADED = True
        return None, None, None

    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)

        _MODEL = model_data["model"]
        _LABEL_NAMES = model_data["label_names"]
        _IMG_SIZE = model_data["img_size"]

        print("✅ CCTV ML model loaded successfully")
        print("📊 Classes:", list(_LABEL_NAMES.values()))
        print("🖼️ Image size:", _IMG_SIZE)

    except Exception as e:
        print(f"❌ Failed to load CCTV ML model: {e}")
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
# PUBLIC API (USED BY app.py)
# =========================

def analyze_visual(media_file):

    # 🔑 Always attempt model load once
    model, label_names, img_size = _load_model_once()

    if media_file is None:
        return "no_feed"

    if model is None:
        return "normal"


    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(media_file.read())
        image_path = tmp.name

    try:
        features = _extract_features(image_path, img_size)

        pred_idx = model.predict(features)[0]
        probs = model.predict_proba(features)[0]

        predicted_class = label_names[pred_idx].lower()
        confidence = probs[pred_idx] * 100

        print(f"🔍 CCTV Prediction: {predicted_class} ({confidence:.1f}%)")

    except Exception as e:
        print(f"❌ CCTV inference error: {e}")
        os.remove(image_path)
        return "normal"

    os.remove(image_path)

    # ---------- LABEL MAPPING (UNCHANGED LOGIC) ----------
    if predicted_class == "normal_track":
        return "normal"

    if predicted_class == "human_on_track":
        return "suspicious(human detected)"

    if predicted_class == "object_on_track":
        return "tampering(object detected)"

    if predicted_class == "broken_track":
        return "tampering(broken track)"

    # Fallback keyword matching
    if "normal" in predicted_class:
        return "normal"
    if "human" in predicted_class:
        return "suspicious(human detected)"
    if "object" in predicted_class:
        return "tampering(object detected)"
    if "broken" in predicted_class:
        return "tampering(broken track)"

    print(f"⚠️ Unknown CCTV class '{predicted_class}' → defaulting to normal")
    return "normal"





