

import numpy as np
import tempfile
import os
import pickle
from PIL import Image


MODEL_PATH = "C:\\Users\\HP\\Desktop\\HACK4DELHI\\my_2class_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ my_2class_model.pkl not found in ml/ folder")

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
label_names = model_data["label_names"]   # {0: classA, 1: classB}
img_size = model_data["img_size"]         # (64, 64)


def extract_features(image_path):
    img = Image.open(image_path)

    
    if img.mode != "RGB":
        img = img.convert("RGB")

    
    img = img.resize(img_size)

    
    img_array = np.array(img).flatten() / 255.0

    return img_array.reshape(1, -1)


def analyze_drone_image(media_file):
    

    if media_file is None:
        return "no_feed"

    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(media_file.read())
        image_path = temp.name

    try:
        features = extract_features(image_path)
        pred_idx = model.predict(features)[0]
        predicted_label = label_names[pred_idx]

        
        confidence = max(model.predict_proba(features)[0])

    except Exception as e:
        print("❌ Drone ML error:", e)
        os.remove(image_path)
        return "normal"

    os.remove(image_path)

    
    label = predicted_label.lower()

    if label in ["normal", "safe", "clear"]:
        return "normal"

    
    return "anomaly"
