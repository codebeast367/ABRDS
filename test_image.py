import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

print("Image test started")

model = tf.keras.models.load_model("biodiversity_model.h5")

class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation',
    'Highway', 'Industrial', 'Pasture',
    'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

RISK_MAP = {
    "Forest": "LOW",
    "HerbaceousVegetation": "LOW",
    "Pasture": "LOW",

    "AnnualCrop": "MEDIUM",
    "PermanentCrop": "MEDIUM",
    "River": "MEDIUM",
    "SeaLake": "MEDIUM",

    "Residential": "HIGH",
    "Industrial": "HIGH",
    "Highway": "HIGH"
}

def adjust_risk(base_risk, confidence):
    if confidence < 60:
        if base_risk == "LOW":
            return "MEDIUM"
        if base_risk == "MEDIUM":
            return "HIGH"
    return base_risk

img_path = "test_images/sample.jpg"

img = load_img(img_path, target_size=(64, 64))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

preds = model.predict(img_array)

predicted_index = np.argmax(preds)
predicted_class = class_names[predicted_index]
confidence = np.max(preds) * 100

base_risk = RISK_MAP[predicted_class]
final_risk = adjust_risk(base_risk, confidence)

print("Land Type :", predicted_class)
print(f"Confidence : {confidence:.2f}%")
print("Risk Level :", final_risk)

