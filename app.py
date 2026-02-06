import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Load model

@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model("biodiversity_model.h5")

model = load_model_cached()


# Class names

class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation',
    'Highway', 'Industrial', 'Pasture',
    'PermanentCrop', 'Residential', 'River', 'SeaLake'
]


# Risk map & logic

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


# UI

st.set_page_config(page_title="Land Risk Detector", layout="centered")

st.title("Satellite Land Risk Detection")
st.write("Upload a satellite image to predict land type & environmental risk")

uploaded_files = st.file_uploader(
    "📤 Upload Satellite Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    st.markdown("---")
    st.subheader("🧪 Batch Prediction Results")

    results = []   # 🔴 IMPORTANT

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)

        img = image.resize((64, 64))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        predicted_index = np.argmax(preds)
        predicted_class = class_names[predicted_index]
        confidence = np.max(preds) * 100

        base_risk = RISK_MAP[predicted_class]
        final_risk = adjust_risk(base_risk, confidence)

        # store for graph & table
        results.append({
            "Image Name": uploaded_file.name,
            "Land Type": predicted_class,
            "Confidence (%)": confidence,
            "Risk Level": final_risk
        })

        # show per image
        st.image(image, width=200)
        st.write(f"🏞 Land Type: **{predicted_class}**")
        st.write(f"📊 Confidence: **{confidence:.2f}%**")

        if final_risk == "LOW":
            st.success("🟢 LOW Risk")
        elif final_risk == "MEDIUM":
            st.warning("🟡 MEDIUM Risk")
        else:
            st.error("🔴 HIGH Risk")

        st.markdown("---")

            # -------------------------
    # TABLE
    # -------------------------
    st.subheader("📋 Prediction Summary Table")
    df = pd.DataFrame(results)
    st.dataframe(df)

    # -------------------------
    # BAR CHART
    # -------------------------
    st.subheader("📊 Risk Distribution - Bar Chart")
    risk_counts = df["Risk Level"].value_counts()
    st.bar_chart(risk_counts)

    # -------------------------
    # PIE CHART
    # -------------------------
    st.subheader("🥧 Risk Distribution - Pie Chart")
    fig, ax = plt.subplots()
    ax.pie(
        risk_counts,
        labels=risk_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)

    # summary table
    st.subheader("Prediction Summary")
    st.dataframe(results)