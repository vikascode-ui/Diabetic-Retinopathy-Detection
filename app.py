import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.keras"

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Diabetic Retinopathy Detection")
st.write("Upload a retinal image to get an AI-based prediction.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Retinal Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Retinal Image")

    if st.button("Predict"):
        image = image.resize((224, 224))

        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.array(image, dtype=np.float32)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)

        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0])) * 100

        class_names = [
            "No Diabetic Retinopathy",
            "Mild",
            "Moderate",
            "Severe",
            "Proliferative Diabetic Retinopathy"
        ]

        st.subheader("Prediction Result")
        st.success(class_names[predicted_class])
        st.write(f"Model confidence: {confidence:.2f}%")

        st.warning(
            "This is an educational AI project, not a medical diagnosis. "
            "Please consult an eye specialist for medical advice."
        )