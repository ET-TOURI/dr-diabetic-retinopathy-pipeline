import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.gradcam import generate_gradcam
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input

st.title("🧠 DR Dashboard — Grad-CAM & Prediction")

uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "png"])
model = load_model("models/resnet_dr_finetuned.h5")

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    img_np = np.array(image)
    input_tensor = preprocess_input(img_np[np.newaxis, ...])
    prediction = model.predict(input_tensor)
    heatmap = generate_gradcam(model, input_tensor)
    st.image(image, caption="Original Image", width=224)
    st.image(heatmap, caption="Grad-CAM", width=224)
    st.markdown(f"**Predicted class:** {np.argmax(prediction)}")