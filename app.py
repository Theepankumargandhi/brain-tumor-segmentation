import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

# Title and description
st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
st.title("🧠 Brain Tumor Segmentation App")
st.write("Upload a brain MRI image to perform tumor segmentation.")

# App status
st.success("✅ App is running!")

# Upload image
uploaded_file = st.file_uploader("📁 Upload a brain MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded MRI Image", use_column_width=True)

    # Convert image to array and print shape
    img_np = np.array(image)
    st.write("Image shape:", img_np.shape)
