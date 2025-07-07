import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow.keras.backend as K
import base64

st.set_page_config(page_title="Iris Match Detector", page_icon="=A", layout="centered")

# Load trained model
model = load_model(r"C:\Users\akash\Downloads\siamese_model.h5")

# Preprocessing function
def preprocess_image(uploaded_file, image_size=(100, 100)):
    image = Image.open(uploaded_file).convert("L")
    image = image.resize(image_size)
    image = np.array(image).astype('float32') / 255.0
    return np.expand_dims(image, axis=(0, -1))

# Background image as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_image = get_base64_image(r"C:\Users\akash\Downloads\image.jpg")  #  Update path as needed

# === Inject CSS ===
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.55), rgba(0, 0, 0, 0.55)),
                    url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    .title {{
        font-size: 40px;
        text-align: center;
        color: #ffffff;
        font-weight: 700;
        text-shadow: 2px 2px 4px #000;
        margin-top: 250px;
    }}
    .subtitle {{ 
        font-size: 20px;
        text-align: center;
        color: #f0f0f0;
        margin-bottom: 20px;
    }}
    .upload-container {{
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        background-color: rgba(0, 0, 0, 0.65);
        padding: 6px 10px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        z-index: 9999;
        width: 65%;
        max-width: 520px;
    }}
    .upload-container .element-container {{
        margin-bottom: 0px !important;
    }}
    .upload-container label {{
        font-size: 12px !important;
        margin-bottom: 1px !important;
    }}
    .upload-container .stFileUploader {{
        transform: scale(0.85);  /* == Shrinks the uploader box */
        transform-origin: top left;
    }}
    .result-box {{
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: white;
        font-size: 22px;
        font-weight: bold;
        margin-top: 30px;
    }}
    .match {{
        background-color: rgba(40, 167, 69, 0.85);
    }}
    .nomatch {{
        background-color: rgba(220, 53, 69, 0.85);
    }}
    </style>
""", unsafe_allow_html=True)

# === Title Section ===
st.markdown('<div class="title">Iris Match Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload two iris images to verify identity</div>', unsafe_allow_html=True)

# === Upload Section ===
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
image1 = st.file_uploader(" Upload First Iris Image", type=["jpg", "jpeg", "png"], key="img1")
image2 = st.file_uploader("Upload Second Iris Image", type=["jpg", "jpeg", "png"], key="img2")
st.markdown('</div>', unsafe_allow_html=True)

# === Matching Logic ===
if image1 and image2:
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption="=A Image 1", use_column_width=True)
    with col2:
        st.image(image2, caption="=A Image 2", use_column_width=True)

    img1 = preprocess_image(image1)
    img2 = preprocess_image(image2)
    prediction = model.predict([img1, img2])[0][0]
    

    st.markdown("---")
    confidence = float(prediction)

    if confidence > 0.5:
        st.markdown(f'<div class="result-box match"> SAME PERSON <br> Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box nomatch">L DIFFERENT PERSONS <br> Confidence: {(1 - confidence):.2f}</div>', unsafe_allow_html=True)

    st.progress(confidence if confidence > 0.5 else (1 - confidence))
