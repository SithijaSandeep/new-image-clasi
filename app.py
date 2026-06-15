import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Page Configuration (App UI Setup)
st.set_page_config(page_title="Student Image Classifier", page_icon="📸", layout="centered")
st.title("📸 Image Classification App")
st.write("Upload an image to predict its class using the trained MobileNetV2 model.")

# 2. Load Model and Class Names
@st.cache_resource  # App eka run vena hemavitema ආයෙ ආයෙ model eka load vena eka valakvanna
def load_my_model():
    # Ara clean training eka nisa kisima custom_objects / safe_mode mukuth ona ne!
    model = tf.keras.models.load_model("student_mobilenetv2_transfer_learning.keras")
    with open("class_names.json", "r") as f:
        classes = json.load(f)
    return model, classes

try:
    model, class_names = load_my_model()
    st.success("Model and Class names loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or files: {e}")

# 3. Image Upload Section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image eka open කර RGB format ekata haravima
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("👁️ Predicting...")
    
    # 4. Preprocessing (Training code eke vidiyatama)
    img_resized = image.resize((160, 160)) # Model input size = 160x160
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Batch dimension eka ekathu kirima (1, 160, 160, 3)
    
    # Preprocess_input (MobileNetV2 eka expect karana range ekata values hadima)
    img_array = preprocess_input(img_array)
    
    # 5. Model Prediction
    predictions = model.predict(img_array)
    
    # Categorical Crossentropy + Softmax thiyena nisa predictions[0] eke thiyenne probability values
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100
    
    # 6. Show Results to User
    st.markdown(f"### Prediction: **{predicted_class}**")
    st.progress(int(confidence))
    st.write(f"Confidence Level: **{confidence:.2f}%**")
