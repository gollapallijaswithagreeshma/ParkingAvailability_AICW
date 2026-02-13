import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


# 1. Page Configuration
st.set_page_config(page_title="Parking Availability Predictor", layout="centered")

# 2. Load the trained model
# Note: Ensure 'model_final.h5' is in your project folder
@st.cache_resource
def load_my_model():
    model = load_model('model_int8.tflite')
    return model

try:
    model = load_my_model()
    class_names = ['empty', 'occupied']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. App UI
st.title("🚗 Parking Spot Availability")
st.write("Upload a photo of a parking spot to check if it's empty or occupied.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # 4. Preprocessing (matching your project specs)
    # Your notebook uses 48x48 resolution
    img_width, img_height = 48, 48
    
    # Prepare image for prediction
    test_image = img.resize((img_width, img_height))
    test_image = image.img_to_array(test_image)
    # Rescale by 1/255 as used in your ImageDataGenerator
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    # 5. Prediction
    if st.button('Predict Status'):
        with st.spinner('Analyzing...'):
            prediction = model.predict(test_image)
            result_index = np.argmax(prediction)
            result = class_names[result_index]
            confidence = np.max(prediction) * 100

            # Display Result
            if result == 'empty':
                st.success(f"### Result: {result.upper()} ✅")
            else:
                st.error(f"### Result: {result.upper()} ❌")
            
            st.info(f"Confidence: {confidence:.2f}%")