import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Image Classifier", page_icon="🖼️")
st.title("AI Image Classification App")
st.write("Upload an image to see the prediction!")

@st.cache_data
def load_class_names():
    with open('class_names.json', 'r') as f:
        return json.load(f)

class_names = load_class_names()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.keras')

model = load_model()

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    confidence = np.max(predictions[0]) * 100
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    
    st.success(f"**Predicted Class:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
