import streamlit as st
import numpy as np
import joblib
import cv2
from skimage.feature import hog
from PIL import Image

# Load the trained model
MODEL_PATH = "model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

# Ensure the correct image size (same as training)
TARGET_SIZE = (64, 64)  # Change this to match training size

# HOG Parameters (Ensure they match training)
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)

# Function to preprocess image and extract HOG features
def extract_features(image):
    image = np.array(image.convert("L"))  # Convert to grayscale
    image = cv2.resize(image, TARGET_SIZE)  # Resize to match training size
    features = hog(image, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, feature_vector=True)
    return features.reshape(1, -1)  # Ensure shape matches training

# Streamlit UI
st.title("🌿 Plant Disease Detection")
st.write("Upload an image of a leaf to detect disease.")

# File upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract features
    features = extract_features(image)

    # Load the model
    model = load_model()

    # Check feature shape consistency
    expected_features = model.n_features_in_
    actual_features = features.shape[1]

    if actual_features != expected_features:
        st.error(f"Feature mismatch! Model expects {expected_features} features but got {actual_features}.")
    else:
        # Make prediction
        prediction = model.predict(features)[0]
        st.success(f"Predicted Disease: {prediction}")
