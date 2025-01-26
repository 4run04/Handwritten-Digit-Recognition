import streamlit as st
import numpy as np
import pickle
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained Random Forest model
with open("mnist_rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load the pre-trained CNN model with data augmentation
with open("mnist_cnn_augmented_model.pkl", "rb") as f:
    cnn_augmented_model = pickle.load(f)

# Streamlit app title
st.title("Handwritten Digit Recognition - Model Comparison")

# Project description
st.write("""
This app uses two pre-trained models (Random Forest and CNN with Data Augmentation) to recognize handwritten digits (0-9). 
Draw a digit in the canvas below and click 'Predict' to see the results from both models.
""")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="black",  # Background color of the canvas
    stroke_width=15,     # Thickness of the drawing stroke
    stroke_color="white",# Color of the drawing stroke
    background_color="black",  # Canvas background color
    width=280,           # Canvas width
    height=280,          # Canvas height
    drawing_mode="freedraw",  # Allow free drawing
    key="canvas",
)

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert the canvas image to grayscale and resize to 28x28
        img = Image.fromarray(canvas_result.image_data.astype("uint8"))
        img = img.convert("L")  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28 pixels

        # Display the processed image
        st.write("**Processed Image (28x28 Grayscale):**")
        st.image(img, caption="Processed Image", width=100)

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Preprocess the image for Random Forest
        img_rf = img_array.reshape(1, -1)  # Flatten to (1, 784)
        img_rf = img_rf / 255.0  # Normalize pixel values

        # Preprocess the image for CNN
        img_cnn = img_array.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)
        img_cnn = img_cnn / 255.0  # Normalize pixel values

        # Make predictions using both models
        rf_prediction = rf_model.predict(img_rf)
        rf_confidence_scores = rf_model.predict_proba(img_rf)[0]

        cnn_prediction = cnn_augmented_model.predict(img_cnn)
        cnn_confidence_scores = cnn_prediction[0]  # CNN outputs probabilities directly

        # Create two columns for side-by-side comparison
        col1, col2 = st.columns(2)

        # Display Random Forest results in the first column
        with col1:
            st.write("### Random Forest Model")
            st.write(f"**Predicted Digit:** {rf_prediction[0]}")
            st.write("**Confidence Scores:**")
            for digit, score in enumerate(rf_confidence_scores):
                st.write(f"Digit {digit}: {score:.4f}")

        # Display CNN results in the second column
        with col2:
            st.write("### CNN Model (with Data Augmentation)")
            st.write(f"**Predicted Digit:** {np.argmax(cnn_confidence_scores)}")
            st.write("**Confidence Scores:**")
            for digit, score in enumerate(cnn_confidence_scores):
                st.write(f"Digit {digit}: {score:.4f}")
    else:
        st.write("Please draw a digit before clicking 'Predict'.")
