import streamlit as st
import numpy as np
import pickle
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained CNN model with data augmentation
try:
    with open("mnist_cnn_augmented_model.pkl", "rb") as f:
        cnn_augmented_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Model file 'mnist_cnn_augmented_model.pkl' not found. Please ensure the file exists in the correct directory.")
    st.stop()

# Streamlit app title
st.title("Handwritten Digit Recognition - Augmented CNN Model")

# Project description
st.write("""
This app uses a pre-trained Convolutional Neural Network (CNN) model with data augmentation to recognize handwritten digits (0-9). 
Draw a digit in the canvas below and click 'Predict' to see the result.
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

        # Convert the image to a numpy array and reshape for CNN input
        img_array = np.array(img)
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)

        # Normalize the pixel values
        img_array = img_array / 255.0

        # Make a prediction using the augmented CNN model
        cnn_prediction = cnn_augmented_model.predict(img_array)
        predicted_digit = np.argmax(cnn_prediction)

        # Get confidence scores (probabilities)
        confidence_scores = cnn_prediction[0]

        # Display the prediction
        st.write(f"**Predicted Digit:** {predicted_digit}")

        # Display confidence scores
        st.write("**Confidence Scores:**")
        for digit, score in enumerate(confidence_scores):
            st.write(f"Digit {digit}: {score:.4f}")
    else:
        st.write("Please draw a digit before clicking 'Predict'.")
