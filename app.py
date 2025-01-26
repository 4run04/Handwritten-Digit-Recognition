import os
import pickle
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# App title
st.title("Handwritten Digit Recognition (CNN trained on augmented Mnist dataset)")
st.write("Draw a digit (0-9) in the box below and click 'Predict' to see the result.")

# Debug
#st.write(f"Current working directory: {os.getcwd()}")
#st.write(f"Does the default model file exist? {'Yes' if os.path.exists('mnist_cnn_model.pkl') else 'No'}")

model = None
try:
    with open("mnist_cnn_augmented_model.pkl", "rb") as file:
        model = pickle.load(file)
    st.success("Default model loaded successfully!")
except Exception as e:
    st.error(f"REFRESH THE APP or Continue drawing on canvas: {e}")

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
    if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
        # Convert the canvas image to grayscale and resize to 28x28
        img = Image.fromarray(canvas_result.image_data.astype("uint8"))
        img = img.convert("L")  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28 pixels

        # Preprocess the image for the model
        img_array = np.array(img)
        img_array = img_array.reshape(1, 28, 28, 1) / 255.0

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        # Display the prediction
        st.write(f"Predicted Digit: **{predicted_digit}**")

        # Display confidence scores
        confidence_scores = {str(i): round(float(score), 3) for i, score in enumerate(prediction[0])}
        st.write("Confidence Scores for Each Digit (0-9):")
        st.write(confidence_scores)

        # Display the processed image
        st.image(img, caption="Processed Image (28x28 Grayscale)", width=100)
    else:
        st.warning("Please draw a digit before clicking 'Predict'.")
