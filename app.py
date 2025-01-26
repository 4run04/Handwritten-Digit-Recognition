import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Disable GPU if you're facing GPU-related issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# App title
st.title("Handwritten Digit Recognition")
st.write("Draw a digit (0-9) in the box below and click 'Predict' to see the result.")

# Load default model
model = None
try:
    model = load_model("/workspaces/Handwritten-Digit-Recognition/mnist_cnn_model.h5")
    st.success("Default model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the default model: {e}")

# Allow user to upload a custom model
uploaded_model = st.file_uploader("Upload your model (.h5 file)", type=["h5"])
if uploaded_model:
    try:
        # Save the uploaded file to a temporary location
        temp_model_path = "temp_model.h5"
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())

        # Load the model from the temporary file
        model = load_model(temp_model_path)
        st.success("Custom model loaded successfully!")

        # Optionally clean up temporary file (optional but recommended)
        os.remove(temp_model_path)

    except Exception as e:
        st.error(f"Error loading the uploaded model: {e}")
        st.stop()

# Ensure a model is loaded
if not model:
    st.error("No model loaded. Please provide a valid model to proceed.")
    st.stop()

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
        # Convert the canvas image to grayscale and resize to 28x28
        img = Image.fromarray(canvas_result.image_data.astype("uint8"))
        img = img.convert("L")
        img = img.resize((28, 28))

        # Preprocess the image for the model
        img_array = np.array(img)
        img_array = img_array.reshape(1, 28, 28, 1) / 255.0

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        # Display the prediction
        st.write(f"Predicted Digit: **{predicted_digit}**")

        # Show confidence scores
        if st.checkbox("Show Confidence Scores"):
            confidence_scores = {str(i): round(float(score), 3) for i, score in enumerate(prediction[0])}
            st.write("Confidence Scores:")
            st.write(confidence_scores)

        # Show the processed image
        if st.checkbox("Show Processed Image"):
            st.image(img, caption="Processed Image (28x28 Grayscale)", width=100)
    else:
        st.warning("Please draw a digit before clicking 'Predict'.")
