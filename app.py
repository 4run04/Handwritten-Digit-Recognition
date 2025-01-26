import os
import pickle
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# App title
st.title("Handwritten Digit Recognition (Pickle Version)")
st.write("Draw a digit (0-9) in the box below and click 'Predict' to see the result.")

# Debugging information
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Does the default model file exist? {'Yes' if os.path.exists('mnist_cnn_model.pkl') else 'No'}")

# Load the model
model = None
try:
    with open("mnist_cnn_model.pkl", "rb") as file:
        model = pickle.load(file)
    st.success("Default model loaded successfully!")
except Exception
