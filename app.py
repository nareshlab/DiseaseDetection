from keras.models import load_model
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

"""
# deep Classifier project
"""

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model_path = 'C:\\Users\\nares\\OneDrive\\Desktop\\DiseaseDetection\\keras_model.h5'
model = load_model(model_path, compile=False)

# Load the labels
labels_path = 'C:\\Users\\nares\\OneDrive\\Desktop\\DiseaseDetection\\labels.txt'
class_names = open(labels_path, 'r').readlines()

# Rest of the code remains unchanged...


# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Streamlit file uploader
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Resize the image to 224x224
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display results in Streamlit
    st.image(image, caption=f'Class: {class_name}')
    st.write(f'Confidence Score: {confidence_score}')
