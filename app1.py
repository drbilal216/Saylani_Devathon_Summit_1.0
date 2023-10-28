#import os
#from flask import Flask, request, render_template
# from keras.preprocessing import image
# from keras.applications.vgg19 import preprocess_input
# from keras.models import load_model

import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

model = keras.models.load_model('model_F.h5')

st.write("## Binary Classification")

def predict(file):

    img = load_img(file, target_size=(150, 150, 3))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    prediction = "Not Infected" if result[0][0] > 0.5 else "Infected"
    st.write(prediction)

img_path = st.file_uploader("Pick an image...")


if st.button("Predict"):
    
    st.title("Image")
    img = Image.open(img_path)
    st.image(img)
    predict(img_path)


