# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil
from datetime import datetime

def get_image(jsonrequest):
    # Get the image from post request
    img = base64_to_pil(jsonrequest)

    # Save the image to ./uploads
    img.save(f"uploads/image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    return img

def run(img, model):
    # Preprocessing
    # Be careful, the input preprocessing must be exact same as on the original model otherwise, it won't make the correct prediction
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x, mode='tf')

    # Evaluation Phase
    probabilities = model.predict(x)
    #probability = "{:.3f}".format(np.amax(probabilities))   # Max probability
    probability = f"{np.amax(probabilities):.3f}"   # Max probability

    prediction = decode_predictions(probabilities, top=1)   # ImageNet Decode

    prediction = str(prediction[0][0][1])  # Convert to string
    prediction = prediction.replace('_', ' ')

    return (prediction, probability)