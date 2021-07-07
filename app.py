import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Some utilites
import numpy as np
from util import base64_to_pil

from tensorflow.keras.models import load_model
import deeplearning_text
import deeplearning_image

# Declare a flask app
app = Flask(__name__)

# You can use pretrained models from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

# Model saved with Keras model.save()
MODEL_PATH_TEXT = 'models/best_model_text.h5'
MODEL_PATH_IMAGE = 'models/ResNet-50.h5'

# Load my text model
model_text = load_model(MODEL_PATH_TEXT)
print(f'Text Model loaded, Start serving... - Check http://127.0.0.1:5000/')

# Load my image model
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
model_image = load_model(MODEL_PATH_IMAGE)
print(f'Image Model loaded, Start serving... - Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predictText', methods=['GET', 'POST'])
def predictText():
    if request.method == 'POST':
        # Get the data from the POST request
        input_test_sentence = [request.json]

        # Make prediction
        print(f'Running preprocessing...\n')
        test_sentence = deeplearning_text.run_preprocessing(input_test_sentence)

        print(f'Input sentence preprocessing ready. Running prediction...\n')
        result = deeplearning_text.evaluate_single_sentence(model_text, test_sentence, multiclass=False)  # a tuple
        result_pred = result[0].capitalize()
        result_prob = f"{result[1][0][0]:.4f}"

        print(f'Prediction is {result_pred}, with a probability of {result_prob}\n')

        # Temp
        if result_pred == "Positive":
            result_pred = result_pred + " Emotion (Grouped, Original: H.)"
        elif result_pred == "Negative":
            result_pred = result_pred + " Emotion (Grouped, Original: A.)"

        # Serialize the result, you can add additional fields
        return jsonify(result=result_pred, probability=result_prob)

    return None


@app.route('/predictImage', methods=['GET', 'POST'])
def predictImage():
    if request.method == 'POST':
        # Get the data from the POST request
        test_image = deeplearning_image.get_image(request.json)

        # Make prediction
        print(f'Running preprocessing... Running prediction...\n')
        result = deeplearning_image.run(test_image, model_image)
        result_pred = result[0].capitalize()
        result_prob = result[1]

        print(f'Prediction is {result_pred}, with a probability of {result_prob}\n')

        # Serialize the result, you can add additional fields
        return jsonify(result=result_pred, probability=result_prob)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
