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
import deeplearning

# Declare a flask app
app = Flask(__name__)

# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

# Model saved with Keras model.save()
MODEL_PATH = 'models/best_model_text.h5'

# Load my own trained model
model = load_model(MODEL_PATH)
print(f'Model loaded, Start serving... - Check http://127.0.0.1:5000/')

# IMAGE RELATED
# def model_predict(img, model):
#     img = img.resize((224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='tf')

#     preds = model.predict(x)
#     return preds

# Get the image from post request
# img = base64_to_pil(request.json)

# Save the image to ./uploads
# img.save("./uploads/image.png")


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predictImage', methods=['GET', 'POST'])
def predictImage():
    if request.method == 'POST':
        # Make prediction
        print(f'Running preprocessing...\n')
        test_sentence = deeplearning.run_deeplearning()
        print(f'Input sentence preprocessing ready. Running prediction...\n')

        result = deeplearning.evaluate_single_sentence(model, test_sentence, multiclass=False)  # a tuple

        print(f'Prediction is {result[0].capitalize()}, with a probability of {result[1][0][0]}\n')

        # Serialize the result, you can add additional fields
        return jsonify(result=result[0].capitalize(), probability=str(result[1][0][0]))

    return None


@app.route('/predictText', methods=['GET', 'POST'])
def predictText():
    if request.method == 'POST':
        # Make prediction
        print(f'Running preprocessing...\n')
        test_sentence = deeplearning.run_deeplearning()
        print(f'Input sentence preprocessing ready. Running prediction...\n')

        result = deeplearning.evaluate_single_sentence(model, test_sentence, multiclass=False)  # a tuple

        print(f'Prediction is {result[0].capitalize()}, with a probability of {result[1][0][0]}\n')

        # Serialize the result, you can add additional fields
        return jsonify(result=result[0].capitalize(), probability=str(result[1][0][0]))

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
