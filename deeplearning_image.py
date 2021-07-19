# -*- coding: utf-8 -*-
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.preprocessing import image

# Some utilites
import numpy as np
import pandas as pd
from util import base64_to_pil
from datetime import datetime

# Model Used
train_model = "ResNet"  # (Inception-v3, Inception-ResNet-v2): Inception, (ResNet-50): ResNet

# Parameters
if train_model == "Inception":
    img_width, img_height =	139, 139
elif train_model == "ResNet":
    img_width, img_height =	197, 197
class_names = list(["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"])

def get_image(jsonrequest):
    # Get the image from post request
    img = base64_to_pil(jsonrequest)

    # Save the image to ./uploads
    img.save(f"uploads/image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    return img

def get_image_from_file(imgpath):
    # Get the image from a file
    img = image.load_img(imgpath)  # PIL Object
    print(type(img))

    plt.imshow(np.asarray(img))  # Plot
    plt.show()

    return img

def preprocess_input(x):
	if train_model == "Inception":
		x /= 127.5
		x -= 1.
		return x
	elif train_model == "ResNet":
		x -= 128.8006  # np.mean(train_dataset)
		x /= 64.6497  # np.std(train_dataset)
	return x

def run(img, model):
    # Preprocessing Phase
    # Be careful, the input preprocessing must be the exact same as on the original model otherwise the model won't operate correctly
    images = np.empty((1, img_height, img_width, 3))
    i = 0
    
    x = image.img_to_array(img)  # Numpy Object
    #single_image = x.reshape(48, 48, 3)
    single_image = resize(x, (img_height, img_width), order=3, mode="constant")  # Bicubic
    # ret = np.empty((img_height, img_width, 3))  
    # ret[:, :, 0] = single_image
    # ret[:, :, 1] = single_image
    # ret[:, :, 2] = single_image
    images[i, :, :, :] = single_image  
    images = preprocess_input(images)

    # Evaluation Phase
    predictions = model.predict(
        images,
        batch_size = 1)

    predicted_classes = np.argmax(predictions, axis=1)  # Returns the class (position of the row) of the maximum probability
    probability = f"{np.amax(predictions):.4f}"   # Max probability
    prediction_string = class_names[predicted_classes[0]]  # Convert to string

    return (prediction_string, probability)  
    