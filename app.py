import os
from flask import Flask, flash, jsonify, request, redirect, url_for, render_template
from flask import send_from_directory
from werkzeug.utils import secure_filename
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import urllib.request
import json
import re

UPLOAD_FOLDER = '/Users/michaelli/Desktop/SEA/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, template_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "5789"

labelList = []
def getPrediction(filename):
    global labelList
    image = cv2.imread('uploads/'+filename)

    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    print("[INFO] loading network...")
    model = load_model('animals.model')
    lb = pickle.loads(open("lb.pickle", "rb").read())

    print("[INFO] classifying image...")
    probs = model.predict(image)[0]
    idx = np.argmax(probs)
    label = lb.classes_[idx]

    label = str(label, 'utf-8')
    print(label)

    labelList.append(label)
    labelList.append(probs[idx]*100)
    probs[idx]= probs[idx]*100

    return labelList[0], labelList[1]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            getPrediction(filename)
            label, acc = getPrediction(filename)
            round(acc)
            flash(label)
            flash(acc)
            flash(filename)
            return render_template('index.html', filename=filename)

@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

if __name__ == "__main__":
    app.run()