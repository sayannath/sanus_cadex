from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import joblib

from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template

# Define a flask app
app = Flask(__name__, template_folder='static')

pneumoniamodel = load_model("./model/Pneumonia-DENSENET.h5")         # Necessary

def HeartValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load(r'./model/heart_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

def pneumoniamodel_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = x / 255.0 

    preds = model.predict(x)
    return preds


@app.route('/pneumonia/', methods=['GET'])
def pneumonia():
    # Main page
    return render_template('pneumonia/home.html')
@app.route('/pneumonia/home.html', methods=['GET'])
def pneumoniahome():
    # Main page
    return render_template('pneumonia/home.html')

@app.route('/pneumonia/test',methods=['GET'])
def pneumoniatest():
    return render_template('pneumonia/predict.html')


@app.route('/pneumonia/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST': 
        print("hanji-------------------------")       
 
        # Get the file from post request
        f = request.files['file']
       
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = pneumoniamodel_predict(file_path, pneumoniamodel)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = preds.argmax(axis = 1)   # ImageNet Decode
        if(pred_class[0] == 0):
            answer = "Normal"
        else:
            answer = "Pneumonia"
        result = answer               # Convert to string
        return result
    return None

@app.route("/heart")
def cancer():
    return render_template("heart/index.html")

@app.route('/heart/predict', methods = ["POST"])
def predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
         #diabetes
        if(len(to_predict_list)==7):
            result = HeartValuePredictor(to_predict_list,7)
    
    if(int(result)==1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("heart/result.html", prediction_text=prediction))       

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)