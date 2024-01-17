import numpy as np 
import os 
from flask import Flask, request, render_template
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.applications.inception_v3 import preprocess_input
import requests
from flask import Flask, request, render_template,redirect, url_for

modeln=load_model(r"C:\Users\indhu\OneDrive\Desktop\tea_leaves\Training files\model_vgg.h5")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/index')
def inde1():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/leaf')
def leaf():
    return render_template('leaf.html')

@app.route('/leaf',methods=["GET","POST"])
def nres():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(modeln.predict(img_data))
        index = ['Anthracnose', 'Algal Leaf', 'bird eye spot', 'brown blight', 'gray light', 'healthy', 'red leaf spot', 'white spot']
        result = index[prediction]
        return render_template("leaf.html", prediction=result)
    return None

if __name__ == '__main__':
    app.run(debug=True)