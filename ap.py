from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import sys
import pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)
model = load_model('Crop-images-classfication-main\croptypes.h5')
model_crop= pickle.load(open('Crop-images-classfication-main\model_crop.pkl', 'rb'))

class_names = ['Cherry','Coffee-plant','Cucumber','Fox_nut(Makhana)','Lemon','Olive-tree','Pearl_millet(bajra)','Tobacco-plant','almond','banana','cardamom','chilli','clove','coconut','cotton','gram','jowar','jute','maize','mustard-oil','papaya','pineapple','rice','soyabean','sugarcane','sunflower','tea','tomato','vigna-radiati(Mung)','wheat']
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/model')
def home():
    return render_template('model.html')


def crop_image(image):
    img = image.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    return img

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    img = crop_image(Image.open(image))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]

    return render_template('result.html', result=predicted_class)

if __name__ == "__main__":
    app.run(debug=True)
