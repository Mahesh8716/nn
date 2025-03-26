from flask import Flask, request, jsonify
from models import load_model, predict_image
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)

# Load the pre-trained model once
try:
    model, weights, bias = load_model()
except FileNotFoundError as e:
    print(e)
    exit(1)

@app.route('/train', methods=['POST'])
def home():
    return "Welcome to the MNIST Prediction Service!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read image and make prediction
    img_bytes = file.read()
    img = Image.open(BytesIO(img_bytes))
    prediction = predict_image(img, model, weights, bias)
    
    return jsonify({"prediction": int(prediction)})  # Convert to int for JSON response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 4000)))
