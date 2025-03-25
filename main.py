# main.py
from flask import Flask, request, jsonify
from models import create_model, predict_image, train_model
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)

# Load the model once
model, weights, bias, train_dl = create_model()

@app.route('/')
def home():
    return "Welcome to the MNIST Prediction Service!"

@app.route('/train', methods=['POST'])
def train():
    # Train the model when the endpoint is hit
    try:
        train_model(model, train_dl, weights, bias, learning_rate=0.1, epochs=5)  # Adjust epochs as needed
        return jsonify({"status": "Model training completed!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # If file is present, get image and predict
    img_bytes = file.read()
    img = Image.open(BytesIO(img_bytes))
    prediction = predict_image(img, model, weights, bias)
    
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 4000)))