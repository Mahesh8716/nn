from fastai.vision.all import *
import torch
from PIL import Image
import os

WEIGHTS_PATH = "model_weights.pth"

# Function to create and load the model
def load_model():
    # Initialize weights and bias for linear model
    weights = torch.randn((28 * 28, 1), requires_grad=False)
    bias = torch.randn(1, requires_grad=False)

    # Load pre-trained weights
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH)
        weights.data = checkpoint['weights']
        bias.data = checkpoint['bias']
        print("✅ Loaded pre-trained model weights.")
    else:
        raise FileNotFoundError("❌ Model weights not found! Please provide 'model_weights.pth'.")

    # Linear model function
    def linear_model(xb, weights, bias):
        return xb @ weights + bias

    return linear_model, weights, bias


# Model prediction function
def predict_image(image, model, weights, bias):
    img = image.convert('L')  # Convert to grayscale
    img = tensor(img).float() / 255  # Normalize to [0, 1]
    img = img.view(-1, 28 * 28)  # Flatten the image to a 1D vector
    
    # Get model output
    output = model(img, weights, bias)
    prediction = torch.sigmoid(output).item()  # Sigmoid output for probability
    
    return 3 if prediction > 0.5 else 7  # Return class label (1 for '3', 0 for '7')
