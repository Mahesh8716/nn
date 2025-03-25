# models.py
from fastai.vision.all import *
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

# Function to create and prepare the model
def create_model():
    path = Path('mnist_sample')
    threes = (path/'train'/'3').ls().sorted()
    sevens = (path/'train'/'7').ls().sorted()

    # Prepare training data
    threes_array = [tensor(Image.open(image)) for image in threes]
    threes_stack = torch.stack(threes_array).float() / 255
    sevens_array = [tensor(Image.open(image)) for image in sevens]
    sevens_stack = torch.stack(sevens_array).float() / 255

    train_x = torch.cat([threes_stack, sevens_stack])
    train_x = train_x.view((-1, 28*28))  # Flatten images to vectors

    train_y = tensor([1]*len(threes_stack) + [0]*len(sevens_stack))  # Labels: 1 for '3', 0 for '7'
    train_y = train_y.unsqueeze(1)  # Add extra dimension for output layer

    # Initialize weights and bias for linear model
    weights = torch.randn((28 * 28, 1), requires_grad=True)
    bias = torch.randn(1, requires_grad=True)

    # Prepare DataLoader
    dataset = TensorDataset(train_x, train_y)
    train_dl = DataLoader(dataset, batch_size=64, shuffle=True)

    # Linear model (simplified version)
    def linear_model(xb, weights, bias):
        return xb @ weights + bias

    return linear_model, weights, bias, train_dl


def train_model(model, train_dl, weights, bias, learning_rate=0.1, epochs=10):
    def loss_fun(pred, targ):
        pred = pred.sigmoid()
        return torch.where(targ == 1, 1 - pred, pred).mean()

    def cal_grad(X, y, weights, bias, model, loss_fun):
        pred = model(X, weights, bias)
        loss = loss_fun(pred, y)
        loss.backward()

    def accuracy(pred, targ):
        preds = pred.sigmoid() > 0.5  # Convert predictions to binary (0 or 1)
        return (preds == targ).float().mean()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for X, y in train_dl:
            cal_grad(X, y, weights, bias, model, loss_fun)
            for p in (weights, bias):
                p.data -= p.grad * learning_rate
                p.grad.zero_()
            
            # Accumulate loss
            pred = model(X, weights, bias)
            loss = loss_fun(pred, y)
            total_loss += loss.item()

            # Calculate accuracy
            acc = accuracy(pred, y)
            total_acc += acc.item()

        avg_loss = total_loss / len(train_dl)
        avg_acc = total_acc / len(train_dl)

        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')




# Model prediction function
def predict_image(image_path, model, weights, bias):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = tensor(img).float() / 255  # Normalize to [0, 1]
    img = img.view(-1, 28 * 28)  # Flatten the image to a 1D vector
    
    # Get model output
    output = model(img, weights, bias)
    prediction = torch.sigmoid(output).item()  # Sigmoid output for probability
    
    return prediction
