import torch
from fastai.vision.all import *
import pickle

# Load dataset
path = untar_data(URLs.MNIST_SAMPLE)
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()

# Convert images to tensors
threes_array = [tensor(Image.open(image)) for image in threes]
sevens_array = [tensor(Image.open(image)) for image in sevens]
threes_stack = torch.stack(threes_array).float() / 255
sevens_stack = torch.stack(sevens_array).float() / 255

# Prepare training data
train_x = torch.cat([threes_stack, sevens_stack])
train_x = train_x.view((-1, 28*28))
train_y = tensor([1] * len(threes_stack) + [0] * len(sevens_stack)).unsqueeze(1)

# Define model
init_params = lambda size, std=1.0: (torch.randn(size) * std).requires_grad_()
weights = init_params((28 * 28, 1))
bias = init_params(1)

def linear_model(xb, weights, bias):
    return xb @ weights + bias

# Save the model
model_data = {"weights": weights, "bias": bias}
with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)