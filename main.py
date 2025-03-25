from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import pickle
import io

app = FastAPI()

# Load the saved model
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

weights = model_data["weights"]
bias = model_data["bias"]

def predict(image: Image.Image):
    img_tensor = torch.tensor(image).float() / 255
    img_tensor = img_tensor.view(-1, 28 * 28)
    pred = torch.sigmoid(img_tensor @ weights + bias)
    return "Three" if pred.item() > 0.5 else "Seven"

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("L").resize((28, 28))
    prediction = predict(image)
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)