from fastapi import FastAPI
from pydantic import BaseModel
from model import predict

app = FastAPI()

# Define request body structure for prediction
class ModelInput(BaseModel):
    features: list  # List of 28x28 pixel values for image input

@app.post("/predict")
def get_prediction(input_data: ModelInput):
    try:
        prediction = predict(input_data.features)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "Neural Network Microservice is Running"}
