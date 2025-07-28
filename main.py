from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model("baseModel.h5")

class Features(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str

def preprocess(features: Features):
    # This is a placeholder. Replace with your actual preprocessing logic.
    input_vector = [
        features.area,
        features.bedrooms,
        features.bathrooms,
        features.stories,
        1 if features.mainroad == "yes" else 0,
        1 if features.guestroom == "yes" else 0,
        1 if features.basement == "yes" else 0,
        1 if features.hotwaterheating == "yes" else 0,
        1 if features.airconditioning == "yes" else 0,
        features.parking,
        1 if features.prefarea == "yes" else 0,
        {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}.get(features.furnishingstatus.lower(), 0)
    ]
    return np.array([input_vector], dtype=np.float32)

@app.post("/predict")
def predict(features: Features):
    processed_input = preprocess(features)
    prediction = model.predict(processed_input)
    return {"predicted_price": float(prediction[0][0])}
