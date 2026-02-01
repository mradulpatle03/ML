from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="IntelliQueue ML API")

model = joblib.load("models/wait_time_xgb.pkl")

class QueueRequest(BaseModel):
    hour: int
    day_of_week: int
    active_counters: int
    customers_in_queue: int
    queue_lag_1: float
    queue_lag_5: float
    service_time: float

@app.post("/predict_wait_time")
def predict_wait_time(data: QueueRequest):
    is_weekend = 1 if data.day_of_week in [5, 6] else 0
    is_peak_hour = 1 if 11 <= data.hour <= 14 else 0

    features = np.array([[
        data.hour,
        data.day_of_week,
        is_weekend,
        is_peak_hour,
        data.active_counters,
        data.customers_in_queue,
        data.queue_lag_1,
        data.queue_lag_5,
        data.service_time
    ]])

    prediction = model.predict(features)[0]

    return {
        "predicted_wait_time_minutes": round(float(prediction), 2)
    }
