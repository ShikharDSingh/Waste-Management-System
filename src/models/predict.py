import joblib
import pandas as pd

def load_model(model_path="../../models/linear_regression_model.pkl"):
    return joblib.load(model_path)

def predict(model, input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return round(prediction, 2)
