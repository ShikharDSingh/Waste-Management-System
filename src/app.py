from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Load the trained model
model = joblib.load("models/linear_regression_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get user input from the form
    input_data = {
        "City/District": request.form["city"],
        "Waste Type": request.form["waste_type"],
        "Waste Generated (Tons/Day)": float(request.form["waste_generated"]),
        "Population Density (People/km²)": float(request.form["population_density"]),
        "Municipal Efficiency Score (1-10)": int(request.form["municipal_efficiency"]),
        "Cost of Waste Management (₹/Ton)": float(request.form["cost"]),
        "Awareness Campaigns Count": int(request.form["campaigns"]),
        "Landfill Capacity (Tons)": float(request.form["landfill_capacity"]),
        "Year": int(request.form["year"]),
        "Disposal Method": request.form["disposal_method"],
        "Landfill Name": request.form["landfill_name"],
        "Latitude": float(request.form["latitude"]),
        "Longitude": float(request.form["longitude"])
    }

    df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(df)[0]
    prediction = round(prediction, 2)

    return render_template("index.html", prediction_text=f"Estimated Recycling Rate: {prediction}%")

if __name__ == "__main__":
    app.run(debug=True)
