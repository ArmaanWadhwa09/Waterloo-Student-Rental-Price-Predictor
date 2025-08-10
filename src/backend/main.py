from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Optional
from feature_for_API import create_features, rule_based_anomaly_check, apply_amenity_adjustments
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
anomaly_model = joblib.load("isolation_forest_model.pkl")
iso_scaler = joblib.load("scaler.pkl")
price_scaler = joblib.load("price_scaler.pkl")
price_model = joblib.load("price_prediction_xgb.pkl")

# Features for each model
iso_features = [
    "Available_Room_Pct", "Furnished_X_Internet", "Is_Luxury_Setup", "Baths",
    "Sqft_per_Room", "Bathroom_Ratio", "Amenity_Score", "Is_Pet_Friendly","Parking_X_Pet"
]
price_features = [
    "Shared_Quality", "Baths", "Is_Shared_Setup", "Bed_Bath_Ratio", "Total_Rooms",
    "Bathroom_Ratio", "Available_Room_Pct", "Is_Balanced", "Is_Spacious", "Rooms_Available"
]

class Listing(BaseModel):
    Price: int
    Baths: int
    Total_Rooms: int
    Rooms_Available: int
    Furnished: Optional[str] = None
    Internet_Available: Optional[str] = None
    Parking_Included: Optional[str] = None
    Pet_Friendly: Optional[str] = None
    Square_Footage: Optional[int] = None

@app.post("/predict")
def predict(listing: Listing):
    data = listing.dict()

    # ----- Fill missing fields with assumptions -----
    messages = []
    default_probs = {
        "Furnished": ("Yes", 0.9),
        "Internet_Available": ("Yes", 0.95),
        "Parking_Included": ("Yes", 0.55),
        "Pet_Friendly": ("No", 0.7)
    }
    for field, (default_val, prob) in default_probs.items():
        if data.get(field) in [None, ""]:
            data[field] = default_val
            messages.append(f"Assumed {field.replace('_', ' ')} = {default_val} (probability: {int(prob * 100)}%)")

    if data.get("Square_Footage") in [None, 0]:
        est = 180
        data["Square_Footage"] = int(np.clip(np.random.normal(est, 100), 250, 1500))
        messages.append(f"Estimated Square Footage a Standard Room: {data['Square_Footage']} sqft")

    # ----- Feature Engineering -----
    df = pd.DataFrame([data])
    df = create_features(df)

    # ----- Rule-Based Checks -----
    rule_flag_series, rule_message_series = rule_based_anomaly_check(df)
    rule_flag = int(rule_flag_series.iloc[0])
    rule_messages = rule_message_series.iloc[0]
    if isinstance(rule_messages, list):
        messages.extend(rule_messages)

    # ----- Anomaly Detection -----
    X_iso = df[iso_features]
    X_iso_scaled = iso_scaler.transform(X_iso)
    anomaly_score = anomaly_model.decision_function(X_iso_scaled)[0]
    iso_flag = int(anomaly_model.predict(X_iso_scaled)[0] == -1)

    # ----- Price Prediction -----
    X_price = df[price_features]
    X_price_scaled = price_scaler.transform(X_price)
    predicted_price = np.exp(price_model.predict(X_price_scaled)[0])
    predicted_price = round(float(predicted_price), 2)
    
    final_price = apply_amenity_adjustments(
        predicted_price,
        amenities={
            'furnished': data['Furnished'] == 'Yes',
            'internet': data['Internet_Available'] == 'Yes',
            'parking': data['Parking_Included'] == 'Yes',
            'pet': data['Pet_Friendly'] == 'Yes'},
         square_footage=data['Square_Footage'])
    
    predicted_price = round(final_price)

    # ----- Price Check -----
    user_price = data["Price"]
    diff = user_price - predicted_price
    if abs(diff) <= 100:
        price_status = "Fairly priced"
    elif diff > 100:
        price_status = f"Overpriced by ${int(diff)}"
    else:
        price_status = f"Underpriced by ${int(abs(diff))}"

    price_is_unusual = abs(diff) > 250

    # ----- Combined Anomaly -----
    combined_anomaly_flag = int(iso_flag or price_is_unusual or rule_flag)

    # ----- Explanations -----
    row = df.iloc[0]
    if row["Is_Luxury_Setup"] and row["Amenity_Score"] < 2:
        messages.append("Luxury setup but lacks basic amenities.")
    if row["Furnished_X_Internet"] == 0:
        messages.append("Both internet and furnishing are missing.")
    if row["Bathroom_Ratio"] > 1.5:
        messages.append("Unusually high number of bathrooms.")

    # ----- Anomaly Summary -----
    if combined_anomaly_flag:
        anomaly_message = "This listing appears unusual based on its features, pricing, or internal consistency."
    else:
        anomaly_message = "This listing looks reasonable compared to similar listings."

    return {
        "predicted_price": predicted_price,
        "price_status": price_status,
        "anomaly_flag": combined_anomaly_flag,
        "anomaly_check": anomaly_message,
        "explanations": messages
    }
