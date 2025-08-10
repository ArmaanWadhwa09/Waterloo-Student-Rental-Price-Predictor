import json

top_features = [
    'Available_Room_Pct',
    'Furnished_X_Internet',
    'Is_Luxury_Setup',
    'Baths',
    'Sqft_per_Room',
    'Bathroom_Ratio',
    'Amenity_Score',
    'Pet_Friendly',
    'Distance_to_Uni_km',
    'Parking_X_Pet'
]

with open("top_features.json", "w") as f:
    json.dump(top_features, f)