import pandas as pd
import numpy as np

def create_features(df):
    # Convert Yes/No strings to booleans
    bool_cols = ['Furnished', 'Internet_Available', 'Parking_Included', 'Pet_Friendly']
    for col in bool_cols:
        df[f'Is_{col}'] = df[col].str.strip().str.lower().map({'yes': True, 'no': False}).fillna(False)

    # Rename for clarity
    df['Has_Internet'] = df['Is_Internet_Available']
    df['Has_Parking'] = df['Is_Parking_Included']
    df['Is_Shared_Setup'] = ((df['Total_Rooms'] > 1) & (df['Baths'] < 2)).astype(int)
    # Core ratios and interactions
    df['Available_Room_Pct'] = df['Rooms_Available'] / (df['Total_Rooms'] + 0.01)
    df['Furnished_X_Internet'] = df['Is_Furnished'].astype(int) * df['Has_Internet'].astype(int)
    df['Is_Luxury_Setup'] = (
        df['Is_Furnished'] & df['Has_Internet'] & (~df['Is_Shared_Setup'])
    ).astype(int)
    df['Sqft_per_Room'] = df['Square_Footage'] / (df['Total_Rooms'] + 0.1)
    df['Bathroom_Ratio'] = df['Baths'] / (df['Total_Rooms'] + 0.1)

    # Amenities
    df['Amenity_Score'] = (
        df['Is_Furnished'].astype(int) +
        df['Has_Internet'].astype(int) +
        df['Has_Parking'].astype(int)
    )

    df['Parking_X_Pet'] = df['Has_Parking'].astype(int) * df['Is_Pet_Friendly'].astype(int)

    # Bed/bath logic
    df['Bed_Bath_Ratio'] = df['Rooms_Available'] / df['Baths'].replace(0, np.nan)
    df['Is_Balanced'] = df['Bed_Bath_Ratio'].between(0.8, 1.2).astype(int)
    df['Is_Spacious'] = (df['Sqft_per_Room'] > df['Sqft_per_Room'].median()).astype(int)

    # Shared quality (a proxy score)
    df['Shared_Quality'] = df['Is_Shared_Setup'].astype(int) * (df['Baths'] / (df['Total_Rooms'] + 0.1))

    return df
def rule_based_anomaly_check_row(row):
    flags = []

    # Room/bath rules
    if row["Baths"] > row["Total_Rooms"]:
        flags.append("More baths than total rooms")
    if row["Baths"] > row["Rooms_Available"]:
        flags.append("More baths than rooms available")
    if row["Rooms_Available"] > row["Total_Rooms"]:
        flags.append("More rooms available than total rooms")
    if row["Total_Rooms"] == 0 or row["Rooms_Available"] == 0:
        flags.append("Zero rooms in the unit")

    # Square footage rules
    if row["Square_Footage"] < 100:
        flags.append("Square footage unrealistically low")
    if row["Square_Footage"] / max(row["Rooms_Available"], 1) > 300:
        flags.append("Square footage per available room too high")

    # Amenities logic
    if row["Furnished"] == "Yes" and row["Square_Footage"] < 100:
        flags.append("Furnished but size is too small")
    if row["Square_Footage"] == 0:
        flags.append("A 0 sqft unit?")

    # Price logic
    if row["Price"] < 2000 and row["Square_Footage"] > 700:
        flags.append("Too cheap for a large unit")

    return flags

def rule_based_anomaly_check(df):
    flags = df.apply(rule_based_anomaly_check_row, axis=1)
    rule_flag = flags.apply(lambda x: len(x) > 0)
    return rule_flag, flags

def apply_amenity_adjustments(base_price, amenities, square_footage=None):
    base_sqft = 180
    amenity_adjustments = {
        'unfurnished': 0.85,
        'no_internet': 0.85,
        'no_parking': 0.88,
        'no_pet': 0.95
        }
    
    adjustment_factor = 1.0
    
    # Apply amenity adjustments
    if not amenities.get('furnished', True):
        adjustment_factor *= amenity_adjustments['unfurnished']
    if not amenities.get('internet', True):
        adjustment_factor *= amenity_adjustments['no_internet']
    if not amenities.get('parking', True):
        adjustment_factor *= amenity_adjustments['no_parking']
    if not amenities.get('pet', True):
        adjustment_factor *= amenity_adjustments['no_pet']
    
    # Continuous size scaling (linear)
    if square_footage is not None:
        size_factor = square_footage / base_sqft
        adjustment_factor *= size_factor
    
    return base_price * adjustment_factor
