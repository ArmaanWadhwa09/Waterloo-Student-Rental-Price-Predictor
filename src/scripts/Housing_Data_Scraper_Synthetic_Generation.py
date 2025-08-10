import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde

# --- Load your data ---
df = pd.read_csv("bamboo_listings_Scaped_Data.csv") 

# --- Step 1: Clean Price ---
df["Price"] = df["Price"].str.replace(r"[^\d]", "", regex=True).astype(float)

# --- Step 2: Extract Rooms_Available and Total_Rooms from Room_Info ---
def extract_room_info(text):
    try:
        parts = text.strip().lower().split(" of ")
        Rooms_Available = int(parts[0])
        total = int(parts[1].split()[0])
        return pd.Series([Rooms_Available, total])
    except:
        return pd.Series([np.nan, np.nan])

df[["Rooms_Available", "Total_Rooms"]] = df["Room_Info"].apply(extract_room_info)
df["Rooms_Available"] = df["Rooms_Available"].astype("Int64")
df["Total_Rooms"] = df["Total_Rooms"].astype("Int64")
df["Price_per_Bed"] = df["Price"] / df["Rooms_Available"]

# --- Step 3: Estimate Baths ---
def compute_price_bins(group):
    tr = group.name
    if pd.isna(tr) or tr < 1:
        return pd.DataFrame()
    if tr == 1:
        return pd.DataFrame({"thresholds": [np.inf]})
    quantiles = np.linspace(0, 1, tr + 1)[1:-1]
    thresholds = group["Price_per_Bed"].quantile(quantiles).values
    return pd.DataFrame({"thresholds": thresholds})

thresholds_dict = (
    df.groupby("Total_Rooms")
    .apply(compute_price_bins)
    .reset_index()
    .groupby("Total_Rooms")["thresholds"]
    .apply(list)
    .to_dict()
)

def estimate_baths(row):
    tr = row["Total_Rooms"]
    ppb = row["Price_per_Bed"]
    rooms_avail = row["Rooms_Available"]
    if pd.isna(tr) or pd.isna(ppb) or pd.isna(rooms_avail):
        return np.nan
    if tr == 1:
        return 1
    thresholds = thresholds_dict.get(tr)
    if not thresholds:
        return np.nan
    for i, t in enumerate(thresholds):
        if ppb < t:
            return min(i + 1, rooms_avail)
    return min(tr, rooms_avail)

df["Baths"] = df.apply(estimate_baths, axis=1).astype("Int64")

# --- Step 4: Fill missing values ---
df["Description"].fillna("Spacious student-friendly unit", inplace=True)
df["Address"].fillna("Near University Ave, Waterloo", inplace=True)
df["Lease_Type"].fillna("Fall Lease", inplace=True)
df["Gender"].fillna("Coed", inplace=True)
df["Scraped_At"] = pd.to_datetime(df["Scraped_At"], errors="coerce")

# --- Step 5: KDE for synthetic Price_per_Bed ---
price_per_bed_kdes = {}
for tr, group in df.dropna(subset=["Price_per_Bed", "Total_Rooms"]).groupby("Total_Rooms"):
    if len(group) > 10:
        price_per_bed_kdes[tr] = gaussian_kde(group["Price_per_Bed"])
    else:
        price_per_bed_kdes[tr] = None

# --- Categorical pools ---
lease_probs = df["Lease_Type"].value_counts(normalize=True).to_dict()
gender_probs = df["Gender"].value_counts(normalize=True).to_dict()
desc_values = df["Description"].dropna().unique()
address_values = df["Address"].dropna().unique()
rooms_avail_dist = {
    tr: g["Rooms_Available"].value_counts(normalize=True).to_dict()
    for tr, g in df.groupby("Total_Rooms")
}

# --- Generate synthetic rows ---
num_needed = max(0, 250000 - len(df))
print(f"Generating {num_needed} synthetic rows...")

def sample_categorical(dist_dict):
    keys, probs = zip(*dist_dict.items())
    return np.random.choice(keys, p=probs)

def generate_synthetic_row():
    total_rooms = int(np.random.choice(df["Total_Rooms"].dropna().unique()))
    rooms_avail = sample_categorical(rooms_avail_dist.get(total_rooms, {1: 1.0})) if total_rooms in rooms_avail_dist else random.randint(1, total_rooms)
    Rooms_Available = int(rooms_avail)

    kde = price_per_bed_kdes.get(total_rooms)
    if kde:
        ppb = max(10, kde.resample(1).flatten()[0])
    else:
        group = df[df["Total_Rooms"] == total_rooms]
        ppb = max(10, np.random.normal(group["Price_per_Bed"].mean(), group["Price_per_Bed"].std()))

    price = round(Rooms_Available * ppb, 2)
    price = np.clip(price, 500, 4000)

    thresholds = thresholds_dict.get(total_rooms)
    baths = 1
    if total_rooms == 1:
        baths = 1
    elif thresholds:
        for i, t in enumerate(thresholds):
            if ppb < t:
                baths = i + 1
                break
        else:
            baths = total_rooms
    baths = min(baths, Rooms_Available)

    lease = sample_categorical(lease_probs)
    gender = sample_categorical(gender_probs)
    desc = random.choice(desc_values)
    address = random.choice(address_values)
    scraped_at = datetime.now() - timedelta(days=random.randint(0, 90))

    # New Columns
    pet_friendly = np.random.choice(["Yes", "No"], p=[0.3, 0.7])
    square_ft = int(np.clip(np.random.normal(400 + 150 * Rooms_Available, 100), 250, 1500))
    furnished = np.random.choice(["Yes", "No"], p=[0.9, 0.1])
    parking = np.random.choice(["Yes", "No"], p=[0.55, 0.45])
    distance_km = round(np.random.uniform(0.3, 4.0), 2)
    walk_score = int(np.clip(100 - (distance_km * random.uniform(12, 20)), 20, 100))
    transit_score = int(np.clip(100 - (distance_km * random.uniform(10, 18)), 30, 100))
    lease_term = np.random.choice([4, 8, 12], p=[0.2, 0.3, 0.5])
    noise_level = round(np.clip(np.random.normal(distance_km * 10, 5), 20, 90), 2)
    internet = np.random.choice(["Yes", "No"], p=[0.95, 0.05])

    # Anomalies (3%)
    if random.random() < 0.03:
        price *= random.uniform(3, 5)
        price = min(price, 10000)
        if random.random() < 0.5:
            Rooms_Available = random.choice([0, total_rooms + random.randint(1, 3)])
        if random.random() < 0.5:
            baths = random.choice([0, total_rooms + random.randint(1, 3)])
        if random.random() < 0.5:
            rooms_avail = random.choice([0, total_rooms + random.randint(1, 3)])

    # Final fix: Baths <= Rooms_Available
    baths = max(1, min(baths, Rooms_Available if Rooms_Available > 0 else total_rooms))

    return {
        "Price": round(price, 2),
        "Address": address,
        "Description": desc,
        "Lease_Type": lease,
        "Gender": gender,
        "Scraped_At": scraped_at.strftime("%Y-%m-%d %H:%M:%S"),
        "Rooms_Available": Rooms_Available,
        "Baths": baths,
        "Total_Rooms": total_rooms,
        "Pet_Friendly": pet_friendly,
        "Square_Footage": square_ft,
        "Furnished": furnished,
        "Parking_Included": parking,
        "Distance_to_Uni_km": distance_km,
        "Walk_Score": walk_score,
        "Transit_Score": transit_score,
        "Lease_Term_Months": lease_term,
        "Noise_Level": noise_level,
        "Internet_Available": internet
    }
synthetic_rows = [generate_synthetic_row() for _ in range(num_needed)]
df_synth = pd.DataFrame(synthetic_rows)
# After generating synthetic_rows and df_synth

new_cols = [
    "Pet_Friendly", "Square_Footage", "Furnished", "Parking_Included",
    "Distance_to_Uni_km", "Walk_Score", "Transit_Score", "Lease_Term_Months",
    "Noise_Level", "Internet_Available"
]

for col in new_cols:
    if col not in df.columns:
        df[col] = np.nan  # create column if missing
    # Sample from synthetic column values to fill NaNs in original df
    missing_mask = df[col].isna()
    num_missing = missing_mask.sum()
    if num_missing > 0:
        samples = np.random.choice(df_synth[col], size=num_missing, replace=True)
        df.loc[missing_mask, col] = samples
# Combine and finalize
df_final = pd.concat([df, df_synth], ignore_index=True)
df_final.drop(columns=["Address","Description", "Scraped_At", "Bed_Bath", "Room_Info"], errors="ignore", inplace=True)
df_final['Price_per_Bed'] = df_final['Price'] / df_final['Rooms_Available']
df_final.Price = df_final.Price.round()
df_final.Distance_to_Uni_km = df_final.Distance_to_Uni_km.round(2)
df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
df_final.dropna(inplace=True)
df_final = df_final[df_final['Baths'] <= df_final['Rooms_Available']]
df_final.to_csv("bamboo_full_raw_data.csv", index=False)

print(f"Final dataset shape: {df_final.shape}")
print(f"Synthetic rows generated: {len(df_synth)}")



assert not (df_final['Baths'] > df_final['Rooms_Available']).any(), "Some baths still exceed available rooms"