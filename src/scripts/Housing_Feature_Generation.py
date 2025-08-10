import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

df = pd.read_csv("bamboo_data_preprocessed.csv")
# Preview
df.head()
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats

def create_150_features(df):
    # =====================================
    # 1. CORE PROPERTY CHARACTERISTICS (30 features)
    # =====================================

    # Space metrics (10 features)
    df['Room_Density'] = df['Total_Rooms'] / (df['Square_Footage'] + 1)
    df['Sqft_per_Room'] = df['Square_Footage'] / (df['Total_Rooms'] + 0.1)
    df['Bathroom_Ratio'] = df['Baths'] / (df['Total_Rooms'] + 0.1)
    df['Available_Room_Pct'] = df['Rooms_Available'] / (df['Total_Rooms'] + 0.01)
    df['Room_Size_Variation'] = df['Square_Footage'] / (df['Total_Rooms'] * df['Rooms_Available'])
    df['Occupancy_Rate'] = 1 - df['Available_Room_Pct']
    df['Bathroom_Accessibility'] = df['Baths'] * df['Square_Footage'] / 1000
    df['Total_Space_per_Bath'] = df['Square_Footage'] / (df['Baths'] + 0.1)
    df['Room_Size_Score'] = np.log1p(df['Square_Footage']) / np.log1p(df['Total_Rooms'])
    df['Compactness_Index'] = (df['Total_Rooms'] ** 2) / (df['Square_Footage'] + 1)

    # Binary flags (10 features)
    df['Is_Studio'] = ((df['Total_Rooms'] == 1) & (df['Square_Footage'] < 500)).astype(int)
    df['Is_High_Density'] = (df['Room_Density'] > 0.05).astype(int)
    df['Luxury_Bath'] = (df['Baths'] > df['Total_Rooms']/2).astype(int)
    df['Is_Overcrowded'] = (df['Total_Rooms'] > df['Square_Footage']/200).astype(int)
    df['Has_Extra_Bath'] = (df['Baths'] > df['Total_Rooms']).astype(int)
    df['Is_Spacious'] = (df['Square_Footage'] > df['Total_Rooms'] * 500).astype(int)
    df['Is_Balanced'] = ((df['Total_Rooms'] >= 2) & (df['Baths'] >= df['Total_Rooms']/2)).astype(int)
    df['Is_New_Renovated'] = (df['Square_Footage'] > df['Total_Rooms'] * 300).astype(int)
    df['Is_Shared_Setup'] = ((df['Total_Rooms'] > 2) & (df['Baths'] < 2)).astype(int)
    df['Is_Luxury_Setup'] = ((df['Baths'] >= 2) & (df['Square_Footage'] > 1000)).astype(int)

    # Binned features (10 features)
    df['Sqft_Tier'] = pd.cut(df['Square_Footage'],
                            bins=[0, 400, 600, 800, 1000, 1200, float('inf')],
                            labels=['micro', 'compact', 'small', 'medium', 'large', 'spacious'])

    df['Room_Tier'] = pd.cut(df['Total_Rooms'],
                            bins=[0, 1, 2, 3, 4, float('inf')],
                            labels=['studio', '1bed', '2bed', '3bed', '4+bed'])

    df['Bath_Tier'] = pd.cut(df['Baths'],
                            bins=[0, 1, 1.5, 2, 3, float('inf')],
                            labels=['0-1', '1', '1.5', '2', '3+'])

    # =====================================
    # 2. LOCATION INTELLIGENCE (30 features)
    # =====================================

    # Proximity features (10 features)
    df['Uni_Proximity'] = pd.cut(df['Distance_to_Uni_km'],
                                bins=[0, 0.5, 1, 2, 3, 5, float('inf')],
                                labels=['on_campus', 'adjacent', 'walkable', 'transit', 'commute', 'remote'])

    df['Walk_Time'] = df['Distance_to_Uni_km'] * 15  # 15 mins per km
    df['Bike_Time'] = df['Distance_to_Uni_km'] * 5  # 5 mins per km

    for dist in [0.5, 1, 2, 3, 5]:
        df[f'Within_{dist}km'] = (df['Distance_to_Uni_km'] <= dist).astype(int)

    # Accessibility profiles (10 features)
    df['Walk_Score_Tier'] = pd.cut(df['Walk_Score'],
                                  bins=[0, 30, 50, 70, 90, 100],
                                  labels=['poor', 'below_avg', 'average', 'good', 'excellent'])

    df['Transit_Score_Tier'] = pd.cut(df['Transit_Score'],
                                     bins=[0, 30, 50, 70, 90, 100],
                                     labels=['poor', 'below_avg', 'average', 'good', 'excellent'])

    df['Accessibility_Gap'] = (df['Walk_Score'] - df['Transit_Score']).abs()
    df['Walk_Transit_Ratio'] = df['Walk_Score'] / (df['Transit_Score'] + 1)
    df['Accessibility_Score'] = (df['Walk_Score'] * 0.6 + df['Transit_Score'] * 0.4)


    # =====================================
    # 3. LEASE DYNAMICS (25 features)
    # =====================================

    # Lease type features (15 features)
    lease_cols = [c for c in df.columns if 'Lease_Type_' in c]
    if lease_cols:
        df['Lease_Type_Count'] = df[lease_cols].sum(axis=1)
        df['Peak_Season_Lease'] = df[[c for c in lease_cols if 'Fall' in c or 'Spring' in c]].max(axis=1)
        df['Off_Season_Lease'] = df[[c for c in lease_cols if 'Winter' in c or 'Summer' in c]].max(axis=1)

        # Create interaction terms
        for lease_type in ['Fall', 'Spring', 'Winter']:
            if f'Lease_Type_{lease_type}' in lease_cols:
                df[f'{lease_type}_Walk_Score'] = df[f'Lease_Type_{lease_type}'] * df['Walk_Score']
                df[f'{lease_type}_Distance'] = df[f'Lease_Type_{lease_type}'] * df['Distance_to_Uni_km']

    # Term features (10 features)
    df['Lease_Term_Tier'] = pd.cut(df['Lease_Term_Months'],
                                  bins=[0, 3, 6, 9, 12, float('inf')],
                                  labels=['0-3', '4-6', '7-9', '10-12', '12+'])

    df['Short_Lease_Flag'] = (df['Lease_Term_Months'] < 4).astype(int)
    df['Long_Lease_Flag'] = (df['Lease_Term_Months'] > 12).astype(int)
    df['Academic_Year_Lease'] = ((df['Lease_Term_Months'] >= 8) & (df['Lease_Term_Months'] <= 10)).astype(int)

    # =====================================
    # 4. AMENITY ANALYSIS (25 features)
    # =====================================

    # Core amenity features (10 features)
    amenities = ['Furnished_int', 'Parking_Included_int', 'Internet_Available_int', 'Pet_Friendly_int']
    existing_amenities = [a for a in amenities if a in df.columns]

    if existing_amenities:
        df['Amenity_Score'] = df[existing_amenities].sum(axis=1)
        df['Missing_Amenities'] = len(existing_amenities) - df['Amenity_Score']
        df['Premium_Amenity_Count'] = df[[a for a in existing_amenities if a != 'Pet_Friendly_int']].sum(axis=1)

        # Create all pairwise interactions (6 features)
        for i, amen1 in enumerate(existing_amenities):
            for j, amen2 in enumerate(existing_amenities):
                if i < j:
                    df[f'{amen1.split("_")[0]}_X_{amen2.split("_")[0]}'] = df[amen1] * df[amen2]

    # Amenity patterns (9 features)
    if 'Amenity_Score' in df.columns:
        for score in [1, 2, 3, 4]:
            df[f'Has_{score}_Amenities'] = (df['Amenity_Score'] >= score).astype(int)

        df['Luxury_Bath_Low_Amenities'] = ((df['Luxury_Bath']) & (df['Amenity_Score'] < 2)).astype(int)
        df['High_Amenity_Small_Space'] = ((df['Amenity_Score'] >= 3) & (df['Square_Footage'] < 600)).astype(int)
        df['Low_Amenity_Large_Space'] = ((df['Amenity_Score'] < 2) & (df['Square_Footage'] > 800)).astype(int)


    # =====================================
    # 6. ANOMALY SIGNATURES (20 features)
    # =====================================

    # Density anomalies (5 features)
    df['High_Density_Small'] = (df['Is_High_Density'] & (df['Square_Footage'] < 600)).astype(int)
    df['Low_Density_Large'] = ((~df['Is_High_Density']) & (df['Square_Footage'] > 1000)).astype(int)
    df['Overcrowded_Flag'] = (df['Room_Density'] > df['Room_Density'].quantile(0.95)).astype(int)

    # Amenity anomalies (5 features)
    if 'Amenity_Score' in df.columns:
        df['Luxury_Bath_Low_Amenities'] = ((df['Luxury_Bath']) & (df['Amenity_Score'] < 2)).astype(int)
        df['High_Amenity_Remote'] = ((df['Amenity_Score'] >= 3) & (df['Distance_to_Uni_km'] > 5)).astype(int)

    # Statistical outliers (10 features)
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns
                   if not any(p in c.lower() for p in ['rent'])]

    for col in numeric_cols[:10]:  # Limit to top 10 numeric columns
        if len(df[col].unique()) > 5:  # Only for columns with sufficient variance
            df[f'Outlier_{col}'] = np.abs(stats.zscore(df[col].fillna(0)))
            df[f'Extreme_{col}'] = (df[f'Outlier_{col}'] > 3).astype(int)

    return df


df = create_150_features(df)

def preprocess_features(df):
    """
    Final optimized preprocessing that:
    1. Explicitly protects binary columns from one-hot encoding
    2. Handles all other column types properly
    3. Maintains all your engineered features
    """
    X = df.drop(columns=['Price','Outlier_Price','Extreme_Price'])

    # 2. First convert all known binary columns (both original and engineered)
    binary_cols = [
        'Pet_Friendly', 'Furnished', 'Parking_Included', 'Internet_Available',
        'Pet_Friendly_int', 'Furnished_int', 'Parking_Included_int', 'Internet_Available_int',
        'Is_Studio', 'Is_High_Density', 'Luxury_Bath', 'Is_Overcrowded',
        'Has_Extra_Bath', 'Is_Spacious', 'Is_Balanced', 'Is_New_Renovated',
        'Is_Shared_Setup', 'Is_Luxury_Setup', 'Short_Lease_Flag', 'Long_Lease_Flag',
        'Academic_Year_Lease', 'High_Density_Small', 'Low_Density_Large',
        'Overcrowded_Flag', 'Luxury_Bath_Low_Amenities', 'High_Amenity_Remote'
    ]

    # Convert using robust binary conversion
    for col in [c for c in binary_cols if c in X.columns]:
        X[col] = X[col].replace(
            {'True': 1, 'False': 0, 'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0, 'y': 1, 'n': 0}
        ).fillna(0).astype(int)

    # 3. Protect binary columns and handle other categoricals
    protected_cols = binary_cols + [c for c in X.columns if '_int' in c]
    categorical_cols = [
        col for col in X.select_dtypes(include=['object', 'category']).columns
        if col not in protected_cols and 1 < X[col].nunique() <= 6
    ]

    # One-hot encode only non-binary categoricals
    if categorical_cols:
        ohe = OneHotEncoder(drop='first', sparse_output=False)
        ohe_features = ohe.fit_transform(X[categorical_cols])
        ohe_df = pd.DataFrame(
            ohe_features,
            columns=ohe.get_feature_names_out(categorical_cols),
            index=X.index
        )
        X = pd.concat([X.drop(columns=categorical_cols), ohe_df], axis=1)

    # 4. Drop remaining non-numeric columns (except protected binary)
    text_cols = X.select_dtypes(include=['object', 'category']).columns
    X = X.drop(columns=text_cols)
    return X

# Usage:
X = preprocess_features(df)
print(f"Processed {len(X.columns)} features")

X.to_csv("Housing_Features.csv")