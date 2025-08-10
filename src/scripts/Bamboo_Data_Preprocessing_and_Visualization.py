import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load and prepare data
df_final = pd.read_csv("bamboo_full_raw_data.csv")
sns.set(style="whitegrid", palette="Set2", font_scale=1.1)

# Create 2x2 grid layout with adjusted size and spacing
fig, axes = plt.subplots(2, 2, figsize=(15, 11))  # Increased figure size
plt.subplots_adjust(wspace=0.8, hspace=0.8)  # Increased spacing between subplots

# --- Plot 1: Price vs Total Rooms (Boxplot) ---
sns.boxplot(data=df_final, x="Total_Rooms", y="Price", ax=axes[0,0])
axes[0,0].set_title("Price Distribution by Room Count", pad=15, fontsize=14)
axes[0,0].set_ylabel("Price ($)", fontsize=12)
axes[0,0].set_xlabel("Total Rooms", fontsize=12)
axes[0,0].tick_params(axis='both', labelsize=10)

# --- Plot 2: Price per Bed vs Lease Term (Boxplot) ---
sns.boxplot(data=df_final, x="Lease_Term_Months", 
            y=df_final["Price"]/df_final["Rooms_Available"], 
            ax=axes[0,1])
axes[0,1].set_title("Price per Bedroom by Lease Term", pad=15, fontsize=14)
axes[0,1].set_ylabel("Price per Bed ($)", fontsize=12)
axes[0,1].set_xlabel("Lease Term (Months)", fontsize=12)
axes[0,1].tick_params(axis='both', labelsize=10)

# --- Plot 3: Price vs Square Footage (Scatter) ---
scatter = sns.scatterplot(data=df_final, x="Square_Footage", y="Price", 
               hue="Furnished", alpha=0.6, ax=axes[1,0])
axes[1,0].set_title("Price vs Property Size", pad=15, fontsize=14)
axes[1,0].set_xlabel("Square Footage", fontsize=12)
axes[1,0].set_ylabel("Price ($)", fontsize=12)
axes[1,0].tick_params(axis='both', labelsize=10)
axes[1,0].legend(title="Furnished", bbox_to_anchor=(1.05, 1), 
                loc='upper left', fontsize=10)

# --- Plot 4: Feature Correlation (Heatmap) ---
corr_data = df_final[["Price", "Rooms_Available", "Baths", 
                     "Square_Footage", "Distance_to_Uni_km"]].corr()
sns.heatmap(corr_data, annot=True, cmap="coolwarm", 
           fmt=".2f", cbar=False, ax=axes[1,1], annot_kws={"size": 10})
axes[1,1].set_title("Feature Correlation Matrix", pad=15, fontsize=14)
axes[1,1].tick_params(axis='x', rotation=45, labelsize=10)
axes[1,1].tick_params(axis='y', labelsize=10)

plt.suptitle("Rental Market Analysis Dashboard", y=1.02, fontsize=18)
plt.tight_layout(pad=3.0)  # Increased padding
plt.show()

# Second dashboard
# Create 2x2 grid layout with adjusted size and spacing
fig, axes = plt.subplots(2, 2, figsize=(15, 11))  # Increased figure size
plt.subplots_adjust(wspace=0.8, hspace=0.8)  # Increased spacing between subplots

# --- Plot 1: Price vs Total Rooms (Boxplot) ---
sns.boxplot(data=df_final, x="Total_Rooms", y="Price", ax=axes[0,0])
axes[0,0].set_title("Price Distribution by Room Count", pad=15, fontsize=14)
axes[0,0].set_ylabel("Price ($)", fontsize=12)
axes[0,0].set_xlabel("Total Rooms", fontsize=12)
axes[0,0].tick_params(axis='both', labelsize=10)

# --- Plot 2: Price per Bed vs Lease Term (Boxplot) ---
sns.boxplot(data=df_final, x="Lease_Term_Months", 
            y=df_final["Price"]/df_final["Rooms_Available"], 
            ax=axes[0,1])
axes[0,1].set_title("Price per Bedroom by Lease Term", pad=15, fontsize=14)
axes[0,1].set_ylabel("Price per Bed ($)", fontsize=12)
axes[0,1].set_xlabel("Lease Term (Months)", fontsize=12)
axes[0,1].tick_params(axis='both', labelsize=10)

# --- Plot 3: Price vs Square Footage (Scatter) ---
scatter = sns.scatterplot(data=df_final, x="Square_Footage", y="Price", 
               hue="Furnished", alpha=0.6, ax=axes[1,0])
axes[1,0].set_title("Price vs Property Size", pad=15, fontsize=14)
axes[1,0].set_xlabel("Square Footage", fontsize=12)
axes[1,0].set_ylabel("Price ($)", fontsize=12)
axes[1,0].tick_params(axis='both', labelsize=10)
axes[1,0].legend(title="Furnished", bbox_to_anchor=(1.05, 1), 
                loc='upper left', fontsize=10)

# --- Plot 4: Feature Correlation (Heatmap) ---
corr_data = df_final[["Price", "Rooms_Available", "Baths", 
                     "Square_Footage", "Distance_to_Uni_km"]].corr()
sns.heatmap(corr_data, annot=True, cmap="coolwarm", 
           fmt=".2f", cbar=False, ax=axes[1,1], annot_kws={"size": 10})
axes[1,1].set_title("Feature Correlation Matrix", pad=15, fontsize=14)
axes[1,1].tick_params(axis='x', rotation=45, labelsize=10)
axes[1,1].tick_params(axis='y', labelsize=10)

plt.suptitle("Rental Market Analysis Dashboard", y=1.02, fontsize=18)
plt.tight_layout(pad=3.0)  # Increased padding
plt.show()

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Copy original DataFrame
df = df_final.copy()

# --- Boolean Columns to Int ---
bool_cols = ['Pet_Friendly', 'Furnished', 'Parking_Included', 'Internet_Available']
for col in bool_cols:
    df[col + '_int'] = df[col].map({
        True: 1, False: 0, 'Yes': 1, 'No': 0,
        'yes': 1, 'no': 0, 'Y': 1, 'N': 0, 'y': 1, 'n': 0
    }).fillna(0).astype(int)


df.to_csv("bamboo_data_preprocessed.csv", index=False)