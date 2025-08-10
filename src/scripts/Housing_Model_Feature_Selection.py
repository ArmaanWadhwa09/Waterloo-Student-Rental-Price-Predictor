from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib

X = pd.read_csv("Housing_Features.csv")
# Standardize all numeric features (skip binary flags if needed)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # X is your preprocessed feature matrix

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns,  # Original feature names
    index=X.index      # Match original DataFrame indices
)
corr_matrix = X_scaled.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Drop columns with correlation > 0.95
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
X_filtered = X_scaled.drop(columns=to_drop)


from sklearn.ensemble import RandomForestRegressor

# Step 1: Get anomaly scores
best_model = IsolationForest(random_state=42, n_estimators=50, contamination=0.01)
best_model.fit(X_filtered)
scores = best_model.decision_function(X_filtered)

# Step 2: Train RF Regressor to mimic IF scores
rf = RandomForestRegressor(random_state=42, n_estimators=50)
rf.fit(X_filtered, scores)

# Step 3: Feature importances
importances = pd.Series(rf.feature_importances_, index=X_filtered.columns)
top_30 = importances.sort_values(ascending=False).head(30).index
X_rf = X_filtered[top_30]
importances = pd.Series(rf.feature_importances_, index=X_filtered.columns)

# Sort descending and pick top 30
top_30_rf = importances.sort_values(ascending=False).head(30)

# Optional: Plot them
plt.figure(figsize=(10, 8))
top_30_rf.sort_values().plot(kind='barh')
plt.title("Top 30 Feature Importances from RF Regressor")
plt.xlabel("Importance")
plt.show()


best_model = IsolationForest(random_state=42, n_estimators=50, contamination = 0.01)
best_model.fit(X_filtered)

baseline_scores = best_model.decision_function(X_filtered)  # Higher = more normal
baseline_var = np.var(baseline_scores)

# Permutation importance
importances = {}

for col in X_filtered.columns: # Iterate over columns of X_filtered
    X_permuted = X_filtered.copy()
    X_permuted[col] = np.random.permutation(X_permuted[col].values)

    permuted_scores = best_model.decision_function(X_permuted)
    var_diff = abs(np.var(permuted_scores) - baseline_var)

    importances[col] = var_diff

# Select top 30 features
importances_series = pd.Series(importances).sort_values(ascending=False)
top_30_perm = importances_series.head(30).index
X_permutaion = X_filtered[top_30_perm]

# Plot top 30 permutation importances
plt.figure(figsize=(10, 8))
importances_series.head(30).sort_values().plot(kind='barh')

plt.title("Permutation Importance (Top 30 Features)")
plt.xlabel("Variance Difference")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Fit and get anomaly scores
model1 = IsolationForest(random_state=42, n_estimators=50, contamination = 0.01).fit(X_rf)
model2 = IsolationForest(random_state=42, n_estimators=50, contamination = 0.01).fit(X_permutaion)

scores1 = model1.decision_function(X_rf)
scores2 = model2.decision_function(X_permutaion)

# Compare std dev (or visual via histogram)
print("Alt1 Std:", np.std(scores1), " | Alt2 Std:", np.std(scores2))

top_features = [
    'Available_Room_Pct',
    'Furnished_X_Internet',
    'Is_Luxury_Setup',
    'Baths',
    'Sqft_per_Room',
    'Bathroom_Ratio',
    'Amenity_Score',
    'Pet_Friendly',
    'Parking_X_Pet'
]
X_combined = X_filtered[top_features]
model_final = IsolationForest(random_state=42, n_estimators=50, contamination=0.01).fit(X_combined)
scores_final = model_final.decision_function(X_combined)
print("Combined Std:", np.std(scores_final))

X_scaled["anomaly_score"] = scores_final
X_scaled["anomaly_flag"] = pd.Series(model_final.predict(X_combined), index=X.index).map({1: 0, -1: 1})

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.heatmap(
    X_scaled[X_scaled["anomaly_flag"] == 1][top_features].corr(),
    annot=True, cmap="coolwarm", center=0
)
plt.title("Correlation Heatmap of Top Features (Anomalies Only)")
plt.tight_layout()
plt.show()


from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

# Filter flagged anomalies
subset = X_scaled[X_scaled["anomaly_flag"] == 1].copy()

# Assign anomaly severity levels using quantiles of the final scores
subset["anomaly_type"] = pd.qcut(
    scores_final[X_scaled["anomaly_flag"] == 1],
    q=3,
    labels=["Mild", "Moderate", "Extreme"]
)

# Prepare data for plotting
subset_plot = subset[top_features].copy()
subset_plot["Anomaly_Type"] = subset["anomaly_type"]

# Plot
plt.figure(figsize=(14, 6))
parallel_coordinates(subset_plot, "Anomaly_Type", colormap="coolwarm", alpha=0.5)
plt.title("Parallel Coordinates Plot of Anomalous Listings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

for col in top_features:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x="anomaly_flag", y=col, data=X_scaled)
    plt.title(f"{col} Distribution by Anomaly Flag (0 = Normal, 1 = Anomaly)")
    plt.tight_layout()
    plt.show()

joblib.dump(model_final, "isolation_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Saved: isolation_forest_model.pkl + scaler.pkl")


