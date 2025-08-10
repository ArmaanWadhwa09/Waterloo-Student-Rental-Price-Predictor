import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

from feature_for_API import create_features

# -------- Load Data --------
df_price = pd.read_csv("Housing_Features_with_Price.csv")
df_price.dropna(inplace=True)

# -------- Feature Engineering --------
X = df_price.drop(columns=['Price'])
y = df_price["Price"]
y_log = np.log1p(y)
X['Shared_Quality'] = X['Is_Shared_Setup'].astype(int) * (X['Baths'] / (X['Total_Rooms'] + 0.1))
X['Bed_Bath_Ratio'] = X['Rooms_Available'] / X['Baths'].replace(0, np.nan)

# -------- Define Top Features for Price --------
top_features = [
    "Shared_Quality",
    "Baths",
    "Is_Shared_Setup",
    "Bed_Bath_Ratio",
    "Total_Rooms",
    "Bathroom_Ratio",
    "Available_Room_Pct",
    "Is_Balanced",
    "Is_Spacious",
    "Rooms_Available",
]

X = X[top_features]

# -------- Scale --------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# -------- Train/Test Split --------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

# -------- XGBoost Model --------
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42
)

model.fit(X_train, y_train)

# -------- Evaluation --------
y_pred_log = model.predict(X_test)
y_test_raw = np.expm1(y_test)
y_pred_raw = np.expm1(y_pred_log)

print(f"\nLog-space metrics:")
print(f"MAE (log): {mean_absolute_error(y_test, y_pred_log):.2f}")
print(f"RMSE (log): {mean_squared_error(y_test, y_pred_log, squared=False):.2f}")
print(f"R² (log): {r2_score(y_test, y_pred_log):.2f}")
print(f"MAPE (log): {mean_absolute_percentage_error(y_test, y_pred_log):.2f}")

print(f"\nReal-world metrics:")
print(f"MAE: ${mean_absolute_error(y_test_raw, y_pred_raw):.2f}")
print(f"RMSE: ${mean_squared_error(y_test_raw, y_pred_raw, squared=False):.2f}")

# -------- Save Model & Scaler --------
joblib.dump(model, "price_prediction_xgb.pkl")
joblib.dump(scaler, "price_scaler.pkl")
print("✅ Saved: price_prediction_xgb.pkl + price_scaler.pkl")

# -------- Feature Importance --------
importance = model.feature_importances_
feat_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop Features:\n", feat_importance_df.head(10))

plt.figure(figsize=(10, 6))
plt.barh(feat_importance_df["Feature"][:20][::-1], feat_importance_df["Importance"][:20][::-1])
plt.xlabel("Importance Score")
plt.title("Top 20 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()