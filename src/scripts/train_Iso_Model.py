from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import json

from feature_for_API import create_features


with open("top_features.json") as f:
    top_features = json.load(f)


df = pd.read_csv('bamboo_data_preprocessed.csv')
df.dropna(inplace=True)
scaler = StandardScaler()
X = create_features(df)

X = X[top_features]  # Only keep top features
X_scaled = scaler.fit_transform(X)
model = IsolationForest(random_state=42, n_estimators=50, contamination=0.01).fit(X_scaled)
score = model.decision_function(X_scaled)[0]
joblib.dump(model, "isolation_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")