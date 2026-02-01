import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("data/raw/queue_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Feature engineering
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["is_peak_hour"] = df["hour"].between(11, 14).astype(int)

df["queue_lag_1"] = df["customers_in_queue"].shift(1)
df["queue_lag_5"] = df["customers_in_queue"].shift(5)
df["wait_lag_1"] = df["wait_time"].shift(1)

df = df.dropna().reset_index(drop=True)

features = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
    "active_counters",
    "customers_in_queue",
    "queue_lag_1",
    "queue_lag_5",
    "service_time"
]

X = df[features]
y = df["wait_time"]

split_idx = int(len(df) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)

print("Validation MAE:", round(mae, 2))

# Save model
joblib.dump(model, "models/wait_time_xgb.pkl")
print("Model saved to models/wait_time_xgb.pkl")
