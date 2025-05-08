import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from haversine import haversine, Unit

# Load your data
df = pd.read_csv("log88.csv")

# --- Step 1: Clean the data ---
df = df.dropna(subset=["GPS[0].Lat", "GPS[0].Lng"])  # Drop rows with missing GPS

# Convert columns
df["timestamp"] = df["timestamp(ms)"] / 1000.0  # to seconds
df["lat"] = df["GPS[0].Lat"] / 1e7
df["lon"] = df["GPS[0].Lng"] / 1e7

# --- Step 2: Calculate movement features ---

# Reorder columns to switch places of 'ATT.Roll' with 'timestamp(ms)'
columns = list(df.columns)
att_roll_index = columns.index("ATT.Roll")
timestamp_index = columns.index("timestamp")

# Swap the positions
columns[att_roll_index], columns[timestamp_index] = columns[timestamp_index], columns[att_roll_index]
df = df[columns]

# Rolling window metrics (optional)
df["roll_std_5"] = df["ATT.Roll"].rolling(window=5).std()
df["roll_mean_5"] = df["ATT.Roll"].rolling(window=5).mean()

# Drop NaNs introduced by rolling
df.dropna(inplace=True)

# Drop original columns you said are unnecessary
df_ml = df.drop(columns=["timestamp(ms)", "GPS[0].Lat", "GPS[0].Lng"])

# Save the final feature set to a CSV file
df_ml.to_csv("df_ml_features_log88.csv", index=False)

print("Final feature set saved to df_ml_features.csv")
