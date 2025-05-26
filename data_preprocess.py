import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine, Unit
import os
import utils


output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "structured_datas"
)

for item in utils.list_of_files:
    file_path = utils.find_file_path("raw_datas", f"{item}.csv")
    # Load data
    df = pd.read_csv(file_path)
    # Normalize gps data
    df["lat"] = df["GPS[0].Lat"] / 1e7
    df["lon"] = df["GPS[0].Lng"] / 1e7
    df = df[df["GPS[0].Lat"] != 0]
    df = df[df["GPS[0].Lng"] != 0] # Delete rows with no GPS data

    # Convert the 'timestamp(ms)' column to seconds
    df["time_s"] = (df["timestamp(ms)"]) / 1000.0
    df["time_diff"] = (df["timestamp(ms)"] - df["timestamp(ms)"].iloc[0]) / 1000.0

    # Clear noise in the data
    df["speed"] = df["GPS[0].Spd"].rolling(window=5).mean()
    df["roll"] = df["ATT.Roll"].rolling(window=5).mean()
    df["yaw"] = df["AHR2.Yaw"].rolling(window=5).mean()

    # After clearing noise, drop the rows with NaN values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["yaw_rad"] = np.deg2rad(df["yaw"])

    df["yaw_rate"] = df["yaw_rad"].diff() / df["time_diff"].diff()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["roll_std"] = df["roll"].rolling(30).std()
    df["yaw_rate_mean"] = df["yaw_rate"].rolling(30).mean()
    df["speed_var"] = df["speed"].rolling(30).var()

    df.drop(
        columns=[
            "timestamp(ms)",
            "GPS[0].Lat",
            "GPS[0].Lng",
            "GPS[0].Spd",
            "ATT.Roll",
            "AHR2.Yaw",
        ],
        inplace=True,
    )
    df.dropna(inplace=True)

    # Save to structured_datas folder
    output_path = os.path.join(output_dir, f"processed_{item}.xlsx")
    df.to_excel(output_path, index=False)
