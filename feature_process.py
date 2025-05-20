import pandas as pd
import numpy as np
import math
import utils
from utils import list_of_files
import os

# Auto labeling loitering data

# --- CONFIG ---
ROLL_THRESHOLD_DEG = 10  # deg
RADIUS_THRESHOLD = 150  # meters
WINDOW_SIZE = 100  # samples (10 sec)
SPEED_STABILITY_THRESHOLD = 2  # m/s standard deviation


output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "structured_datas"
)

for item in list_of_files:
    # --- Load data ---
    file_name = utils.find_file_path("structured_datas", f"processed_{item}.xlsx")
    df = pd.read_excel(file_name)  # or use the uploaded file

    df["loiter_radius"] = df.apply(
        lambda row: utils.loiter_radius(row["speed"], row["roll"]), axis=1
    )

    # --- Add rolling labels ---
    labels = []
    # New DataFrame to hold summary rows
    summary_rows = []

    for i in range(0, len(df), WINDOW_SIZE):
        window = df.iloc[i : i + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            break  # Skip incomplete window

        speed_std = window["speed"].std()
        avg_roll = window["roll"].mean()
        avg_radius = (
            window["loiter_radius"].replace([np.inf, -np.inf], np.nan).dropna().mean()
        )

        if (
            (avg_radius < RADIUS_THRESHOLD)
            and (abs(avg_roll) > ROLL_THRESHOLD_DEG) # use abs() because roll can be negative for left turn
            and (speed_std < SPEED_STABILITY_THRESHOLD)
        ):

            label = "loiter"
        else:
            label = "non-loiter"

        labels.extend([label] * WINDOW_SIZE)

        summary = {
            "average_roll": avg_roll,
            "speed_std": speed_std,
            "loiter_radius": avg_radius,
            "label": label,
        }
        summary_rows.append(summary)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # --- Save the processed data ---
    output_dir = utils.find_file_path(
        "structured_datas", f"processed_{item}_with_labels.xlsx"
    )
    summary_df.to_excel(output_dir, index=False)
    print(f"Processed data saved to {output_dir}")
