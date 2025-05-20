import utils
import pandas as pd
import os

output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "structured_datas"
)
file_path = utils.find_file_path("raw_datas", f"90.csv")
# Load your data
df = pd.read_csv(file_path)

# Convert the 'timestamp(ms)' column to seconds
df["time_s"] = (df["timestamp(ms)"].iloc[0]) / 1000.0

print(df.info())
print(df[df == 0].any(axis=0))