import os
import pandas as pd

# Directory containing the processed files
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "structured_datas")

# List all files that match the pattern 'processed_*_with_labels.xlsx'
files = [f for f in os.listdir(data_dir) if f.startswith("processed_") and f.endswith("_with_labels.xlsx")]

# Read and combine all DataFrames
df_list = [pd.read_excel(os.path.join(data_dir, f)) for f in files]
combined_df = pd.concat(df_list, ignore_index=True)

# Save the combined DataFrame to a new Excel file
output_path = os.path.join(data_dir, "all_processed_with_labels_combined.xlsx")
combined_df.to_excel(output_path, index=False)

print(f"Combined data saved to {output_path}")