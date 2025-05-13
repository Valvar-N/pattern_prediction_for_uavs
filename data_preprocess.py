import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine, Unit
import os
from geopy.distance import geodesic

# Get the directory of the current Python file
current_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_dir, 'raw_datas')

# Specify the filename
filename = '67.csv'

# Get the full path to the file
file_path = os.path.join(folder_path, filename)

# Load your data
df = pd.read_csv(file_path)

# Convert the 'timestamp(ms)' column to seconds
df['time_s'] = (df['timestamp(ms)'].iloc[0]) / 1000.0
df['time_diff'] = (df['timestamp(ms)'] - df['timestamp(ms)'].iloc[0]) / 1000.0

# Clear noise in the data
df['speed'] = df['GPS[0].Spd'].rolling(window=5).mean()
df['roll'] = df['ATT.Roll'].rolling(window=5).mean()
df['yaw'] = df['AHR2.Yaw'].rolling(window=5).mean()

# After clearing noise, drop the rows with NaN values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df['yaw_rad'] = np.deg2rad(df['yaw'])
df['yaw_unwrapped'] = np.unwrap(df['yaw_rad'])  # fix 360 → 0 wrap

df['yaw_rate'] = df['yaw_unwrapped'].diff() / df['time_diff'].diff()
df['yaw_rate_deg'] = np.rad2deg(df['yaw_rate'])

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

def calc_distance(row1, row2):
    return geodesic((row1['GPS[0].Lat'], row1['GPS[0].Lng']), (row2['GPS[0].Lat'], row2['GPS[0].Lng'])).meters

distances = [0]
for i in range(1, len(df)):
    dist = calc_distance(df.iloc[i], df.iloc[i - 1])
    distances.append(dist)

df['distance_m'] = distances
df['total_distance'] = df['distance_m'].cumsum()

df.to_csv('processed_data.csv', index=False)

# TODO setup module for more readbility and maintainability
# TODO continue preprocessing for distance 