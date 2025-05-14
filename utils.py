import os
from geopy.distance import geodesic
import numpy as np
import math

list_of_files = [
    67,
    68,
    69,
    79,
    81,
    86,
    88,
    89,
    90,
    102,
    106,
    109,
    111
]

def find_file_path(target_folder, file_name):
    # Get the directory of the current Python file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, target_folder)
    # Get the full path to the file
    file_path = os.path.join(folder_path, file_name)
    return file_path


def calc_distance(row1, row2):
    return geodesic(
        (row1["lat"], row1["lon"]),
        (row2["lat"], row2["lon"]),
    ).meters


def haversine(row1, row2):
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = np.radians([row1["lat"], row1["lon"]])
    lat2, lon2 = np.radians([row2["lat"], row2["lon"]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def loiter_radius(speed_mps, roll_deg):
    g = 9.81  # m/s^2
    roll_rad = math.radians(roll_deg)
    if abs(math.tan(roll_rad)) < 1e-6:
        return float('inf')  # straight flight
    return (speed_mps ** 2) / (g * math.tan(roll_rad))
