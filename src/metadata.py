"""
What is metadata.py

When the classifier identifies an aircraft say "F22", the pipeline needs its specs: missile speed, missile range, enemy generation, maneuverability. Those live in your aircraft_metadata.csv.
metadata.py has one job: take an aircraft name, look it up in the CSV, return its specs as a dictionary.
"""

import pandas as pd
from schemas import BASE_DIR

CSV_PATH = BASE_DIR / "data" / "aircraft_metadata.csv"

# Load the aircraft metadata
aircraft_df = pd.read_csv(CSV_PATH)

# Function to get the Aircraft specs
def get_aircraft_metadata(aircraft_name:str) -> dict:
    """
    Look up aircraft specifications from the metadata CSV by aircraft name.

    Args:
        aircraft_name (str): Aircraft name (case-insensitive). e.g. "F22", "rafale"

    Returns:
        dict: All CSV columns for that aircraft. Empty dict if not found.
    """
    
    # Upper case everything to search
    aircraft_name = aircraft_name.upper()
    
    # Find the row
    aircraft_row = aircraft_df[aircraft_df['aircraft'].str.upper() == aircraft_name]
    
    # Handling egde cases
    if not aircraft_row.empty:
        return aircraft_row.iloc[0].to_dict()
    else:
        return {}

    