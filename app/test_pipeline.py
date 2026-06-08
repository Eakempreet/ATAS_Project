"""
test_pipeline.py -- End-to-End Pipeline Verification Script

WHAT:
    Sends a real POST request to the running FastAPI server at /analyze.
    Uses a real aircraft image and hardcoded engagement parameters.

WHY:
    Confirms the full pipeline runs without errors before any UI work begins.
    If this script returns clean JSON with all expected keys, the backend is ready.
    If it fails, we fix the backend here, not inside the HUD.

HOW TO USE:
    1. Start the FastAPI server in one terminal:
       uvicorn app.main:app --reload

    2. Run this script in a second terminal:
       python app/test_pipeline.py

EXPECTED OUTPUT:
    A JSON response containing:
    aircraft_name, missile_speed, missile_range, enemy_generation,
    maneuverability, no_aa_capability, eta_seconds, hit_probability, recommendation
"""


# Import the necessary libraries
import requests
from pathlib import Path

# Define image test path
IMAGE_PATH = str(Path(__file__).parent.parent / 'assets'/ 'Su57.jpg')
print(IMAGE_PATH)

# Define Testing parameters
params = {
    "your_speed": 500.0,
    "your_altitude": 10000.0,
    "enemy_altitude": 8000.0,
    "countermeasure_deployed": 0,
    "launch_distance": 15000.0,
    "remaining_distance": 10000.0,
    "azimuth": 45.0,
    "elevation": 5.0,
    "friendly_aircraft": "Tejas"
}


# Send the data to the server
# and 
# store what the server sends back
response = requests.post(
    url='http://localhost:8000/analyze',   # Address of the running FastAPI server
    data=params,                           # Form fields - speed, altitude, etc.
    files={
        "image": (                         # "image" matches the field name in main.py
            "Su57.jpg",                    # Filename the server will see
            open(IMAGE_PATH, "rb"),        # Open the image in read-binary mode
            "image/jpeg"                   # Tell the server what type of file this is
        )
    }
)

print(response.status_code)   # should be 200 if everything worked
print(response.json())        # the actual pipeline output




"""
VERIFIED OUTPUT - June 8, 2026
End-to-end pipeline confirmed working. Status 200.

Test image : assets/Su57.jpg
Server     : uvicorn app.main:app (no --reload)
Run with   : python -m app.test_pipeline

Response:
{
    "aircraft_name"   : "Su57",
    "missile_speed"   : 2058,
    "missile_range"   : 400000,
    "enemy_generation": 5.0,
    "maneuverability" : 2,
    "no_aa_capability": 0,
    "eta_seconds"     : 3.77,
    "hit_probability" : 0.969,
    "recommendation"  : "BREAK HARD + DEPLOY CM + DISENGAGE IMMEDIATELY"
}

All 6 src/ modules executed without error.
Aircraft correctly identified from image.
Recommendation matches expected threat profile for Gen-5 enemy at 10km.
"""