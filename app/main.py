"""
main.py
-----------------------------------------------------
# WHAT: FastAPI application entry point for the ATAS backend.
#       Exposes a single endpoint - POST /analyze - that accepts an aircraft
#       image and engagement parameters, runs the full pipeline, and returns
#       a JSON response.
#
# WHY:  All five src/ modules are isolated units. main.py is the glue layer
#       that wires them together in the correct order and exposes the result
#       to the outside world via HTTP.
#
# USE:  Called by the HUD frontend via fetch(). Also used directly during
#       testing via curl or a Python test script to verify end-to-end pipeline
#       behavior before any UI work begins.

# User controls / HUD inputs: + 1 Image
# your_speed, your_altitude, enemy_altitude, countermeasure_deployed,
# launch_distance, remaining_distance, azimuth, elevation

# Comes from aircraft_metadata.csv:
# missile_speed, missile_range, enemy_generation, maneuverability

# Derived by build_feature_array() internally:
# closure_rate, missile_phase
"""



# Importing the necessary libraries
from fastapi import FastAPI
from fastapi import UploadFile, File, Form
import tempfile
import os
from src.classifier import predict_aircraft
from src.metadata import get_aircraft_metadata
from src.physics_generator import build_feature_array
from src.models import make_predictions
from src.decision import get_recommendation


# Creating a server
app = FastAPI()


# Function to take the inputs
# image -> as temp file
# 8 features -> form object
@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),             # The ... just means required. No default value.
    your_speed: float = Form(...),             # name : type = where_it_comes_from(required_or_default) 
    your_altitude: float = Form(...),
    enemy_altitude: float = Form(...),
    countermeasure_deployed: int = Form(...),
    launch_distance: float = Form(...),
    remaining_distance: float = Form(...),
    azimuth: float = Form(...),
    elevation: float = Form(...),
    friendly_aircraft: str = Form(...)
):
    
    # Browser sends image as raw bytes, not a file path
    # predict_aircraft() needs a real file path on disk
    # So we write the bytes to a temp file and grab its path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())  # dump bytes from memory to disk
        tmp_path = tmp.name            # grab the path e.g. /tmp/abc123.jpg
        
        
    # 1. Calling the function to predict the aircraft
    aircraft_name = predict_aircraft(image_path=tmp_path)
    
    # 2. Get the metadata of predicted aircraft
    metadata_dict = get_aircraft_metadata(aircraft_name=aircraft_name)
    friendly_metadata = get_aircraft_metadata(aircraft_name=friendly_aircraft)
    
    # 3. Build the features to feed the models
    feature_dict = build_feature_array(
        # From HUD sliders
        launch_distance=launch_distance,
        remaining_distance=remaining_distance,
        azimuth=azimuth,
        elevation=elevation,
        your_altitude=your_altitude,
        your_speed=your_speed,
        enemy_altitude=enemy_altitude,
        countermeasure_deployed=countermeasure_deployed,
        # From metadata lookup (enemy aircraft)
        missile_speed=metadata_dict['missile_speed'],
        missile_range=metadata_dict['missile_range'],
        enemy_generation=metadata_dict['enemy_generation'],
        # From metadata lookup (friendly aircraft)
        your_maneuverability=friendly_metadata['maneuverability']
    )
    
    # 4. Make predictions using models
    predictions_dict = make_predictions(feature_dict=feature_dict)
    
    # 5. Providing recommendations to the pilot
    suggestion = get_recommendation(
        eta_seconds=predictions_dict['eta_seconds'],
        hit_probability=predictions_dict['hit_probability'],
        no_aa_capability=metadata_dict['no_aa_capability'],
        enemy_generation=metadata_dict['enemy_generation'],
        friendly_generation=friendly_metadata['enemy_generation']
    )
    
    # 6. Delete the temparary file since work is done
    os.remove(tmp_path)
    
    # 7. Return the value predicted and derived
    return {
        'aircraft_name' : aircraft_name,
        'missile_speed' : metadata_dict['missile_speed'],
        'missile_range' : metadata_dict['missile_range'],
        'enemy_generation' : metadata_dict['enemy_generation'],
        'maneuverability'  : metadata_dict['maneuverability'],
        'no_aa_capability' : metadata_dict['no_aa_capability'],
        'eta_seconds'     : predictions_dict['eta_seconds'],
        'hit_probability' : predictions_dict['hit_probability'],
        'recommendation'  : suggestion
    }
    
    
    
"""
Request comes in
      ↓
Image saved to temp file
      ↓
predict_aircraft()     → aircraft_name
      ↓
get_aircraft_metadata() x 2  → enemy + friendly metadata
      ↓
build_feature_array()  → 14 feature dict
      ↓
make_predictions()     → eta + hit probability
      ↓
get_recommendation()   → tactical suggestion
      ↓
Temp file deleted
      ↓
JSON returned to HUD
"""