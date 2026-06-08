"""
models.py — Inference Layer
----------------------------
Loads the trained ETA regressor and hit classifier from disk once at module
import time. Exposes a single function that accepts a feature dict, converts
it to an ordered numpy array using FEATURE_COLUMNS, runs both models, and
returns ETA in seconds and hit probability as a float.

This module knows nothing about physics, metadata, or decisions.
Its only job is: feature array in, predictions out.
"""


from src.schemas import FEATURE_COLUMNS, MODEL_PATHS
from joblib import load
import numpy as np

# Empty dictionary to hold the active loaded model
loaded_model = {}

# Load the model once in the memory
for model_name, path in MODEL_PATHS.items():
    if model_name in ('eta', 'hit'):
        try:
            loaded_model[model_name] = load(path)
            print(f"Successfully loaded: {model_name} model")
        except FileNotFoundError:
            print(f"Error: The file at {path} could not be found.")
        except Exception as e:
            print(f"An error occurred while loading model {model_name}: {e}")
            
            
            
# Function to make predictions using loaded models
def make_predictions(feature_dict):
    """
    Converts a feature dict to a numpy array and runs both models.

    Args:
        feature_dict (dict): 14-feature dict in FEATURE_COLUMNS order,
                             produced by build_feature_array().

    Returns:
        dict: {
            "eta_seconds": float,       # predicted evasion time, clipped at 0
            "hit_probability": float    # probability of hit after evasion (0.0 - 1.0)
        }
    """
    
    # Convert the dictionary into ndarray using the order of FEATURE_COLUMNS
    feature_values = np.array([feature_dict[col] for col in FEATURE_COLUMNS])
    
    # Reshape the the feature_values into (1, 14) as a line of table
    feature_values = feature_values.reshape(1, -1)
    
    # Performing predictions
    for model_name, model in loaded_model.items():
        if model_name == 'eta':
            eta_prediction = model.predict(feature_values)
            # Clip the negative time value to 0
            eta_prediction = np.maximum(eta_prediction, 0)[0]
            
        else:
            hit_prediction = model.predict_proba(feature_values)[0][1]   # [probability of miss, probability of hit]
            
            
    return {
        "eta_seconds": float(eta_prediction),
        "hit_probability": float(hit_prediction)
    }
    