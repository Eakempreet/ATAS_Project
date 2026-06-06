"""
What is schemas.py
A single Python file that holds shared constants — column names, their order, and any fixed values the pipeline depends on.

Why it exists
Three files will each build or consume the 14-feature array: physics_generator.py, models.py, and main.py. If each one hardcodes its own column list, one typo in one file breaks the whole pipeline silently.
schemas.py means they all import from the same list. One source of truth.

What goes inside it
Three things:

FEATURE_COLUMNS - the 14 feature names in exact training order
MODEL_PATHS - file paths to your two saved .joblib models
DECISION_THRESHOLDS - the ETA and hit probability cutoffs your decision layer will use
"""


from pathlib import Path

# Path pointing towards project ATAS main folder
BASE_DIR = Path(__file__).parent.parent
# Path pointing towards saved models
MODEL_DIR = BASE_DIR / "release_models"

# Feature columns to be feed to models
FEATURE_COLUMNS = [
    'launch_distance', 'remaining_distance', 'closure_rate', 'azimuth', 'elevation',
    'missile_phase', 'your_speed', 'your_altitude', 'your_maneuverability',
    'enemy_altitude', 'missile_speed', 'missile_range', 'enemy_generation',
    'countermeasure_deployed'
]

# Model paths as dict to use them to make predictions
MODEL_PATHS = {
    'classifier' : MODEL_DIR / "aircraft_classifier" / "atas_final_fine_tuned_aircraft_classifier_model.keras",
    'eta'        : MODEL_DIR / "eta" / "atas_final_eta_regressor_model.joblib",
    'hit'        : MODEL_DIR / "hit" / "atas_final_hit_classifier_model.joblib"
}

# Cutoffs used by decision.py to convert model outputs into pilot recommendations
DECISION_THRESHOLD = {
    'hit_prob_high'   : 0.75,   # BREAK HARD + DEPLOY COUNTERMEASURES
    'hit_prob_medium' : 0.60,   # DEPLOY COUNTERMEASURES
    'eta_critical'    : 10      # seconds - urgency threshold
}