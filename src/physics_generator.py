# Importing the required libraries
import numpy as np
import pandas as pd
import math
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_ROOT     = PROJECT_ROOT / "data"
OUTPUT_PATH   = DATA_ROOT / "synthetic_engagements.csv"
METADATA_PATH = DATA_ROOT / "aircraft_metadata.csv"

# ── Generation config ─────────────────────────────────────────────────────────
N_ROWS = 1_000_000           # Python ignores underscores in numbers, just readability

# ── Feature ranges ────────────────────────────────────────────────────────────
# Launch distance: how far the missile was fired from (metres)
# Floor is 500m — below this, guns are more effective than missiles and
# the missile doesn't have enough distance to arm and track properly
# Upper bound is capped to each aircraft's missile_range at row generation
LAUNCH_DISTANCE_MIN = 500

# Your aircraft speed range - 61 m/s (TB2 drone) → 983 m/s (SR-71), full metadata range
YOUR_SPEED_RANGE       = (61, 983)

# Your aircraft altitude (metres)
# 0 = ground level, 30,000m = upper combat/operational ceiling
# SR-71 operates at ~24,000m, most fighters top out around 20,000m
# 30,000m gives headroom for all 102 aircraft in metadata
YOUR_ALTITUDE_RANGE    = (0, 30_000)

# Enemy aircraft altitude (metres) — same envelope as your own aircraft
# Determines vertical geometry of the engagement alongside elevation angle
ENEMY_ALTITUDE_RANGE   = (0, 30_000)

# Azimuth: horizontal angle of incoming threat, clockwise from North
# 0° = head-on, 90° = right side, 180° = tail-chase, 270° = left side
AZIMUTH_RANGE          = (0, 360)

# Elevation: vertical angle of incoming threat
# 0° = same altitude, +90° = directly above, -90° = directly below
ELEVATION_RANGE        = (-90, 90)

# Maneuverability: 0 = low (bomber/transport), 1 = medium (older jets),
#                  2 = high (modern fighters)
MANEUVERABILITY_VALUES = [0, 1, 2]

# Countermeasure: 0 = not deployed, 1 = deployed (flares/chaff)
COUNTERMEASURE_VALUES  = [0, 1]


# ── Runs once — loads combat-capable aircraft from metadata CSV ───────────────
def _load_metadata():
    """
    Loads combat-capable aircraft from the metadata CSV (no_aa_capability == 0).

    Returns:
        pd.DataFrame: Filtered metadata containing only aircraft with air-to-air missile capability.
    """
    
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_PATH}")
    metadata_df = pd.read_csv(METADATA_PATH)
    metadata_df = metadata_df[metadata_df["no_aa_capability"]==0]
    return metadata_df


# ── Derives missile phase from how far it has already traveled ────────────────
# phase 0 = boost (just launched), 1 = mid-course, 2 = terminal (final approach)
def _derive_missile_phase(remaining_distance, launch_distance):
    """
    Derives missile flight phase from how much of its journey it has completed.

    Args:
        remaining_distance (float): Distance remaining between missile and target (metres).
        launch_distance (float): Total distance at the moment of launch (metres).

    Returns:
        int: 0 = boost, 1 = mid-course, 2 = terminal.
        
        Phase 0 — engine burning, accelerating, just launched
        Phase 1 — flying toward your predicted position, guided but not actively tracking you yet
        Phase 2 — active seeker on, tracking you specifically, hardest to fool
    """
    
    traveled_distance = launch_distance - remaining_distance
    ratio = traveled_distance / launch_distance
    
    if ratio < 0.33:
        return 0
    elif 0.33 <= ratio < 0.66:
        return 1
    else:
        return 2


# ── Derives closure rate using 3D geometry (azimuth + elevation + speeds) ─────
# How fast the gap between you and the missile is closing, in m/s
# Enemy altitude adds the vertical dimension to the engagement geometry
def _derive_closure_rate(missile_speed, your_speed,
                         azimuth, elevation):
    """
    Derives the closure rate - how fast the gap between the missile and the target is closing (m/s).

    Uses 3D geometry: azimuth accounts for horizontal approach angle,
    elevation accounts for vertical approach angle.

    Args:
        missile_speed (float): Speed of the incoming missile (m/s).
        your_speed (float): Speed of the friendly aircraft (m/s).
        azimuth (float): Horizontal angle of incoming threat in degrees (0° = head-on).
        elevation (float): Vertical angle of incoming threat in degrees (0° = same altitude).

    Returns:
        float: Closure rate in m/s.
    """
    
    # Convert the angles into radians and prepare for cosine
    azimuth = math.radians(azimuth)
    elevation = math.radians(elevation)
    
    # Extract the closure rate
    closure_rate = missile_speed + (your_speed * (math.cos(azimuth) * math.cos(elevation)))
    
    return closure_rate


def _derive_evasion_time(remaining_distance, closure_rate,
                         missile_phase, enemy_generation,
                         your_speed, your_altitude, enemy_altitude):
    """
    Derives the minimum evasion time - seconds before the missile reaches you.

    Base calculation is pure kinematics: remaining_distance / closure_rate.
    Four modifiers are applied to account for factors the base formula cannot capture.

    Modifiers:
        - missile_phase == 2 (terminal): seeker has locked on, countermeasures
          need 2-3s overhead to be effective. Shrinks window by 15%. (x 0.85)
        - enemy_generation == 5: HOBS (High Off-Boresight)
          seeker + ECCM make the missile harder to
          defeat, compressing effective reaction time by ~10%. (x 0.90)
        - your_speed > 522 m/s (above median): high energy state gives more
          lateral geometry per second during evasion. Slight expansion. (x 1.05)
        - abs(your_altitude - enemy_altitude) > 5000m: large altitude gap pushes
          engagement toward edge of missile performance envelope, degrading
          terminal accuracy. Slight expansion. (x 1.10)

    Args:
        remaining_distance (float): Distance between missile and you right now (metres).
        closure_rate (float): Combined closing speed from _derive_closure_rate() (m/s).
        missile_phase (int): 0 = boost, 1 = mid-course, 2 = terminal.
        enemy_generation (float): Enemy aircraft generation (3.5, 4, 4.5, or 5).
        your_speed (float): Your current airspeed (m/s).
        your_altitude (float): Your current altitude (metres).
        enemy_altitude (float): Enemy aircraft altitude (metres).

    Returns:
        float: Evasion time in seconds.
    """
    
    # Calculate the evasion time
    evasion_time = remaining_distance / closure_rate
    
    # Terminal phase - seeker locked on, countermeasures need 2–3s overhead
    if missile_phase == 2:
        evasion_time *= 0.85
        
    # Gen 5 aircraft missile (HOBS + ECCM) compresses effective reaction time
    if enemy_generation == 5:
        evasion_time *= 0.90
        
    # High speed — more room to maneuver
    if your_speed > 522:
        evasion_time *= 1.05
        
    # Large altitude gap - missile at edge of performance envelope
    if abs(your_altitude - enemy_altitude) > 5000:
        evasion_time *= 1.10
        
    return evasion_time
    
    

def _derive_hit_label(countermeasure_deployed, your_maneuverability,
                      azimuth, elevation, missile_phase, enemy_generation,
                      your_altitude, enemy_altitude):
    """
    Derives whether the missile hits after an evasion attempt.

    Starts from a neutral survival score of 0.5. Modifiers push it up (more
    likely to hit) or down (more likely to miss). Final label is 1 if the
    missile hits, 0 if it misses.

    This is where azimuth and elevation belong — they affect survival after
    evasion, not when the missile arrives (that is handled by closure_rate
    in _derive_evasion_time).

    Modifiers:
        - countermeasure_deployed == 1: active countermeasures significantly
          reduce hit probability. Score drops.
        - your_maneuverability == 2: high maneuverability makes evasion more
          effective. Score drops.
        - azimuth near 0° (head-on): least time and geometry to evade.
          Score rises.
        - missile_phase == 2 (terminal): seeker locked on, hardest to defeat.
          Score rises.
        - enemy_generation == 5: HOBS + ECCM make the missile more lethal.
          Score rises.
        - elevation far from 0°: steep approach angle reduces evasion options.
          Score rises.
        - large altitude differential > 5000m: pushes missile toward edge of
          performance envelope. Score drops.

    Args:
        countermeasure_deployed (int): 0 = not deployed, 1 = deployed.
        your_maneuverability (int): 0 = low, 1 = medium, 2 = high.
        azimuth (float): Horizontal angle of incoming threat in degrees (0° = head-on).
        elevation (float): Vertical angle of incoming threat in degrees (0° = level).
        missile_phase (int): 0 = boost, 1 = mid-course, 2 = terminal.
        enemy_generation (float): Enemy aircraft generation (3.5, 4, 4.5, or 5).
        your_altitude (float): Your current altitude (metres).
        enemy_altitude (float): Enemy aircraft altitude (metres).

    Returns:
        int: 1 if missile hits, 0 if missile misses.
    """
    
    # Neutral starting score - modifiers push it toward hit or miss
    score = 0.5
    
    # Countermeasures active - significantly reduces hit chance
    if countermeasure_deployed == 1:
        score -= 0.20
        
    # High maneuverability - evasion more effective
    if your_maneuverability == 2:
        score -= 0.10
        
    # Head-on approach - least time and geometry to evade
    if azimuth < 30:
        score += 0.15
        
    # Terminal phase - seeker locked on, hardest to defeat
    if missile_phase == 2:
        score += 0.15
        
    # Gen 5 missile - HOBS + ECCM make it more lethal
    if enemy_generation == 5:
        score += 0.10
        
    # Steep approach angle - reduces evasion options
    if abs(elevation) > 45:
        score += 0.10
        
    # Large altitude gap - missile at edge of performance envelope
    if abs(your_altitude - enemy_altitude) > 5000:
        score -= 0.10

    # Hit if score crosses 0.5
    return 1 if score >= 0.5 else 0
    

# ── Generates one complete engagement scenario as a dict ─────────────────────
def _generate_row(aircraft_row):
    """
    Generates one synthetic engagement scenario for a given aircraft.

    Samples random values for all situational features, derives computed
    features using physics, and returns a complete row with labels.

    Args:
        aircraft_row (pd.Series): One row from the metadata CSV.

    Returns:
        dict: All 14 features plus evasion_time and hit labels.
    """
    
    # Sample situational features
    your_speed    = np.random.uniform(*YOUR_SPEED_RANGE)
    your_altitude = np.random.uniform(*YOUR_ALTITUDE_RANGE)
    enemy_altitude = np.random.uniform(*ENEMY_ALTITUDE_RANGE)
    azimuth       = np.random.uniform(*AZIMUTH_RANGE)
    elevation     = np.random.uniform(*ELEVATION_RANGE)
    
    # Sample categorical features
    your_maneuverability    = np.random.choice(MANEUVERABILITY_VALUES)
    countermeasure_deployed = np.random.choice(COUNTERMEASURE_VALUES)
    
    # Pull threat specs from metadata
    missile_speed    = aircraft_row["missile_speed"]
    missile_range    = aircraft_row["missile_range"]
    enemy_generation = aircraft_row["enemy_generation"]
    
    # Sample engagement distances
    launch_distance    = np.random.uniform(LAUNCH_DISTANCE_MIN, missile_range)
    remaining_distance = np.random.uniform(0, launch_distance)    # remaining_distance <= launch_distance always
    
    # Derive computed features
    missile_phase = _derive_missile_phase(remaining_distance, launch_distance)
    closure_rate  = _derive_closure_rate(missile_speed, your_speed,
                                         azimuth, elevation)

    # Derive labels
    evasion_time = _derive_evasion_time(remaining_distance, closure_rate,
                                        missile_phase, enemy_generation,
                                        your_speed, your_altitude, enemy_altitude)
    hit          = _derive_hit_label(countermeasure_deployed, your_maneuverability,
                                     azimuth, elevation, missile_phase, enemy_generation,
                                     your_altitude, enemy_altitude)
    
    # Return complete row
    return {
        "launch_distance":        launch_distance,
        "remaining_distance":     remaining_distance,
        "closure_rate":           closure_rate,
        "azimuth":                azimuth,
        "elevation":              elevation,
        "missile_phase":          missile_phase,
        "your_speed":             your_speed,
        "your_altitude":          your_altitude,
        "your_maneuverability":   your_maneuverability,
        "enemy_altitude":         enemy_altitude,
        "missile_speed":          missile_speed,
        "missile_range":          missile_range,
        "enemy_generation":       enemy_generation,
        "countermeasure_deployed": countermeasure_deployed,
        "evasion_time":           evasion_time,
        "hit":                    hit
    }
    


# ── Saves completed DataFrame to CSV ─────────────────────────────────────────
def _save_dataset(df):
    """
    Saves the generated dataset to a CSV file.

    Args:
        df (pd.DataFrame): The complete synthetic engagement dataset.
    """
    
    # Create data directory if it doesn't exist
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(OUTPUT_PATH, index=False)


# ── Public entry point — the only function the notebook calls ─────────────────
def generate_dataset():
    """
    Generates 1,000,000 synthetic engagement scenarios and saves to CSV.

    Loads combat-capable aircraft from metadata, samples each aircraft
    equally across all rows, derives all features and labels using physics,
    and writes the final dataset to disk.

    Only valid threat scenarios are included (closure_rate > 0).

    Returns:
        pd.DataFrame: The complete synthetic engagement dataset.
    """
    metadata = _load_metadata()
    
    rows_per_aircraft = N_ROWS // len(metadata)  # 1_000_000 // 56 → 17,857 rows per aircraft
    rows = []
    
    for _, aircraft_row in metadata.iterrows():
        count = 0
        while count < rows_per_aircraft:
            row = _generate_row(aircraft_row)
            if row["closure_rate"] > 0:
                rows.append(row)
                count += 1
    
    # Fill remaining rows to hit exactly N_ROWS
    remaining_rows = N_ROWS - len(rows)
    if remaining_rows:
        count = 0
        while count < remaining_rows:
            row = _generate_row(metadata.sample(1).iloc[0])
            if row["closure_rate"] > 0:
                rows.append(row)
                count += 1
    
    df = pd.DataFrame(rows)
    _save_dataset(df)
    print(f"Done. {len(rows):,} rows saved to {OUTPUT_PATH}")