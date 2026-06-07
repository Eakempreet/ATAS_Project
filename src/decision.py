"""
decision.py — Tactical Decision Layer
---------------------------------------
Receives ETA and hit probability from models.py and converts them into
a human-readable threat level and tactical recommendation.

This module knows nothing about models, physics, or metadata.
Its only job is: predictions in, decision out.
"""

from schemas import DECISION_THRESHOLDS

def get_recommendation(eta_seconds, hit_probability, no_aa_capability, enemy_generation, friendly_generation):
    """
    Converts model predictions and platform data into a tactical recommendation.

    Args:
        eta_seconds (float): Predicted evasion time in seconds, clipped at 0.
        hit_probability (float): Probability of hit after evasion maneuvers (0.0 - 1.0).
        no_aa_capability (int): 1 if enemy has no air-to-air capability, 0 otherwise.
        enemy_generation (float): Enemy aircraft generation (e.g. 4.0, 4.5, 5.0).
        friendly_generation (float): Friendly aircraft generation (e.g. 4.0, 4.5, 5.0).

    Returns:
        str: Tactical recommendation string for the pilot.
    """
    
    # Check first air-to-air capability
    if no_aa_capability == 1:
        return "NON-THREAT PLATFORM - MONITOR"
    
    # Calculate the generational gap
    gen_gap = friendly_generation - enemy_generation
    
    # Define combat posture
    if gen_gap >= DECISION_THRESHOLDS['gen_advantage_threshold']:
        combat_posture = "Friendly advantage"
    elif gen_gap <= -DECISION_THRESHOLDS['gen_disadvantage_threshold']:
        combat_posture = "Enemy advantage"
    else:
        combat_posture = "Parity"
        
    # Get the ETA status
    if eta_seconds <= DECISION_THRESHOLDS['eta_critical']:
        eta_status = "Low ETA"
    else:
        eta_status = "High ETA"
        
        
    # Get the Hit probability status
    if hit_probability >= DECISION_THRESHOLDS['hit_prob_high']:
        hit_status = "High hit probability"
    elif DECISION_THRESHOLDS['hit_prob_medium'] <= hit_probability <= DECISION_THRESHOLDS['hit_prob_high']:
        hit_status = "Medium hit probability"
    else:
        hit_status = "Low hit probability"
        
        
    # Get the recommendation based on decisions
    if eta_status == "Low ETA":
        if hit_status == "High hit probability":
            # Critical threat - missile close, high chance of hit
            if combat_posture == "Friendly advantage":
                return "BREAK HARD + DEPLOY CM + COUNTER-ENGAGE"
            elif combat_posture == "Parity":
                return "BREAK HARD + DEPLOY CM"
            else:
                # Outmatched - survive first, disengage
                return "BREAK HARD + DEPLOY CM + DISENGAGE IMMEDIATELY"

        elif hit_status == "Medium hit probability":
            # Elevated threat - missile close, uncertain outcome
            if combat_posture == "Friendly advantage":
                return "DEPLOY CM + DEFENSIVE MANEUVERS + ENGAGE"
            elif combat_posture == "Parity":
                return "DEPLOY CM + DEFENSIVE MANEUVERS"
            else:
                # Disadvantaged - don't risk engagement
                return "DEPLOY CM + DEFENSIVE MANEUVERS + DISENGAGE"

        else:
            # Missile close but likely evadable - redirect and reassess
            if combat_posture == "Friendly advantage":
                return "DEPLOY CM + CHANGE FLIGHT PATH + ENGAGE"
            elif combat_posture == "Parity":
                return "DEPLOY CM + CHANGE FLIGHT PATH"
            else:
                return "DEPLOY CM + CHANGE FLIGHT PATH + DISENGAGE"

    else:
        if hit_status == "High hit probability":
            # Time available but threat is serious - prepare now
            if combat_posture == "Friendly advantage":
                return "DEFENSIVE MANEUVERS + PREPARE CM + COUNTER-ENGAGE"
            elif combat_posture == "Parity":
                return "DEFENSIVE MANEUVERS + PREPARE CM"
            else:
                # Outmatched - use the time to create distance
                return "DEFENSIVE MANEUVERS + PREPARE CM + MAINTAIN DISTANCE"

        elif hit_status == "Medium hit probability":
            # Uncertain threat with time to monitor - stay ready
            if combat_posture == "Friendly advantage":
                return "PREPARE CM + MONITOR + ENGAGE WHEN READY"
            elif combat_posture == "Parity":
                return "PREPARE CM + MONITOR TRAJECTORY"
            else:
                return "PREPARE CM + MAINTAIN DISTANCE"

        else:
            # Low threat - missile unlikely to hit, time available
            if combat_posture == "Friendly advantage":
                return "STAY SHARP + CLOSE AND ENGAGE"
            elif combat_posture == "Parity":
                return "STAY SHARP + MONITOR TRAJECTORY"
            else:
                # Disadvantaged even in low threat - keep distance
                return "STAY SHARP + MAINTAIN DISTANCE"