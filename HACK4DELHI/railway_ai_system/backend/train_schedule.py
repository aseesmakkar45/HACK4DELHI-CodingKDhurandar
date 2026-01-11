import random

_previous_distance = None

def get_train_status():
    
    return {
        "train_id": "NDLS_EXP_12951",
        "distance_km": round(random.uniform(0.2, 6.0), 2),
        "speed_kmph": random.choice([60, 80, 100])
    }

def get_direction(current_distance):
    
    global _previous_distance

    if _previous_distance is None:
        direction = "unknown"
    elif current_distance < _previous_distance:
        direction = "approaching"
    else:
        direction = "moving_away"

    _previous_distance = current_distance
    return direction