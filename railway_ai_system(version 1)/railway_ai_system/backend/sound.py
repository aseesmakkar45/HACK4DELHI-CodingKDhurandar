# backend/sound.py

def analyze_sound(sound_file):
    """
    Analyzes uploaded sound file.
    Returns: normal / suspicious
    """

    if sound_file is None:
        return "no_data"

    filename = sound_file.name.lower()

    # Demo logic based on filename
    if any(keyword in filename for keyword in ["cut", "grind", "hammer"]):
        return "suspicious"

    return "normal"