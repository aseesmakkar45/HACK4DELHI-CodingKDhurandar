

def analyze_sound(sound_file):
    

    if sound_file is None:
        return "no_data"

    filename = sound_file.name.lower()

    
    if any(keyword in filename for keyword in ["cut", "grind", "hammer"]):
        return "suspicious"

    return "normal"