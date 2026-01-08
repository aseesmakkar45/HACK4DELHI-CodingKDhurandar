# backend/vision.py

def analyze_visual(media_file):
    """
    Analyze CCTV or Drone visual input.
    Supports both IMAGE and VIDEO uploads.
    Currently simulated using filename keywords.
    """

    # If no feed is provided
    if media_file is None:
        return "no_feed"

    # Get filename safely
    filename = media_file.name.lower()

    # --- Simulated detection logic ---
    if "tamper" in filename or "damage" in filename or "break" in filename:
        return "tampering"

    if "flood" in filename or "water" in filename or "rain" in filename:
        return "flood"

    if "normal" in filename or "clear" in filename:
        return "normal"

    # Default fallback (important for unknown files)
    return "normal"