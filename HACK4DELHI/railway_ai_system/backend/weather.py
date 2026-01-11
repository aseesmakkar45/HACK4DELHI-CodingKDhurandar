import os
import requests

# =========================
# INTERNAL STATE
# =========================

_API_KEY = None
_API_KEY_LOADED = False


# =========================
# LAZY API KEY LOADER
# =========================

def _get_api_key():
    global _API_KEY, _API_KEY_LOADED

    if _API_KEY_LOADED:
        return _API_KEY

    # Try Streamlit / system env first
    _API_KEY = os.getenv("OPENWEATHER_API_KEY")

    # Optional: fallback to dotenv ONLY if running locally
    if not _API_KEY:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            _API_KEY = os.getenv("OPENWEATHER_API_KEY")
        except Exception:
            _API_KEY = None

    if _API_KEY:
        print("✅ Weather API key loaded")
    else:
        print("⚠️ Weather API key not found → using fallback data")

    _API_KEY_LOADED = True
    return _API_KEY


# =========================
# WEATHER FETCH
# =========================

def fetch_weather_by_coordinates(lat, lon):
    api_key = _get_api_key()

    # ---------- FALLBACK MODE ----------
    if not api_key:
        return {
            "temperature_c": 25,
            "humidity_percent": 60,
            "wind_speed_mps": 3.5,
            "condition": "Clear",
            "description": "clear sky",
            "rain_mm_last_1h": 0,
        }

    try:
        url = (
            "https://api.weatherapi.com/v1/current.json"
            f"?key={api_key}&q={lat},{lon}&aqi=no"
        )

        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            raise RuntimeError(f"API returned status {response.status_code}")

        data = response.json()

        return {
            "temperature_c": data["current"]["temp_c"],
            "humidity_percent": data["current"]["humidity"],
            "wind_speed_mps": data["current"]["wind_kph"] / 3.6,
            "condition": data["current"]["condition"]["text"],
            "description": data["current"]["condition"]["text"].lower(),
            "rain_mm_last_1h": data["current"].get("precip_mm", 0),
        }

    except Exception as e:
        print(f"⚠️ Weather API Error: {e}")

        return {
            "temperature_c": 25,
            "humidity_percent": 60,
            "wind_speed_mps": 3.5,
            "condition": "unavailable",
            "description": "weather data not available",
            "rain_mm_last_1h": 0,
        }


# =========================
# ZONE WEATHER (UNCHANGED)
# =========================

def get_zone_weather(zone_sensors):
    zone_weather = {}

    for sensor_id, sensor in zone_sensors.items():
        weather = fetch_weather_by_coordinates(
            sensor["lat"],
            sensor["lon"]
        )

        zone_weather[sensor_id] = {
            "location": sensor["name"],
            "weather": weather
        }

    return zone_weather
