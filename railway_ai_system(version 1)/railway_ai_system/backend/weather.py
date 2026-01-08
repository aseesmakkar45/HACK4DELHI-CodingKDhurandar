# backend/weather.py
import random

def fetch_weather_forecast(city="Delhi"):
    """
    Simulates fetching weather forecast for a city.
    Hackathon-safe (no API dependency).
    """

    possible_conditions = [
        "low_risk",     # clear weather
        "medium_risk",  # light rain
        "high_risk"     # heavy rain / storm
    ]

    return random.choice(possible_conditions)