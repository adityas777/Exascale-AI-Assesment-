import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import pytz

logger = logging.getLogger(__name__)

DHANBAD_LAT = 23.7957
DHANBAD_LON = 86.4304

# Simple server-side cache dictionary
_weather_cache = {
    "data": None,      # pd.DataFrame
    "timestamp": 0.0,  # float timestamp
    "stale": False     # boolean
}

def generate_synthetic_weather() -> pd.DataFrame:
    """
    Generates synthetic weather forecast for Dhanbad as a robust offline fallback.
    """
    logger.warning("Generating synthetic weather fallback.")
    kolkata_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(kolkata_tz)
    start_date = datetime(now.year, now.month, now.day) # tz-naive representing local date
    
    # 2 days at 10-minute resolution
    idx = pd.date_range(start=start_date, end=start_date + timedelta(days=2), freq='10min')
    
    # Simple temperature cycle (peak ~14:00, low ~05:00)
    doy = idx.dayofyear
    seasonal_temp = 28 + 8 * np.sin(2 * np.pi * (doy - 80) / 365)
    diurnal = 6 * np.sin(2 * np.pi * (idx.hour - 9) / 24)
    wx_temp = seasonal_temp + diurnal + np.random.normal(0, 0.5, len(idx))
    
    wx_humidity = np.clip(55 + 20 * np.sin(2 * np.pi * (idx.hour - 3) / 24) + np.random.normal(0, 2, len(idx)), 10, 100)
    wx_cloud = np.clip(30 + np.random.normal(0, 5, len(idx)), 0, 100)
    wx_wind = np.clip(2.0 + np.random.normal(0, 0.2, len(idx)), 0.1, None)
    
    wdf = pd.DataFrame({
        'wx_temperature': wx_temp,
        'wx_humidity': wx_humidity,
        'wx_cloud_cover': wx_cloud,
        'wx_wind_speed': wx_wind
    }, index=idx)
    wdf.index.name = 'time'
    return wdf

def fetch_live_weather() -> tuple[pd.DataFrame, bool]:
    """
    Fetches weather from Open-Meteo API, processes it, and updates cache.
    Returns (weather_df, is_stale).
    """
    global _weather_cache
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": DHANBAD_LAT,
        "longitude": DHANBAD_LON,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m",
        "timezone": "Asia/Kolkata",
        "forecast_days": 2
    }
    
    try:
        logger.info(f"Fetching weather from {url} for coordinates ({DHANBAD_LAT}, {DHANBAD_LON})")
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        
        j = resp.json()
        if "hourly" not in j:
            raise ValueError("Invalid response from Open-Meteo: 'hourly' key not found.")
            
        hourly = j["hourly"]
        
        wdf = pd.DataFrame({
            "wx_temperature": hourly["temperature_2m"],
            "wx_humidity": hourly["relative_humidity_2m"],
            "wx_cloud_cover": hourly["cloud_cover"],
            "wx_wind_speed": hourly["wind_speed_10m"]
        }, index=pd.to_datetime(hourly["time"]))
        
        # Resample from hourly to 10-minute resolution using linear interpolation
        wdf_10m = wdf.resample("10min").interpolate(method="linear")
        
        # Cache success
        _weather_cache["data"] = wdf_10m
        _weather_cache["timestamp"] = time.time()
        _weather_cache["stale"] = False
        
        logger.info("Successfully fetched and cached live weather data.")
        return wdf_10m, False
        
    except Exception as e:
        logger.error(f"Failed to fetch live weather: {e}")
        
        if _weather_cache["data"] is not None:
            logger.info("Serving stale weather data from cache.")
            _weather_cache["stale"] = True
            return _weather_cache["data"], True
        else:
            logger.warning("No cache available. Falling back to synthetic weather data.")
            synthetic_df = generate_synthetic_weather()
            _weather_cache["data"] = synthetic_df
            _weather_cache["timestamp"] = time.time()
            _weather_cache["stale"] = True
            return synthetic_df, True

def get_weather_data() -> tuple[pd.DataFrame, bool]:
    """
    Returns cached weather data if cache TTL (10 minutes) is valid.
    Otherwise fetches fresh data.
    """
    global _weather_cache
    
    now_ts = time.time()
    
    if _weather_cache["data"] is None:
        logger.info("Weather cache is empty. Initiating first fetch...")
    else:
        cache_age = now_ts - _weather_cache["timestamp"]
        if cache_age < 600:
            logger.info(f"Cache hit: Serving cached weather data (age: {cache_age:.1f} seconds, stale: {_weather_cache['stale']})")
            return _weather_cache["data"], _weather_cache["stale"]
        else:
            logger.info(f"Cache expired (age: {cache_age:.1f} seconds). Requesting fresh fetch...")
        
    return fetch_live_weather()
