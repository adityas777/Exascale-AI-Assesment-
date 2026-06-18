import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_feature_frame(weather_df: pd.DataFrame, holidays_loader, metadata: dict) -> pd.DataFrame:
    """
    Builds the feature frame for the 96 timestamps in weather_df.
    Replicates the exact training notebook features and ordering.
    """
    # weather_df index is pd.DatetimeIndex (tz-naive local times)
    times = weather_df.index
    
    hour = times.hour
    dow = times.dayofweek
    month = times.month
    
    is_weekend = (dow >= 5).astype(int)
    
    # Cyclical encodings
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    
    # Weather-derived features
    # cooling_degree: max(wx_temperature - 24, 0)
    cooling_degree = (weather_df['wx_temperature'] - 24).clip(lower=0)
    
    # Holidays features
    is_holiday = [int(holidays_loader.is_holiday(d.date())) for d in times]
    days_to_nearest_holiday = [holidays_loader.days_to_nearest_holiday(d.date()) for d in times]
    
    # Combine into dictionary
    features = {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "is_weekend": is_weekend,
        "month": month,
        "is_holiday": is_holiday,
        "days_to_nearest_holiday": days_to_nearest_holiday,
        "wx_temperature": weather_df["wx_temperature"].values,
        "wx_humidity": weather_df["wx_humidity"].values,
        "wx_cloud_cover": weather_df["wx_cloud_cover"].values,
        "wx_wind_speed": weather_df["wx_wind_speed"].values,
        "cooling_degree": cooling_degree.values
    }
    
    # Create DataFrame
    df_feats = pd.DataFrame(features, index=times)
    
    # Retrieve correct column order from metadata
    expected_cols = metadata["calendar_weather_features"]
    
    # Reorder columns to match model training signature exactly
    df_feats = df_feats[expected_cols]
    
    return df_feats
