import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from app.schemas import (
    HealthResponse,
    WeatherResponse,
    WeatherDataItem,
    HolidaysResponse,
    HolidayItem,
    NearestUpcomingHoliday,
    ForecastResponse,
    ForecastDataItem
)
from app.weather_client import get_weather_data
from app.holidays_loader import HolidaysLoader
from app.features import build_feature_frame

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals
model_b = None
model_a = None
metadata = {}
holidays_loader = None
artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
_backtest_data_cache = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_b, model_a, metadata, holidays_loader
    
    # Load metadata.json
    meta_path = os.path.join(artifacts_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        logger.info("Loaded metadata.json successfully.")
    else:
        logger.error(f"metadata.json not found at {meta_path}")
        
    # Load model_b_live.pkl
    model_b_path = os.path.join(artifacts_dir, "model_b_live.pkl")
    if os.path.exists(model_b_path):
        model_b = joblib.load(model_b_path)
        logger.info("Loaded model_b_live.pkl successfully.")
    else:
        logger.error(f"model_b_live.pkl not found at {model_b_path}")
        
    # Load model_a_full.pkl
    model_a_path = os.path.join(artifacts_dir, "model_a_full.pkl")
    if os.path.exists(model_a_path):
        model_a = joblib.load(model_a_path)
        logger.info("Loaded model_a_full.pkl successfully.")
    else:
        logger.warning(f"model_a_full.pkl not found at {model_a_path}")
        
    # Load holidays
    holidays_path = os.path.join(artifacts_dir, "jharkhand_holidays.csv")
    holidays_loader = HolidaysLoader(holidays_path)
    
    yield

app = FastAPI(title="APU Demand Forecasting API", lifespan=lifespan)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}

@app.get("/weather", response_model=WeatherResponse)
async def weather():
    logger.info("GET /weather - Request received")
    df_weather, stale = get_weather_data()
    
    # Generate index starting from rounded down "now" in Asia/Kolkata
    kolkata_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(kolkata_tz)
    discard = timedelta(minutes=now.minute % 10, seconds=now.second, microseconds=now.microsecond)
    start_time = now - discard
    
    # We create a naive local index for matching the naive index of weather_client
    start_time_naive = start_time.replace(tzinfo=None)
    forecast_index = pd.date_range(start=start_time_naive, periods=96, freq='10min')
    
    # Slice/reindex to get the exact forecast window
    df_window = df_weather.reindex(forecast_index, method='nearest')
    
    data_items = []
    for t, row in df_window.iterrows():
        # Re-attach timezone when returning to clients
        t_aware = kolkata_tz.localize(t)
        data_items.append(WeatherDataItem(
            timestamp=t_aware.isoformat(),
            temperature=float(row["wx_temperature"]),
            humidity=float(row["wx_humidity"]),
            cloud_cover=float(row["wx_cloud_cover"]),
            wind_speed=float(row["wx_wind_speed"])
        ))
    logger.info(f"GET /weather - Successfully returned {len(data_items)} weather items. (Stale: {stale})")
    return WeatherResponse(
        generated_at=now.isoformat(),
        stale=stale,
        data=data_items
    )

@app.get("/holidays", response_model=HolidaysResponse)
async def holidays():
    logger.info("GET /holidays - Request received")
    # Calculate current window
    kolkata_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(kolkata_tz)
    discard = timedelta(minutes=now.minute % 10, seconds=now.second, microseconds=now.microsecond)
    start_time = now - discard
    end_time = start_time + timedelta(minutes=10 * 95)
    
    in_window_raw = holidays_loader.holidays_in_range(start_time.date(), end_time.date())
    
    in_window = [
        HolidayItem(
            date=item["date"],
            name=item["name"],
            type=item["type"],
            confidence=item["confidence"]
        ) for item in in_window_raw
    ]
    
    nearest_raw = holidays_loader.get_nearest_upcoming(now.date())
    nearest = NearestUpcomingHoliday(
        date=nearest_raw.get("date", ""),
        name=nearest_raw.get("name", ""),
        days_away=nearest_raw.get("days_away", 999)
    )
    logger.info(f"GET /holidays - Found {len(in_window)} holidays in window. Nearest upcoming: {nearest.name} in {nearest.days_away} days.")
    return HolidaysResponse(
        in_window=in_window,
        nearest_upcoming=nearest
    )

def run_backtest_forecast() -> ForecastResponse:
    global model_a, metadata, holidays_loader, _backtest_data_cache
    
    kolkata_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(kolkata_tz)
    
    if _backtest_data_cache is not None:
        logger.info("Serving cached historical backtest forecast (fast-path).")
        return ForecastResponse(
            generated_at=now.isoformat(),
            model="model_a_full",
            stale=False,
            data=_backtest_data_cache
        )
        
    # Try different paths to find utility_consumption_clean.csv
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    paths_to_try = [
        os.path.join(workspace_root, "data", "utility_consumption_clean.csv"),
        os.path.join(workspace_root, "utility_consumption_clean.csv"),
        os.path.join(artifacts_dir, "utility_consumption_clean.csv"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "utility_consumption_clean.csv"),
        "../utility_consumption_clean.csv",
        "utility_consumption_clean.csv"
    ]
    
    data_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            data_path = p
            break
            
    if data_path is None:
        logger.error("utility_consumption_clean.csv not found in any expected location.")
        raise HTTPException(status_code=404, detail="Historical backtest data file utility_consumption_clean.csv not found.")
        
    try:
        # Load dataset
        logger.info(f"Loading backtest data from {data_path} (cache miss)...")
        df_hist = pd.read_csv(data_path, parse_dates=['Datetime']).set_index('Datetime')
        
        # Calculate total_load
        df_hist['total_load'] = df_hist[['F1_132KV_PowerConsumption','F2_132KV_PowerConsumption','F3_132KV_PowerConsumption']].sum(axis=1)
        
        # Lags & Rollings (Must be computed on full dataset before slicing)
        df_hist['lag_1step'] = df_hist['total_load'].shift(1)
        df_hist['lag_1day'] = df_hist['total_load'].shift(144)
        df_hist['lag_1week'] = df_hist['total_load'].shift(1008)
        df_hist['roll_mean_1h'] = df_hist['total_load'].shift(1).rolling(6).mean()
        df_hist['roll_mean_24h'] = df_hist['total_load'].shift(1).rolling(144).mean()
        
        # Slice to the last 96 timestamps FIRST before doing expensive row-mapping
        df_test_window = df_hist.tail(96).copy()
        
        # Rename weather columns to wx_*
        df_test_window.rename(columns={
            'Temperature': 'wx_temperature',
            'Humidity': 'wx_humidity',
            'WindSpeed': 'wx_wind_speed'
        }, inplace=True)
        
        df_test_window['wx_cloud_cover'] = 20.0  # Fallback for cloud cover since it's missing in clean dataset
        
        # Compute cyclical/calendar features only on 96 rows
        df_test_window['hour'] = df_test_window.index.hour
        df_test_window['dow'] = df_test_window.index.dayofweek
        df_test_window['month'] = df_test_window.index.month
        df_test_window['is_weekend'] = (df_test_window['dow'] >= 5).astype(int)
        
        df_test_window['hour_sin'] = np.sin(2 * np.pi * df_test_window['hour'] / 24)
        df_test_window['hour_cos'] = np.cos(2 * np.pi * df_test_window['hour'] / 24)
        df_test_window['dow_sin'] = np.sin(2 * np.pi * df_test_window['dow'] / 7)
        df_test_window['dow_cos'] = np.cos(2 * np.pi * df_test_window['dow'] / 7)
        
        df_test_window['cooling_degree'] = (df_test_window['wx_temperature'] - 24).clip(lower=0)
        
        # Holiday features computed only on the 96 window rows
        df_test_window['is_holiday'] = df_test_window.index.map(lambda d: int(holidays_loader.is_holiday(d.date())))
        df_test_window['days_to_nearest_holiday'] = df_test_window.index.map(lambda d: holidays_loader.days_to_nearest_holiday(d.date()))
        
        # Select target features and drop NaNs
        expected_features = metadata["full_features"]
        df_feats = df_test_window[expected_features].dropna()
        
        if model_a is None:
            raise ValueError("Backtest Model A is not loaded.")
            
        predictions = model_a.predict(df_feats)
        
        # Generate JSON items
        data_items = []
        for i, (t, _) in enumerate(df_feats.iterrows()):
            t_aware = kolkata_tz.localize(t) if t.tzinfo is None else t
            data_items.append(ForecastDataItem(
                timestamp=t_aware.isoformat(),
                predicted_load_kw=float(predictions[i])
            ))
            
        # Cache the result
        _backtest_data_cache = data_items
        logger.info("Cached backtest predictions successfully.")
        
        return ForecastResponse(
            generated_at=now.isoformat(),
            model="model_a_full",
            stale=False,
            data=data_items
        )
    except Exception as e:
        logger.error(f"Error in backtest calculation: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing backtest: {str(e)}")

@app.get("/forecast", response_model=ForecastResponse)
async def forecast(mode: str = Query("live", description="Prediction mode: live or backtest")):
    global model_b, metadata, holidays_loader
    
    logger.info(f"GET /forecast - Request received (mode={mode})")
    kolkata_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(kolkata_tz)
    
    if mode == "backtest":
        return run_backtest_forecast()
        
    # Get weather data (either cached or freshly fetched)
    df_weather, stale = get_weather_data()
    
    discard = timedelta(minutes=now.minute % 10, seconds=now.second, microseconds=now.microsecond)
    start_time = now - discard
    
    start_time_naive = start_time.replace(tzinfo=None)
    forecast_index = pd.date_range(start=start_time_naive, periods=96, freq='10min')
    
    # Reindex weather df to forecast window
    df_window = df_weather.reindex(forecast_index, method='nearest')
    
    # Calculate inference features
    df_feats = build_feature_frame(df_window, holidays_loader, metadata)
    
    if model_b is None:
        raise HTTPException(status_code=500, detail="Live forecast Model B is not loaded.")
        
    # Predict load demand
    predictions = model_b.predict(df_feats)
    
    data_items = []
    for i, t in enumerate(forecast_index):
        t_aware = kolkata_tz.localize(t)
        data_items.append(ForecastDataItem(
            timestamp=t_aware.isoformat(),
            predicted_load_kw=float(predictions[i])
        ))
    logger.info(f"GET /forecast - Successfully generated live predictions. (Stale: {stale})")
    return ForecastResponse(
        generated_at=now.isoformat(),
        model="model_b_live",
        stale=stale,
        data=data_items
    )

# Serve static dashboard
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=FileResponse)
async def get_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Dashboard Not Found</h1><p>Ensure backend/app/static/index.html is created.</p>")
