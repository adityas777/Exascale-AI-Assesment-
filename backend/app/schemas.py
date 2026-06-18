from pydantic import BaseModel
from typing import List, Dict, Any

class HealthResponse(BaseModel):
    status: str

class WeatherDataItem(BaseModel):
    timestamp: str
    temperature: float
    humidity: float
    cloud_cover: float
    wind_speed: float

class WeatherResponse(BaseModel):
    generated_at: str
    stale: bool
    data: List[WeatherDataItem]

class HolidayItem(BaseModel):
    date: str
    name: str
    type: str
    confidence: str

class NearestUpcomingHoliday(BaseModel):
    date: str
    name: str
    days_away: int

class HolidaysResponse(BaseModel):
    in_window: List[HolidayItem]
    nearest_upcoming: NearestUpcomingHoliday

class ForecastDataItem(BaseModel):
    timestamp: str
    predicted_load_kw: float

class ForecastResponse(BaseModel):
    generated_at: str
    model: str
    stale: bool
    data: List[ForecastDataItem]
