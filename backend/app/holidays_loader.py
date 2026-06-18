import os
import pandas as pd
import datetime
import logging

logger = logging.getLogger(__name__)

class HolidaysLoader:
    def __init__(self, holidays_csv_path: str):
        self.holiday_dict = {}
        self.holiday_dates = set()
        self.load_holidays(holidays_csv_path)

    def load_holidays(self, path: str):
        if not os.path.exists(path):
            logger.error(f"Holidays CSV not found at: {path}")
            return
        
        try:
            df = pd.read_csv(path)
            # dates are in YYYY-MM-DD format
            df['date'] = pd.to_datetime(df['date']).dt.date
            for _, row in df.iterrows():
                d = row['date']
                name = row['name']
                h_type = row['type']
                conf = row['confidence']
                self.holiday_dict[d] = {
                    "date": d.isoformat(),
                    "name": name,
                    "type": h_type,
                    "confidence": conf
                }
                self.holiday_dates.add(d)
            logger.info(f"Loaded {len(self.holiday_dates)} holidays from {path}")
        except Exception as e:
            logger.error(f"Error loading holidays: {e}")

    def is_holiday(self, date: datetime.date) -> bool:
        return date in self.holiday_dates

    def days_to_nearest_holiday(self, date: datetime.date) -> int:
        if not self.holiday_dates:
            return 999
        diffs = [abs((date - hd).days) for hd in self.holiday_dates]
        return min(diffs)

    def holidays_in_range(self, start: datetime.date, end: datetime.date) -> list[dict]:
        res = []
        # sort dates
        for hd in sorted(self.holiday_dates):
            if start <= hd <= end:
                res.append(self.holiday_dict[hd])
        return res

    def get_nearest_upcoming(self, date: datetime.date) -> dict:
        """
        Returns the nearest holiday where holiday_date >= date, with 'days_away' added.
        If no upcoming holiday exists, returns the absolute nearest one in the past.
        """
        if not self.holiday_dates:
            return {}
        
        upcoming = [hd for hd in self.holiday_dates if hd >= date]
        if not upcoming:
            # Fallback to absolute closest if none in the future
            nearest_date = min(self.holiday_dates, key=lambda hd: abs((date - hd).days))
            days_away = (nearest_date - date).days
            res = self.holiday_dict[nearest_date].copy()
            res["days_away"] = days_away
            return res
        
        nearest_date = min(upcoming)
        days_away = (nearest_date - date).days
        res = self.holiday_dict[nearest_date].copy()
        res["days_away"] = days_away
        return res
