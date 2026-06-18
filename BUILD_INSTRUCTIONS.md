# Build Spec — APU Power Demand Forecasting: Live Backend + Auto-Refreshing Dashboard + Docker

Hand this whole document to your IDE's coding agent as the task brief. It assumes Milestones 1–2
(EDA, cleaning, feature engineering, model training) are already done. This covers Milestones 3 & 4:
the FastAPI backend, the live auto-refreshing dashboard, and Docker packaging.

---

## 0. Files to give your IDE, and why each one matters

Upload/attach all of these — the backend cannot be built correctly without them:

| File | Why it's required |
|---|---|
| `backend/artifacts/model_b_live.pkl` | **The model the live API actually serves.** Trained on weather+calendar features only (no lag features), because no live SCADA feed exists for this prototype. |
| `backend/artifacts/model_a_full.pkl` | Optional but recommended — include it so you can also expose a `/forecast?mode=backtest` endpoint that demonstrates the more accurate lag-based model against historical holdout data. Not used for the live dashboard itself. |
| `backend/artifacts/metadata.json` | **Critical.** Contains the exact ordered feature-name lists (`calendar_weather_features`, `full_features`) the models were trained on, plus Dhanbad's lat/lon. The live feature-computation code must reproduce these exact column names — a mismatch here is the single most common cause of silently wrong predictions in deployed ML systems. |
| `backend/artifacts/jharkhand_holidays.csv` | Holiday calendar — but **it currently only has regional tribal-festival estimates for 2017** (the training year). Section 1 below gives you the 2026 rows to append before building. |
| `notebooks/02_features_and_model.ipynb` | **Required as a reference, not to run.** Your IDE needs to read the feature-engineering cells (cyclical hour/day encoding, `cooling_degree`, `is_holiday`, `days_to_nearest_holiday`) and reproduce that *exact* logic at inference time. Re-deriving it independently risks subtle inconsistencies between training and serving. |
| `data/utility_consumption_clean.csv` | Only needed if you build the optional backtest endpoint/demo. Not needed for live mode. |

Give the agent this instruction explicitly: **"Read `metadata.json` and the feature-engineering cells in
`02_features_and_model.ipynb` first, and use those exact feature names/order — do not invent your own
feature names."**

---

## 1. Close the data gap first: 2026 regional festival dates

`jharkhand_holidays.csv` needs these rows added for live-mode 2026 dates (the gazetted national/state
holidays are already correct for both years via the `holidays` Python library — only the four regional
tribal festivals need manual extension, same as was done for 2017):

| date | name | type | confidence |
|---|---|---|---|
| 2026-03-21 | Sarhul (Chaitra Shukla Tritiya) | regional_tribal | high — multiple independent sources agree |
| 2026-09-22 | Karam / Karma Parab (Bhadrapada Shukla Ekadashi) | regional_tribal | high — multiple independent sources agree |
| 2026-01-13 | Tusu Parab (Poush Sankranti period) | regional_tribal | medium |
| 2026-11-09 | Sohrai (traditional rule: day after Diwali, Kartik Amavasya) | regional_tribal | **conflicting** — see note |

**Note on Sohrai:** sources disagree. The Jharkhand government's own festivals page and Wikipedia both
describe Sohrai as falling on the new-moon day immediately after Diwali (→ 9 Nov 2026, since Diwali 2026
is 8 Nov). At least one third-party holiday-list site instead gives "12–13 January 2026," which is
suspiciously close to Tusu Parab/Makar Sankranti and may reflect a mix-up between the two festivals rather
than a real second date. **Instruct your IDE to add both candidate rows with `confidence: conflicting`**,
and flag in the README that this should be verified against the official Jharkhand government gazette
before the date the project is actually demoed/submitted.

Tell your IDE: *"Append these rows to `backend/artifacts/jharkhand_holidays.csv` before writing any
code that reads it."*

---

## 2. Backend — FastAPI

### 2.1 Project structure to create
```
backend/
  app/
    main.py
    weather_client.py
    features.py
    holidays_loader.py
    schemas.py
  artifacts/            (already provided)
  requirements.txt
```

### 2.2 `requirements.txt`
```
fastapi
uvicorn[standard]
scikit-learn
joblib
pandas
numpy
requests
apscheduler
```

### 2.3 `weather_client.py` — live weather fetch
Exact instructions for the agent:
- Call `GET https://api.open-meteo.com/v1/forecast` (this is the **live forecast** endpoint, different
  host from the historical archive endpoint used in training — no API key needed for either).
- Params: `latitude=23.7957`, `longitude=86.4304` (pull from `metadata.json`, don't hardcode twice),
  `hourly=temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m`, `timezone=Asia/Kolkata`,
  `forecast_days=2` (use 2, not 1 — guarantees a full 96-block window even if "now" is late in the day).
- Parse the JSON `hourly` block into a DataFrame indexed by time, rename columns to
  `wx_temperature`, `wx_humidity`, `wx_cloud_cover`, `wx_wind_speed` (must match training names exactly).
- Resample hourly → 10-minute via linear interpolation (same method as the training notebook), then
  slice to exactly the next 96 blocks starting from "now" rounded **down** to the nearest 10-minute mark.
- **Caching requirement:** wrap this in a server-side cache with a 10-minute TTL (use a simple
  module-level dict with a timestamp, or `apscheduler` to refresh on a schedule). The dashboard will poll
  every 10 minutes, but the backend must not hit Open-Meteo on every single poll from every browser tab —
  cache the upstream response and serve from cache between refreshes.
- **Resilience requirement:** wrap the actual HTTP call in try/except. On failure, serve the last
  successfully cached response and add `"stale": true` to the response payload rather than crashing or
  returning an error to the dashboard. Log the failure.

### 2.4 `holidays_loader.py`
- Load `jharkhand_holidays.csv` once at startup into a set of dates (plus name/confidence lookup).
- Function `is_holiday(date) -> bool`, `days_to_nearest_holiday(date) -> int`, and
  `holidays_in_range(start, end) -> list[dict]` for the `/holidays` endpoint.

### 2.5 `features.py` — must mirror the notebook exactly
Instruct the agent: *"Implement `build_feature_frame(weather_df, holidays_loader)` that reproduces,
feature-for-feature, the calendar/weather feature engineering in `02_features_and_model.ipynb` Section 3
— do not redesign it."* Specifically, for each of the 96 forecast timestamps:
- `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` (cyclical encodings, same formulas as the notebook)
- `is_weekend`, `month`
- `is_holiday`, `days_to_nearest_holiday` (from `holidays_loader`)
- `wx_temperature`, `wx_humidity`, `wx_cloud_cover`, `wx_wind_speed` (from the live weather fetch)
- `cooling_degree = max(wx_temperature - 24, 0)`
- Final column order/names must exactly match `metadata['calendar_weather_features']` — load that list
  from `metadata.json` rather than hardcoding it, so the two never drift apart.

### 2.6 `main.py` — endpoints
Build exactly these:

**`GET /health`** → `{"status": "ok"}`

**`GET /weather`** → next 24h of weather at 10-min resolution:
```json
{"generated_at": "...", "stale": false,
 "data": [{"timestamp": "...", "temperature": 0, "humidity": 0, "cloud_cover": 0, "wind_speed": 0}, ...]}
```

**`GET /holidays`** → holidays overlapping the forecast window, plus the nearest one if none overlap:
```json
{"in_window": [{"date": "...", "name": "...", "confidence": "..."}],
 "nearest_upcoming": {"date": "...", "name": "...", "days_away": 0}}
```

**`GET /forecast`** → the core endpoint: fetch cached weather → build features → `model_b.predict()` →
return:
```json
{"generated_at": "...", "model": "model_b_live", "stale": false,
 "data": [{"timestamp": "...", "predicted_load_kw": 0}, ...]}
```
Load `model_b_live.pkl` and `metadata.json` once at startup (FastAPI `lifespan` or module-level globals),
not on every request.

Add CORS middleware (`fastapi.middleware.cors.CORSMiddleware`) allowing the frontend's origin (or `*` for
this prototype) since the dashboard will be a separate static page calling these endpoints.

---

## 3. Frontend — auto-refreshing dashboard

Single HTML file, vanilla JS + Chart.js (no build step → simplest to Dockerize). Exact requirements:

1. **Auto-refresh every 10 minutes**: on page load, call `/forecast`, `/weather`, `/holidays` immediately,
   then `setInterval(fetchAll, 10 * 60 * 1000)`. Show a visible "Last updated: HH:MM:SS" timestamp that
   updates on every refresh, so it's obviously live rather than static.
2. **Forecast chart**: Chart.js line chart, x-axis = the 96 timestamps, y-axis = `predicted_load_kw`.
3. **Weather panel**: four small cards/sparklines for temperature, humidity, cloud cover, wind speed,
   pulled from `/weather`, aligned to the same time axis as the forecast chart (or at minimum show the
   current/next values clearly).
4. **Holiday annotations**: if `/holidays` returns anything in `in_window`, draw a vertical marker line on
   the forecast chart at that timestamp with a label (Chart.js `annotation` plugin, loaded from
   `cdnjs.cloudflare.com`), and show a small banner: *"Upcoming: [festival name] on [date] — demand may
   deviate from the typical weekday pattern."* If nothing's in the window, show the nearest upcoming one
   in a quieter, secondary spot.
5. **States to handle explicitly**: loading spinner on first load; a visible "stale data" badge if any
   response has `"stale": true`; a clear error banner if a fetch fails outright (don't fail silently).

---

## 4. Docker

Single container, simplest possible story for grading:
- `python:3.11-slim` base
- Copy `backend/` and the frontend's static files into the image
- `pip install -r requirements.txt`
- Serve the frontend as static files from the **same** FastAPI app (`StaticFiles` mount) so one container,
  one port, does everything
- `EXPOSE 8000`, `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`
- Add a `.dockerignore` for `notebooks/`, `__pycache__/`, `.git/`

---

## 5. README.md — must include
1. One-paragraph architecture overview (live weather → feature pipeline → Model B → dashboard)
2. `docker build` / `docker run` commands, and the local-dev (no Docker) alternative
3. API reference for the four endpoints with example responses
4. **Documented limitations**, stated plainly rather than hidden: Model B's accuracy trade-off vs Model A
   (9.2% vs 0.65% MAPE in backtesting — restate from notebook 2's summary table), the conflicting Sohrai
   date sourcing, and the fact that holiday dates beyond 2026 will need the same manual extension process.

---

## 6. Verification checklist before you submit
- [ ] `/forecast` was hit at least once with real internet and you saw real Open-Meteo data (not a fallback/error)
- [ ] Dashboard visibly updates its "Last updated" timestamp every 10 minutes without a manual page reload
- [ ] Temporarily test the holiday banner by hardcoding a near-term fake holiday date and confirming it renders, then remove the hardcode
- [ ] `docker build . && docker run -p 8000:8000 <image>` works from a clean clone and the dashboard loads at `localhost:8000`
- [ ] Killing your internet connection briefly and refreshing shows the "stale data" badge instead of crashing
