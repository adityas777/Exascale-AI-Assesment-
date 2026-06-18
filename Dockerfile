FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY backend/app/ ./app/
COPY backend/artifacts/ ./artifacts/
# Copy backtest dataset to artifacts so backtesting works in-container
COPY data/utility_consumption_clean.csv ./artifacts/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
