FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY feature_for_API.py .      
COPY price_scaler.pkl .
COPY price_prediction_xgb.pkl .
COPY scaler.pkl .
COPY isolation_forest_model.pkl .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]