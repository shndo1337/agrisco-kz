FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model if artifacts missing
RUN python -c "from pathlib import Path; \
    exit(0) if Path('data/processed/model.joblib').exists() else exit(1)" \
    || echo "Model artifacts not found — mount data/ or run train.py"

EXPOSE 8501 8000

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
