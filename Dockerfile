FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .
RUN python scripts/ingest.py

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
