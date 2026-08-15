FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ ./serving/
COPY model/artifacts/tft.onnx ./model/artifacts/tft.onnx
COPY model/artifacts/tft.ckpt ./model/artifacts/tft.ckpt
COPY model/artifacts/scalers.json ./model/artifacts/scalers.json

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "serving.api"]