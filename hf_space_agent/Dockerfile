FROM python:3.15-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Real source, copied in (not re-implemented) -- avoids the drift risk the
# main repo's hf_space/app.py has against serving/api.py.
COPY agents/ ./agents/
COPY ehr/ ./ehr/
COPY serving/__init__.py ./serving/__init__.py
COPY serving/healthcare/ ./serving/healthcare/
COPY app.py .

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "app.py"]
