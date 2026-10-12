# One image, three entrypoints (api | worker | mcp). Non-root, no build tools in the final layer.
FROM python:3.12-slim AS build
WORKDIR /w
COPY priorauth/requirements.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
RUN useradd --system --uid 10001 --create-home pa
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY priorauth ./priorauth
USER 10001
EXPOSE 8080 8765 9102
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:%s/healthz'%os.getenv('PORT','8080'))" || exit 1
CMD ["uvicorn", "--factory", "priorauth.api:app_factory", "--host", "0.0.0.0", "--port", "8080"]
