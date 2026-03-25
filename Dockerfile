# API only — run Streamlit locally or add a second stage later.
FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "incident_memory_engine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
