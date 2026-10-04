# Runtime image for the scrape + extract steps run by Airflow (see airflow/).
# Not used to run Airflow itself — that's the separate apache/airflow image in
# airflow/docker-compose.yaml. This image just needs the project's own deps.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY scraper/requirements.txt scraper/requirements.txt
COPY second_staged_pipeline/requirements.txt second_staged_pipeline/requirements.txt
RUN pip install --no-cache-dir \
    -r scraper/requirements.txt \
    -r second_staged_pipeline/requirements.txt

COPY scraper/ scraper/
COPY second_staged_pipeline/ second_staged_pipeline/
COPY full_llm/ full_llm/

ENV PYTHONUNBUFFERED=1
