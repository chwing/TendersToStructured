"""Daily: scrape new tenders from the procurement portals, then run the
Staged 2 extraction pipeline over just that day's newly-scraped documents.

    scrape_tenders  ->  stage_new_docs  ->  extract_tenders

- scrape_tenders and extract_tenders run in sibling containers built from the
  tender-pipeline image (../../Dockerfile), via DockerOperator — see
  ../docker-compose.yaml for why (keeps torch/transformers out of Airflow).
- stage_new_docs runs in-process in the scheduler (pure stdlib, no heavy
  deps needed): it reads the manifest scrape_tenders wrote and copies just
  those documents into a per-run input folder, so extraction doesn't
  reprocess previously-scraped tenders.

All three tasks share one bind-mounted host directory at container path
/data: tender_docs_new/, manifests/<ds>.json, and runs/<ds>/{input,output}/.
The Airflow containers get that mount straight from docker-compose.yaml;
scrape_tenders/extract_tenders get it via DATA_MOUNT below, which must point
at the same directory but as a path the *host* Docker daemon can resolve
(see docker-compose.yaml's comment on docker-outside-of-docker).
"""

import json
import os
import shutil
from datetime import timedelta
from pathlib import Path

from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

TENDER_PIPELINE_IMAGE = os.environ.get("TENDER_PIPELINE_IMAGE", "tender-pipeline:latest")
DATA_HOST_PATH = os.environ.get(
    "TENDER_PIPELINE_DATA_HOST_PATH",
    str(Path(__file__).resolve().parents[1] / "data"),
)
DATA_MOUNT = Mount(source=DATA_HOST_PATH, target="/data", type="bind")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


FAST_MODE = _env_bool("TENDER_PIPELINE_FAST_MODE", False)
SCRAPER_SOURCE = os.environ.get("TENDER_PIPELINE_SCRAPER_SOURCE", "all")
# Default to 10 tenders for DAG runs; override with env `TENDER_PIPELINE_SCRAPER_LIMIT`
SCRAPER_LIMIT = os.environ.get("TENDER_PIPELINE_SCRAPER_LIMIT", "10")
try:
    SCRAPER_LIMIT = int(SCRAPER_LIMIT) if SCRAPER_LIMIT not in (None, "") else None
except Exception:
    SCRAPER_LIMIT = 10
SCRAPER_MAX_PAGES = _env_int("TENDER_PIPELINE_SCRAPER_MAX_PAGES", 10)
FAST_EXTRACTOR_MODEL = os.environ.get("TENDER_PIPELINE_FAST_EXTRACTOR_MODEL", "mistral")
FAST_EXTRACTOR_TIMEOUT = _env_int("TENDER_PIPELINE_FAST_EXTRACTOR_TIMEOUT", 600)
FAST_EXTRACTOR_NUM_CTX = _env_int("TENDER_PIPELINE_FAST_EXTRACTOR_NUM_CTX", 8192)
FAST_MIN_CONFIDENCE = os.environ.get("TENDER_PIPELINE_FAST_MIN_CONFIDENCE", "0.0")

# Ollama is assumed to run on the host, as in the main README.
OLLAMA_BASE_URL = os.environ.get("TENDER_PIPELINE_OLLAMA_URL", "http://host.docker.internal:11434")

default_args = {
    "owner": "tendersToStructured",
    "retries": 1,
    "retry_delay": timedelta(minutes=40),
}


def _stage_new_docs(ds: str, **_):
    manifest_path = Path(f"/data/manifests/{ds}.json")
    input_dir = Path(f"/data/runs/{ds}/input")
    input_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}, nothing to stage")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied = 0
    for tender in manifest.get("tenders", []):
        for doc_path in tender.get("documents", []):
            src = Path(doc_path)
            if not src.exists():
                print(f"WARNING: manifest references missing file {src}")
                continue
            shutil.copy2(src, input_dir / src.name)
            copied += 1

    print(
        f"Staged {copied} document(s) from {len(manifest.get('tenders', []))} "
        f"tender(s) into {input_dir}"
    )


with DAG(
    dag_id="tender_scrape_and_extract",
    description="Daily: scrape new tenders, extract structured fields from the new documents",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["tenders"],
) as dag:

    scrape_tenders = DockerOperator(
        task_id="scrape_tenders",
        image=TENDER_PIPELINE_IMAGE,
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mounts=[DATA_MOUNT],
        network_mode="bridge",
        command=" ".join(
            part for part in [
                "python scraper/run_scraper.py",
                f"--source {SCRAPER_SOURCE}",
                "--output-dir /data/tender_docs_new",
                f"--manifest-out /data/manifests/{{{{ ds }}}}.json",
                f"--max-pages {SCRAPER_MAX_PAGES}",
                f"--limit {SCRAPER_LIMIT}" if SCRAPER_LIMIT is not None else "",
                "-v",
            ]
            if part
        ),
    )

    stage_new_docs = PythonOperator(
        task_id="stage_new_docs",
        python_callable=_stage_new_docs,
        op_kwargs={"ds": "{{ ds }}"},
    )

    extract_tenders = DockerOperator(
        task_id="extract_tenders",
        image=TENDER_PIPELINE_IMAGE,
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mounts=[DATA_MOUNT],
        network_mode="bridge",
        extra_hosts={"host.docker.internal": "host-gateway"},
        command=(
            (
                "python full_llm/run_extractor.py "
                "/data/runs/{{ ds }}/input "
                "--output-dir /data/runs/{{ ds }}/output "
                f"--base-url {OLLAMA_BASE_URL} "
                f"--model {FAST_EXTRACTOR_MODEL} "
                f"--timeout {FAST_EXTRACTOR_TIMEOUT} "
                f"--num-ctx {FAST_EXTRACTOR_NUM_CTX} "
                f"--min-confidence {FAST_MIN_CONFIDENCE}"
            )
            if FAST_MODE
            else (
                "python second_staged_pipeline/run_pipeline.py "
                "/data/runs/{{ ds }}/input "
                "--output-dir /data/runs/{{ ds }}/output "
                f"--extractor-base-url {OLLAMA_BASE_URL} "
                f"--judge-base-url {OLLAMA_BASE_URL}"
            )
        ),
    )

    scrape_tenders >> stage_new_docs >> extract_tenders
