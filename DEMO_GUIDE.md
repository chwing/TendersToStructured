# Demo Guide — reset docs, rerun extraction, show Airflow

Two ways to demo the pipeline: **without Docker** (fast, just the extractor) and
**with Docker + Airflow** (full orchestration, scrape → stage → extract, shown in a web UI).

---

## Prerequisites (both methods)

- Ollama running locally with the models pulled:
  ```bash
  ollama serve                      # if not already running as a service
  ollama pull qwen2.5:14b           # or qwen2.5:7b if 14b is too heavy for your RAM
  ollama pull mistral
  ```
  Check it's up: `curl http://localhost:11434/api/tags`

- Python deps for the pipeline you're demoing:
  ```bash
  pip install -r second_staged_pipeline/requirements.txt
  ```

**Memory note:** this machine has 16GB RAM and often runs low on free memory
(PyCharm + browser + WSL/Docker VM can eat most of it). The Docker image build
below installs torch/transformers and got OOM-killed the first time it was
tried with only ~1.6GB free. Before building, close heavy apps (PyCharm,
extra browser tabs) or check free RAM with:
```bash
powershell -Command "Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory"
```
(value is in KB — want at least ~3-4GB free before `docker build`).

---

## Resetting the docs / outputs

To rerun extraction cleanly, clear only the generated outputs — **never delete
`tender_docs/`** (the input sample documents), and keep in mind everything under
`tender_docs/`, `tender_docs_new/`, and `**/output/` is already gitignored (not tracked, so this is local-only cleanup):

```bash
# clear previous extraction results (keeps input docs untouched)
rm -rf second_staged_pipeline/output/*

# if you also want to reset the Airflow-side state (scraped docs, manifests, run folders)
rm -rf airflow/data/*
```

---

## Method 1 — Without Docker (manual run)

Fastest way to show the extractor working end to end, no containers involved.

```bash
cd second_staged_pipeline
python run_pipeline.py ../tender_docs
```

Common flags for the demo:
```bash
# skip the LLM judge step for speed
python run_pipeline.py ../tender_docs --skip-judge

# use the smaller local model if 14b is too slow/heavy
python run_pipeline.py ../tender_docs --extractor-model qwen2.5:7b --judge-model mistral
```

**Output:** `second_staged_pipeline/output/YYYY-MM-DD/extractions.{json,xlsx}` — open the
`.xlsx` to show the structured fields per document.

To re-demo from scratch: delete `second_staged_pipeline/output/*` and rerun the command above.

---

## Method 2 — With Docker + Airflow (full orchestration)

Shows the scheduled DAG: `scrape_tenders → stage_new_docs → extract_tenders`.

### 1. Build the pipeline runtime image (used by the DAG's DockerOperator tasks)
```bash
docker build -t tender-pipeline:latest .
```
This installs torch/transformers/sentence-transformers — it's the heavy step. Make sure
you have free RAM (see prerequisites) before running it, and expect a few minutes.

### 2. Build and start Airflow
```bash
cd airflow
docker compose build              # builds the Airflow webserver/scheduler image
docker compose up airflow-init     # one-off: db migrate + create admin user
docker compose up -d               # start webserver + scheduler in the background
```

### 3. Open the UI
- http://localhost:8080
- login: `admin` / `admin`
- find the DAG `tender_scrape_and_extract`, unpause it (toggle on the left), then click
  the "trigger DAG" (play) button to run it immediately instead of waiting for the daily schedule.
- click into the run to watch the three tasks (`scrape_tenders`, `stage_new_docs`,
  `extract_tenders`) go green in the Graph view — `extract_tenders` runs the same
  Staged 2 pipeline as Method 1, just inside a sibling container.

### 4. Check the results
Output lands on the host at:
```
airflow/data/runs/<run-date>/output/extractions.{json,xlsx}
```
(`airflow/data/manifests/<run-date>.json` has the list of tenders scraped that day, and
`airflow/data/tender_docs_new/` has the raw scraped documents.)

### 5. Reset for a fresh demo run
```bash
cd airflow
docker compose down                # stop containers (add -v to also wipe the Postgres volume)
rm -rf data/*                      # clear scraped docs, manifests, run outputs
cd ..
docker compose -f airflow/docker-compose.yaml up -d   # or repeat step 2's `up -d`
```
Then trigger the DAG again from the UI (step 3).

### Notes
- `extract_tenders` reaches your local Ollama via `host.docker.internal:11434` — keep
  `ollama serve` running on the host while the DAG runs.
- The Airflow containers talk to your **host** Docker daemon (via the `docker.sock` mount)
  to launch `scrape_tenders`/`extract_tenders` as sibling containers — that's why step 1's
  image must be built and tagged `tender-pipeline:latest` on the host before triggering the DAG.
- To fully tear down: `docker compose -f airflow/docker-compose.yaml down -v` (also drops
  the Postgres metadata volume — next `airflow-init` starts clean).
