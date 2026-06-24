# TenderExtractor

Structured extraction from public procurement tender documents (French/Arabic, PDF/DOCX). Two extraction strategies, a merge mode, and a comparison tool.

---

## Overview

TenderExtractor reads tender documents and outputs a structured JSON/Excel file with 36 fields per document — deadlines, budgets, evaluation criteria, required profiles, and more. It targets North African tenders (Tunisia, Morocco, Algeria) funded by World Bank and African Development Bank.

Two strategies are available:

| | Strategy A — LLM | Strategy B — Staged |
|---|---|---|
| How | Full document → single LLM prompt | Chunk → BM25 + BGE-M3 retrieval → LLM on curated context |
| Strength | High precision, no hallucination | Higher recall on mid-size docs |
| Weakness | Fails on very large docs (OOM) | Can invent values at low confidence |
| Script | `run_llm_extractor.py` | `run_staged_extractor.py` |

A third script (`run_merge.py`) combines both: LLM output as primary, staged backfills only the null fields that clear a confidence threshold.

---

## Requirements

- Python 3.12
- [Ollama](https://ollama.com/) running locally with a model pulled (tested with `qwen2.5:7b`)
- Dependencies: `pip install -r requirements.txt`

Pull the model once:
```bash
ollama pull qwen2.5:7b
```

---

## Usage

### Strategy A — Full LLM

```bash
python run_llm_extractor.py tender_docs --model qwen2.5:7b
```

### Strategy B — Staged pipeline

```bash
python run_staged_extractor.py tender_docs --model qwen2.5:7b
```

### Merge (recommended)

Run both strategies first, then merge:

```bash
python run_merge.py
```

This keeps every LLM value and backfills nulls from the staged pipeline where confidence ≥ 0.6.

### Compare strategies

```bash
python run_comparison.py
```

Outputs a semantic field-by-field comparison report to `output/comparison/YYYY-MM-DD/`.

---

## Key options

| Flag | Default | Description |
|---|---|---|
| `--model` | `mistral` | Ollama model name |
| `--provider` | `ollama` | `ollama` or `openrouter` |
| `--num-ctx` | `32768` | Ollama context window (tokens). Raise for large docs. |
| `--timeout` | `600` | Per-request timeout in seconds |
| `--min-confidence` | `0.6` | (Staged/Merge) Drop fields below this confidence |
| `--no-embeddings` | off | Use BM25 only, skip BGE-M3 (faster, lower quality) |
| `--no-gliner` | off | Skip GLiNER NER stage |
| `--top-k` | `5` | Chunks per field for retrieval |

---

## Output

All runs write to `output/{strategy}/YYYY-MM-DD/`:

```
output/
  llm/2026-06-23/
    extractions.json
    extractions.xlsx
  staged/2026-06-23/
    extractions.json
    extractions.xlsx
  merged/2026-06-23/
    extractions.json
    extractions.xlsx
  comparison/2026-06-23/
    report.json
    report.xlsx
```

Processed files are copied to `treated_docs/` to avoid reprocessing on re-runs.

---

## Extracted fields (36 total)

**Identity:** `tender_id`, `title`, `reference_number`

**Issuing party:** `issuing_organization`, `department`, `country`, `city_region`

**Dates:** `publication_date`, `submission_deadline`, `questions_deadline`, `award_date`

**Financial:** `budget`, `currency`, `payment_terms`, `financial_guarantee`

**Technical scope:** `project_description`, `domain`, `required_technologies`, `deliverables`, `scope_of_work`, `hosting_requirements`

**Team / HR:** `num_profiles`, `roles_profiles`, `seniority_level`, `certifications`, `mission_duration`

**Eligibility:** `required_experience`, `company_size`, `required_documents`, `geographic_restrictions`, `legal_requirements`

**Evaluation:** `evaluation_criteria`, `lot_number`

**Relevance:** `is_tech_relevant`, `relevance_reason`

Each field is returned as `{"value": ..., "confidence": 0.40 | 0.65 | 0.90}`.

---

## Project structure

```
src/
  extractor/
    fields.py          # ALL_FIELDS list
    models.py          # Pydantic models (TenderExtraction, ExtractedField)
    document_reader.py # PDF (pdfplumber) + DOCX reader
    output.py          # JSON + Excel serialization
  llm_pipeline/
    extractor.py       # LLMPipelineExtractor (Strategy A)
    prompt.py          # System + user prompt templates
  staged_pipeline/
    ingestor.py        # Docling → pdfplumber fallback
    chunker.py         # Section-aware chunking
    retriever.py       # BM25 + BGE-M3 hybrid retrieval
    field_queries.py   # Per-field query strings
    gliner_extractor.py# Optional GLiNER NER hints
    llm_finalizer.py   # Sends curated context to LLM
    pipeline.py        # StagedPipelineExtractor (Strategy B)
run_llm_extractor.py
run_staged_extractor.py
run_merge.py
run_comparison.py
report.md              # Comparison report (Claude ground truth vs both pipelines)
```

---

## Known limitations

- **Very large documents (300+ pages):** qwen2.5:7b runs out of memory at `--num-ctx 32768`. The staged pipeline handles these better via chunking, but Docling's OCR stage also OOMs on page images. Use `--no-embeddings` to reduce memory pressure, or switch to a cloud model via `--provider openrouter`.
- **Non-tender documents:** Bank appraisal reports, policy documents, and similar non-procurement files correctly return all-null (they are not tenders).
- **Arabic documents:** Language detection and extraction are supported but accuracy is lower than French.