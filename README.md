# TenderExtractor

Structured extraction from public procurement tender documents (French/Arabic, PDF/DOCX).
Targets North African tenders (Tunisia, Morocco, Algeria)

Outputs a structured JSON/Excel file with 36 fields per document — deadlines, budgets, evaluation criteria, required profiles, and more.

---

## Strategies

Three extraction strategies are available, ordered from simplest to most powerful:

| | Full LLM | Staged 1 | Staged 2 |
|---|---|---|---|
| **Folder** | `full_llm/` | `first_staged_pipeline/` | `second_staged_pipeline/` |
| **Best for** | Short docs (< 10 000 chars) | Mid-size docs | All doc sizes |
| **Pre-processing** | None | GLiNER NER + BM25/BGE-M3 retrieval | XLM-RoBERTa NER + multilingual-e5 embeddings + section ranking |
| **Context compression** | No | Yes — field-by-field retrieval | Yes — semantic retrieval + scoring |
| **LLM validation** | No | No | Yes — LLM Judge + self-correction |
| **Token cost** | High on large docs | Medium | Low on large docs |
| **Quality** | Good | Good | Best |

### When to use which

- **Full LLM** — doc is short (a few pages). No overhead, direct and fast.
- **Staged 1** — doc is mid-size, you want retrieval-augmented extraction without heavy transformer models.
- **Staged 2** — doc is long or quality matters most. Compresses context before the LLM call, then validates the output with a judge model.

> Staged 2 also has a built-in router: docs under `--staged-threshold` (default 10 000 chars) skip the compression stages and go directly to the LLM, so you get the best of both approaches automatically.

---

## Requirements

- Python 3.12
- [Ollama](https://ollama.com/) running locally

Pull the recommended models once:
```bash
ollama pull qwen2.5:14b   # extraction model (Staged 2)
ollama pull mistral        # judge model (Staged 2) / default for Full LLM and Staged 1
```

Install dependencies (each folder has its own `requirements.txt`):
```bash
pip install -r full_llm/requirements.txt
pip install -r first_staged_pipeline/requirements.txt   # heavier — includes GLiNER, sentence-transformers
pip install -r second_staged_pipeline/requirements.txt  # heaviest — includes XLM-RoBERTa, multilingual-e5
```

---

## Strategy 1 — Full LLM

**How it works:** reads the document, builds a single prompt with the full text, sends it to the LLM, and parses the JSON response. No pre-processing.

```
Document → read text → LLM prompt → JSON output
```

**Run:**
```bash
python full_llm/run_extractor.py ./tender_docs
```

**Common options:**
```bash
# change model
python full_llm/run_extractor.py ./tender_docs --model qwen2.5:14b

# use OpenRouter instead of local Ollama
python full_llm/run_extractor.py ./tender_docs --provider openrouter --api-key sk-...

# larger context window for bigger docs
python full_llm/run_extractor.py ./tender_docs --num-ctx 65536

# drop low-confidence fields
python full_llm/run_extractor.py ./tender_docs --min-confidence 0.4
```

**Output:** `full_llm/output/YYYY-MM-DD/extractions.{json,xlsx}`

**Limitation:** large documents hit the context window limit and return all-null. Switch to Staged 2 for those.

---

## Strategy 2 — Staged 1

**How it works:** a multi-step pipeline that pre-processes the document before the LLM call.

```
Document
  → Stage 1: ingest (Docling / pdfplumber)
  → Stage 2: chunk into sections
  → Stage 3: GLiNER NER hints (named entities)
  → Stage 4: BM25 + BGE-M3 retrieval — per-field context selection
  → Stage 5: LLM on curated context → JSON output
```

The key idea: instead of sending the whole document to the LLM, each field gets only the most relevant chunks retrieved by keyword (BM25) and semantic (BGE-M3) search. This dramatically reduces the prompt size for large documents.

**Run:**
```bash
cd first_staged_pipeline
python run_staged_extractor.py ../tender_docs --model qwen2.5:14b
```

**Common options:**
```bash
# faster — skip BGE-M3 embeddings, BM25 only
python run_staged_extractor.py ../tender_docs --no-embeddings

# skip GLiNER NER hints
python run_staged_extractor.py ../tender_docs --no-gliner

# more chunks per field (higher recall, more tokens)
python run_staged_extractor.py ../tender_docs --top-k 8

# drop fields below confidence threshold
python run_staged_extractor.py ../tender_docs --min-confidence 0.6
```

**Output:** `first_staged_pipeline/output/staged/YYYY-MM-DD/extractions.{json,xlsx}`

**Also available in this folder:**

Merge LLM + Staged 1 results (LLM primary, Staged 1 backfills nulls):
```bash
python run_merge.py
```

Compare the two outputs side by side:
```bash
python run_comparison.py
```

---

## Strategy 3 — Staged 2

**How it works:** a 6-stage hybrid pipeline with stronger models and a validation loop.

```
Document
  → Stage 1: document processing (PyMuPDF / pdfplumber / python-docx)
  → Stage 2: classical extraction (regex — dates, references, budgets)
  → Stage 3: XLM-RoBERTa NER + multilingual-e5 embeddings  ┐
  → Stage 4: context compression — section ranking + semantic retrieval ┘ (skipped if doc < threshold)
  → Stage 5: extraction LLM (Qwen2.5) → structured JSON
  → Stage 6: LLM Judge (Mistral) — validates output, flags issues, triggers self-correction
```

**The router:** after Stage 1, doc length is compared against `--staged-threshold` (default 10 000 chars).
- Short doc → Stages 3 and 4 are skipped, full text goes straight to Stage 5.
- Long doc → full pipeline runs, compressing context before Stage 5.

This means Staged 2 is efficient on short docs and powerful on long docs automatically.

**Run:**
```bash
cd second_staged_pipeline
python run_pipeline.py ../tender_docs
```

**Common options:**
```bash
# change the length-based routing threshold (chars)
python run_pipeline.py ../tender_docs --staged-threshold 15000

# different extraction and judge models
python run_pipeline.py ../tender_docs --extractor-model qwen2.5:14b --judge-model mistral

# skip the judge entirely (faster)
python run_pipeline.py ../tender_docs --skip-judge

# use OpenRouter for the extraction LLM
python run_pipeline.py ../tender_docs \
  --extractor-provider openrouter --extractor-api-key sk-... \
  --extractor-model anthropic/claude-3-5-haiku

# tune semantic retrieval
python run_pipeline.py ../tender_docs --min-similarity 0.3 --max-context-chars 12000

# disable transformer models (much faster, lower quality)
python run_pipeline.py ../tender_docs --no-ner --no-embeddings
```

**Output:** `second_staged_pipeline/output/YYYY-MM-DD/extractions.{json,xlsx}`

---

## Extracted fields (36 total)

Each field is returned as `{"value": ..., "confidence": 0.40 | 0.65 | 0.90}`.

| Group | Fields |
|---|---|
| Identity | `tender_id`, `title`, `reference_number` |
| Issuing party | `issuing_organization`, `department`, `country`, `city_region` |
| Dates | `publication_date`, `submission_deadline`, `questions_deadline`, `award_date` |
| Financial | `budget`, `currency`, `payment_terms`, `financial_guarantee` |
| Technical scope | `project_description`, `domain`, `required_technologies`, `deliverables`, `scope_of_work`, `hosting_requirements` |
| Team / HR | `num_profiles`, `roles_profiles`, `seniority_level`, `certifications`, `mission_duration` |
| Eligibility | `required_experience`, `company_size`, `required_documents`, `geographic_restrictions`, `legal_requirements` |
| Evaluation | `evaluation_criteria`, `lot_number` |
| Relevance | `is_tech_relevant`, `relevance_reason` |

---

## Project structure

```
tendersToStructured/
│
├── full_llm/                        # Strategy 1 — Full LLM
│   ├── run_extractor.py
│   └── src/
│       ├── extractor/               # document reader, models, output
│       └── llm_pipeline/            # LLM client + prompt templates
│
├── first_staged_pipeline/           # Strategy 2 — Staged 1
│   ├── run_staged_extractor.py
│   ├── run_llm_extractor.py         # LLM-only runner (kept for comparison)
│   ├── run_merge.py                 # merge LLM + staged outputs
│   ├── run_comparison.py            # field-by-field comparison report
│   └── src/
│       ├── extractor/               # shared models + document reader
│       ├── llm_pipeline/            # LLM client + prompts
│       └── staged_pipeline/         # chunker, retriever, GLiNER, pipeline
│
├── second_staged_pipeline/          # Strategy 3 — Staged 2
│   ├── run_pipeline.py
│   └── src/
│       ├── stage1_document.py       # document processing
│       ├── stage2_classical.py      # regex extraction
│       ├── stage3_transformer.py    # XLM-RoBERTa NER + e5 embeddings
│       ├── stage4_context.py        # context compression
│       ├── stage5_extractor.py      # extraction LLM
│       ├── stage6_judge.py          # LLM judge + self-correction
│       ├── pipeline.py              # orchestrator + length router
│       ├── models.py                # Pydantic models
│       └── output.py                # JSON + Excel serialization
│
└── tender_docs/                     # input documents (PDF / DOCX)
```
