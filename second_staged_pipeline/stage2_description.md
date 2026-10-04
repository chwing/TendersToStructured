# Staged 2 Pipeline — How It Works

6-stage hybrid AI system for extracting structured data from tender documents (French/Arabic, PDF/DOCX).

---

## Overview

```
Document
  → Stage 1: Document Processing        (PyMuPDF / pdfplumber / python-docx)
  → Stage 2: Classical Extraction       (regex — dates, references, budgets)
  → [Router: short doc? skip to Stage 5]
  → Stage 3: Transformer NLP            (XLM-RoBERTa NER + multilingual-e5 embeddings)
  → Stage 4: Context Compression        (~95% token reduction via semantic retrieval)
  → Stage 5: Extraction LLM             (Qwen2.5 — structured JSON output)
  → Stage 6: LLM Judge + self-correction (Mistral — validation loop, up to 2 rounds)
  → Output: extractions.json + extractions.xlsx
```

---

## Stage 1 — Document Processing

**File:** `src/stage1_document.py`

Reads the input file and prepares it for the rest of the pipeline:

- **PDF** → PyMuPDF (primary) or pdfplumber (fallback)
- **DOCX** → python-docx
- Detects document language (French / Arabic / English)
- Splits the text into **chunks** (sections / paragraphs)

Output: `ProcessedDocument` with `.text` (full raw text) and `.chunks` (list of sections).

---

## Stage 2 — Classical Extraction

**File:** `src/stage2_classical.py`

Runs **regex and rule-based patterns** over the raw text before any LLM is involved. Extracts:

- Tender reference numbers
- Dates and submission deadlines
- Budget amounts and currencies
- Contact information

These facts are reliable and deterministic. They are collected into a `ClassicalFacts` object and later passed as **hints** to Stage 5 alongside NER results, so the LLM does not have to re-discover them from scratch.

---

## Router — Length-Based Routing

After Stage 2, the pipeline checks `len(doc.text)` against `--staged-threshold` (default: **10 000 chars**):

| Doc length | Path |
|---|---|
| < 10 000 chars | Stages 3 and 4 are **skipped** — full raw text goes directly to Stage 5 |
| ≥ 10 000 chars | Full pipeline continues through Stages 3 and 4 |

This means the pipeline is efficient on short documents and powerful on long ones automatically. The threshold can be changed with `--staged-threshold <N>`.

---

## Stage 3 — Transformer NLP *(long docs only)*

**File:** `src/stage3_transformer.py`

Runs two transformer models over the document chunks:

1. **XLM-RoBERTa NER** (`Davlan/xlm-roberta-base-wikiann-ner`)
   - Extracts named entities (organizations, locations) from each chunk
   - Results become additional hints for Stage 5

2. **multilingual-e5 embeddings** (`intfloat/multilingual-e5-base`)
   - Encodes every chunk into a dense vector
   - Used by Stage 4 for semantic similarity search

Both models support French and Arabic natively.

---

## Stage 4 — Context Compression *(long docs only)*

**File:** `src/stage4_context.py`

Takes all chunks and their embeddings, then selects only the most relevant content for the LLM:

1. Scores each chunk by **semantic similarity** to a domain profile string ("AI software engineering data analytics cloud")
2. Retrieves the highest-scoring chunks per field using vector similarity
3. Drops chunks below `--min-similarity` threshold (default: 0.25)
4. Caps the output at `--max-context-chars` (default: 16 000 chars)

Result: a compressed context that typically achieves **~95% token reduction** vs. the full document, while preserving the most relevant sections.

---

## Stage 5 — Extraction LLM

**File:** `src/stage5_extractor.py`  
**Model:** `qwen2.5:14b` (default)

Sends the compressed context (or full text for short docs) to the LLM with a structured prompt:

- Lists all **36 fields** to extract (dates, budgets, profiles, deliverables, etc.)
- Prepends **hints** from Stage 2 (regex facts) and Stage 3 (NER entities) so the LLM can confirm rather than discover
- Expects a JSON response where each field is `{"value": ..., "confidence": 0.40 | 0.65 | 0.90}`

Confidence levels:
- `0.90` — high: explicitly stated in the document
- `0.65` — medium: inferred or partially stated
- `0.40` — low: guessed or uncertain

---

## Stage 6 — LLM Judge + Self-Correction

**File:** `src/stage6_judge.py`  
**Model:** `mistral` (default)

Validates the extraction from Stage 5. **Skipped automatically if average confidence ≥ 0.80** (the extraction is already high-quality).

### Validation
The judge reviews the extraction against the compressed context and checks for:
- Hallucinated values (stated in extraction but not in the document)
- Missing mandatory fields
- Low-confidence answers that could be improved

### Self-Correction Loop
If the judge flags issues:
1. High-severity issues are collected
2. Stage 5 is called again with the **previous extraction + judge feedback** as additional context
3. The loop repeats up to **2 revision rounds**
4. Stops early if the judge approves or no high-severity issues remain

---

## Output

**File:** `src/output.py`

After the pipeline completes, results are written to `output/YYYY-MM-DD/`:

- `extractions.json` — full structured output with all fields and confidence scores
- `extractions.xlsx` — tabular view, one row per document

---

## Configuration

All key parameters can be tuned via CLI:

```bash
# Change routing threshold (chars)
python run_pipeline.py ../tender_docs --staged-threshold 15000

# Different models
python run_pipeline.py ../tender_docs --extractor-model qwen2.5:14b --judge-model mistral

# Skip the judge entirely
python run_pipeline.py ../tender_docs --skip-judge

# Tune context compression
python run_pipeline.py ../tender_docs --min-similarity 0.3 --max-context-chars 12000

# Disable transformer models (faster, lower quality)
python run_pipeline.py ../tender_docs --no-ner --no-embeddings
```
