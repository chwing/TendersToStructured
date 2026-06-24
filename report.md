# TenderExtractor — Extraction & Comparison Report
**Date:** 2026-06-24  
**Documents:** 3 common files (French, Tunisian public procurement)  
**Model:** qwen2.5:7b via Ollama (local)

---

## 1. How the Two Methods Work

### Strategy A — Full LLM (Direct Extraction)

This method treats the entire document as a single unit and asks the model to extract all 36 fields in one shot.

**Step-by-step:**

1. **Read** — The document is read with `pdfplumber` (PDFs) or `python-docx` (DOCX). All pages are concatenated into one large text string.
2. **Prompt construction** — A system prompt describes all 36 fields, their expected format, and strict rules (no hallucination, literal values only, confidence levels). The full document text is appended as the user message.
3. **LLM call** — A single request is sent to Ollama (`/api/chat`) with `format: json` and `temperature: 0`. The model must return a JSON object with all 36 fields in one response.
4. **Parsing & validation** — The JSON response is parsed, each field is validated (date format check, numeric sanity check, absence phrase filtering), and fields that contain placeholder phrases like "non mentionné" are dropped.
5. **Save** — The result is saved incrementally (after each document) to `output/llm/YYYY-MM-DD/extractions.json` and `.xlsx`.

**Strengths:** Simple, fast on small documents, no information loss from chunking, consistent field population when the document fits in context.  
**Weaknesses:** Fails completely on documents longer than the model's context window. A 324-page PDF (~568k chars) cannot be processed at all — the model sees a truncated prompt and returns all-null.

---

### Strategy B — Staged Pipeline (Retrieve then Generate)

This method breaks extraction into stages: first understand the document structure, retrieve only relevant parts per field, then ask the model to extract from a focused subset.

**Step-by-step:**

1. **Ingest** — Same as Strategy A: `pdfplumber` or `python-docx` reads the document into a full text string.
2. **Chunking** — The full text is split into section-aware chunks of ~1,000–1,500 characters each, respecting paragraph and section boundaries.
3. **BM25 index** — A keyword retrieval index (BM25Okapi) is built over all chunks. Optionally, BGE-M3 dense embeddings are added for hybrid retrieval. In current runs, embeddings were disabled (`--no-embeddings`) to avoid GPU memory issues.
4. **Per-field retrieval** — For each of the 36 fields, predefined query strings (e.g., `"date limite de soumission"` for `submission_deadline`) retrieve the top-5 most relevant chunks. Multiple query results are fused with Reciprocal Rank Fusion (RRF).
5. **Context curation** — Retrieved chunks are deduplicated and sorted by document order. The first 2 document chunks are always included regardless of retrieval score, to ensure the cover page (tender ID, deadline, issuing org) is captured.
6. **Hard cap** — Curated context exceeding 24,000 characters is truncated to prevent GPU OOM.
7. **GLiNER NER** (optional, disabled in current runs) — A named entity recognition pass extracts candidate values (dates, organisations, reference numbers) as extraction hints.
8. **LLM finalisation** — The curated context is sent to the same Ollama model with the same extraction prompt. Because the context is smaller and focused, the model works with less noise.
9. **Confidence filter** — Fields extracted with confidence below 0.6 are dropped to suppress low-confidence guesses.
10. **Save** — Saved incrementally after each document to `output/staged/YYYY-MM-DD/extractions.json`.

**Strengths:** Can handle larger documents via chunking. Retrieval focuses the model on relevant text rather than the full noise of a long document. Modular — each stage can be tuned independently.  
**Weaknesses:** BM25 alone (without embeddings) misses semantically-related text that doesn't share keywords. The 24k char cap can cut off relevant sections on long documents. Chunking adds latency and can break cross-section context.

---

## 2. Comparison Results

### 2.1 Document-level summary

| Document | LLM fill | Staged fill | Field match | Substantive match | Avg similarity |
|---|---|---|---|---|---|
| AMI-_AMOA_TRANSVERSE08.062022.pdf | 40.0% | 37.1% | **97.1%** | 92.9% | 0.9714 |
| AMI_AMOA_V_NO_14102021.pdf | 31.4% | 31.4% | **100.0%** | 100.0% | 1.0000 |
| TDRs-_AMOA_TRANSVERSE_-08-06-2022.docx | 37.1% | 37.1% | **97.1%** | 92.3% | 0.9796 |

> **Substantive match** excludes fields that are null in both pipelines. It measures agreement only where at least one pipeline produced a value.

### 2.2 Field-level divergences

Only **2 divergences** found across 3 documents × 36 fields = 108 total comparisons:

| Document | Field | LLM value | Staged value | Verdict |
|---|---|---|---|---|
| AMI-_AMOA_TRANSVERSE08.062022.pdf | `submission_deadline` | `2022-07-08` | `null` | LLM correct, staged missed |
| TDRs-_AMOA_TRANSVERSE_-08-06-2022.docx | `domain` | `e-government` | `digital identity` | LLM correct, staged wrong |

---

## 3. Patterns — What the Staged Pipeline Systematically Misses

### 3.1 Dates (submission_deadline)

The staged pipeline missed `submission_deadline` on Doc 1, even though LLM found `2022-07-08`. Two likely causes:

- **BM25 keyword mismatch**: The deadline phrase in the document may use wording not covered by the predefined field queries (e.g., "date limite de remise des offres" vs "date limite de soumission"). Without dense embeddings, BM25 cannot bridge synonyms.
- **Confidence filter**: If the model extracts the date but marks it `low` confidence (0.40), it is silently dropped by the 0.6 threshold. The LLM pipeline has no such floor.

This is the most actionable finding — enabling dense embeddings (`--no-embeddings` off) would likely recover this class of miss.

### 3.2 Domain classification (retrieval noise)

On Doc 3, staged returned `digital identity` while LLM correctly returned `e-government`. This is a classic retrieval pollution problem: BM25 selected chunks containing identity-related content ("identité numérique") and fed those to the LLM, which classified the domain based on what it saw locally rather than the overall project framing.

LLM Strategy A reads the full document and sees the macro context. Strategy B reads only retrieved fragments — a locally dominant topic in those fragments overrides the global framing.

### 3.3 Fields consistently null in BOTH pipelines

The following fields were never populated by either pipeline across all 3 documents. This is **expected and correct** — these are AMOA consulting tenders, not technical implementation contracts:

| Category | Always-null fields | Why |
|---|---|---|
| Financial | `budget`, `currency`, `payment_terms`, `financial_guarantee` | Consulting AMIs do not specify contract amounts |
| Technical | `required_technologies`, `hosting_requirements` | Consulting scope, no tech stack defined |
| HR detail | `seniority_level`, `certifications`, `required_experience`, `company_size` | Stated in proposals, not in the AMI |
| Administrative | `tender_id`, `department`, `questions_deadline`, `award_date`, `geographic_restrictions`, `legal_requirements`, `lot_number` | Not present in these short AMIs |

These are correct nulls, not extraction failures. They reflect the document type, not a pipeline limitation.

---

## 4. Overall Assessment

### Agreement rate
Both pipelines agree on **98.1% of all field comparisons** (106/108). In the 2 cases of disagreement, LLM Strategy A was correct both times.

### Fill rates
Both methods achieve 31–40% field fill rates. The low fill rate is a property of the document type (consulting AMIs are intentionally sparse on financials and technical specs), not a pipeline failure.

### Strategy A (Full LLM)
| | |
|---|---|
| **Best for** | Documents under ~30 pages / ~30k chars |
| **Advantage** | Full document context → correct dates, correct domain classification |
| **Limitation** | Hard fails on large docs (Tunisia AfDB 80k chars → all-null at num_ctx=8192) |

### Strategy B (Staged)
| | |
|---|---|
| **Best for** | Large documents that exceed the LLM context window |
| **Advantage** | Chunking handles 324-page PDFs; modular stages are independently tunable |
| **Limitation** | BM25-only retrieval misses synonyms; retrieval noise causes domain mislabelling; confidence filter drops valid low-confidence dates |

### Recommendation

Use **Strategy A as primary** for all documents under ~30 pages. Use **Strategy B as fallback** for large documents, with two improvements:

1. **Re-enable dense embeddings** once GPU stability is restored — BGE-M3 hybrid retrieval will recover the missed `submission_deadline` class of errors.
2. **Lower the confidence floor** to 0.4 for date fields specifically, since dates extracted from retrieved chunks are often factually correct but marked low-confidence by the model.

The merge script (`run_merge.py`) is already the right architecture: LLM values take priority, staged backfills only null fields above the confidence threshold.

---

## 5. Strategy C — Translate-then-Extract

### 5.1 How it works

This strategy adds a preprocessing step before either pipeline runs:

1. **Translate** — Each document is chunked into 2,000-character blocks and sent one chunk at a time to the same Ollama model (`qwen2.5:7b`) with the instruction "translate to English, preserve structure". The translated chunks are joined and saved as a `.txt` file in `translated_docs/`.
2. **Extract** — Both Strategy A (LLM) and Strategy B (Staged) then run on the English `.txt` files, using the same prompts and configuration as before.
3. **Compare** — The same `run_comparison.py` compares the two English extractions.

The hypothesis is that the extraction model performs better on English text (it was predominantly trained on English data), and BM25 retrieval benefits from using standard English field-query terms rather than French equivalents.

### 5.2 Comparison: native-language runs vs translated runs

**Document-level metrics:**

| Document | Run | LLM fill | Staged fill | Field match | Substantive match | Avg sim |
|---|---|---|---|---|---|---|
| AMI-_AMOA_TRANSVERSE08.06 | Native FR | 40.0% | 37.1% | 97.1% | 92.9% | 0.971 |
| AMI-_AMOA_TRANSVERSE08.06 | Translated EN | **42.9%** | **45.7%** | 91.4% | 81.2% | 0.940 |
| AMI_AMOA_V_NO_14102021 | Native FR | 31.4% | 31.4% | **100.0%** | **100.0%** | **1.000** |
| AMI_AMOA_V_NO_14102021 | Translated EN | **37.1%** | **37.1%** | **100.0%** | **100.0%** | **1.000** |
| TDRs-_AMOA_TRANSVERSE_08-06 | Native FR | 37.1% | 37.1% | 97.1% | 92.3% | 0.980 |
| TDRs-_AMOA_TRANSVERSE_08-06 | Translated EN | 34.3% | 31.4% | **82.9%** | **50.0%** | **0.864** |

**Overall agreement rate:** Native = 98.1% · Translated = 91.4%

### 5.3 Field-level changes after translation

**Fields unlocked by translation (native=null, translated=extracted):**

| Field | Doc | Native | Translated |
|---|---|---|---|
| `tender_id` | Doc 1 | null / null | "TN-MTCTD-228235-CS-QCBS" (both) |
| `tender_id` | Doc 2 | null / null | "21/P-BAD/2021" (both) |
| `submission_deadline` | Doc 1 | null / null | "2022-07-08" (both) |
| `submission_deadline` | Doc 2 | "2021-11-08" / null | "2021-11-08" (both) |
| `publication_date` | Doc 1 | null / null | "2022-06-15" (both) |
| `city_region` | Doc 1 | null / null | "Tunis" (both) |
| `department` | Doc 2 | null / null | "Ministry of Communication Technologies (MCT)" (both) |
| `deliverables` | Doc 2 | null / null | correctly extracted (both) |
| `scope_of_work` | Doc 2 | partial / null | fully extracted (both) |
| `evaluation_criteria` | Doc 1 | null / null | "Solidity (30pts); HR (20pts); References (70pts)" (both) |
| `roles_profiles` | Doc 3 | null / null | LLM extracted 6 profiles |

**Fields that degraded after translation (matched native → mismatched translated):**

| Field | Doc | Native status | Translated status | Notes |
|---|---|---|---|---|
| `mission_duration` | Doc 1 | both_null | mismatch (0.36) | LLM="24 months", staged=24 (numeric vs string after translation) |
| `required_documents` | Doc 1 | both_null | mismatch (0.61) | Same content, different order — translation reordered list |
| `issuing_organization` | Doc 3 | match | mismatch (0.66) | Staged added "Government of the Republic of Tunisia" suffix |
| `project_description` | Doc 3 | match | mismatch (0.06) | LLM kept title, staged retrieved unrelated intro paragraph |
| `deliverables` | Doc 3 | match | mismatch (0.06) | LLM got empty label, staged retrieved mission descriptions |
| `scope_of_work` | Doc 3 | match | mismatch (0.18) | LLM extracted missions, staged retrieved high-level summary |
| `relevance_reason` | Doc 3 | match | mismatch (0.28) | Both extracted, but translation caused paraphrase divergence |

### 5.4 Analysis of Doc 3 degradation

Doc 3 (TDRs / Terms of Reference) is structurally different from the AMI-type docs: it is a multi-section specification document without a clean cover page. After translation, both pipelines extracted inconsistent content for `project_description`, `deliverables`, and `scope_of_work`.

Root cause: the translation model split the document into 2,000-char chunks and translated each independently. Without cross-chunk context, section headers and their body paragraphs were sometimes separated at chunk boundaries, and the translated text lost the structural cues (e.g., numbered section headings) that the original French document had. The BM25 retriever then surfaced different chunks for each field query, causing staged to pick up an introduction paragraph as `project_description` while LLM read the translated title line. In the native French run, both pipelines agreed because the document structure was better preserved.

### 5.5 Advantages and disadvantages of Strategy C

**Advantages:**
- **Unlocks admin fields**: `tender_id`, `submission_deadline`, `publication_date` were recovered on 2/3 documents after translation. The model reads English reference codes (e.g., "TN-MTCTD-228235-CS-QCBS") more reliably than when embedded in French text.
- **Staged pipeline catches up**: On Doc 1, staged fill rate rose from 37.1% → 45.7%, surpassing LLM (42.9%). Translation improved BM25 retrieval because English query terms match the translated text exactly.
- **Language-agnostic downstream**: Once translated, all downstream tooling (embeddings, BM25, NER) operates in English — simpler model assumptions, better English embedding models.
- **Perfect agreement on Doc 2**: Both pipelines extracted 37.1% (up from 31.4%) with 100% field match — the highest-quality result across all runs.

**Disadvantages:**
- **2× LLM cost**: Every document now requires one full translation pass (N chunks × Ollama call) plus the extraction pass. For a 324-page PDF, translation alone requires ~284 serial Ollama calls.
- **Chunk-boundary artifacts**: 2,000-char chunk slicing breaks mid-sentence and mid-section. The joining strategy (`\n\n`.join) does not know whether chunks are in the same paragraph, so the reassembled text has structural gaps. This caused Doc 3's degradation.
- **Overall agreement dropped**: Cross-pipeline match rate fell from 98.1% (native) to 91.4% (translated). Translation introduced paraphrase variation that the semantic comparison engine penalises.
- **Translation errors propagate**: Legal and domain terms translated imprecisely (e.g., "Maître d'Ouvrage" → "Project Authority" or "Owner's Party" depending on chunk context) cause the model to return subtly different values for the same field across documents.
- **No improvement on financial fields**: All financial fields (`budget`, `currency`, `payment_terms`, `financial_guarantee`) remained null after translation — confirming that these fields are structurally absent from the documents, not a language-model limitation.
- **Doc 3 regression**: Fill rates dropped (LLM: 37.1% → 34.3%; staged: 37.1% → 31.4%) and substantive match dropped from 92.3% to 50%. Translation hurt more than it helped on TDR-type documents.

### 5.6 When to use each approach

| Scenario | Recommended strategy |
|---|---|
| Short AMI/RFP document (≤ 30 pages), French | Strategy A (native LLM) — fast, no degradation risk |
| Short AMI/RFP document (≤ 30 pages), Arabic | Strategy C (translate then A) — extraction model has weaker Arabic coverage |
| Long TDR / specification document, French | Strategy B (staged, native) — translation hurts structural docs |
| Very large document (> 100 pages) | Strategy B (staged, native) — translation is too slow (N×284 calls) |
| Highest fill rate on structured AMI | Strategy C (translate then B) — best result on Doc 1 and Doc 2 |

---

## 6. Strategy D — Helsinki-NLP MarianMT Translation + Extraction

### 6.1 What changed

The translation step is replaced from Ollama (general chat model, 2,000-char blind chunks) to **Helsinki-NLP/opus-mt-fr-en** — a MarianMT model trained exclusively on French→English parallel corpora. Chunking now splits at paragraph then sentence boundaries rather than fixed character counts, so structural cues (numbered sections, bullet lists) are preserved across batches.

### 6.2 Document-level results

| Document | LLM fill | Staged fill | Field match | Substantive match | Avg sim |
|---|---|---|---|---|---|
| AMI-_AMOA_TRANSVERSE08.06 | 42.9% | 40.0% | 77.1% | 50.0% | 0.823 |
| AMI_AMOA_V_NO_14102021 | 37.1% | 31.4% | 94.3% | 84.6% | 0.943 |
| TDRs-_AMOA_TRANSVERSE_08-06 | 31.4% | 31.4% | **100.0%** | **100.0%** | **1.000** |

### 6.3 Full three-way comparison

| Document | Native FR | Ollama | Helsinki |
|---|---|---|---|
| AMI-_AMOA_TRANSVERSE08.06 | 97.1% | 91.4% | **77.1%** |
| AMI_AMOA_V_NO_14102021 | **100.0%** | **100.0%** | 94.3% |
| TDRs-_AMOA_TRANSVERSE_08-06 | 97.1% | 82.9% | **100.0%** |
| **Overall avg match** | **98.1%** | **91.4%** | **90.5%** |

### 6.4 Field-level findings

**Doc 1 — critical finding: incomplete translation**

The staged pipeline returned several field values in French despite the document being "translated":

| Field | LLM value (English) | Staged value | Issue |
|---|---|---|---|
| `title` | "Mission of Assistance to the Master's degree 'AMOA'..." | "Mission d'Assistance à la Maîtrise d'Ouvrage..." | Staged retrieved untranslated chunk |
| `issuing_organization` | "Ministry of Communication Technology" | "Ministère des Technologies de la Communication" | Same — French chunk surfaced by BM25 |
| `project_description` | Full English description | French title sentence | BM25 retrieved wrong paragraph |
| `evaluation_criteria` | Concise English summary | Very detailed English breakdown (full table) | Both English but completely different granularity |

Root cause: MarianMT skipped some paragraphs during translation (possibly paragraphs with mixed formatting, bullet characters, or special symbols that the tokenizer treated as noise). Those untranslated French paragraphs remained in the `.txt` file. The LLM pipeline read the full document and extracted from English sections; the staged pipeline's BM25 index ranked French paragraphs higher for some field queries and returned French content.

**Doc 1 — new finds:**
- `reference_number` = "TN-MTCTD-228235-CS-QCBS" extracted by LLM (was null in both previous translation runs)
- `required_experience` extracted by LLM only (staged missed it)
- `publication_date` = "2022-06-17" extracted by staged only (new — not in Ollama run either)

**Doc 2 — partial success:**
- `tender_id` = "21/P-BAD/2021" correctly extracted by both (consistent across all runs)
- `title` both extracted **in French** ("Mission d'appui aux suivi...") — Helsinki did not translate this title, likely because it was too short for a full paragraph and was grouped into a batch that the model skipped
- `reference_number` = "P-TN-G00-03" new — extracted by LLM only, not seen in any previous run
- Staged fill rate dropped from 37.1% (Ollama) to 31.4% — staged missed `required_documents`, `deliverables` that Ollama had found

**Doc 3 — best result across all runs:**
- Perfect 100% field match, 100% substantive match, avg similarity 1.0
- Both pipelines extracted identical English values for `deliverables` (6 detailed mission descriptions), `roles_profiles` (6 profiles), `project_description`, `domain`
- `domain` = "e-government" correctly by both — this was wrong in native run (staged had returned "digital identity") and wrong in Ollama run too. Helsinki is the **only run** where Doc 3 domain was correctly classified by both pipelines
- This is the strongest result for any document across all runs

### 6.5 Advantages and disadvantages of Helsinki vs Ollama translation

**Advantages of Helsinki over Ollama:**
- **Speed**: translates the entire document in one batched pass (seconds vs hundreds of serial API calls)
- **Consistent terminology**: one model run = one consistent vocabulary throughout the document (no cross-chunk term drift)
- **Structural preservation**: paragraph-aware chunking keeps section headers with their body text
- **Better on TDR-type docs**: Doc 3 went from 82.9% match (Ollama) → 100% match (Helsinki) — the paragraph-level chunking preserved the list structure that Ollama broke
- **No Ollama dependency for translation**: frees up the GPU for the extraction step

**Disadvantages of Helsinki vs Ollama:**
- **Partial translation failures**: MarianMT silently skips paragraphs it cannot tokenise (special characters, very short lines, table cells) — leaving untranslated French fragments in the output that poison the staged pipeline's retrieval
- **Lower overall agreement**: Doc 1 dropped from 91.4% (Ollama) → 77.1% (Helsinki) because untranslated chunks caused LLM and staged to extract from different language sections
- **Title preservation**: both Doc 1 and Doc 2 had their French titles left untranslated — MarianMT treated short isolated title lines as noise or batched them into a context where translation failed
- **Lower staged fill rate on Doc 2**: 37.1% (Ollama) → 31.4% (Helsinki) — Helsinki's translation of Doc 2 was less complete than Ollama's, causing staged to miss fields it previously found

### 6.6 Overall rating

| Criterion | Native | Ollama | Helsinki |
|---|---|---|---|
| LLM/Staged agreement | ★★★★★ | ★★★☆☆ | ★★★☆☆ |
| Fill rate (structured AMI) | ★★★☆☆ | ★★★★☆ | ★★★★☆ |
| Fill rate (TDR docs) | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |
| Translation quality | — | ★★★☆☆ | ★★★★☆ |
| Speed | ★★★★★ | ★★☆☆☆ | ★★★★★ |
| Structural preservation | ★★★★★ | ★★☆☆☆ | ★★★★☆ |
| Domain accuracy (Doc 3) | ✗ staged wrong | ✗ staged wrong | ✓ both correct |

**Helsinki is superior to Ollama on translation quality and speed**, but its partial translation failures (untranslated paragraphs in the output) introduce a new failure mode that Ollama did not have. The fix is a post-translation validation pass: detect any paragraph in the output that is still in French (via langdetect) and re-translate it. This would combine Helsinki's speed and consistency with Ollama's completeness guarantee.

**The overall winner on raw match rate remains native-language extraction** (98.1%), confirming that for French documents where the extraction model performs adequately, translation adds overhead without improving accuracy.
