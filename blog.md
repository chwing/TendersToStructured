# Building TenderExtractor: From Raw PDFs to Structured Data (and Everything That Went Wrong)

*A technical write-up on extracting structured information from French and Arabic public procurement documents using local LLMs, retrieval pipelines, and translation models — and the bugs, crashes, and dead ends along the way.*

---

## The Problem

Public tenders in Tunisia and across the MENA region are published as PDFs and Word documents — written in French or Arabic, following no standard format, ranging from 3 pages to 324 pages. Each document potentially contains dozens of structured fields: tender ID, issuing organisation, submission deadline, budget, required profiles, evaluation criteria, and more.

Reading these manually is slow. The goal was to automate extraction: feed a document in, get a structured JSON out, with every field populated from the document content — no guessing, no hallucination.

We settled on 36 target fields covering administration, financials, technical requirements, HR profiles, and eligibility criteria. The extraction model: **qwen2.5:7b running locally via Ollama** — no cloud APIs, no data leaving the machine.

---

## Strategy A: Just Ask the LLM

The first approach was the simplest one imaginable. Read the document, dump the full text into a prompt, and ask the model to return all 36 fields as JSON.

```
System: You are a tender extraction assistant. Extract the following fields...
User: [full document text — 40,000 characters]
```

The model returns a JSON blob. We parse it, validate each field (date format checks, dropping "non mentionné" placeholders, sanity-checking numeric values), and save it to disk.

**It worked.** For a 7-page French AMI document, the model extracted the title, issuing organisation, submission deadline, and scope of work correctly — in under 30 seconds.

The fill rate was 31–40% across our test documents. That sounds low, but it reflects the documents themselves: consulting AMI tenders genuinely don't contain budget figures, technology requirements, or HR seniority levels. Empty fields are not extraction failures — they're correct.

**The wall we hit:** a 324-page PDF from CNSS. At ~568,000 characters, the document is approximately 140,000 tokens — roughly 17× the model's 8,192-token context window. The model silently truncated the input and returned all nulls. No error, no warning. Just 36 empty fields.

---

## Strategy B: Retrieve, Then Extract

The 324-page document forced a rethink. Instead of feeding the full text to the model, we built a pipeline that retrieves only the relevant sections before calling the model.

**The stages:**

1. **Ingest** — pdfplumber extracts text page by page.
2. **Chunk** — The full text is split into ~1,000–1,500 character chunks, respecting paragraph boundaries.
3. **Index** — A BM25Okapi keyword index is built over all chunks.
4. **Retrieve** — For each of the 36 fields, predefined query strings (e.g., `"date limite de soumission"` for the submission deadline field) retrieve the top-5 most relevant chunks. Multiple queries per field are fused with Reciprocal Rank Fusion.
5. **Curate** — Retrieved chunks are deduplicated and assembled into a focused context. The first 2 chunks of the document are always included to guarantee the cover page (which holds the tender ID, deadline, and issuing organisation) is present.
6. **Extract** — The curated context — not the full document — is sent to the same Ollama model with the same extraction prompt.
7. **Filter** — Fields extracted with confidence below 0.6 are dropped to suppress guesses.

This lets us process arbitrarily long documents: a 324-page PDF gets chunked into ~400 chunks, BM25 selects the 5 most relevant per field, and the model sees a focused 24,000-character context instead of 568,000.

---

## The Crash That Changed Everything

The first test of Strategy B ended with Python dying mid-run — no traceback, no error message, just silence.

The culprit was **Docling**, a document parsing library we were using as the primary PDF reader. For the 324-page CNSS document, Docling allocated image buffers for each page, hit a native C++ memory limit, threw `std::bad_alloc`, and killed the Python process. Since we only saved results at the end of the loop, every extraction that had succeeded before the crash was lost.

Two fixes:

1. **Remove Docling entirely.** pdfplumber handles everything Docling was doing, without the native memory risk. The 324-page PDF processed cleanly with pdfplumber: 568,000 characters extracted from 324 pages.

2. **Incremental saves.** After every successful document, write the extraction to disk immediately. If anything crashes, we keep what we had.

---

## The Hang That Took Hours to Diagnose

After fixing the crash, the staged pipeline started a new kind of failure: it would launch, print one line, and then stop. Not crash — stop. No output, no heartbeat, and Ctrl+C wouldn't kill it.

We wrote a small import trace script that printed a message after importing each module, then ran it to find the last successful print before the hang. The output stopped at:

```
importing src.extractor.output ...
```

The culprit: `import pandas as pd` at the top of `output.py`. On Windows, pandas triggers a chain of imports that eventually loads a native library that calls into the Windows thread scheduler — and with certain GPU driver states active from Ollama, this blocked indefinitely.

The same problem existed in three places:
- `output.py` — pandas at module level
- `retriever.py` — `from sentence_transformers import SentenceTransformer` at module level  
- `pipeline.py` — `from src.staged_pipeline.gliner_extractor import GlinerExtractor` at module level, which triggered torch/CUDA initialisation

The fix in all three cases: **lazy imports**. Move the import inside the function that actually needs it, so it only runs when that code path is reached — not at startup.

```python
# Before (hangs on import)
import pandas as pd

# After (loads only when save_extractions() is called)
def save_extractions(extractions, ...):
    import pandas as pd
    ...
```

After this fix, the pipeline started and ran. We added a heartbeat thread — printing a tick every 10 seconds — so we could always see that Python was alive and not just silently stuck:

```python
def _heartbeat():
    t0 = time.time()
    while not _stop_heartbeat.wait(10):
        print(f"  [heartbeat] still running ... ({int(time.time()-t0)}s elapsed)")
threading.Thread(target=_heartbeat, daemon=True).start()
```

---

## The GPU OOM Spiral

With both pipelines working, we ran them on 5 documents. Midway through the LLM pipeline, Ollama returned a 500 Internal Server Error. The logs showed: `CUDA error: out of memory`.

What happened: earlier long-running requests had fragmented GPU VRAM. Ollama's CUDA backend crashed and stayed crashed until the server was restarted. The fix: restart Ollama, reduce `num_ctx` from 32,768 to 8,192 (cutting VRAM usage from ~8GB to ~2GB), and add a hard 24,000-character cap on curated contexts before they reach the LLM.

The 24,000-character cap means roughly 6,000 tokens — well within the 8,192-token context window with room left for the prompt and response. The tradeoff: for very long documents, we might cut off relevant sections. For the document sizes we were working with (3–15 pages), the cap was never hit.

---

## Timeout That Didn't Work

One document took 18 minutes before we killed the process manually. We had set a 300-second timeout on the `requests` call — so why didn't it fire?

The answer is how `requests` timeouts work: they're **socket idle timeouts**, not wall-clock timeouts. With `stream=False`, Ollama holds the HTTP connection open while generating (sending no bytes back), so from `requests`'s perspective the socket is idle — but the timeout never fires because no data has been received yet to trigger the idle check.

We tried switching to `stream=True` with line-by-line reading and a deadline check:

```python
for line in resp.iter_lines():
    if time.time() > deadline:
        break
```

This caused a 405 Method Not Allowed error from Ollama — apparently the `format: json` option and `stream: true` don't mix cleanly in that version. We reverted to `stream=False` and instead addressed the root cause: smaller contexts mean faster generation, so the 24k cap and 8,192 num_ctx together kept generation time to 30–90 seconds per document.

---

## Comparing the Two Pipelines

After both pipelines ran successfully on 3 common documents, we built a comparison script that:
- Loads both extraction JSONs
- Matches documents by filename
- For each of the 36 fields, computes semantic similarity between the LLM value and the staged value using Ollama embeddings (nomic-embed-text)
- Labels each field as `match`, `mismatch`, `llm_only`, `staged_only`, or `both_null`
- Saves a full report to JSON and Excel

**Results across 3 documents:**

| Document | Field match rate | Avg similarity |
|---|---|---|
| AMI-_AMOA_TRANSVERSE08.06 | 97.1% | 0.971 |
| AMI_AMOA_V_NO_14102021 | 100.0% | 1.000 |
| TDRs-_AMOA_TRANSVERSE_08-06 | 97.1% | 0.980 |

98.1% overall agreement. Only 2 real divergences across 108 comparisons, and in both cases LLM Strategy A was correct:

- **Submission deadline** (Doc 1): LLM extracted `2022-07-08`, staged returned null. BM25 without embeddings couldn't find the right chunk because the deadline phrase in the document used different wording than our predefined query.
- **Domain** (Doc 3): LLM returned `e-government`, staged returned `digital identity`. The BM25 retriever had ranked identity-related chunks highly for the domain field — and the model classified the domain based on what it saw in those chunks, not the overall project framing.

The ~60% null rate across fields is not a bug. Consulting AMI tenders don't contain budgets, tech stacks, or HR seniority levels. Those nulls are correct.

---

## Strategy C: Translate First, Then Extract

A natural next question: does the extraction model perform better on English? qwen2.5:7b is trained predominantly on English data, so French legal terminology might be harder for it to parse correctly.

We added a translation preprocessing step: chunk the document into 2,000-character blocks, send each to Ollama with "translate to English", join the results, save as a `.txt` file. Then run both pipelines on the English text.

**What improved:**
- `tender_id` unlocked on 2/3 documents (was always null in native French runs)
- `submission_deadline` recovered on the document where staged had missed it
- `publication_date`, `city_region`, `evaluation_criteria` newly extracted on Doc 1
- Doc 2 fill rate rose from 31.4% → 37.1% with still-perfect 100% field agreement

**What broke:**
- Doc 3 (a Terms of Reference document) fell from 97.1% → 82.9% match. The 2,000-char blind chunking split section headers from their bodies. Without that structural context, BM25 retrieved different paragraphs for LLM and staged, causing them to extract different values for `project_description`, `deliverables`, and `scope_of_work`.
- Overall agreement rate dropped from 98.1% to 91.4%
- Translation adds ~284 serial Ollama calls for a 300-page document — hours of work

The pattern was clear: translation helps on short, well-structured AMI documents. It hurts on longer specification documents where the document structure is load-bearing.

---

## Strategy D: Helsinki-NLP MarianMT

The Ollama-based translation had two problems: speed (serial API calls) and chunk boundary artifacts (2,000-char blind cuts). We replaced it with **Helsinki-NLP/opus-mt-fr-en** — a MarianMT model trained specifically on French→English translation, running locally via HuggingFace Transformers.

Key differences:
- **Paragraph-aware chunking**: instead of slicing at 2,000 chars, we split at paragraph boundaries (`\n\n`), then further at sentence boundaries if a paragraph exceeds the model's 512-token limit. Structure is preserved.
- **Batched inference**: all paragraphs are processed in one model pass. A 15-page document translates in seconds instead of minutes.
- **Dedicated translation training**: the model has seen millions of French→English parallel sentences — it knows that "Maître d'Ouvrage" is "Project Owner" and it will use that consistently throughout the document.
- **No Ollama dependency for translation**: the GPU is free for the extraction step.

**Results:**

| Document | Native | Ollama | Helsinki |
|---|---|---|---|
| AMI-_AMOA_TRANSVERSE08.06 | 97.1% | 91.4% | 77.1% |
| AMI_AMOA_V_NO_14102021 | 100.0% | 100.0% | 94.3% |
| TDRs-_AMOA_TRANSVERSE_08-06 | 97.1% | 82.9% | **100.0%** |

Helsinki fixed Doc 3 completely — 100% field match, and for the first time both pipelines correctly classified the domain as `e-government` (previously staged had returned `digital identity` in both native and Ollama runs). Paragraph-aware chunking preserved the document's list structure perfectly, and both pipelines retrieved identical chunks.

But Doc 1 got worse: 77.1%, down from 91.4% with Ollama. The reason: **MarianMT silently left some paragraphs untranslated**. Paragraphs with bullet characters, mixed formatting, or very short lines were skipped by the tokenizer. Those French fragments stayed in the output file. The LLM pipeline read the full document and extracted from the English sections; the staged pipeline's BM25 index ranked the French fragments highly for some field queries and returned French values — causing LLM and staged to disagree on `title`, `issuing_organization`, and `project_description`.

This is a solvable problem. A post-translation validation pass — running `langdetect` on every paragraph of the output and re-translating any still-French paragraph — would fix it. But it's an extra step we hadn't built yet.

---

## What We Learned

**On the pipeline design:**

1. **Lazy imports are not optional on Windows.** Any module that loads torch, pandas, or sentence-transformers at import time will hang indefinitely in certain GPU driver states. Move all heavy imports inside the functions that need them.

2. **Save after every document, not at the end.** Native crashes (from C++ memory failures) and process kills don't give Python a chance to clean up. Incremental saves are the only protection.

3. **BM25 without embeddings is brittle.** It misses synonyms, variant phrasings, and cross-section context. The `submission_deadline` miss in the native French run was purely a keyword gap. Dense embeddings (BGE-M3 or similar) would recover this class of error.

4. **The confidence filter is a double-edged sword.** Dropping fields below 0.6 confidence reduces hallucinations but silently discards valid low-confidence extractions (especially dates, which the model often extracts correctly but marks uncertain). Per-field confidence thresholds would be better than a single global floor.

**On translation:**

5. **General chat models make bad translators.** Sending 2,000-char chunks to qwen2.5:7b works, but terminology drifts across chunks, structure is lost at cut points, and it's slow. A dedicated translation model is always the better choice.

6. **MarianMT is fast and high-quality but not complete.** It silently skips paragraphs it can't tokenise. Any production use needs a validation pass to detect and re-translate skipped content.

7. **Translation helps structured documents and hurts specification documents.** AMI-style tenders (short, cover-page-first, well-structured) benefit from translation. TDR-style documents (multi-section, no cover page) lose structural cues when translated and chunked.

8. **The best overall approach depends on document type:**
   - Short French AMI → native LLM (Strategy A)
   - Short Arabic AMI → Helsinki translate → LLM (Strategy D)
   - Long specification document → native staged pipeline (Strategy B)
   - Large document (100+ pages) → staged pipeline, native language, embeddings enabled

**On measurement:**

9. **Semantic similarity is a better comparison metric than string equality.** "Ministry of Communication Technologies" and "Ministère des Technologies de la Communication" are the same thing — string equality marks them as a mismatch, cosine similarity over embeddings marks them correctly as equivalent.

10. **Fill rate tells you about document type, not pipeline quality.** A 35% fill rate on a consulting AMI is correct. The same rate on a technical implementation tender would be a failure. Context matters.

---

## Where Things Stand

Three runs. Four strategies. Twelve document comparisons. Summary of what each approach achieved:

| Strategy | Best match rate | Best fill rate | Speed | Works on large docs |
|---|---|---|---|---|
| A — Native LLM | 100% (Doc 2) | 42.9% | Fast | No |
| B — Staged native | 100% (Doc 2) | 45.7% | Medium | Yes |
| C — Ollama translate + extract | 100% (Doc 2) | 45.7% | Slow | No |
| D — Helsinki translate + extract | 100% (Doc 3) | 42.9% | Fast | Partial |

No single strategy dominates all dimensions. The right architecture is a **routing layer**: detect document language and length at intake, pick the appropriate strategy, run it, and merge with the fallback strategy's output wherever the primary returned null.

That merge step — LLM values take priority, staged fills in the gaps — is already implemented in the codebase. The next step is wiring the router to make the strategy selection automatic.

---

*Built with: Python 3.12 · Ollama · qwen2.5:7b · pdfplumber · python-docx · BM25Okapi · Helsinki-NLP/opus-mt-fr-en · HuggingFace Transformers · Pydantic v2 · pandas · nomic-embed-text*
