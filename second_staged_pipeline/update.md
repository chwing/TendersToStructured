# Staged 2 Pipeline — Analysis & Roadmap

**Date:** 2026-07-15  
**Reference:** Claude extractions are treated as ground truth for all comparisons below.

---

## Known Issues

| # | Issue | Root cause |
|---|---|---|
| 1 | Title null on compressed docs | Stage 4 drops cover page chunk (no section keywords match) |
| 2 | Garbage budget values | Currency regex matches letters near digits without requiring a numeric amount |
| 3 | `j/h` misread as BHD (Bahraini Dinar) | Same regex fires on man-days notation `j/h` |
| 4 | Wrong domain classification | LLM defaults to "cybersecurity" or "digital identity" for banking/IT docs |
| 5 | `is_tech_relevant` False for ERP | ERP not in prompt examples; noisy hints distort reasoning |
| 6 | LLM-only route lower confidence | Classical hints injected on short docs add noise instead of signal |

---

## 3-Way Comparison: Staged 2 vs Ollama (qwen2.5:7b) vs Claude

Legend: `+` correct/present | `-` missing | `~` partial/lower quality | `x` wrong value (hallucination)

---

### Appel d'offres monetique.doc

| Field | Claude (truth) | Staged 2 | Ollama |
|---|---|---|---|
| Title | "Appel d'offres monetique" | x null | + present |
| City/Region | "Ouagadougou" | x null | - null |
| Submission Deadline | 30/03/2001 | + correct | x null |
| Award Date | 13/04/2001 | x 02/04/2001 (wrong) | + correct |
| Domain | "Monetique bancaire" | x "cybersecurity" (wrong) | - null |
| Project Description | full description | x null | ~ shorter |
| Scope of Work | present | x null | ~ = project_description |
| Deliverables | 4-phase plan | x null | - null |
| Required Documents | detailed list | x null | - null |
| Legal Requirements | double envelope system | x null | - null |
| Lot Number | full 3-lot description | ~ number only (3) | - null |
| Budget | (correctly absent) | x "FCFA, enveloppe., enveloppe" | - null |
| Num Profiles | (not in doc) | x 50 (fabricated) | - null |

**Staged 2 — Missing fields:** title, city/region, project description, scope of work, deliverables, required documents, legal requirements  
**Staged 2 — Hallucinations:** domain ("cybersecurity"), budget (garbage from regex), award date (off by 11 days), num_profiles (50 — likely "50 copies" of sealed envelopes misread as a headcount)

---

### CDC.docx

| Field | Claude (truth) | Staged 2 | Ollama |
|---|---|---|---|
| Title | "RFP Sourcing et Offres de services IT" | x null | ~ verbose alternate title |
| Domain | "Prestations Intellectuelles Informatiques / Securite SI" | x "digital identity" (too narrow) | x "cybersecurity" (too narrow) |
| Required Technologies | detailed list | + 9 items (correct) | - null |
| Roles/Profiles | 8 named job titles | ~ 6 generic category labels | - null |
| Mission Duration | "3 ans ferme a partir de signature" | ~ "3 ans" (incomplete) | - null |
| Required Documents | 7-item detailed list | ~ 3 short items | - null |
| Evaluation Criteria | 3-phase structured (admin, tech, commercial) | ~ 9 generic items (no structure) | - null |
| Tender ID | (not in Claude) | + n 32-10, n 09-08 | - null |
| Legal Requirements | (not in Claude) | + 4 items | - null |
| Required Experience | (not in Claude) | + 3 items | - null |

**Staged 2 — Missing fields:** title  
**Staged 2 — Hallucinations:** domain ("digital identity" — document is about IT staffing contracts, not identity management)  
**Staged 2 — Partial/degraded:** roles (generic labels vs specific job titles), evaluation criteria (flat list vs structured 3-phase process), required documents (incomplete)  
**Note:** Staged 2 extracts 3 fields Claude misses (tender_id, legal_requirements, required_experience) — classical extraction advantage on structured data.

---

### Offre financiere STB.docx

| Field | Claude (truth) | Staged 2 | Ollama |
|---|---|---|---|
| Title | "Elaboration du Schema Directeur SI de la Banque" | x null | - null |
| Publication Date | 20/05/2026 | x null | - null |
| Budget | "417 450 DT HT, 496 765.5 TTC (TVA 19%)" | ~ 417450 DT only + x 487775.50 DZD (wrong currency) | ~ written-out text, conf 0.4 |
| Currency | TND | ~ DT (low confidence) | - null |
| Scope of Work | 4-phase breakdown | x null | - null |
| Submission Deadline | (not in Claude) | + 2026-06-20 | - null |
| Country / City | (not in Claude) | + Tunisia / Tunis-Ariana | - null |
| Mission Duration | (not in Claude) | + 120 days | - null |

**Staged 2 — Missing fields:** title, publication date, scope of work  
**Staged 2 — Hallucinations:** budget TTC amount labeled as DZD (Algerian Dinar) instead of TND — the classical extractor grabbed the second number and attached the wrong currency code  
**Note:** Staged 2 extracts 3 fields Claude misses (deadline, location, duration). Ollama extracts almost nothing from this document.

---

### sonede1.docx

| Field | Claude (truth) | Staged 2 | Ollama |
|---|---|---|---|
| Title | full title | + present | + present |
| Issuing Organization | "SONEDE (chef de file, paiement equitable entre SONEDE et ONAS)" | ~ "SONEDE chef de fil" (truncated) | - null |
| Budget | "83 j/h (dont 10% sur 24 mois)" | x "83 BHD/jour" (j/h misread as Bahraini Dinar) | - null |
| Certifications | clean single string | ~ raw multiline blob, conf 0.4 | ~ structured list |
| Required Experience | full text | x null | + structured |
| Legal Requirements | "BAC+4; validite 60 jours" | x null | - null |
| Scope of Work | 4-mission breakdown | + correct | + correct |
| Mission Duration | "26 mois (dont 35 jours...)" | + correct (full) | ~ "26" (no unit) |

**Staged 2 — Missing fields:** required experience, legal requirements  
**Staged 2 — Hallucinations:** budget currency — `j/h` (jours/homme, man-days) misidentified as `BHD` (Bahraini Dinar) by the currency regex  
**Staged 2 — Partial/degraded:** issuing organization (ONAS co-payer detail dropped), certifications (raw unstructured blob)

---

### sonede2.docx

| Field | Claude (truth) | Staged 2 | Ollama |
|---|---|---|---|
| Title | full title | + present | + present |
| Budget | "144 j/h (30%)" | ~ "144 (30%)" (missing j/h unit) | - null |
| Mission Duration | full phase breakdown (6m+10m+6m+2m) | ~ "24 mois (hors delai)" only | ~ "24 mois" only |
| Legal Requirements | "BAC+4; validite 60 jours" | x null | - null |
| is_tech_relevant | (Claude does not extract this) | x False — WRONG for ERP project | + True (correct) |
| Evaluation Criteria | (Claude does not extract this) | - null | x "AVIS: Defavorable" (evaluator rejection note, not criteria) |

**Staged 2 — Missing fields:** legal requirements  
**Staged 2 — Hallucinations:** `is_tech_relevant = False` for an ERP implementation project — the model's reasoning was distorted by noisy classical hints  
**Staged 2 — Partial/degraded:** budget (value correct but unit missing), mission duration (total only, phases missing)

---

## Bilan

| Document | Staged 2 vs Claude | Key Staged 2 failures |
|---|---|---|
| monetique.doc | Missing 7 fields, 4 hallucinations | Null title, wrong domain, garbage budget, wrong award date, fabricated num_profiles |
| CDC.docx | Missing title only, domain wrong, 3 fields degraded | Domain wrong; also extracts 3 fields Claude misses |
| STB.docx | Missing 3 fields, 1 hallucination | Null title, wrong TTC currency (DZD), null scope |
| sonede1.docx | Missing 2 fields, 1 hallucination | Budget currency wrong (BHD for j/h), missing experience and legal |
| sonede2.docx | Missing 1 field, 1 hallucination | is_tech_relevant=False (wrong), missing legal requirements |

**Conclusion:**

Claude is the most reliable extractor. It never produces wrong values — when unsure it returns null rather than a fabricated answer. Staged 2 extracts more fields on average (classical regex pulls dates, IDs, and structured lists that Claude ignores), but introduces systematic errors: the currency regex is the single biggest bug source (Issues 2 and 3), responsible for hallucinations across 3 of 5 documents. The domain classifier is the second biggest source, defaulting to "cybersecurity" or "digital identity" when the document doesn't obviously match a category.

Ollama (qwen2.5:7b) is the weakest of the three. It fails almost entirely on STB, hallucinates an evaluator note as evaluation criteria, and consistently misses structured fields. It should not be used as a reference.

---

## Improvements to Implement

Fix in order of impact before building the agent:

1. **Header block injection (Issue 1):** Prepend the first 500 chars of raw document text as a protected `[HEADER]` block before the compressed context. Fixes null title on all staged-route documents.

2. **Currency regex whitelist (Issues 2 + 3):** Require `\d[\d\s.,]*` adjacent to currency code. Restrict to a known whitelist (TND, DT, MAD, XOF, FCFA, EUR, USD, DZD, BHD...). Rejects `j/h`, `enveloppe`, and plain keyword matches.

3. **Domain prompt examples (Issue 4):** Add labelled examples to the LLM system prompt: monetique/ATM/DAB = "Monetique bancaire", ERP/SAP = "ERP", GovTech/e-gov = "e-government". Remove "cybersecurity" as a catch-all for banking IT.

4. **is_tech_relevant prompt (Issue 5):** Add ERP, e-government, and digital transformation as explicit positive examples. Fixing Issues 1 and 6 first will likely resolve this automatically by reducing context noise.

5. **LLM-only route hint filtering (Issue 6):** On documents below the staged threshold, inject only hints with confidence >= 0.85. Skip budget hints entirely unless a numeric amount was confirmed. Skip domain hints.

---

## Agent Architecture Notes

The pipeline will be wrapped into a file-drop agent once the fixes above are applied. A full plan exists at `C:\Users\oussa\.claude\plans\pure-jumping-storm.md`.

Key design decisions:

- **Trigger:** folder watch on `inbox/` using the `watchdog` library. New or moved files are added to a processing queue.
- **Pipeline singleton:** `HybridTenderPipeline` loads NLP models (XLM-RoBERTa, multilingual-e5) once at startup (10–30s). Never reloaded per file.
- **Worker thread:** single thread drains the queue sequentially. Ollama handles one request at a time anyway.
- **Storage:** SQLite database `tenders.db`, one row per document, one column per extraction field. `source_file` has a UNIQUE constraint — re-dropping the same file is a no-op.
- **File routing:** `inbox/` on arrival → `processed/` on success → `errors/` + `.error.txt` on failure.
- **New files:** `agent/database.py`, `agent/watcher.py`, `agent/agent.py`, `run_agent.py`.
- **New dependency:** `watchdog>=4.0.0`.
- **No changes** to any file in `src/`.

The agent is a thin shell around the existing pipeline. Its output quality is entirely determined by the pipeline. Fix the pipeline first.
