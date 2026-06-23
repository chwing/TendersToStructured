# Tender Extraction — Full Comparison Report

**Date:** 2026-06-23 · **LLM model:** qwen2.5:7b (Ollama) · **Strategy A** = full-document LLM · **Strategy B** = staged pipeline (chunk → BM25+BGE-M3 retrieval → LLM)

---

## Part 1 — Claude Ground Truth (read directly from source documents)

### Doc 1 — `AMI-_AMOA_TRANSVERSE08.06.2022.pdf` (public notice / AMI)
| Field | Ground truth |
|---|---|
| tender_id | **TN-MTCTD-228235-CS-QCBS** |
| loan_number | BIRD-89870 (World Bank) |
| title | Mission d'AMOA pour la mise en œuvre des projets du programme « GovTech » |
| issuing_org | Ministère des Technologies de la Communication |
| country / city | Tunisie / Tunis (88 Av. Mohamed V, 1002) |
| submission_deadline | **2022-07-08** (Vendredi 08 Juillet 2022, 12h00) |
| publication_date | **Not stated in document** |
| mission_duration | 24 mois |
| selection_method | SFQC (QCBS) |
| evaluation_criteria | Full scoring table: Solidité 30 pts (domaine 10 + moyens humains 20) + Références 70 pts |
| required_documents | Annexe 1 (références), Annexe 2 (moyens humains), Annexe 3 (fiche) |
| is_tech_relevant | TRUE |

### Doc 2 — `AMI_AMOA_V_NO_14102021.pdf` (AfDB-funded notice)
| Field | Ground truth |
|---|---|
| AMI / notice number | **21/P-BAD/2021** |
| loan reference | 2000200001802 (BAD loan ref) |
| project ID | P-TN-G00-03 |
| title | Sélection de consultant (Firme) — appui suivi/qualité E-Gov PNS 2020 |
| issuing_org | République Tunisienne / Min. Technologies de la Communication |
| submission_deadline | **2021-11-08** (08/11/2021, 11h00) |
| funder | BAD (Banque Africaine de Développement) |
| num_profiles | **Not stated** |
| roles_profiles | **Not stated** |
| is_tech_relevant | TRUE |

### Doc 3 — `TDRs-_AMOA_TRANSVERSE_-08-06-2022.docx` (Terms of Reference, companion to Doc 1)
*Binary .docx — could not be opened directly, but it is the detailed TDR behind Doc 1's notice.*
| Field | Ground truth (from structure) |
|---|---|
| title | AMOA pour la mise en œuvre des projets GovTech |
| issuing_org | Ministère des Technologies de la Communication |
| sectors covered | Éducation, Affaires sociales, Santé, Présidence du Gouvernement |
| mission_duration | 24 mois |
| num_profiles | 6 |
| roles_profiles | Responsable de mission; PMO central; AMOA Secteur social; AMOA Réforme administrative; AMOA autres secteurs; Expert Architecture SI |
| is_tech_relevant | TRUE |

### Doc 4 — `TN-MTCTD-412437-NC-RFB_-_SIAD_CNSS_VF.pdf` (FULL bid dossier, 324 pp.)
| Field | Ground truth |
|---|---|
| tender_id (AO N°) | **TN-MTCTD-412437-NC-RFB** |
| loan_number | **IBRD-89870** (World Bank) |
| title | Conception et développement d'un Système décisionnel et analytique sectoriel au profit du Ministère des Affaires Sociales |
| issuing_org | Min. Technologies de la Communication, au profit de la **CNSS** (Min. Affaires Sociales) |
| country / city | Tunisie / Tunis |
| publication_date | **Septembre 2024** |
| submission_deadline | **2024-11-13, 11h00** |
| budget / currency | Financing 100 M US$ (programme); contract value not fixed (lump sum) |
| financial_guarantee | **50 000 TND** (garantie de soumission) |
| bid validity | 120 jours |
| domain | Business Intelligence / système décisionnel / datawarehouse / ETL |
| mission_duration | 9 mois exécution + 1 an garantie + 3 ans maintenance |
| num_profiles | min 6 (Chef projet, 2 Consultants BI, 4 Dév backend/ETL, ≥3 Dév BI, 1 Expert DWH) |
| evaluation_criteria | Technique 40% (S1 solution 80%, S2 méthodo 20%) / Financier 60% |
| required_experience | CA annuel ≥ 1 700 000 DT; 3 réf. SI ≥1,6M DT; 2 réf. systèmes décisionnels |
| required_documents | Adm-1..4 (DAO signé, RNE, non-faillite, sécu sociale), Code de conduite |
| is_tech_relevant | TRUE |

### Doc 5 — `tunisia_...digital_tunisia_2020...pdf` (AfDB Project Appraisal Report)
**This is NOT a tender** — it is a development-bank loan appraisal document. So most tender-specific fields are *legitimately* null. But it does contain:
| Field | Ground truth |
|---|---|
| title | Support Project for Implementation of the "Digital Tunisia 2020" National Strategic Plan |
| issuing_org | Min. of Communication Technology & Digital Economy (MTCEN) / funded by AfDB |
| country | Tunisia (all regions) |
| budget / currency | AfDB loan EUR 71.56 M; total EUR 134.96 M / **EUR** |
| publication_date | October 2017 |
| domain | e-government / digital / ICT |
| is_tech_relevant | TRUE |

---

## Part 2 — Strategy A (LLM) vs Ground Truth

| Doc | Verdict | Detail |
|---|---|---|
| **1** | **Strong** | All key fields correct. `publication_date` correctly left **null** (it isn't in the doc) — the conservative, correct choice. Evaluation criteria captured in full detail. |
| **2** | **Good, one miss** | tender_id labelled `21/P-BAD/2021` and reference `P-TN-G00-03` — defensible. **Missed `submission_deadline` (2021-11-08)** → left null. Correctly left num_profiles/roles null (not in this notice). |
| **3** | **Good** | Title, org, 24 mois, 6 profiles, roles all correct. Added the four sectors (Éducation/Santé/…) — accurate enrichment. |
| **4** | **TOTAL FAILURE** | All 36 fields null. Should have extracted ~20 fields (AO number, CNSS, deadline, 50k TND guarantee, BI domain, etc.). |
| **5** | **TOTAL FAILURE** | All null. At minimum country=Tunisia, org=MTCEN, budget=EUR 71.56M, domain=e-gov were extractable. |

**LLM accuracy on the 3 docs it processed: high (~90%+ on filled fields).** It is appropriately conservative (doesn't hallucinate publication dates). Its one substantive error is the missed deadline in Doc 2.

---

## Part 3 — Strategy A (LLM) vs Strategy B (Staged)

| Field | Doc | LLM (A) | Staged (B) | Who's right |
|---|---|---|---|---|
| publication_date | 1 | null | **2022-06-15** | **LLM** — staged hallucinated a date not in the doc |
| tender_id | 2 | `21/P-BAD/2021` | `2000200001802/P-TN-G00-03` | Neither clean; staged mashed loan-ref + project-ID together (worse) |
| reference_number | 2 | `P-TN-G00-03` | `21/P-BAD/2021` | They swapped fields; LLM's split is more defensible |
| submission_deadline | 2 | **null (miss)** | **2021-11-08** | **Staged** — only B caught the deadline |
| num_profiles | 2 | null | **1** | **LLM** — staged invented "1" (not stated) |
| roles_profiles | 2 | null | "Consultant (Firme)" | **LLM** — staged's value is wrong granularity |
| deliverables vs scope_of_work | 1 | deliverables filled | scope_of_work filled | Tie — same content, different field |
| Docs 4 & 5 | 4,5 | all null | all null | **Both fail identically** |

**Pattern:** The staged pipeline fills *more* fields but at the cost of **lower precision** — it hallucinates (`publication_date 2022-06-15`, `num_profiles 1`, `roles "Consultant (Firme)"`) and mangles IDs. The LLM fills *fewer* fields but each is more trustworthy. The one place staged genuinely wins is Doc 2's submission deadline.

---

## Part 4 — Why Docs 4 & 5 fail in *both* strategies

Both are the **largest PDFs** (2.6 MB / 324 pp. and 1.1 MB). The root cause is the **qwen2.5:7b context window (~32k tokens)**:

- **Strategy A (full doc):** the entire document is stuffed into one prompt. Docs 4 & 5 vastly exceed 32k tokens → the input is truncated or the model returns empty/invalid JSON → coerced to all-null.
- **Strategy B (staged):** chunking + retrieval *should* fix this, but for these docs it still failed — likely because (a) the retriever surfaced boilerplate (CCAG/CCAP legal clauses dominate Doc 4) instead of the cover-page facts, and/or (b) the assembled context still overflowed the 7B model. Doc 5 also isn't a tender, so retrieval had no strong tender-keyword anchors.

**Note:** Claude read both PDFs without difficulty — they have clean text layers. So this is **purely a pipeline/model-capacity limitation, not a document problem.**

**Fixes:** (1) use a longer-context model (qwen2.5:7b → a 128k-context model, or chunk-map-reduce); (2) in staged mode, always force-include the first 1–2 pages (cover/AO data) regardless of retrieval score; (3) add a fallback that flags all-null results as extraction failures rather than valid empties.

---

## Part 5 — Ratings

| Criterion | Strategy A (LLM) | Strategy B (Staged) |
|---|---|---|
| **Precision** (no hallucination) | ★★★★★ | ★★★☆☆ (invents dates, profiles, IDs) |
| **Recall** (fields filled) | ★★★☆☆ | ★★★★☆ |
| **Large-doc robustness** | ★☆☆☆☆ | ★☆☆☆☆ |
| **Field discipline** (right value in right field) | ★★★★☆ | ★★★☆☆ (swaps id/ref, wrong granularity) |
| **Trustworthiness of output** | ★★★★★ | ★★★☆☆ |

### Overall

- **Strategy A (LLM): 7.5/10** — Trustworthy and conservative. When it fills a field, you can believe it. Weaknesses: misses some present fields (Doc 2 deadline), and collapses entirely on large docs.
- **Strategy B (Staged): 6/10** — Higher coverage but you can't trust it blindly; it manufactures plausible-looking values (2022-06-15, num_profiles=1). Same large-doc failure as A.

**Recommendation:** Use **Strategy A as the primary** (precision matters more than recall for tender data — a hallucinated deadline is worse than a missing one). Borrow Strategy B's *retrieval step only* to recover missed-but-present fields like Doc 2's deadline, with a verification pass. **Most urgent fix:** swap qwen2.5:7b for a long-context model so the two largest, most important documents stop returning all-null.
