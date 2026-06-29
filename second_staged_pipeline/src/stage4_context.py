"""Stage 4 — Intelligent Context Compression.

Reduces large tenders (50,000+ tokens) down to ~1,500 relevant tokens.
Methods:
  1. Section ranking — prioritize important sections, reduce boilerplate
  2. Chunk filtering — discard chunks below similarity threshold
  3. Semantic retrieval — per-field relevant chunk selection using embeddings + BM25
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from .stage1_document import DocumentChunk
from .stage3_transformer import TransformerStage


def _log(msg: str):
    print(f"  [stage4] {msg}", flush=True)


# ── Section priority scores ────────────────────────────────────────────────────

_HIGH_PRIORITY_KEYWORDS = [
    r"objet", r"objet\s+du\s+march[eé]", r"scope", r"périmètre",
    r"cahier\s+des?\s+charges?", r"spécifications?\s+techniques?",
    r"technical\s+requirements?", r"requirements?",
    r"critères?\s+d[''']éligibilité", r"eligibility",
    r"budget", r"montant", r"prix",
    r"délai", r"date\s+limite", r"deadline",
    r"critères?\s+d[''']évaluation", r"evaluation",
    r"références?\s+requises?", r"expérience\s+requise",
    r"qualifications?", r"compétences?\s+requises?",
    r"certifications?", r"ISO",
    r"contact", r"adresse", r"coordonnées?",
    r"الشروط", r"المواصفات", r"التقنية", r"الميزانية", r"الموعد",
]

_LOW_PRIORITY_KEYWORDS = [
    r"clause\s+générale", r"dispositions?\s+générales?",
    r"préambule", r"introduction\s+générale",
    r"glossaire", r"définitions?",
    r"annexe\s+(?:juridique|légale|administrative)",
    r"conditions?\s+générales?\s+de\s+vente",
    r"règlement\s+(?:intérieur|général)",
]

_HIGH_RE = re.compile("|".join(_HIGH_PRIORITY_KEYWORDS), re.IGNORECASE | re.UNICODE)
_LOW_RE = re.compile("|".join(_LOW_PRIORITY_KEYWORDS), re.IGNORECASE | re.UNICODE)

# ── Field queries for semantic retrieval ───────────────────────────────────────

FIELD_QUERIES: dict[str, list[str]] = {
    "tender_id": ["numéro appel offres référence AO", "tender reference number AO"],
    "title": ["intitulé du marché titre projet", "tender title project name"],
    "reference_number": ["référence interne numéro dossier", "internal reference number"],
    "issuing_organization": ["maître d'ouvrage pouvoir adjudicateur organisme", "contracting authority issuing organization"],
    "department": ["département division direction service", "department division"],
    "country": ["pays localisation géographie", "country location"],
    "city_region": ["ville région wilaya gouvernorat", "city region province"],
    "publication_date": ["date publication émission avis", "publication date issued"],
    "submission_deadline": ["date limite dépôt soumission offres délai", "submission deadline closing date"],
    "questions_deadline": ["date limite questions demandes clarifications", "questions deadline clarifications"],
    "award_date": ["date attribution résultats notification", "award date contract notification"],
    "budget": ["budget montant enveloppe prix maximum estimé", "budget amount maximum price"],
    "currency": ["devise monnaie DT EUR MAD", "currency denomination"],
    "payment_terms": ["conditions paiement modalités règlement", "payment terms conditions"],
    "financial_guarantee": ["caution cautionnement garantie financière bancaire", "financial guarantee bank guarantee"],
    "project_description": ["description projet mission objectifs", "project description objectives"],
    "domain": ["domaine secteur activité TI informatique", "domain sector IT technology"],
    "required_technologies": ["technologies requises outils frameworks plateformes", "required technologies tools frameworks"],
    "deliverables": ["livrables résultats attendus produits", "deliverables expected results"],
    "scope_of_work": ["périmètre travaux prestations missions", "scope of work services"],
    "hosting_requirements": ["hébergement infrastructure cloud datacenter", "hosting cloud infrastructure"],
    "num_profiles": ["nombre de profils postes ressources humaines", "number of profiles positions"],
    "roles_profiles": ["profils recherchés postes expertises rôles", "roles profiles expertise required"],
    "seniority_level": ["niveau séniorité expérience années junior senior", "seniority level years experience"],
    "certifications": ["certifications requises ISO CMMI PMP AWS", "required certifications ISO"],
    "mission_duration": ["durée mission contrat mois années", "mission duration contract period"],
    "required_experience": ["expérience requise références similaires projets", "required experience similar projects"],
    "company_size": ["taille entreprise effectif chiffre affaires", "company size employees revenue"],
    "required_documents": ["documents administratifs pièces dossier", "required documents administrative"],
    "geographic_restrictions": ["restrictions géographiques nationalité implantation", "geographic restrictions nationality"],
    "legal_requirements": ["exigences légales juridiques réglementaires", "legal requirements regulatory"],
    "evaluation_criteria": ["critères évaluation notation pondération technique financier", "evaluation criteria scoring"],
    "lot_number": ["lot numéro tranche partie", "lot number tranche"],
    "is_tech_relevant": ["informatique technologie numérique logiciel", "IT technology software digital"],
    "relevance_reason": ["domaine pertinence compétence technologie", "relevance domain competency"],
}


@dataclass
class ContextResult:
    compressed_text: str
    total_chars_before: int
    total_chars_after: int
    chunks_used: int
    chunks_total: int
    reduction_pct: float


def build_compressed_context(
    chunks: list[DocumentChunk],
    transformer: TransformerStage,
    domain_profile: str = "AI software engineering data analytics",
    min_similarity: float = 0.25,
    max_chars: int = 16_000,
    header_chunks: int = 3,
) -> ContextResult:
    """
    Build a compressed context from all chunks using:
    1. Always-include header chunks (cover page, AO header)
    2. Section ranking (high-priority sections)
    3. Semantic similarity filter against domain profile
    """
    total_chars_before = sum(len(c.text) for c in chunks)
    n = len(chunks)

    if n == 0:
        return ContextResult("", 0, 0, 0, 0, 0.0)

    # Score each chunk
    scores = _score_chunks(chunks, transformer, domain_profile, min_similarity)

    # Always include first `header_chunks` chunks (cover page)
    always_include = set(range(min(header_chunks, n)))

    # Select chunks above threshold + forced header chunks
    selected = sorted(
        {i for i, s in enumerate(scores) if s >= min_similarity} | always_include
    )

    _log(f"Selected {len(selected)}/{n} chunks above similarity threshold {min_similarity}")

    # Build context in document order
    context = _assemble_context(chunks, selected, max_chars)

    total_chars_after = len(context)
    reduction = 1.0 - (total_chars_after / max(total_chars_before, 1))

    _log(
        f"Context compressed: {total_chars_before} → {total_chars_after} chars "
        f"({reduction*100:.0f}% reduction)"
    )

    return ContextResult(
        compressed_text=context,
        total_chars_before=total_chars_before,
        total_chars_after=total_chars_after,
        chunks_used=len(selected),
        chunks_total=n,
        reduction_pct=reduction * 100,
    )


def retrieve_for_fields(
    chunks: list[DocumentChunk],
    transformer: TransformerStage,
    fields: list[str],
    top_k: int = 3,
) -> dict[str, list[DocumentChunk]]:
    """Retrieve top-k chunks for each requested field using semantic search + BM25 fusion."""
    result: dict[str, list[DocumentChunk]] = {}

    try:
        from rank_bm25 import BM25Okapi
        tokenized = [c.text.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        _bm25_available = True
    except ImportError:
        bm25 = None
        _bm25_available = False

    for field in fields:
        queries = FIELD_QUERIES.get(field, [field.replace("_", " ")])
        all_ranked: list[list[int]] = []

        for query in queries:
            ranked = _rank_query(query, chunks, bm25, transformer, top_k=top_k)
            all_ranked.append(ranked)

        fused = _rrf_fuse(all_ranked)
        result[field] = [chunks[i] for i in fused[:top_k]]

    return result


def build_field_context(
    field_chunks: dict[str, list[DocumentChunk]],
    chunks: list[DocumentChunk],
    header_count: int = 2,
    max_chars: int = 16_000,
) -> str:
    """Build a deduplicated context from retrieved field chunks."""
    seen: set[int] = set()
    for chunk in sorted(chunks, key=lambda c: c.index)[:header_count]:
        seen.add(chunk.index)
    for chunk_list in field_chunks.values():
        for chunk in chunk_list:
            seen.add(chunk.index)

    ordered = sorted([c for c in chunks if c.index in seen], key=lambda c: c.index)
    return _assemble_context_from_list(ordered, max_chars)


def _score_chunks(
    chunks: list[DocumentChunk],
    transformer: TransformerStage,
    domain_profile: str,
    min_similarity: float,
) -> list[float]:
    """Score each chunk by: section rank + domain similarity."""
    n = len(chunks)
    section_scores = np.array([_section_score(c) for c in chunks])

    if transformer.is_embedding_ready:
        sim_scores = transformer.similarity_scores(domain_profile)
        if sim_scores is not None and len(sim_scores) == n:
            combined = 0.4 * _normalize(section_scores) + 0.6 * _normalize(sim_scores)
            return combined.tolist()

    return section_scores.tolist()


def _section_score(chunk: DocumentChunk) -> float:
    """Score a chunk based on its section title and content."""
    text = (chunk.section_title or "") + " " + chunk.text[:200]
    if _HIGH_RE.search(text):
        return 0.8
    if _LOW_RE.search(text):
        return 0.1
    return 0.4


def _rank_query(
    query: str,
    chunks: list[DocumentChunk],
    bm25,
    transformer: TransformerStage,
    top_k: int,
) -> list[int]:
    n = len(chunks)
    bm25_scores = np.zeros(n)
    if bm25 is not None:
        tokens = query.lower().split()
        bm25_scores = np.array(bm25.get_scores(tokens))

    dense_scores = np.zeros(n)
    if transformer.is_embedding_ready:
        sim = transformer.similarity_scores(query)
        if sim is not None and len(sim) == n:
            dense_scores = sim

    if bm25 is not None and transformer.is_embedding_ready:
        combined = _normalize(bm25_scores) * 0.35 + _normalize(dense_scores) * 0.65
    elif bm25 is not None:
        combined = bm25_scores
    else:
        combined = dense_scores

    return np.argsort(combined)[::-1][:top_k].tolist()


def _assemble_context(
    chunks: list[DocumentChunk],
    selected_indices: list[int],
    max_chars: int,
) -> str:
    ordered = [chunks[i] for i in selected_indices]
    return _assemble_context_from_list(ordered, max_chars)


def _assemble_context_from_list(
    ordered: list[DocumentChunk],
    max_chars: int,
) -> str:
    parts: list[str] = []
    last_section: str | None = None
    total = 0

    for chunk in ordered:
        if chunk.section_title and chunk.section_title != last_section:
            header = f"\n## {chunk.section_title}\n"
            parts.append(header)
            total += len(header)
            last_section = chunk.section_title

        if max_chars > 0 and total + len(chunk.text) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(chunk.text[:remaining])
            break

        parts.append(chunk.text)
        total += len(chunk.text)

    return "\n\n".join(parts)


def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def _rrf_fuse(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)
