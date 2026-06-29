from __future__ import annotations

import time
from typing import Optional

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

from src.staged_pipeline.chunker import Chunk
from src.staged_pipeline.field_queries import FIELD_QUERIES
from first_staged_pipeline.src.extractor import ALL_FIELDS, FIELD_TIER

_EMBED_MODEL_NAME = "BAAI/bge-m3"
_SHORT_DOC_THRESHOLD = 6000  # chars

# Chunks retrieved per field per tier (for unified single-pass)
TIER_TOP_K: dict[str, int] = {
    "mandatory": 3,
    "nice_to_have": 1,
    "extra": 1,
}


def _log(msg: str):
    print(f"  [retriever] {msg}", flush=True)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], use_embeddings: bool = True):
        self.chunks = chunks
        self.use_embeddings = use_embeddings
        self._bm25: Optional[BM25Okapi] = None
        self._embeddings: Optional[np.ndarray] = None
        self._embed_model: Optional[SentenceTransformer] = None
        self._build_index()

    def _build_index(self):
        n = len(self.chunks)
        _log(f"Building BM25 index over {n} chunks ...")
        t0 = time.time()
        tokenized = [c.text.lower().split() for c in self.chunks]
        if _BM25_AVAILABLE:
            self._bm25 = BM25Okapi(tokenized)
            _log(f"BM25 index ready ({time.time()-t0:.1f}s)")

        if self.use_embeddings and self.chunks:
            try:
                from sentence_transformers import SentenceTransformer
                _ST_AVAILABLE = True
            except ImportError:
                _ST_AVAILABLE = False
            if not _ST_AVAILABLE:
                _log("sentence-transformers not installed — using BM25 only")
                return
            _log(f"Loading embedding model '{_EMBED_MODEL_NAME}' (first run downloads ~1.5 GB — please wait) ...")
            t1 = time.time()
            self._embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
            _log(f"Model loaded ({time.time()-t1:.1f}s) — encoding {n} chunks ...")
            t2 = time.time()
            texts = [c.text for c in self.chunks]
            self._embeddings = self._embed_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=16,
            )
            _log(f"Embeddings ready ({time.time()-t2:.1f}s)")
        elif not self.use_embeddings:
            _log("Embeddings disabled (--no-embeddings) — using BM25 only")

    def retrieve_for_field(self, field_name: str, top_k: int = 5) -> list[Chunk]:
        queries = FIELD_QUERIES.get(field_name, [field_name])
        all_ranked: list[list[int]] = []
        for query in queries:
            ranked = self._rank_query(query, top_k=top_k)
            all_ranked.append(ranked)
        fused = _rrf_fuse(all_ranked, k=60)
        return [self.chunks[i] for i in fused[:top_k]]

    def _rank_query(self, query: str, top_k: int) -> list[int]:
        n = len(self.chunks)
        if n == 0:
            return []

        bm25_scores = np.zeros(n)
        if self._bm25 is not None:
            tokens = query.lower().split()
            bm25_scores = np.array(self._bm25.get_scores(tokens))

        dense_scores = np.zeros(n)
        if self._embed_model is not None and self._embeddings is not None:
            q_emb = self._embed_model.encode([query], normalize_embeddings=True)[0]
            dense_scores = self._embeddings @ q_emb

        if self._bm25 is not None and self._embed_model is not None:
            combined = _normalize(bm25_scores) * 0.4 + _normalize(dense_scores) * 0.6
        elif self._bm25 is not None:
            combined = bm25_scores
        else:
            combined = dense_scores

        return np.argsort(combined)[::-1][:top_k].tolist()

    def retrieve_all_fields(self, top_k: int = 5) -> dict[str, list[Chunk]]:
        _log(f"Retrieving top-{top_k} chunks for {len(ALL_FIELDS)} fields ...")
        t0 = time.time()
        result = {field: self.retrieve_for_field(field, top_k=top_k) for field in ALL_FIELDS}
        _log(f"Retrieval done ({time.time()-t0:.1f}s)")
        return result

    def retrieve_all_fields_tiered(
        self, tier_top_k: dict[str, int] = None
    ) -> dict[str, list[Chunk]]:
        """Retrieve with per-tier top_k: mandatory gets more chunks, extra gets fewer."""
        tk = tier_top_k or TIER_TOP_K
        _log(f"Tiered retrieval — mandatory={tk['mandatory']}, "
             f"nice_to_have={tk['nice_to_have']}, extra={tk['extra']}")
        t0 = time.time()
        result = {}
        for field in ALL_FIELDS:
            tier = FIELD_TIER.get(field, "nice_to_have")
            result[field] = self.retrieve_for_field(field, top_k=tk[tier])
        _log(f"Tiered retrieval done ({time.time()-t0:.1f}s)")
        return result


def build_unified_context(
    field_chunks: dict[str, list[Chunk]],
    chunks: list[Chunk],
    header_count: int = 2,
    max_chars: int = 12_000,
) -> str:
    """Single deduped context from all retrieved chunks across all tiers.

    Mandatory fields contribute up to 3 chunks each, others 1 — but all share
    the same pool so there is zero repetition. System prompt is paid once.
    """
    seen_indices: set[int] = set()
    for chunk in sorted(chunks, key=lambda c: c.index)[:header_count]:
        seen_indices.add(chunk.index)
    for chunk_list in field_chunks.values():
        for chunk in chunk_list:
            seen_indices.add(chunk.index)

    ordered = sorted([c for c in chunks if c.index in seen_indices], key=lambda c: c.index)

    parts: list[str] = []
    last_section: Optional[str] = None
    for chunk in ordered:
        if chunk.section_title and chunk.section_title != last_section:
            parts.append(f"\n## {chunk.section_title}\n")
            last_section = chunk.section_title
        parts.append(chunk.text)

    curated = "\n\n".join(parts)
    if max_chars > 0 and len(curated) > max_chars:
        curated = curated[:max_chars]

    _log(f"Unified context: {len(seen_indices)}/{len(chunks)} chunks -> {len(curated)} chars")
    return curated


def build_curated_context_for_fields(
    field_chunks: dict[str, list[Chunk]],
    chunks: list[Chunk],
    fields: list[str],
    header_count: int = 2,
    max_chars: int = 0,
) -> str:
    """Build a curated context using only chunks retrieved for `fields`.

    header_count: how many leading document chunks to always include.
    max_chars: if > 0, truncate the result to this many characters.
    """
    seen_indices: set[int] = set()
    for field in fields:
        for chunk in field_chunks.get(field, []):
            seen_indices.add(chunk.index)
    for chunk in sorted(chunks, key=lambda c: c.index)[:header_count]:
        seen_indices.add(chunk.index)

    ordered = sorted([c for c in chunks if c.index in seen_indices], key=lambda c: c.index)

    parts: list[str] = []
    last_section: Optional[str] = None
    for chunk in ordered:
        if chunk.section_title and chunk.section_title != last_section:
            parts.append(f"\n## {chunk.section_title}\n")
            last_section = chunk.section_title
        parts.append(chunk.text)

    curated = "\n\n".join(parts)
    if max_chars > 0 and len(curated) > max_chars:
        curated = curated[:max_chars]

    _log(f"Context [{', '.join(fields[:3])}...] : "
         f"{len(seen_indices)}/{len(chunks)} chunks -> {len(curated)} chars")
    return curated


def build_curated_context(field_chunks: dict[str, list[Chunk]], chunks: list[Chunk]) -> str:
    seen_indices: set[int] = set()
    for field_chunk_list in field_chunks.values():
        for chunk in field_chunk_list:
            seen_indices.add(chunk.index)

    # Fix 3 — always include the first chunks (cover page / AO header). These
    # hold tender_id, deadline, issuing org and guarantee; retrieval often
    # misses them on long docs where legal boilerplate dominates, causing the
    # LLM to hallucinate those exact fields.
    for chunk in sorted(chunks, key=lambda c: c.index)[:2]:
        seen_indices.add(chunk.index)

    ordered = sorted([c for c in chunks if c.index in seen_indices], key=lambda c: c.index)

    parts: list[str] = []
    last_section: Optional[str] = None
    for chunk in ordered:
        if chunk.section_title and chunk.section_title != last_section:
            parts.append(f"\n## {chunk.section_title}\n")
            last_section = chunk.section_title
        parts.append(chunk.text)

    curated = "\n\n".join(parts)
    _log(f"Curated context: {len(seen_indices)}/{len(chunks)} chunks -> {len(curated)} chars")
    return curated


def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _rrf_fuse(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)
