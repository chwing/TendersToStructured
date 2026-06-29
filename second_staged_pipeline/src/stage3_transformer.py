"""Stage 3 — Transformer-Based Semantic Understanding.

Uses:
- XLM-RoBERTa for multilingual NER (organizations, locations, requirements, certifications)
- Sentence-transformers for semantic embeddings (multilingual-e5 or BGE-M3)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .stage1_document import DocumentChunk


def _log(msg: str):
    print(f"  [stage3] {msg}", flush=True)


# ── NER ────────────────────────────────────────────────────────────────────────

@dataclass
class NEREntity:
    text: str
    label: str
    score: float
    chunk_index: int


@dataclass
class NERResult:
    organizations: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    misc: list[str] = field(default_factory=list)
    all_entities: list[NEREntity] = field(default_factory=list)

    def to_hint_text(self) -> str:
        lines = ["=== NER ENTITIES (from XLM-RoBERTa) ==="]
        if self.organizations:
            lines.append(f"Organizations: {', '.join(self.organizations[:8])}")
        if self.locations:
            lines.append(f"Locations: {', '.join(self.locations[:5])}")
        if self.dates:
            lines.append(f"Date entities: {', '.join(self.dates[:5])}")
        if self.misc:
            lines.append(f"Other entities: {', '.join(self.misc[:8])}")
        lines.append("=== END NER ENTITIES ===")
        return "\n".join(lines)


_NER_MODEL_NAME = "Davlan/xlm-roberta-base-wikiann-ner"
_EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"

_LABEL_MAP = {
    "ORG": "organizations",
    "B-ORG": "organizations",
    "I-ORG": "organizations",
    "LOC": "locations",
    "B-LOC": "locations",
    "I-LOC": "locations",
    "PER": "misc",
    "B-PER": "misc",
    "I-PER": "misc",
    "MISC": "misc",
    "B-MISC": "misc",
    "I-MISC": "misc",
    "DATE": "dates",
}


class TransformerStage:
    """Handles NER and embedding computation."""

    def __init__(
        self,
        use_ner: bool = True,
        use_embeddings: bool = True,
        ner_model: str = _NER_MODEL_NAME,
        embed_model: str = _EMBED_MODEL_NAME,
        ner_batch_size: int = 8,
        embed_batch_size: int = 16,
    ):
        self.use_ner = use_ner
        self.use_embeddings = use_embeddings
        self.ner_model_name = ner_model
        self.embed_model_name = embed_model
        self.ner_batch_size = ner_batch_size
        self.embed_batch_size = embed_batch_size

        self._ner_pipeline = None
        self._embed_model = None
        self._chunk_embeddings: Optional[np.ndarray] = None
        self._chunks: list[DocumentChunk] = []

        if use_ner:
            self._load_ner()
        if use_embeddings:
            self._load_embeddings()

    def _load_ner(self):
        try:
            from transformers import pipeline as hf_pipeline
            _log(f"Loading NER model '{self.ner_model_name}' ...")
            t0 = time.time()
            self._ner_pipeline = hf_pipeline(
                "ner",
                model=self.ner_model_name,
                aggregation_strategy="simple",
            )
            _log(f"NER model ready ({time.time()-t0:.1f}s)")
        except Exception as e:
            _log(f"NER model load failed ({e}) — NER disabled")
            self._ner_pipeline = None

    def _load_embeddings(self):
        try:
            from sentence_transformers import SentenceTransformer
            _log(f"Loading embedding model '{self.embed_model_name}' ...")
            t0 = time.time()
            self._embed_model = SentenceTransformer(self.embed_model_name)
            _log(f"Embedding model ready ({time.time()-t0:.1f}s)")
        except Exception as e:
            _log(f"Embedding model load failed ({e}) — embeddings disabled")
            self._embed_model = None

    def run_ner(self, chunks: list[DocumentChunk], max_chars_per_chunk: int = 512) -> NERResult:
        """Run NER on chunks and aggregate entities."""
        if self._ner_pipeline is None:
            _log("NER skipped (model not loaded)")
            return NERResult()

        result = NERResult()
        seen: set[str] = set()

        _log(f"Running NER on {len(chunks)} chunks ...")
        t0 = time.time()

        for chunk in chunks:
            text = chunk.text[:max_chars_per_chunk]
            try:
                entities = self._ner_pipeline(text)
                for ent in entities:
                    label = ent.get("entity_group", ent.get("entity", ""))
                    word = ent.get("word", "").strip()
                    score = float(ent.get("score", 0.0))

                    if not word or len(word) < 2 or score < 0.7:
                        continue

                    key = word.lower()
                    if key in seen:
                        continue
                    seen.add(key)

                    bucket = _LABEL_MAP.get(label.upper())
                    if bucket:
                        getattr(result, bucket).append(word)
                    result.all_entities.append(NEREntity(
                        text=word, label=label, score=score, chunk_index=chunk.index
                    ))
            except Exception as e:
                _log(f"  NER error on chunk {chunk.index}: {e}")

        _log(
            f"NER done ({time.time()-t0:.1f}s) — "
            f"orgs={len(result.organizations)}, locs={len(result.locations)}"
        )
        return result

    def build_chunk_embeddings(self, chunks: list[DocumentChunk]) -> np.ndarray:
        """Compute and cache embeddings for all chunks."""
        if self._embed_model is None:
            _log("Embeddings skipped (model not loaded)")
            return np.array([])

        self._chunks = chunks
        _log(f"Encoding {len(chunks)} chunks ...")
        t0 = time.time()
        texts = [f"passage: {c.text}" for c in chunks]
        self._chunk_embeddings = self._embed_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=self.embed_batch_size,
        )
        _log(f"Embeddings ready ({time.time()-t0:.1f}s) — shape={self._chunk_embeddings.shape}")
        return self._chunk_embeddings

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """Embed a single query string."""
        if self._embed_model is None:
            return None
        return self._embed_model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
        )[0]

    def similarity_scores(self, query: str) -> Optional[np.ndarray]:
        """Return cosine similarity between query and all chunk embeddings."""
        if self._chunk_embeddings is None or len(self._chunk_embeddings) == 0:
            return None
        q_emb = self.embed_query(query)
        if q_emb is None:
            return None
        return self._chunk_embeddings @ q_emb

    @property
    def chunk_embeddings(self) -> Optional[np.ndarray]:
        return self._chunk_embeddings

    @property
    def is_embedding_ready(self) -> bool:
        return self._chunk_embeddings is not None and len(self._chunk_embeddings) > 0

    @property
    def is_ner_ready(self) -> bool:
        return self._ner_pipeline is not None
