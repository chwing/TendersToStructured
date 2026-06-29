import time
from typing import Optional

from first_staged_pipeline.src.extractor import TenderExtraction
from src.staged_pipeline.ingestor import ingest
from src.staged_pipeline.chunker import chunk_document
from src.staged_pipeline.retriever import (
    HybridRetriever, build_curated_context, _SHORT_DOC_THRESHOLD,
)
from src.staged_pipeline.llm_finalizer import LLMFinalizer

def _try_import_gliner():
    try:
        from src.staged_pipeline.gliner_extractor import GlinerExtractor
        return GlinerExtractor
    except (ImportError, Exception):
        return None


def _log(msg: str):
    print(f"  [pipeline] {msg}", flush=True)


class StagedPipelineExtractor:
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        use_gliner: bool = True,
        use_embeddings: bool = True,
        top_k: int = 5,
        max_retries: int = 3,
        timeout: int = 300,
        num_ctx: int = 32768,
        min_confidence: float = 0.6,
    ):
        self.use_embeddings = use_embeddings
        self.top_k = top_k
        self.max_chars = 24_000  # hard cap before sending to LLM (~6k tokens)

        self._finalizer = LLMFinalizer(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            num_ctx=num_ctx,
            min_confidence=min_confidence,
        )
        self._gliner = None
        if use_gliner:
            _log("Loading GLiNER model ...")
            GlinerExtractor = _try_import_gliner()
            if GlinerExtractor:
                try:
                    self._gliner = GlinerExtractor()
                    _log("GLiNER ready.")
                except Exception as e:
                    _log(f"GLiNER init failed ({e}) — skipping.")
            else:
                _log("GLiNER not available — skipping.")

    @property
    def last_prompt_chars(self) -> int:
        return self._finalizer._last_prompt_chars

    def extract_file(self, file_path: str) -> TenderExtraction:
        _log("Stage 1 — Ingest")
        t0 = time.time()
        doc = ingest(file_path)
        _log(f"Ingest done ({time.time()-t0:.1f}s) — {len(doc.text)} chars via {doc.parser}")
        return self._run_pipeline(doc.text, source_file=file_path)

    def _run_pipeline(self, text: str, source_file: str = "") -> TenderExtraction:
        from first_staged_pipeline.src.extractor import _detect_language
        language = _detect_language(text)
        _log(f"Detected language: {language}")

        use_tiered = len(text) >= _SHORT_DOC_THRESHOLD
        field_chunks: Optional[dict] = None
        chunks: list = []

        if not use_tiered:
            _log(f"Short document ({len(text)} chars) — skipping retrieval, using full text")
            combined_ctx = text
        else:
            _log("Stage 2 — Chunking")
            t0 = time.time()
            chunks = chunk_document(text)
            _log(f"Chunking done ({time.time()-t0:.1f}s) — {len(chunks)} chunks")

            _log("Stage 3 — Tiered retrieval (mandatory top_k=8 / nice=5 / extra=3)")
            retriever = HybridRetriever(chunks, use_embeddings=self.use_embeddings)
            field_chunks = retriever.retrieve_all_fields_tiered()

            combined_ctx = build_curated_context(field_chunks, chunks)
            if len(combined_ctx.strip()) < 500:
                _log("Curated context too short — falling back to full text single-pass")
                use_tiered = False
                field_chunks = None
                combined_ctx = text

        ner_hints = ""
        if self._gliner is not None:
            _log("Stage 5 — GLiNER NER")
            t0 = time.time()
            try:
                candidates = self._gliner.extract_candidates(combined_ctx[:self.max_chars])
                ner_hints = self._gliner.format_hints(candidates)
                _log(f"GLiNER done ({time.time()-t0:.1f}s) — {len(candidates)} entity types found")
            except Exception as e:
                _log(f"GLiNER failed ({e}) — continuing without hints")

        _log("Stage 6 — LLM finalization")
        t0 = time.time()
        if use_tiered and field_chunks is not None:
            _log("  mode: single-pass unified (1 call, per-tier confidence post-filter)")
            result = self._finalizer.finalize_single_pass(
                field_chunks=field_chunks,
                chunks=chunks,
                source_file=source_file,
                language=language,
                ner_hints=ner_hints,
            )
        else:
            _log("  mode: short-doc fallback (full text, 1 call)")
            result = self._finalizer.finalize(
                curated_text=combined_ctx[:self.max_chars],
                source_file=source_file,
                language=language,
                ner_hints=ner_hints,
            )
        _log(f"LLM done ({time.time()-t0:.1f}s)")
        return result
