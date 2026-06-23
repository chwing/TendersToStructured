import time
from typing import Optional

from src.extractor.models import TenderExtraction
from src.staged_pipeline.ingestor import ingest
from src.staged_pipeline.chunker import chunk_document
from src.staged_pipeline.retriever import HybridRetriever, build_curated_context, _SHORT_DOC_THRESHOLD
from src.staged_pipeline.llm_finalizer import LLMFinalizer

try:
    from src.staged_pipeline.gliner_extractor import GlinerExtractor
    _GLINER_OK = True
except ImportError:
    _GLINER_OK = False


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
        self.use_gliner = use_gliner and _GLINER_OK
        self.use_embeddings = use_embeddings
        self.top_k = top_k

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
        self._gliner: Optional[GlinerExtractor] = None
        if self.use_gliner:
            try:
                from src.staged_pipeline.gliner_extractor import GlinerExtractor
                self._gliner = GlinerExtractor()
            except Exception:
                self._gliner = None

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
        from src.extractor.document_reader import _detect_language
        language = _detect_language(text)
        _log(f"Detected language: {language}")

        if len(text) < _SHORT_DOC_THRESHOLD:
            _log(f"Short document ({len(text)} chars) — skipping retrieval, using full text")
            curated = text
        else:
            _log("Stage 2 — Chunking")
            t0 = time.time()
            chunks = chunk_document(text)
            _log(f"Chunking done ({time.time()-t0:.1f}s) — {len(chunks)} chunks")

            _log("Stage 3 — Building hybrid index + retrieval")
            retriever = HybridRetriever(chunks, use_embeddings=self.use_embeddings)
            field_chunks = retriever.retrieve_all_fields(top_k=self.top_k)

            _log("Stage 4 — Curating context")
            curated = build_curated_context(field_chunks, chunks)

            if len(curated.strip()) < 500:
                _log("Curated context too short — falling back to full text")
                curated = text

        if self._gliner is not None:
            _log("Stage 5 — GLiNER NER")
            t0 = time.time()
            ner_hints = ""
            try:
                candidates = self._gliner.extract_candidates(curated)
                ner_hints = self._gliner.format_hints(candidates)
                _log(f"GLiNER done ({time.time()-t0:.1f}s) — {len(candidates)} entity types found")
            except Exception as e:
                _log(f"GLiNER failed ({e}) — continuing without hints")
                ner_hints = ""
        else:
            ner_hints = ""

        _log("Stage 6 — LLM finalization (sending to model ...)")
        t0 = time.time()
        result = self._finalizer.finalize(
            curated_text=curated,
            source_file=source_file,
            language=language,
            ner_hints=ner_hints,
        )
        _log(f"LLM done ({time.time()-t0:.1f}s)")
        return result
