"""Main pipeline orchestrator — 6-stage hybrid AI system.

Stage 1: Document Processing   (PyMuPDF / pdfplumber / python-docx)
Stage 2: Classical Extraction  (regex, rules, pattern detection)
Stage 3: Transformer NLP       (XLM-RoBERTa NER + multilingual-e5 embeddings)
Stage 4: Context Compression   (section ranking + semantic retrieval → ~95% token reduction)
Stage 5: Extraction LLM        (Qwen2.5 — structured field extraction)
Stage 6: LLM Judge             (Mistral/Llama — validation + self-correction loop)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .models import TenderExtraction, JudgeResult, ALL_FIELDS
from .stage1_document import ProcessedDocument, process_document
from .stage2_classical import ClassicalFacts, extract_classical
from .stage3_transformer import TransformerStage, NERResult
from .stage4_context import (
    ContextResult,
    build_compressed_context,
    retrieve_for_fields,
    build_field_context,
)
from .stage5_extractor import ExtractionLLM
from .stage6_judge import LLMJudge


def _log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


# Docs shorter than this go directly to the LLM (skip Stage 3 + 4).
# Adjust here or pass staged_threshold= to HybridTenderPipeline.
STAGED_PIPELINE_THRESHOLD = 10_000  # chars

_HIGH_CONFIDENCE_THRESHOLD = 0.80  # skip judge if avg confidence is high
_MAX_REVISION_ROUNDS = 2


@dataclass
class PipelineResult:
    extraction: TenderExtraction
    judge_result: Optional[JudgeResult]
    classical_facts: ClassicalFacts
    ner_result: NERResult
    context_result: Optional[ContextResult]
    total_time: float
    revision_rounds: int

    def summary(self) -> str:
        populated = sum(
            1 for f in ALL_FIELDS if getattr(self.extraction, f) is not None
        )
        lines = [
            f"Fields populated: {populated}/{len(ALL_FIELDS)}",
            f"Total time: {self.total_time:.1f}s",
            f"Revision rounds: {self.revision_rounds}",
        ]
        if self.judge_result:
            lines.append(
                f"Judge: valid={self.judge_result.is_valid}, "
                f"confidence={self.judge_result.overall_confidence:.2f}, "
                f"issues={len(self.judge_result.issues)}"
            )
        if self.context_result:
            lines.append(
                f"Token reduction: {self.context_result.reduction_pct:.0f}% "
                f"({self.context_result.total_chars_before} → {self.context_result.total_chars_after} chars)"
            )
        return "\n".join(lines)


class HybridTenderPipeline:
    """6-stage hybrid AI pipeline for tender key extraction and validation."""

    def __init__(
        self,
        # Stage 3 — Transformer
        use_ner: bool = True,
        use_embeddings: bool = True,
        ner_model: str = "Davlan/xlm-roberta-base-wikiann-ner",
        embed_model: str = "intfloat/multilingual-e5-base",
        # Stage 5 — Extraction LLM
        extractor_provider: str = "ollama",
        extractor_model: str = "qwen2.5:14b",
        extractor_base_url: str = "http://localhost:11434",
        extractor_api_key: Optional[str] = None,
        extractor_timeout: int = 300,
        extractor_num_ctx: int = 8192,
        extractor_max_retries: int = 3,
        # Stage 6 — Judge LLM (different model)
        judge_provider: str = "ollama",
        judge_model: str = "mistral",
        judge_base_url: str = "http://localhost:11434",
        judge_api_key: Optional[str] = None,
        judge_timeout: int = 240,
        judge_num_ctx: int = 16384,
        skip_judge: bool = False,
        # Context optimization
        domain_profile: str = "AI software engineering data analytics cloud",
        min_similarity: float = 0.25,
        max_context_chars: int = 16_000,
        # Routing
        staged_threshold: int = STAGED_PIPELINE_THRESHOLD,
        # General
        min_confidence: float = 0.40,
    ):
        self.domain_profile = domain_profile
        self.min_similarity = min_similarity
        self.max_context_chars = max_context_chars
        self.staged_threshold = staged_threshold
        self.min_confidence = min_confidence
        self.skip_judge = skip_judge

        _log("Initialising Stage 3 — Transformer models ...")
        self._transformer = TransformerStage(
            use_ner=use_ner,
            use_embeddings=use_embeddings,
            ner_model=ner_model,
            embed_model=embed_model,
        )

        _log("Initialising Stage 5 — Extraction LLM ...")
        self._extractor = ExtractionLLM(
            provider=extractor_provider,
            model=extractor_model,
            base_url=extractor_base_url,
            api_key=extractor_api_key,
            timeout=extractor_timeout,
            num_ctx=extractor_num_ctx,
            max_retries=extractor_max_retries,
        )

        if not skip_judge:
            _log("Initialising Stage 6 — LLM Judge ...")
            self._judge = LLMJudge(
                provider=judge_provider,
                model=judge_model,
                base_url=judge_base_url,
                api_key=judge_api_key,
                timeout=judge_timeout,
                num_ctx=judge_num_ctx,
            )
        else:
            self._judge = None

    def process_file(self, file_path: str) -> PipelineResult:
        """Full 6-stage pipeline for a single file."""
        t_total = time.time()

        # ── Stage 1: Document Processing ──────────────────────────────────────
        _log(f"Stage 1 — Document Processing: {file_path}")
        t0 = time.time()
        doc = process_document(file_path)
        _log(f"Stage 1 done ({time.time()-t0:.1f}s)")

        # ── Stage 2: Classical Extraction ─────────────────────────────────────
        _log("Stage 2 — Classical Extraction")
        t0 = time.time()
        classical_facts = extract_classical(doc.text)
        _log(f"Stage 2 done ({time.time()-t0:.1f}s)")

        # ── Route: LLM-only vs full staged pipeline ────────────────────────────
        doc_len = len(doc.text)
        use_staged = doc_len >= self.staged_threshold
        _log(
            f"Routing — doc length={doc_len} chars, threshold={self.staged_threshold} → "
            f"{'STAGED (Stage 3+4)' if use_staged else 'LLM-only (Stage 3+4 skipped)'}"
        )

        context_result: Optional[ContextResult] = None
        ner_result = NERResult(organizations=[], locations=[])  # empty fallback

        if use_staged:
            # ── Stage 3: Transformer NLP ───────────────────────────────────────
            _log("Stage 3 — Transformer NLP (NER + embeddings)")
            t0 = time.time()
            ner_result = self._transformer.run_ner(doc.chunks)
            if doc.chunks:
                self._transformer.build_chunk_embeddings(doc.chunks)
            _log(f"Stage 3 done ({time.time()-t0:.1f}s)")

            # ── Stage 4: Context Compression ──────────────────────────────────
            _log("Stage 4 — Context Compression")
            t0 = time.time()
            context_result = build_compressed_context(
                chunks=doc.chunks,
                transformer=self._transformer,
                domain_profile=self.domain_profile,
                min_similarity=self.min_similarity,
                max_chars=self.max_context_chars,
            )
            compressed_context = context_result.compressed_text
            _log(f"Stage 4 done ({time.time()-t0:.1f}s)")
        else:
            compressed_context = doc.text

        # Build combined hints from Stages 2 + 3
        hints = _build_hints(classical_facts, ner_result)

        # ── Stage 5: Extraction LLM ────────────────────────────────────────────
        _log("Stage 5 — Extraction LLM")
        t0 = time.time()
        extraction = self._extractor.extract(
            context=compressed_context,
            source_file=file_path,
            language=doc.language,
            hints=hints,
            min_confidence=self.min_confidence,
        )
        _log(f"Stage 5 done ({time.time()-t0:.1f}s)")

        # ── Confidence-based routing ───────────────────────────────────────────
        avg_conf = _average_confidence(extraction)
        _log(f"Average extraction confidence: {avg_conf:.2f}")

        judge_result: Optional[JudgeResult] = None
        revision_rounds = 0

        if self.skip_judge:
            _log("Stage 6 — Judge skipped (--skip-judge)")
        elif avg_conf >= _HIGH_CONFIDENCE_THRESHOLD:
            _log(
                f"Stage 6 — Judge skipped (high confidence={avg_conf:.2f} "
                f">= threshold={_HIGH_CONFIDENCE_THRESHOLD})"
            )
        else:
            # ── Stage 6: LLM Judge + self-correction loop ──────────────────────
            _log("Stage 6 — LLM Judge")
            extraction_dict = self._extractor.to_dict(extraction)

            for round_num in range(_MAX_REVISION_ROUNDS + 1):
                t0 = time.time()
                judge_result = self._judge.validate(
                    extraction_dict=extraction_dict,
                    context=compressed_context,
                )
                _log(f"Stage 6 round {round_num+1} ({time.time()-t0:.1f}s)")

                if not judge_result.needs_revision or round_num >= _MAX_REVISION_ROUNDS:
                    break

                # Self-correction loop
                issues = self._judge.high_severity_issues(judge_result)
                if not issues:
                    break

                _log(
                    f"Stage 6 — Self-correction round {round_num+1}: "
                    f"{len(issues)} issue(s) to fix"
                )
                t0 = time.time()
                extraction = self._extractor.extract_with_revision(
                    context=compressed_context,
                    previous_extraction=extraction_dict,
                    judge_issues=issues,
                    source_file=file_path,
                    language=doc.language,
                    min_confidence=self.min_confidence,
                )
                extraction_dict = self._extractor.to_dict(extraction)
                revision_rounds += 1
                _log(f"Revision done ({time.time()-t0:.1f}s)")

        total_time = time.time() - t_total
        _log(f"Pipeline complete — {total_time:.1f}s total")

        result = PipelineResult(
            extraction=extraction,
            judge_result=judge_result,
            classical_facts=classical_facts,
            ner_result=ner_result,
            context_result=context_result,
            total_time=total_time,
            revision_rounds=revision_rounds,
        )
        _log(result.summary())
        return result


def _build_hints(classical_facts: ClassicalFacts, ner_result: NERResult) -> str:
    parts = []
    if any([
        classical_facts.tender_references,
        classical_facts.deadlines,
        classical_facts.budgets,
        classical_facts.contacts,
    ]):
        parts.append(classical_facts.to_hint_text())

    if any([ner_result.organizations, ner_result.locations]):
        parts.append(ner_result.to_hint_text())

    return "\n\n".join(parts)


def _average_confidence(extraction: TenderExtraction) -> float:
    confidences = [
        getattr(extraction, f).confidence
        for f in ALL_FIELDS
        if getattr(extraction, f) is not None
    ]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)
