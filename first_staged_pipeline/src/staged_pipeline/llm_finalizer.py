from typing import Optional

from src.llm_pipeline.extractor import LLMPipelineExtractor
from first_staged_pipeline.src.extractor import TenderExtraction


class LLMFinalizer(LLMPipelineExtractor):
    """Sends curated (reduced) context to the LLM instead of the full document."""

    def finalize(
        self,
        curated_text: str,
        source_file: str = "",
        language: Optional[str] = None,
        ner_hints: str = "",
    ) -> TenderExtraction:
        if ner_hints:
            curated_text = f"{ner_hints}\n\n---\n\n{curated_text}"

        return self.extract_text(
            text=curated_text,
            source_file=source_file,
            language=language,
        )

    def finalize_single_pass(
        self,
        field_chunks: dict,
        chunks: list,
        source_file: str = "",
        language: Optional[str] = None,
        ner_hints: str = "",
    ) -> TenderExtraction:
        """One LLM call, one system prompt, all 36 fields.

        Context is built from tiered retrieval (mandatory=3 chunks, others=1),
        deduplicated into a single unified context. Per-tier confidence thresholds
        are applied post-extraction per field — mandatory 0.4, nice-to-have 0.6,
        extra 0.5.
        """
        from first_staged_pipeline.src.extractor import FIELD_TIER
        from src.staged_pipeline.retriever import build_unified_context

        TIER_THRESHOLDS = {"mandatory": 0.4, "nice_to_have": 0.6, "extra": 0.5}
        per_field_conf = {f: TIER_THRESHOLDS[FIELD_TIER[f]] for f in FIELD_TIER}

        context = build_unified_context(field_chunks, chunks, header_count=2, max_chars=12_000)
        if ner_hints:
            context = f"{ner_hints}\n\n---\n\n{context}"

        print("  [finalizer] Single-pass — 1 LLM call, all 36 fields, per-tier confidence", flush=True)
        return self.extract_text(
            text=context,
            source_file=source_file,
            language=language,
            fields=None,          # all 36 fields in one prompt
            temperature=0.0,
            per_field_min_confidence=per_field_conf,
        )

    def finalize_tiered(
        self,
        field_chunks: dict,
        chunks: list,
        source_file: str = "",
        language: Optional[str] = None,
        ner_hints: str = "",
    ) -> TenderExtraction:
        """Three-pass extraction: mandatory → nice_to_have → extra.

        Each tier gets its own focused context, temperature, and confidence threshold.
        Results are merged — mandatory values take precedence.
        """
        from first_staged_pipeline.src.extractor import MANDATORY_FIELDS, NICE_TO_HAVE_FIELDS, EXTRA_FIELDS
        from src.staged_pipeline.retriever import build_curated_context_for_fields

        def _add_hints(text: str) -> str:
            return f"{ner_hints}\n\n---\n\n{text}" if ner_hints else text

        # --- Tier 1: mandatory ---
        # Full retrieval (top_k=8 already applied), always include first 4 chunks,
        # no temperature randomness, lenient confidence floor so dates/IDs survive.
        print("  [finalizer] Tier 1 — mandatory fields (temp=0, min_conf=0.4)", flush=True)
        ctx_mandatory = build_curated_context_for_fields(
            field_chunks, chunks,
            fields=MANDATORY_FIELDS,
            header_count=4,
            max_chars=16_000,
        )
        ext_mandatory = self.extract_text(
            text=_add_hints(ctx_mandatory),
            source_file=source_file,
            language=language,
            fields=MANDATORY_FIELDS,
            temperature=0.0,
            min_confidence=0.4,
        )

        # --- Tier 2: nice to have ---
        # Standard retrieval (top_k=5), standard confidence, deterministic.
        print("  [finalizer] Tier 2 — nice-to-have fields (temp=0, min_conf=0.6)", flush=True)
        ctx_nice = build_curated_context_for_fields(
            field_chunks, chunks,
            fields=NICE_TO_HAVE_FIELDS,
            header_count=2,
            max_chars=12_000,
        )
        ext_nice = self.extract_text(
            text=_add_hints(ctx_nice),
            source_file=source_file,
            language=language,
            fields=NICE_TO_HAVE_FIELDS,
            temperature=0.0,
            min_confidence=0.6,
        )

        # --- Tier 3: extra ---
        # Light retrieval (top_k=3), higher temperature so the model can infer
        # implicit values (e.g. hosting from architecture description).
        print("  [finalizer] Tier 3 — extra fields (temp=0.3, min_conf=0.5)", flush=True)
        ctx_extra = build_curated_context_for_fields(
            field_chunks, chunks,
            fields=EXTRA_FIELDS,
            header_count=1,
            max_chars=6_000,
        )
        ext_extra = self.extract_text(
            text=_add_hints(ctx_extra),
            source_file=source_file,
            language=language,
            fields=EXTRA_FIELDS,
            temperature=0.3,
            min_confidence=0.5,
        )

        # Merge: mandatory wins, then nice_to_have, then extra
        merged = self.merge_extractions(
            [ext_mandatory, ext_nice, ext_extra],
            source_file=source_file,
            language=language,
        )

        # Accumulate token counts across all 3 calls
        total_chars = sum(
            e.prompt_chars or 0
            for e in [ext_mandatory, ext_nice, ext_extra]
        )
        merged.prompt_chars = total_chars
        merged.prompt_tokens_est = total_chars // 4
        return merged
