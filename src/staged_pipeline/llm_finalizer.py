from typing import Optional

from src.llm_pipeline.extractor import LLMPipelineExtractor
from src.extractor.models import TenderExtraction
from src.llm_pipeline.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


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
            text_with_hints = f"{ner_hints}\n\n---\n\n{curated_text}"
        else:
            text_with_hints = curated_text

        return self.extract_text(
            text=text_with_hints,
            source_file=source_file,
            language=language,
        )
