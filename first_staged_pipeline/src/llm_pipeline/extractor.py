import json
import re
import time
from typing import Any, Optional

import requests

from first_staged_pipeline.src.extractor import read_document
from first_staged_pipeline.src.extractor import ALL_FIELDS
from first_staged_pipeline.src.extractor import ExtractedField, TenderExtraction
from src.llm_pipeline.prompt import SYSTEM_PROMPT, make_user_prompt

_ABSENCE_PHRASES = {
    "non mentionné", "non mentionnée", "non spécifié", "non spécifiée",
    "non disponible", "introuvable", "not found", "not mentioned",
    "not specified", "n/a", "none", "aucun", "aucune", "pas mentionné",
    "غير مذكور", "غير متوفر",
}

_TENDER_ID_PATTERN = re.compile(r"[A-Z0-9]{2,}(?:[/\-][A-Z0-9]{1,}){1,}")
_DATE_PATTERN = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


class LLMPipelineExtractor:
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 300,
        num_ctx: int = 32768,
        min_confidence: float = 0.0,
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.min_confidence = min_confidence

        self._last_prompt_chars: int = 0

    def extract_file(self, file_path: str) -> TenderExtraction:
        text, language = read_document(file_path)
        return self.extract_text(text, source_file=file_path, language=language)

    def extract_text(
        self,
        text: str,
        source_file: str = "",
        language: Optional[str] = None,
        fields: Optional[list[str]] = None,
        temperature: float = 0.0,
        min_confidence: Optional[float] = None,
        per_field_min_confidence: Optional[dict[str, float]] = None,
    ) -> TenderExtraction:
        user_prompt = make_user_prompt(text, fields)
        self._last_prompt_chars = len(SYSTEM_PROMPT) + len(user_prompt)

        raw_json = self._call_llm_with_retry(SYSTEM_PROMPT, user_prompt, temperature=temperature)
        data = self._parse_json(raw_json)

        effective_min_conf = min_confidence if min_confidence is not None else self.min_confidence
        extraction = self._build_extraction(
            data, source_file=source_file, language=language,
            min_confidence=effective_min_conf,
            per_field_min_confidence=per_field_min_confidence,
        )
        extraction.prompt_chars = self._last_prompt_chars
        extraction.prompt_tokens_est = self._last_prompt_chars // 4

        # Skip all-null check when extracting a field subset — partial results are expected.
        if fields is None and self._populated_count(extraction) == 0 and len(text) > 3000:
            raise RuntimeError(
                f"All-null extraction on a {len(text)}-char document — likely context "
                f"truncation. Increase num_ctx (current={self.num_ctx}) or use the staged "
                f"pipeline for this file."
            )

        return extraction

    @staticmethod
    def _populated_count(extraction: TenderExtraction) -> int:
        return sum(
            1
            for name in ALL_FIELDS
            if getattr(extraction, name, None) is not None
        )

    def _call_llm_with_retry(self, system: str, user: str, temperature: float = 0.0) -> str:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._call_llm(system, user, temperature=temperature)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

    def _call_llm(self, system: str, user: str, temperature: float = 0.0) -> str:
        if self.provider == "ollama":
            return self._call_ollama(system, user, temperature=temperature)
        elif self.provider == "openrouter":
            return self._call_openrouter(system, user, temperature=temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_ollama(self, system: str, user: str, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _call_openrouter(self, system: str, user: str, temperature: float = 0.0) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _parse_json(self, raw: str) -> dict:
        # Strategy 1: direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 2: strip markdown fences
        stripped = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        stripped = re.sub(r"\s*```$", "", stripped)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Strategy 3: extract first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]}")

    def _build_extraction(
        self, data: dict, source_file: str, language: Optional[str],
        min_confidence: Optional[float] = None,
        per_field_min_confidence: Optional[dict[str, float]] = None,
    ) -> TenderExtraction:
        default_conf = min_confidence if min_confidence is not None else self.min_confidence
        fields: dict[str, Any] = {}
        for field_name in ALL_FIELDS:
            raw = data.get(field_name)
            field_conf = (
                per_field_min_confidence[field_name]
                if per_field_min_confidence and field_name in per_field_min_confidence
                else default_conf
            )
            fields[field_name] = self._coerce_field(field_name, raw, min_confidence=field_conf)

        return TenderExtraction(
            source_file=source_file,
            document_language=language,
            **fields,
        )

    def _coerce_field(
        self, field_name: str, raw: Any, min_confidence: Optional[float] = None
    ) -> Optional[ExtractedField]:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            return None

        value = raw.get("value")
        confidence = raw.get("confidence", "low")

        value = self._post_process_value(field_name, value)
        if value is None:
            return None

        field = ExtractedField(value=value, confidence=confidence)

        effective_min_conf = min_confidence if min_confidence is not None else self.min_confidence
        if field.confidence < effective_min_conf:
            return None

        return field

    @staticmethod
    def merge_extractions(
        extractions: list["TenderExtraction"],
        source_file: str,
        language: Optional[str],
    ) -> "TenderExtraction":
        """Merge tier extractions — first non-None value per field wins."""
        merged: dict[str, Any] = {}
        for field_name in ALL_FIELDS:
            for ext in extractions:
                val = getattr(ext, field_name, None)
                if val is not None:
                    merged[field_name] = val
                    break
            else:
                merged[field_name] = None
        return TenderExtraction(source_file=source_file, document_language=language, **merged)

    def _post_process_value(self, field_name: str, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            if stripped.lower() in _ABSENCE_PHRASES:
                return None
            value = stripped

        if field_name == "tender_id" and isinstance(value, str):
            if not _TENDER_ID_PATTERN.search(value):
                return None

        if field_name in ("publication_date", "submission_deadline", "questions_deadline", "award_date"):
            if isinstance(value, str):
                if not _DATE_PATTERN.match(value):
                    return None
                if "XX" in value or value.endswith("-"):
                    return None

        if field_name == "num_profiles":
            try:
                n = int(value)
                if n <= 0 or n > 200:
                    return None
                return n
            except (TypeError, ValueError):
                return None

        if field_name == "is_tech_relevant" and isinstance(value, str):
            return value.lower() in ("true", "yes", "oui", "1")

        return value
