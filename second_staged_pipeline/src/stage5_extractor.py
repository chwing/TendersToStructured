"""Stage 5 — Extraction LLM.

Uses Qwen2.5 (via Ollama) to extract structured fields from the compressed context.
Receives: relevant sections + extraction schema + pre-detected facts from Stages 2 & 3.
Returns: TenderExtraction with confidence scores.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import requests

from .models import ExtractedField, TenderExtraction, ALL_FIELDS


def _log(msg: str):
    print(f"  [stage5] {msg}", flush=True)


_ABSENCE_PHRASES = {
    "non mentionné", "non mentionnée", "non spécifié", "non spécifiée",
    "non disponible", "introuvable", "not found", "not mentioned",
    "not specified", "n/a", "none", "aucun", "aucune", "pas mentionné",
    "غير مذكور", "غير متوفر", "null", "undefined", "unknown",
}

_TENDER_ID_RE = re.compile(r"[A-Z0-9]{2,}(?:[/\-][A-Z0-9]{1,}){1,}")
_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")

EXTRACTION_SYSTEM_PROMPT = """You are an expert extraction engine for public tender documents (North Africa, Middle East, Europe).

Your ONLY job: extract specific fields from the document excerpts provided and return a structured JSON object.

## Rules
1. Extract ONLY what is explicitly stated in the provided text. Do NOT invent or guess values.
2. If a field is not present, return {"value": null, "confidence": "high"}.
3. Dates MUST follow the format YYYY-MM-DD (or YYYY-MM if day unknown, YYYY if month unknown).
4. tender_id: must be a reference code with separators (/, -) present VERBATIM. Never return a project title.
5. budget: include the amount AND currency (e.g., "500000 DT", "2 million EUR").
6. is_tech_relevant: true if an IT/software/data company can respond; false for construction, catering, transport.
7. domain: choose ONE from: cybersecurity, digital identity, AI, data, cloud, ERP, network, health, agriculture, fintech, e-government, mobility, education, HR, other.
8. num_profiles: small integer (1-50). Never confuse with user counts or financial amounts.
9. Pre-extracted facts provided in the context (marked with ===) are reliable — use them to fill fields.
10. Return ONLY valid JSON — no markdown fences, no explanations.

## Confidence levels
- "high": value explicitly stated in the document
- "medium": value reasonably inferred from context
- "low": value uncertain or partially inferred
"""

_USER_PROMPT_TEMPLATE = """Extract all fields from the tender document below.

{hints_block}

Return a JSON object with EXACTLY these keys, each mapping to {{"value": ..., "confidence": "high"|"medium"|"low"}}:

tender_id, title, reference_number,
issuing_organization, department, country, city_region,
publication_date, submission_deadline, questions_deadline, award_date,
budget, currency, payment_terms, financial_guarantee,
project_description, domain, required_technologies, deliverables, scope_of_work, hosting_requirements,
num_profiles, roles_profiles, seniority_level, certifications, mission_duration,
required_experience, company_size, required_documents, geographic_restrictions, legal_requirements,
evaluation_criteria, lot_number,
is_tech_relevant, relevance_reason

---
DOCUMENT EXCERPTS:
{context}
---

Return ONLY the JSON object."""


class ExtractionLLM:
    """Stage 5: Qwen2.5-based structured extraction."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:14b",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 300,
        num_ctx: int = 32768,
        temperature: float = 0.0,
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.temperature = temperature
        self._last_prompt_chars = 0

    def extract(
        self,
        context: str,
        source_file: str = "",
        language: Optional[str] = None,
        hints: str = "",
        min_confidence: float = 0.0,
    ) -> TenderExtraction:
        hints_block = hints if hints else ""
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            hints_block=hints_block,
            context=context,
        )
        self._last_prompt_chars = len(EXTRACTION_SYSTEM_PROMPT) + len(user_prompt)

        _log(
            f"Calling {self.model} — "
            f"context={len(context)} chars (~{len(context)//4} tokens), "
            f"total prompt={self._last_prompt_chars} chars"
        )
        t0 = time.time()
        raw = self._call_with_retry(EXTRACTION_SYSTEM_PROMPT, user_prompt)
        _log(f"LLM response received ({time.time()-t0:.1f}s)")

        data = self._parse_json(raw)
        extraction = self._build_extraction(data, source_file, language, min_confidence)
        extraction.prompt_chars = self._last_prompt_chars
        extraction.prompt_tokens_est = self._last_prompt_chars // 4
        return extraction

    def extract_with_revision(
        self,
        context: str,
        previous_extraction: dict,
        judge_issues: list[dict],
        source_file: str = "",
        language: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> TenderExtraction:
        """Re-extract with judge feedback for self-correction."""
        issues_text = "\n".join(
            f"- Field '{i['field']}': {i['issue']} (severity: {i['severity']})"
            for i in judge_issues
        )
        revision_prompt = (
            f"The previous extraction had the following issues:\n{issues_text}\n\n"
            f"Previous extraction:\n{json.dumps(previous_extraction, ensure_ascii=False, indent=2)}\n\n"
            f"Please re-extract these fields more carefully from the document below.\n\n"
            f"---\nDOCUMENT EXCERPTS:\n{context}\n---\n\n"
            f"Return ONLY the corrected JSON object with ALL {len(ALL_FIELDS)} fields."
        )
        self._last_prompt_chars = len(EXTRACTION_SYSTEM_PROMPT) + len(revision_prompt)

        _log(f"Revision call — correcting {len(judge_issues)} issues")
        raw = self._call_with_retry(EXTRACTION_SYSTEM_PROMPT, revision_prompt)
        data = self._parse_json(raw)
        extraction = self._build_extraction(data, source_file, language, min_confidence)
        extraction.prompt_chars = self._last_prompt_chars
        extraction.prompt_tokens_est = self._last_prompt_chars // 4
        return extraction

    def _call_with_retry(self, system: str, user: str) -> str:
        last_error = None
        num_ctx = self.num_ctx
        for attempt in range(self.max_retries):
            try:
                return self._call(system, user, num_ctx=num_ctx)
            except requests.exceptions.HTTPError as e:
                last_error = e
                # Ollama returns 500 when num_ctx exceeds available VRAM.
                # Halve the context window on each retry so the model can load.
                if e.response is not None and e.response.status_code == 500:
                    new_ctx = max(num_ctx // 2, 2048)
                    _log(
                        f"Ollama 500 — likely OOM with num_ctx={num_ctx}. "
                        f"Retrying with num_ctx={new_ctx} (attempt {attempt+1}/{self.max_retries})"
                    )
                    num_ctx = new_ctx
                elif attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    _log(f"Retry {attempt+1}/{self.max_retries} after {wait}s (error: {e})")
                    time.sleep(wait)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    _log(f"Retry {attempt+1}/{self.max_retries} after {wait}s (error: {e})")
                    time.sleep(wait)
        raise RuntimeError(f"Extraction LLM failed after {self.max_retries} attempts: {last_error}")

    def _call(self, system: str, user: str, num_ctx: int | None = None) -> str:
        ctx = num_ctx if num_ctx is not None else self.num_ctx
        if self.provider == "ollama":
            return self._call_ollama(system, user, num_ctx=ctx)
        elif self.provider == "openrouter":
            return self._call_openrouter(system, user)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_ollama(self, system: str, user: str, num_ctx: int | None = None) -> str:
        ctx = num_ctx if num_ctx is not None else self.num_ctx
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_ctx": ctx},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        _log(f"Ollama request — model={self.model}, num_ctx={ctx}")
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _call_openrouter(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
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
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        stripped = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        stripped = re.sub(r"\s*```$", "", stripped)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON from LLM response: {raw[:300]}")

    def _build_extraction(
        self,
        data: dict,
        source_file: str,
        language: Optional[str],
        min_confidence: float,
    ) -> TenderExtraction:
        fields: dict[str, Any] = {}
        for field_name in ALL_FIELDS:
            raw = data.get(field_name)
            fields[field_name] = self._coerce_field(field_name, raw, min_confidence)
        return TenderExtraction(
            source_file=source_file,
            document_language=language,
            **fields,
        )

    def _coerce_field(
        self,
        field_name: str,
        raw: Any,
        min_confidence: float,
    ) -> Optional[ExtractedField]:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            return None

        value = raw.get("value")
        confidence = raw.get("confidence", "low")

        value = self._post_process(field_name, value)
        if value is None:
            return None

        ef = ExtractedField(value=value, confidence=confidence)
        if ef.confidence < min_confidence:
            return None
        return ef

    def _post_process(self, field_name: str, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() in _ABSENCE_PHRASES:
                return None
            value = stripped

        if field_name == "tender_id" and isinstance(value, str):
            if not _TENDER_ID_RE.search(value):
                return None

        if field_name in ("publication_date", "submission_deadline", "questions_deadline", "award_date"):
            if isinstance(value, str):
                if not _DATE_RE.match(value):
                    return None

        if field_name == "num_profiles":
            try:
                n = int(value)
                return n if 1 <= n <= 200 else None
            except (TypeError, ValueError):
                return None

        if field_name == "is_tech_relevant" and isinstance(value, str):
            return value.lower() in ("true", "yes", "oui", "1", "vrai")

        return value

    def to_dict(self, extraction: TenderExtraction) -> dict:
        """Convert extraction to flat dict for judge input."""
        result = {}
        for field_name in ALL_FIELDS:
            ef = getattr(extraction, field_name, None)
            if ef is not None:
                result[field_name] = {"value": ef.value, "confidence": ef.confidence}
            else:
                result[field_name] = {"value": None, "confidence": "high"}
        return result
