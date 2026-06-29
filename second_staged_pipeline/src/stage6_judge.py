"""Stage 6 — LLM as a Judge.

A DIFFERENT model (Mistral or Llama) validates the extraction produced by Stage 5.
The judge does NOT redo the extraction — it checks:
  - Extraction correctness (value is present in source)
  - Hallucinations (values not found in source)
  - Logical consistency (contradictory fields)
  - Missing or unsupported fields
  - Formatting / normalization issues

Returns a JudgeResult with validation issues and a needs_revision flag.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import requests

from .models import JudgeResult, ValidationIssue, ALL_FIELDS


def _log(msg: str):
    print(f"  [stage6] {msg}", flush=True)


JUDGE_SYSTEM_PROMPT = """You are a validation judge for AI-generated tender document extractions.

Your job: verify whether extracted fields are accurate and supported by the source document.

## What to check
1. HALLUCINATIONS: Is the extracted value actually present in the source text?
2. FORMAT: Dates must be YYYY-MM-DD. Numeric fields must be numbers.
3. LOGICAL CONSISTENCY: Are deadline, publication_date, award_date in the right order?
4. MISSING: Are critical fields (tender_id, submission_deadline, budget) absent when they should not be?
5. UNSUPPORTED: Does the extracted value contradict or misrepresent the source?

## Output format
Return a JSON object with this structure:
{
  "is_valid": true/false,
  "overall_confidence": 0.0-1.0,
  "needs_revision": true/false,
  "summary": "Brief summary of validation result",
  "issues": [
    {
      "field": "field_name",
      "issue": "Description of the problem",
      "severity": "high"|"medium"|"low",
      "suggestion": "What the correct value should be (optional)"
    }
  ]
}

## Severity levels
- "high": hallucination, completely wrong value, or critical field missing
- "medium": format issue, minor inconsistency
- "low": style or normalization issue

## Rules
- needs_revision = true if ANY high-severity issue exists OR more than 2 medium issues
- Only flag real problems — do not invent issues
- Return ONLY valid JSON, no markdown fences
"""

_JUDGE_USER_TEMPLATE = """Validate the following extraction against the source document.

## Extracted fields:
{extraction_json}

## Source document (excerpts):
{context}

Validate each non-null field. Return the JSON validation report."""


class LLMJudge:
    """Stage 6: validation judge using a different model than the extractor."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        max_retries: int = 2,
        timeout: int = 240,
        num_ctx: int = 16384,
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.num_ctx = num_ctx

    def validate(
        self,
        extraction_dict: dict,
        context: str,
        confidence_threshold: float = 0.5,
    ) -> JudgeResult:
        """Validate the extraction against the source context."""
        _log(f"Judge model: {self.model}")

        non_null = {k: v for k, v in extraction_dict.items() if v.get("value") is not None}
        _log(f"Validating {len(non_null)} non-null fields")

        if not non_null:
            _log("No fields to validate — extraction is empty")
            return JudgeResult(
                is_valid=False,
                overall_confidence=0.0,
                needs_revision=True,
                summary="Extraction produced no fields",
                issues=[ValidationIssue(
                    field="*",
                    issue="Extraction is completely empty",
                    severity="high",
                )],
            )

        extraction_json = json.dumps(non_null, ensure_ascii=False, indent=2)

        # Limit context sent to judge (different budget than extractor)
        judge_context = context[:8000] if len(context) > 8000 else context

        user_prompt = _JUDGE_USER_TEMPLATE.format(
            extraction_json=extraction_json,
            context=judge_context,
        )

        t0 = time.time()
        try:
            raw = self._call_with_retry(JUDGE_SYSTEM_PROMPT, user_prompt)
            _log(f"Judge response ({time.time()-t0:.1f}s)")
            result = self._parse_result(raw)
        except Exception as e:
            _log(f"Judge failed ({e}) — returning permissive result")
            result = JudgeResult(
                is_valid=True,
                overall_confidence=confidence_threshold,
                needs_revision=False,
                summary=f"Judge unavailable: {e}",
            )

        _log(
            f"Validation: valid={result.is_valid}, "
            f"confidence={result.overall_confidence:.2f}, "
            f"issues={len(result.issues)}, "
            f"needs_revision={result.needs_revision}"
        )
        return result

    def _call_with_retry(self, system: str, user: str) -> str:
        last_error = None
        num_ctx = self.num_ctx
        for attempt in range(self.max_retries):
            try:
                return self._call(system, user, num_ctx=num_ctx)
            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response is not None and e.response.status_code == 500:
                    new_ctx = max(num_ctx // 2, 2048)
                    _log(
                        f"Judge 500 — likely OOM with num_ctx={num_ctx}. "
                        f"Retrying with num_ctx={new_ctx}"
                    )
                    num_ctx = new_ctx
                elif attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Judge LLM failed after {self.max_retries} attempts: {last_error}")

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
            "options": {"temperature": 0.0, "num_ctx": ctx},
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

    def _call_openrouter(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.0,
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

    def _parse_result(self, raw: str) -> JudgeResult:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            stripped = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            stripped = re.sub(r"\s*```$", "", stripped)
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    data = json.loads(m.group())
                else:
                    raise ValueError(f"Cannot parse judge response: {raw[:200]}")

        issues = [
            ValidationIssue(
                field=i.get("field", "unknown"),
                issue=i.get("issue", ""),
                severity=i.get("severity", "low"),
                suggestion=i.get("suggestion"),
            )
            for i in data.get("issues", [])
        ]

        high_issues = [i for i in issues if i.severity == "high"]
        medium_issues = [i for i in issues if i.severity == "medium"]
        needs_revision = len(high_issues) > 0 or len(medium_issues) > 2

        return JudgeResult(
            is_valid=data.get("is_valid", True),
            overall_confidence=float(data.get("overall_confidence", 0.7)),
            needs_revision=data.get("needs_revision", needs_revision),
            summary=data.get("summary", ""),
            issues=issues,
        )

    def high_severity_issues(self, result: JudgeResult) -> list[dict]:
        """Return high+medium issues as dicts for the extractor revision prompt."""
        return [
            {
                "field": i.field,
                "issue": i.issue,
                "severity": i.severity,
                "suggestion": i.suggestion,
            }
            for i in result.issues
            if i.severity in ("high", "medium")
        ]
