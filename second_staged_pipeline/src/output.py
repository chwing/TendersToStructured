"""Output serialization — JSON + Excel."""
from __future__ import annotations

import json
import pathlib
from datetime import date
from typing import Any

from .models import TenderExtraction, ALL_FIELDS
from .pipeline import PipelineResult


def _field_to_value(ef) -> Any:
    if ef is None:
        return None
    return ef.value


def _field_to_confidence(ef) -> Any:
    if ef is None:
        return None
    return ef.confidence


def extraction_to_record(result: PipelineResult) -> dict:
    ext = result.extraction
    record: dict[str, Any] = {
        "source_file": ext.source_file,
        "document_language": ext.document_language,
        "prompt_chars": ext.prompt_chars,
        "prompt_tokens_est": ext.prompt_tokens_est,
        "pipeline_time_s": round(result.total_time, 1),
        "revision_rounds": result.revision_rounds,
    }

    if result.judge_result:
        record["judge_valid"] = result.judge_result.is_valid
        record["judge_confidence"] = result.judge_result.overall_confidence
        record["judge_issues"] = len(result.judge_result.issues)
        record["judge_summary"] = result.judge_result.summary

    if result.context_result:
        record["context_chars_before"] = result.context_result.total_chars_before
        record["context_chars_after"] = result.context_result.total_chars_after
        record["context_reduction_pct"] = round(result.context_result.reduction_pct, 1)

    # Classical facts
    record["classical_refs"] = ", ".join(result.classical_facts.tender_references[:3])
    record["classical_deadlines"] = ", ".join(result.classical_facts.deadlines[:3])
    record["classical_budgets"] = ", ".join(result.classical_facts.budgets[:3])

    for field_name in ALL_FIELDS:
        ef = getattr(ext, field_name, None)
        record[field_name] = _field_to_value(ef)
        record[f"{field_name}_conf"] = _field_to_confidence(ef)

    return record


def save_results(
    results: list[PipelineResult],
    output_root: str = "output",
) -> dict[str, str]:
    today = date.today().isoformat()
    out_dir = pathlib.Path(output_root) / today
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [extraction_to_record(r) for r in results]

    json_path = out_dir / "extractions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)

    xlsx_path = None
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        xlsx_path = out_dir / "extractions.xlsx"
        df.to_excel(str(xlsx_path), index=False)
    except ImportError:
        pass

    paths = {"json": str(json_path)}
    if xlsx_path:
        paths["xlsx"] = str(xlsx_path)
    return paths
