"""Helpers for saving extractions to JSON and Excel."""
import json
import pathlib
from datetime import date
from typing import Any

from first_staged_pipeline.src.extractor import ALL_FIELDS
from first_staged_pipeline.src.extractor import TenderExtraction


def _field_to_row(extraction: TenderExtraction) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_file": extraction.source_file,
        "document_language": extraction.document_language,
    }
    for field_name in ALL_FIELDS:
        ef = getattr(extraction, field_name, None)
        if ef is None:
            row[field_name] = None
            row[f"{field_name}_confidence"] = None
        else:
            row[field_name] = ef.value
            row[f"{field_name}_confidence"] = ef.confidence
    return row


def save_extractions(
    extractions: list[TenderExtraction],
    strategy: str,
    output_root: str = "output",
) -> dict[str, str]:
    today = date.today().isoformat()
    out_dir = pathlib.Path(output_root) / strategy / today
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "extractions.json"
    xlsx_path = out_dir / "extractions.xlsx"

    # JSON
    data = [e.model_dump() for e in extractions]
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Excel
    import pandas as pd
    rows = [_field_to_row(e) for e in extractions]
    df = pd.DataFrame(rows)
    df.to_excel(str(xlsx_path), index=False)

    return {"json": str(json_path), "xlsx": str(xlsx_path)}
