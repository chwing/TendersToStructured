#!/usr/bin/env python3
"""Quick smoke-test extractor that emits minimal extractions for debugging.

Creates a `extractions.json`/`extractions.xlsx` under `output/<today>/` so the
DAG / downstream checks can see generated artifacts without contacting LLMs.
"""
import pathlib
import sys
from datetime import date

# Ensure project root is on sys.path so package imports work when run from scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from full_llm.src.extractor.models import TenderExtraction, ExtractedField
from full_llm.src.extractor.output import save_extractions


def main(input_dir: str = "tender_docs_new", output_root: str = "output"):
    p = pathlib.Path(input_dir)
    if not p.exists():
        print(f"Input directory does not exist: {p}")
        return

    files = [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in (".pdf", ".docx", ".doc", ".txt")]
    if not files:
        print(f"No supported files found in {p}")
        return

    extractions = []
    for f in files:
        te = TenderExtraction(
            source_file=str(f),
            document_language=None,
            title=ExtractedField(value=f.stem, confidence=0.90),
        )
        extractions.append(te)

    paths = save_extractions(extractions, output_root=output_root)
    print(f"Wrote dummy extractions: {paths}")


if __name__ == "__main__":
    main()