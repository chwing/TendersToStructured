#!/usr/bin/env python3
"""Merge — LLM (Strategy A) primary, backfilled from Staged (Strategy B).

The LLM pipeline is precise but misses some present fields (e.g. a submission
deadline). The staged pipeline has higher recall. This script keeps every LLM
value and only fills fields that the LLM left null, using the staged value when
its confidence clears --min-confidence. The result is high precision + recovered
recall, as recommended in report.md (Part 5).
"""
import argparse
import json
import pathlib
from datetime import date

from src.extractor.fields import ALL_FIELDS
from src.extractor.models import TenderExtraction
from src.extractor.output import save_extractions


def _latest_dir(root: pathlib.Path, strategy: str) -> pathlib.Path:
    base = root / strategy
    dirs = sorted([d for d in base.iterdir() if d.is_dir()]) if base.exists() else []
    if not dirs:
        raise FileNotFoundError(f"No dated output found under {base}")
    return dirs[-1]


def _load(path: pathlib.Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {rec["source_file"]: rec for rec in data}


def main():
    parser = argparse.ArgumentParser(description="Merge LLM + Staged extractions")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--llm-date", default=None, help="YYYY-MM-DD (default: latest)")
    parser.add_argument("--staged-date", default=None, help="YYYY-MM-DD (default: latest)")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="Only backfill from staged fields at/above this confidence")
    args = parser.parse_args()

    root = pathlib.Path(args.output_dir)
    llm_dir = (root / "llm" / args.llm_date) if args.llm_date else _latest_dir(root, "llm")
    staged_dir = (root / "staged" / args.staged_date) if args.staged_date else _latest_dir(root, "staged")

    llm_recs = _load(llm_dir / "extractions.json")
    staged_recs = _load(staged_dir / "extractions.json")

    print(f"LLM    : {llm_dir / 'extractions.json'} ({len(llm_recs)} docs)")
    print(f"Staged : {staged_dir / 'extractions.json'} ({len(staged_recs)} docs)")

    merged: list[TenderExtraction] = []
    total_backfilled = 0

    for source_file, llm_rec in llm_recs.items():
        staged_rec = staged_recs.get(source_file, {})
        rec = dict(llm_rec)
        filled_here = []

        for field in ALL_FIELDS:
            if rec.get(field) is not None:
                continue  # keep LLM value — precision wins
            staged_val = staged_rec.get(field)
            if staged_val is None:
                continue
            conf = staged_val.get("confidence", 0.0) if isinstance(staged_val, dict) else 0.0
            if conf >= args.min_confidence:
                rec[field] = staged_val
                filled_here.append(field)

        if filled_here:
            total_backfilled += len(filled_here)
            print(f"  {pathlib.Path(source_file).name}: backfilled {len(filled_here)} "
                  f"field(s) from staged -> {', '.join(filled_here)}")

        merged.append(TenderExtraction(**rec))

    paths = save_extractions(merged, strategy="merged", output_root=args.output_dir)
    print(f"\nBackfilled {total_backfilled} field(s) total across {len(merged)} docs.")
    print(f"Saved merged extraction(s):")
    print(f"  JSON : {paths['json']}")
    print(f"  Excel: {paths['xlsx']}")


if __name__ == "__main__":
    main()
