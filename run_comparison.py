#!/usr/bin/env python3
"""
Compare LLM vs Staged extractions using semantic similarity.

Loads output/llm/.../extractions.json and output/staged/.../extractions.json
(latest dated folder by default) and compares field by field with embeddings.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import argparse
import json
import pathlib
import time
from datetime import date
from typing import Any, Optional

import numpy as np

from src.extractor.fields import ALL_FIELDS
from src.extractor.models import TenderExtraction, ExtractedField


# ── Semantic similarity engine ───────────────────────────────────────────────

class SemanticEngine:
    """Wraps sentence-transformers or Ollama embeddings for cosine similarity."""

    def __init__(self, backend: str = "st", model: str = "", base_url: str = "http://localhost:11434"):
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self._cache: dict[str, np.ndarray] = {}

        if backend == "st":
            from sentence_transformers import SentenceTransformer
            model = model or "BAAI/bge-m3"
            print(f"  [semantic] Loading sentence-transformers model '{model}' ...", flush=True)
            t0 = time.time()
            self._st_model = SentenceTransformer(model)
            print(f"  [semantic] Model ready ({time.time()-t0:.1f}s)", flush=True)
        elif backend == "ollama":
            import requests as _req
            self._requests = _req
            model = model or "nomic-embed-text"
            self._ollama_model = model
            print(f"  [semantic] Using Ollama embeddings model '{model}'", flush=True)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.model_name = model

    def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        if self.backend == "st":
            vec = self._st_model.encode(text, normalize_embeddings=True)
        else:
            vec = self._ollama_embed(text)
        self._cache[text] = vec
        return vec

    def _ollama_embed(self, text: str) -> np.ndarray:
        resp = self._requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self._ollama_model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def similarity(self, a: Any, b: Any) -> float:
        """Cosine similarity between two values. Null handling included."""
        if a is None and b is None:
            return 1.0
        if a is None or b is None:
            return 0.0
        sa, sb = str(a).strip(), str(b).strip()
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        if sa == sb:
            return 1.0
        try:
            ea, eb = self.embed(sa), self.embed(sb)
            return float(np.clip(np.dot(ea, eb), -1.0, 1.0))
        except Exception:
            # Fallback to character-level similarity
            from difflib import SequenceMatcher
            return SequenceMatcher(None, sa, sb).ratio()


# ── Load helpers ─────────────────────────────────────────────────────────────

def find_latest_json(strategy: str, output_root: str) -> Optional[str]:
    base = pathlib.Path(output_root) / strategy
    if not base.exists():
        return None
    dated_dirs = sorted(
        (d for d in base.iterdir() if d.is_dir()),
        key=lambda d: d.name,
        reverse=True,
    )
    for d in dated_dirs:
        candidate = d / "extractions.json"
        if candidate.exists():
            return str(candidate)
    return None


def load_extractions(json_path: str) -> list[TenderExtraction]:
    data = json.loads(pathlib.Path(json_path).read_text(encoding="utf-8"))
    return [TenderExtraction(**item) for item in data]


def _get_value(extraction: TenderExtraction, field: str) -> Optional[Any]:
    ef: Optional[ExtractedField] = getattr(extraction, field, None)
    return ef.value if ef is not None else None


# ── Comparison metrics ────────────────────────────────────────────────────────

MATCH_THRESHOLD = 0.70


def compute_field_statuses(
    llm: TenderExtraction,
    staged: TenderExtraction,
    engine: SemanticEngine,
) -> dict[str, dict]:
    statuses = {}
    for field in ALL_FIELDS:
        lv = _get_value(llm, field)
        sv = _get_value(staged, field)

        if lv is None and sv is None:
            status = "both_null"
            sim = 1.0
        elif lv is None:
            status = "staged_only"
            sim = 0.0
        elif sv is None:
            status = "llm_only"
            sim = 0.0
        else:
            sim = engine.similarity(lv, sv)
            status = "match" if sim >= MATCH_THRESHOLD else "mismatch"

        statuses[field] = {
            "llm_value": lv,
            "staged_value": sv,
            "status": status,
            "semantic_similarity": round(sim, 4),
        }
    return statuses


def fill_rate(extraction: TenderExtraction) -> float:
    filled = sum(1 for f in ALL_FIELDS if _get_value(extraction, f) is not None)
    return filled / len(ALL_FIELDS)


def compute_doc_metrics(
    file_name: str,
    llm: TenderExtraction,
    staged: TenderExtraction,
    engine: SemanticEngine,
) -> dict:
    print(f"  Comparing {file_name} ...", flush=True)
    field_statuses = compute_field_statuses(llm, staged, engine)

    total = len(ALL_FIELDS)
    match_count = sum(1 for s in field_statuses.values() if s["status"] in ("match", "both_null"))
    substantive = [(f, s) for f, s in field_statuses.items() if s["status"] != "both_null"]
    sub_match = sum(1 for _, s in substantive if s["status"] == "match")
    avg_sim = np.mean([s["semantic_similarity"] for s in field_statuses.values()])

    return {
        "file": file_name,
        "fill_rate_llm": round(fill_rate(llm), 3),
        "fill_rate_staged": round(fill_rate(staged), 3),
        "field_match_rate": round(match_count / total, 3),
        "substantive_match_rate": round(sub_match / len(substantive), 3) if substantive else 1.0,
        "avg_semantic_similarity": round(float(avg_sim), 4),
        "field_statuses": field_statuses,
    }


# ── Report builders ───────────────────────────────────────────────────────────

def build_summary_df(doc_metrics: list[dict]):
    import pandas as pd
    cols = [
        "file", "fill_rate_llm", "fill_rate_staged",
        "field_match_rate", "substantive_match_rate", "avg_semantic_similarity",
    ]
    return pd.DataFrame([{c: m[c] for c in cols} for m in doc_metrics])


def build_fields_df(doc_metrics: list[dict]):
    import pandas as pd
    rows = []
    for m in doc_metrics:
        for field, s in m["field_statuses"].items():
            rows.append({
                "file": m["file"],
                "field": field,
                "llm_value": s["llm_value"],
                "staged_value": s["staged_value"],
                "status": s["status"],
                "semantic_similarity": s["semantic_similarity"],
            })
    return pd.DataFrame(rows)


def build_field_aggregate_df(fields_df):
    import pandas as pd
    rows = []
    for field in ALL_FIELDS:
        sub = fields_df[fields_df["field"] == field]
        n = len(sub)
        rows.append({
            "field": field,
            "n_docs": n,
            "llm_fill_rate": round((sub["llm_value"].notna()).sum() / n, 3) if n else 0,
            "staged_fill_rate": round((sub["staged_value"].notna()).sum() / n, 3) if n else 0,
            "match_rate": round((sub["status"] == "match").sum() / n, 3) if n else 0,
            "mismatch_rate": round((sub["status"] == "mismatch").sum() / n, 3) if n else 0,
            "llm_only_rate": round((sub["status"] == "llm_only").sum() / n, 3) if n else 0,
            "staged_only_rate": round((sub["status"] == "staged_only").sum() / n, 3) if n else 0,
            "both_null_rate": round((sub["status"] == "both_null").sum() / n, 3) if n else 0,
            "avg_semantic_similarity": round(sub["semantic_similarity"].mean(), 4) if n else 0,
        })
    return pd.DataFrame(rows)


def save_report(doc_metrics: list[dict], output_root: str = "output") -> dict[str, str]:
    today = date.today().isoformat()
    out_dir = pathlib.Path(output_root) / "comparison" / today
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "report.json"
    xlsx_path = out_dir / "report.xlsx"

    report_data = []
    for m in doc_metrics:
        entry = {k: v for k, v in m.items() if k != "field_statuses"}
        entry["fields"] = m["field_statuses"]
        report_data.append(entry)
    json_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")

    import pandas as pd
    summary_df = build_summary_df(doc_metrics)
    fields_df = build_fields_df(doc_metrics)
    agg_df = build_field_aggregate_df(fields_df)

    with pd.ExcelWriter(str(xlsx_path), engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        fields_df.to_excel(writer, sheet_name="Fields", index=False)
        agg_df.to_excel(writer, sheet_name="Field Aggregate", index=False)

    return {"json": str(json_path), "xlsx": str(xlsx_path)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM vs Staged extractions (loads from existing output JSONs)"
    )
    parser.add_argument("--output-dir", default="output", help="Root output directory")
    parser.add_argument("--llm-json", default=None, help="Explicit path to LLM extractions.json")
    parser.add_argument("--staged-json", default=None, help="Explicit path to staged extractions.json")
    parser.add_argument(
        "--semantic-backend", default="st", choices=["st", "ollama"],
        help="Embedding backend: 'st' = sentence-transformers BGE-M3 (default), 'ollama' = Ollama embeddings",
    )
    parser.add_argument("--embed-model", default="", help="Override embedding model name")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    args = parser.parse_args()

    # Resolve input files
    llm_json = args.llm_json or find_latest_json("llm", args.output_dir)
    staged_json = args.staged_json or find_latest_json("staged", args.output_dir)

    if not llm_json or not pathlib.Path(llm_json).exists():
        print(f"ERROR: No LLM extractions.json found in {args.output_dir}/llm/")
        print("  Run: python run_llm_extractor.py tender_docs/ first")
        return
    if not staged_json or not pathlib.Path(staged_json).exists():
        print(f"ERROR: No staged extractions.json found in {args.output_dir}/staged/")
        print("  Run: python run_staged_extractor.py tender_docs/ first")
        return

    print(f"LLM source   : {llm_json}")
    print(f"Staged source: {staged_json}")

    llm_extractions = load_extractions(llm_json)
    staged_extractions = load_extractions(staged_json)
    print(f"Loaded {len(llm_extractions)} LLM extractions, {len(staged_extractions)} staged extractions")

    # Match by filename
    staged_by_file = {pathlib.Path(e.source_file).name: e for e in staged_extractions}
    pairs = []
    for llm_e in llm_extractions:
        fname = pathlib.Path(llm_e.source_file).name
        staged_e = staged_by_file.get(fname)
        if staged_e is None:
            print(f"  WARNING: no staged result for '{fname}' — skipping")
            continue
        pairs.append((fname, llm_e, staged_e))

    if not pairs:
        print("No matching files between LLM and staged outputs.")
        return

    # Build semantic engine
    engine = SemanticEngine(
        backend=args.semantic_backend,
        model=args.embed_model,
        base_url=args.base_url,
    )

    # Compare
    print(f"\nComparing {len(pairs)} document(s) field by field (semantic threshold = {MATCH_THRESHOLD}) ...\n")
    doc_metrics = []
    for fname, llm_e, staged_e in pairs:
        m = compute_doc_metrics(fname, llm_e, staged_e, engine)
        doc_metrics.append(m)
        print(
            f"    match={m['field_match_rate']:.0%}  "
            f"substantive={m['substantive_match_rate']:.0%}  "
            f"avg_sim={m['avg_semantic_similarity']:.3f}  "
            f"fill LLM={m['fill_rate_llm']:.0%} staged={m['fill_rate_staged']:.0%}"
        )

    paths = save_report(doc_metrics, output_root=args.output_dir)
    print(f"\nReport saved:")
    print(f"  JSON : {paths['json']}")
    print(f"  Excel: {paths['xlsx']}")


if __name__ == "__main__":
    main()
