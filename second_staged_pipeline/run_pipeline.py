#!/usr/bin/env python3
"""
Hybrid AI System for Tender Key Extraction and Validation
6-stage pipeline: Document Processing → Classical → Transformer NLP →
                  Context Compression → Extraction LLM → LLM Judge

Usage:
  python run_pipeline.py <input_dir> [options]

Examples:
  python run_pipeline.py ./tenders
  python run_pipeline.py ./tenders --extractor-model qwen2.5:7b --judge-model mistral
  python run_pipeline.py ./tenders --no-ner --no-embeddings --skip-judge   # fast mode
  python run_pipeline.py ./tenders --extractor-provider openrouter --extractor-api-key sk-...
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import argparse
import pathlib
import sys
import threading
import time

# Allow running from project root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from second_staged_pipeline.src.pipeline import HybridTenderPipeline
from second_staged_pipeline.src.output import save_results


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid AI Tender Extraction — 6-Stage Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", help="Directory containing tender documents (PDF/DOCX/TXT)")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")

    # Stage 3 — Transformer
    g3 = parser.add_argument_group("Stage 3 — Transformer NLP")
    g3.add_argument("--no-ner", action="store_true", help="Disable XLM-RoBERTa NER")
    g3.add_argument("--no-embeddings", action="store_true", help="Disable semantic embeddings")
    g3.add_argument("--ner-model", default="Davlan/xlm-roberta-base-wikiann-ner",
                    help="HuggingFace NER model name")
    g3.add_argument("--embed-model", default="intfloat/multilingual-e5-base",
                    help="HuggingFace embedding model name")

    # Stage 5 — Extraction LLM
    g5 = parser.add_argument_group("Stage 5 — Extraction LLM")
    g5.add_argument("--extractor-provider", default="ollama",
                    choices=["ollama", "openrouter"], help="LLM provider")
    g5.add_argument("--extractor-model", default="qwen2.5:14b",
                    help="Model for extraction (default: qwen2.5:14b)")
    g5.add_argument("--extractor-base-url", default="http://localhost:11434",
                    help="Ollama base URL")
    g5.add_argument("--extractor-api-key", default=None, help="API key for OpenRouter")
    g5.add_argument("--extractor-timeout", type=int, default=300,
                    help="Request timeout in seconds")
    g5.add_argument("--extractor-num-ctx", type=int, default=8192,
                    help="Context window size in tokens (default 8192; reduce if Ollama returns 500)")
    g5.add_argument("--extractor-retries", type=int, default=3,
                    help="Max retries on failure")

    # Stage 6 — Judge LLM
    g6 = parser.add_argument_group("Stage 6 — LLM Judge")
    g6.add_argument("--judge-provider", default="ollama",
                    choices=["ollama", "openrouter"], help="Judge LLM provider")
    g6.add_argument("--judge-model", default="mistral",
                    help="Model for validation (default: mistral, must differ from extractor)")
    g6.add_argument("--judge-base-url", default="http://localhost:11434",
                    help="Judge Ollama base URL")
    g6.add_argument("--judge-api-key", default=None, help="Judge API key for OpenRouter")
    g6.add_argument("--judge-timeout", type=int, default=240,
                    help="Judge request timeout in seconds")
    g6.add_argument("--judge-num-ctx", type=int, default=8000,
                    help="Judge context window in tokens (default 8000)")
    g6.add_argument("--skip-judge", action="store_true",
                    help="Skip Stage 6 validation entirely")

    # Routing
    gr = parser.add_argument_group("Routing")
    gr.add_argument(
        "--staged-threshold", type=int, default=10_000,
        help="Doc char length above which Stage 3+4 run (default: 10000). "
             "Shorter docs go directly to LLM.",
    )

    # Context optimization
    gc = parser.add_argument_group("Context Optimization")
    gc.add_argument("--domain-profile",
                    default="AI software engineering data analytics cloud",
                    help="Domain keywords for semantic relevance scoring")
    gc.add_argument("--min-similarity", type=float, default=0.25,
                    help="Minimum chunk similarity to include in context (0-1)")
    gc.add_argument("--max-context-chars", type=int, default=16000,
                    help="Maximum context characters sent to LLM")

    # General
    parser.add_argument("--min-confidence", type=float, default=0.40,
                        help="Drop fields below this confidence (default: 0.40)")

    args = parser.parse_args()

    # Validate: extractor and judge should be different models
    if (
        not args.skip_judge
        and args.extractor_model == args.judge_model
        and args.extractor_provider == args.judge_provider
    ):
        print(
            f"WARNING: Extractor and judge are using the same model ({args.extractor_model}). "
            f"Using the same model reduces validation reliability. "
            f"Consider using different models (e.g., qwen2.5:14b + mistral).",
            flush=True,
        )

    input_path = pathlib.Path(args.input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {input_path}")
        sys.exit(1)

    files = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in (".pdf", ".docx", ".doc", ".txt")
    )
    if not files:
        print(f"No PDF/DOCX/TXT files found in {input_path}")
        sys.exit(0)

    print(f"Found {len(files)} file(s) in {input_path}", flush=True)

    # Heartbeat thread
    _stop = threading.Event()
    def _heartbeat():
        t0 = time.time()
        while not _stop.wait(15):
            print(f"  [heartbeat] running ... ({int(time.time()-t0)}s)", flush=True)
    threading.Thread(target=_heartbeat, daemon=True).start()

    print("Initialising pipeline ...", flush=True)
    pipeline = HybridTenderPipeline(
        use_ner=not args.no_ner,
        use_embeddings=not args.no_embeddings,
        ner_model=args.ner_model,
        embed_model=args.embed_model,
        extractor_provider=args.extractor_provider,
        extractor_model=args.extractor_model,
        extractor_base_url=args.extractor_base_url,
        extractor_api_key=args.extractor_api_key,
        extractor_timeout=args.extractor_timeout,
        extractor_num_ctx=args.extractor_num_ctx,
        extractor_max_retries=args.extractor_retries,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        judge_timeout=args.judge_timeout,
        judge_num_ctx=args.judge_num_ctx,
        skip_judge=args.skip_judge,
        domain_profile=args.domain_profile,
        min_similarity=args.min_similarity,
        max_context_chars=args.max_context_chars,
        staged_threshold=args.staged_threshold,
        min_confidence=args.min_confidence,
    )

    results = []
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {f.name}", flush=True)
        t0 = time.time()
        try:
            result = pipeline.process_file(str(f))
            elapsed = time.time() - t0
            populated = sum(
                1 for field in __import__(
                    "second_staged_pipeline.src.models", fromlist=["ALL_FIELDS"]
                ).ALL_FIELDS
                if getattr(result.extraction, field) is not None
            )
            print(
                f"  OK — {elapsed:.1f}s, {populated} fields populated, "
                f"{result.revision_rounds} revision(s)",
                flush=True,
            )
            results.append(result)
            paths = save_results(results, output_root=args.output_dir)
            print(f"  -> Saved: {paths['json']}", flush=True)
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}", flush=True)
            traceback.print_exc()

    _stop.set()
    print(f"\nDone. {len(results)}/{len(files)} document(s) extracted successfully.")


if __name__ == "__main__":
    main()
