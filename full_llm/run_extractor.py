#!/usr/bin/env python3
"""Full LLM extraction — feeds the entire document directly to the LLM.

Best for short documents (< 10 000 chars). For longer documents use
second_staged_pipeline which compresses context before the LLM call.

Usage:
  python run_extractor.py <input_dir> [options]

Examples:
  python run_extractor.py ./tender_docs
  python run_extractor.py ./tender_docs --model qwen2.5:14b --num-ctx 32768
  python run_extractor.py ./tender_docs --provider openrouter --api-key sk-...
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from full_llm.src.llm_pipeline.extractor import LLMPipelineExtractor
from full_llm.src.extractor.output import save_extractions


def main():
    parser = argparse.ArgumentParser(
        description="Full LLM Tender Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", help="Directory containing tender documents (PDF/DOCX/TXT)")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openrouter"])
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-request timeout in seconds (default 600)")
    parser.add_argument("--num-ctx", type=int, default=32768,
                        help="Ollama context window in tokens (default 32768)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Drop fields below this confidence (default: 0.0 = keep all)")
    args = parser.parse_args()

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

    extractor = LLMPipelineExtractor(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_retries=args.retries,
        num_ctx=args.num_ctx,
        timeout=args.timeout,
        min_confidence=args.min_confidence,
    )

    extractions = []
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {f.name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            extraction = extractor.extract_file(str(f))
            elapsed = time.time() - t0
            tokens_est = extractor._last_prompt_chars // 4
            print(f"OK ({elapsed:.1f}s, ~{tokens_est} tokens)")
            extractions.append(extraction)
            paths = save_extractions(extractions, output_root=args.output_dir)
            print(f"  -> Saved: {paths['json']}")
        except Exception as e:
            import traceback
            print(f"FAILED: {e}")
            traceback.print_exc()

    print(f"\nDone. {len(extractions)}/{len(files)} document(s) extracted successfully.")


if __name__ == "__main__":
    main()
