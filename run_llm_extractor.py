#!/usr/bin/env python3
"""Strategy A — Full LLM extraction."""
import argparse
import pathlib
import shutil
import sys
import time

from src.llm_pipeline.extractor import LLMPipelineExtractor
from src.extractor.output import save_extractions


def main():
    parser = argparse.ArgumentParser(description="TenderExtractor — Strategy A: Full LLM")
    parser.add_argument("input_dir", help="Directory containing tender documents")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openrouter"])
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--treated-dir", default="treated_docs")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-request timeout in seconds (default 600)")
    parser.add_argument("--num-ctx", type=int, default=32768,
                        help="Ollama context window in tokens (default 32768; raise for large docs)")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_dir)
    treated_path = pathlib.Path(args.treated_dir)
    treated_path.mkdir(exist_ok=True)

    already_processed = {f.name for f in treated_path.iterdir()} if treated_path.exists() else set()

    files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in (".pdf", ".docx", ".doc") and f.name not in already_processed
    ]

    if not files:
        print("No new files to process.")
        return

    extractor = LLMPipelineExtractor(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_retries=args.retries,
        num_ctx=args.num_ctx,
        timeout=args.timeout,
    )

    extractions = []
    for f in files:
        print(f"[LLM] Processing {f.name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            extraction = extractor.extract_file(str(f))
            elapsed = time.time() - t0
            tokens_est = extractor._last_prompt_chars // 4
            print(f"OK ({elapsed:.1f}s, ~{tokens_est} tokens)")
            extractions.append(extraction)
            shutil.copy2(f, treated_path / f.name)
        except Exception as e:
            print(f"FAILED: {e}")

    if extractions:
        paths = save_extractions(extractions, strategy="llm", output_root=args.output_dir)
        print(f"\nSaved {len(extractions)} extraction(s):")
        print(f"  JSON : {paths['json']}")
        print(f"  Excel: {paths['xlsx']}")
    else:
        print("No successful extractions.")


if __name__ == "__main__":
    main()
