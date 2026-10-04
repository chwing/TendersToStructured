#!/usr/bin/env python3
"""Runs ONE extraction approach over a list of files and records per-document results.

Spawned by run_benchmark.py (one subprocess per approach, because the three
pipelines use conflicting `src` import layouts). Not meant to be run by hand.
"""
import argparse
import json
import os
import pathlib
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from metrics import FIELDS, has_value  # noqa: E402


class PeakRss:
    def __init__(self):
        import psutil
        self._p = psutil.Process()
        self.peak = self._p.memory_info().rss
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            self.peak = max(self.peak, self._p.memory_info().rss)
            time.sleep(0.2)

    def reset(self):
        self.peak = self._p.memory_info().rss

    @property
    def mb(self):
        return self.peak / 1048576


def mark(proxy, key):
    try:
        urllib.request.urlopen(f"{proxy}/__mark?key={urllib.parse.quote(key)}", timeout=10).read()
    except Exception:
        pass


def build(args):
    """Returns a callable: path -> (extraction, extras dict)."""
    if args.approach == "full_llm":
        from full_llm.src.llm_pipeline.extractor import LLMPipelineExtractor
        ex = LLMPipelineExtractor(
            provider="ollama", model=args.model, base_url=args.proxy, max_retries=3,
            num_ctx=args.num_ctx, timeout=args.timeout, min_confidence=args.min_confidence,
        )
        return lambda p: (ex.extract_file(p), {})

    if args.approach == "staged1":
        sys.path.insert(2, str(ROOT / "first_staged_pipeline"))
        from src.staged_pipeline.pipeline import StagedPipelineExtractor
        ex = StagedPipelineExtractor(
            provider="ollama", model=args.model, base_url=args.proxy, use_gliner=True,
            use_embeddings=True, top_k=5, max_retries=1, timeout=args.timeout,
            num_ctx=args.num_ctx, min_confidence=args.min_confidence,
        )
        return lambda p: (ex.extract_file(p), {})

    from second_staged_pipeline.src.pipeline import HybridTenderPipeline
    pipe = HybridTenderPipeline(
        extractor_model=args.model, extractor_base_url=args.proxy, extractor_num_ctx=args.num_ctx,
        extractor_timeout=args.timeout, extractor_max_retries=3,
        judge_model=args.judge_model, judge_base_url=args.proxy, judge_num_ctx=args.num_ctx,
        judge_timeout=args.timeout, skip_judge=(args.approach == "staged2"),
        min_confidence=args.min_confidence,
    )

    def run(p):
        res = pipe.process_file(p)
        cr = res.context_result
        extras = {
            "route": "staged" if cr is not None else "llm_only",
            "reduction_pct": cr.reduction_pct if cr is not None else None,
            "judge_invoked": res.judge_result is not None,
            "judge_valid": res.judge_result.is_valid if res.judge_result is not None else None,
            "revision_rounds": res.revision_rounds,
        }
        return res.extraction, extras

    return run


def collect(extraction):
    out = {}
    for f in FIELDS:
        ef = getattr(extraction, f, None)
        if ef is not None and has_value(ef.value):
            out[f] = {"value": ef.value, "confidence": ef.confidence}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True, choices=["full_llm", "staged1", "staged2", "staged2_judge"])
    ap.add_argument("--files", required=True, help="JSON file containing a list of document paths")
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--judge-model", default="mistral")
    ap.add_argument("--num-ctx", type=int, required=True)
    ap.add_argument("--min-confidence", type=float, required=True)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = json.loads(pathlib.Path(args.files).read_text(encoding="utf-8"))
    rss = PeakRss()

    print(f"[bench] {args.approach}: loading pipeline ...", flush=True)
    t0 = time.time()
    run = build(args)
    init_seconds = time.time() - t0
    baseline = rss.mb
    print(f"[bench] {args.approach}: ready in {init_seconds:.1f}s, worker RAM {baseline:.0f} MB", flush=True)

    records = []
    for i, path in enumerate(files, 1):
        name = pathlib.Path(path).name
        mark(args.proxy, f"{args.approach}|{name}")
        rss.reset()
        rec = {"doc": name, "ok": True, "error": None, "fields": {}, "extras": {}, "prompt_chars": None}
        t0 = time.time()
        try:
            extraction, extras = run(path)
            rec["fields"] = collect(extraction)
            rec["extras"] = extras
            rec["prompt_chars"] = getattr(extraction, "prompt_chars", None)
        except Exception as e:
            traceback.print_exc()
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        rec["seconds"] = time.time() - t0
        rec["peak_rss_mb"] = rss.mb
        mark(args.proxy, "__idle")
        records.append(rec)
        pathlib.Path(args.out).write_text(
            json.dumps({"init_seconds": init_seconds, "baseline_rss_mb": baseline, "records": records},
                       ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        status = "OK" if rec["ok"] else "FAILED"
        print(f"[bench] {args.approach} {i}/{len(files)} {name}: {status}, {rec['seconds']:.1f}s, "
              f"{len(rec['fields'])} fields", flush=True)


if __name__ == "__main__":
    main()
