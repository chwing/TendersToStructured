#!/usr/bin/env python3
"""Benchmark Full LLM vs Staged 1 vs Staged 2 on the same documents, same model, same settings.

Every LLM call goes through a small logging proxy in front of Ollama, so calls,
real token counts and LLM time are measured per document without touching the
pipelines. Each approach runs in its own subprocess (worker.py), one after the other.

Usage (from the project root, with Ollama running):
  python benchmark/run_benchmark.py
  python benchmark/run_benchmark.py --approaches full_llm,staged2 --limit 3
  python benchmark/run_benchmark.py --approaches full_llm,staged1,staged2,staged2_judge
  python benchmark/run_benchmark.py --analyze-only benchmark/output/<run-folder>
"""
import argparse
import http.server
import json
import pathlib
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import metrics  # noqa: E402

DOC_SUFFIXES = (".pdf", ".docx", ".doc", ".txt")
ALL_APPROACHES = ["full_llm", "staged1", "staged2", "staged2_judge"]

STATE = {"key": "__idle", "ollama": "http://localhost:11434"}
LOCK = threading.Lock()
STATS = defaultdict(lambda: {
    "calls": 0, "failed_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
    "llm_seconds": 0.0, "load_seconds": 0.0, "req_prompt_chars": 0, "configs": set(),
})


def _record(key, body, data, code):
    try:
        req = json.loads(body or b"{}")
    except Exception:
        req = {}
    chars = sum(len(m.get("content", "")) for m in req.get("messages", [])) + len(req.get("prompt", "") or "")
    opts = req.get("options") or {}
    cfg = f"{req.get('model')} num_ctx={opts.get('num_ctx')} temperature={opts.get('temperature')}"
    resp = {}
    if code == 200:
        try:
            resp = json.loads(data)
        except Exception:
            resp = {}
    with LOCK:
        s = STATS[key]
        s["calls"] += 1
        s["req_prompt_chars"] += chars
        s["configs"].add(cfg)
        if code != 200:
            s["failed_calls"] += 1
            return
        s["prompt_tokens"] += resp.get("prompt_eval_count", 0) or 0
        s["completion_tokens"] += resp.get("eval_count", 0) or 0
        s["llm_seconds"] += (resp.get("total_duration", 0) or 0) / 1e9
        s["load_seconds"] += (resp.get("load_duration", 0) or 0) / 1e9


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def _handle(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/__mark":
            STATE["key"] = urllib.parse.parse_qs(parsed.query).get("key", ["__idle"])[0]
            return self._send(200, b"ok", "text/plain")
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        is_llm = parsed.path in ("/api/chat", "/api/generate")
        if is_llm and body:
            try:
                obj = json.loads(body)
                obj.setdefault("keep_alive", "30m")
                body = json.dumps(obj).encode("utf-8")
            except Exception:
                pass
        req = urllib.request.Request(STATE["ollama"] + self.path, data=body, method=self.command)
        if self.headers.get("Content-Type"):
            req.add_header("Content-Type", self.headers["Content-Type"])
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                data, code = resp.read(), resp.status
        except urllib.error.HTTPError as e:
            data, code = e.read(), e.code
        except Exception as e:
            data, code = json.dumps({"error": str(e)}).encode(), 502
        if is_llm:
            _record(STATE["key"], body, data, code)
        self._send(code, data)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def ollama_get(path):
    with urllib.request.urlopen(STATE["ollama"] + path, timeout=10) as r:
        return json.loads(r.read())


def preflight(args, approaches):
    try:
        tags = ollama_get("/api/tags")
    except Exception as e:
        sys.exit(f"ERROR: cannot reach Ollama at {STATE['ollama']} ({e}). Start it with `ollama serve`.")
    have = {m["name"] for m in tags.get("models", [])}

    def present(model):
        return model in have or (":" not in model and f"{model}:latest" in have)

    needed = [args.model] + ([args.judge_model] if "staged2_judge" in approaches else [])
    for m in needed:
        if not present(m):
            sys.exit(f"ERROR: model '{m}' is not pulled. Run `ollama pull {m}`. Installed: {sorted(have)}")
    try:
        import psutil
        free = psutil.virtual_memory().available / 2**30
        print(f"Free RAM: {free:.1f} GB", flush=True)
        if free < 4:
            print("WARNING: under 4 GB free. Close PyCharm/browsers first or the worker may be OOM-killed.", flush=True)
    except ImportError:
        pass


def discover(input_dir, only, limit):
    files = sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in DOC_SUFFIXES)
    if only:
        wanted = [s.lower() for s in only.split(",")]
        files = [f for f in files if any(w in f.name.lower() for w in wanted)]
    return files[:limit] if limit else files


def doc_length(path):
    try:
        sys.path.insert(0, str(ROOT))
        from full_llm.src.extractor import read_document
        text, _ = read_document(str(path))
        return len(text)
    except Exception:
        return None


def run_worker(approach, files_json, args, out_path, log_path):
    cmd = [
        sys.executable, str(HERE / "worker.py"), "--approach", approach, "--files", str(files_json),
        "--proxy", f"http://127.0.0.1:{args.proxy_port}", "--model", args.model,
        "--judge-model", args.judge_model, "--num-ctx", str(args.num_ctx),
        "--min-confidence", str(args.min_confidence), "--timeout", str(args.timeout), "--out", str(out_path),
    ]
    env = {**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=env, cwd=str(ROOT))
        try:
            for line in proc.stdout:
                log.write(line)
                if line.startswith("[bench]"):
                    print(line.rstrip(), flush=True)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            raise
    if proc.returncode != 0:
        print(f"WARNING: worker for {approach} exited with code {proc.returncode}; see {log_path}", flush=True)


def merge_proxy(records, approach):
    conditions = set()
    for rec in records:
        s = STATS.get(f"{approach}|{rec['doc']}")
        if s:
            conditions |= s["configs"]
        rec.update({
            "calls": s["calls"] if s else 0,
            "failed_calls": s["failed_calls"] if s else 0,
            "prompt_tokens": s["prompt_tokens"] if s else 0,
            "completion_tokens": s["completion_tokens"] if s else 0,
            "llm_seconds": s["llm_seconds"] if s else 0.0,
            "load_seconds": s["load_seconds"] if s else 0.0,
            "req_prompt_chars": s["req_prompt_chars"] if s else 0,
        })
    return sorted(conditions)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(ROOT / "tender_docs"))
    ap.add_argument("--output-dir", default=str(HERE / "output"))
    ap.add_argument("--approaches", default="full_llm,staged1,staged2",
                    help=f"comma list from: {', '.join(ALL_APPROACHES)}")
    ap.add_argument("--model", default="qwen2.5:7b", help="Ollama model used by every approach")
    ap.add_argument("--judge-model", default="mistral", help="Only used by staged2_judge")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--min-confidence", type=float, default=0.40)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--only", default=None, help="comma list of filename substrings to include")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--size-threshold", type=int, default=10_000)
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--proxy-port", type=int, default=11500)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--analyze-only", default=None, help="regenerate the report from a finished run folder")
    args = ap.parse_args()

    if args.analyze_only:
        run_dir = pathlib.Path(args.analyze_only)
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        text = metrics.build_report(results)
        (run_dir / "report.txt").write_text(text, encoding="utf-8")
        metrics.write_csv(results, run_dir / "per_doc.csv")
        print(text)
        return

    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]
    bad = [a for a in approaches if a not in ALL_APPROACHES]
    if bad:
        sys.exit(f"Unknown approach(es): {bad}. Choose from {ALL_APPROACHES}")

    STATE["ollama"] = args.ollama_url.rstrip("/")
    preflight(args, approaches)

    input_dir = pathlib.Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")
    files = discover(input_dir, args.only, args.limit)
    if not files:
        sys.exit("No documents to process.")
    print(f"{len(files)} document(s), approaches: {', '.join(approaches)}, model: {args.model}", flush=True)

    docs = [{"name": f.name, "chars": doc_length(f)} for f in files]

    run_dir = pathlib.Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    files_json = run_dir / "files.json"
    files_json.write_text(json.dumps([str(f) for f in files]), encoding="utf-8")

    server = Server(("127.0.0.1", args.proxy_port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    results = {
        "meta": {
            "started": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model": args.model, "judge_model": args.judge_model, "num_ctx": args.num_ctx,
            "min_confidence": args.min_confidence, "size_threshold": args.size_threshold,
        },
        "docs": docs, "order": approaches, "approaches": {},
    }

    try:
        if not args.no_warmup:
            print("Warming up the model ...", flush=True)
            try:
                body = json.dumps({"model": args.model, "prompt": "ok", "stream": False, "keep_alive": "30m",
                                   "options": {"num_ctx": args.num_ctx}}).encode()
                urllib.request.urlopen(urllib.request.Request(
                    STATE["ollama"] + "/api/generate", data=body,
                    headers={"Content-Type": "application/json"}), timeout=900).read()
            except Exception as e:
                print(f"WARNING: warm-up failed ({e}); first document will include model load time.", flush=True)

        for approach in approaches:
            print(f"\n=== {metrics.LABELS[approach]} ===", flush=True)
            out_path = run_dir / f"{approach}.json"
            run_worker(approach, files_json, args, out_path, run_dir / "logs" / f"{approach}.log")
            if not out_path.exists():
                print(f"No output from {approach}; skipping.", flush=True)
                continue
            data = json.loads(out_path.read_text(encoding="utf-8"))
            data["conditions"] = merge_proxy(data["records"], approach)
            results["approaches"][approach] = data
    except KeyboardInterrupt:
        print("\nInterrupted; reporting what was collected.", flush=True)
    finally:
        server.shutdown()

    (run_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    text = metrics.build_report(results)
    (run_dir / "report.txt").write_text(text, encoding="utf-8")
    metrics.write_csv(results, run_dir / "per_doc.csv")
    print("\n" + text)
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
