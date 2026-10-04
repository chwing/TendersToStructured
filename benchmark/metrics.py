"""KPI computation and text report for the extraction benchmark (stdlib only)."""
import csv
import re
import unicodedata
from itertools import combinations

FIELDS = [
    "tender_id", "title", "reference_number",
    "issuing_organization", "department", "country", "city_region",
    "publication_date", "submission_deadline", "questions_deadline", "award_date",
    "budget", "currency", "payment_terms", "financial_guarantee",
    "project_description", "domain", "required_technologies", "deliverables",
    "scope_of_work", "hosting_requirements",
    "num_profiles", "roles_profiles", "seniority_level", "certifications", "mission_duration",
    "required_experience", "company_size", "required_documents",
    "geographic_restrictions", "legal_requirements",
    "evaluation_criteria", "lot_number",
    "is_tech_relevant", "relevance_reason",
]
MANDATORY = [
    "tender_id", "title", "reference_number", "issuing_organization",
    "publication_date", "submission_deadline", "budget",
    "project_description", "domain", "scope_of_work", "is_tech_relevant",
]
CRITICAL = ["title", "issuing_organization", "submission_deadline", "budget"]
DATE_FIELDS = ("publication_date", "submission_deadline", "questions_deadline", "award_date")
NUMERIC_FIELDS = ("budget", "num_profiles", "mission_duration")

LABELS = {
    "full_llm": "Full LLM",
    "staged1": "Staged 1",
    "staged2": "Staged 2",
    "staged2_judge": "Staged 2+judge",
}

_CURRENCIES = {
    "tnd", "dt", "mad", "dh", "dhs", "xof", "xaf", "fcfa", "cfa", "eur", "usd", "dzd", "da",
    "gbp", "chf", "sar", "aed", "qar", "egp", "lyd", "mru", "cad",
    "dinar", "dinars", "dirham", "dirhams", "euro", "euros", "dollar", "dollars", "francs", "franc",
}
_ABSENT = {
    "non mentionne", "non mentionnee", "non specifie", "non specifiee", "non disponible",
    "introuvable", "not found", "not mentioned", "not specified", "aucun", "aucune",
    "pas mentionne", "unknown", "inconnu",
}


def has_value(v):
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() not in ("", "null", "none", "n/a")
    if isinstance(v, dict):
        return any(has_value(x) for x in v.values())
    if isinstance(v, (list, tuple, set)):
        return any(has_value(x) for x in v)
    return True


def to_text(v):
    if isinstance(v, (list, tuple)):
        return "; ".join(to_text(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {to_text(x)}" for k, x in v.items())
    return str(v)


def _ascii(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def suspects(field, v):
    """Cheap no-gold sanity checks: returns a list of reasons the value looks wrong."""
    t = to_text(v).strip()
    a = _ascii(t).strip(" .")
    if a in _ABSENT:
        return ["absence phrase kept as a value"]
    out = []
    if field == "budget" and not re.search(r"\d", t):
        out.append("budget without a number")
    if field == "currency":
        toks = set(re.findall(r"[a-z]+", a))
        if not (toks & _CURRENCIES or "€" in t or "$" in t):
            out.append("currency not in whitelist")
    if field in DATE_FIELDS and not re.search(r"(19|20)\d{2}", t):
        out.append("date without a year")
    if field == "num_profiles":
        m = re.fullmatch(r"\s*(\d+)(?:\.0+)?\s*", t)
        if not m or int(m.group(1)) > 100:
            out.append("num_profiles not a plausible integer")
    if field in ("tender_id", "reference_number") and not re.search(r"\d", t):
        out.append("identifier without digits")
    return out


def _canon_date(t):
    m = re.search(r"(\d{4})-(\d{2})(?:-(\d{2}))?", t)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.search(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", t)
    if m:
        return m.group(3), m.group(2).zfill(2), m.group(1).zfill(2)
    return None


def values_agree(field, a, b):
    ta, tb = to_text(a), to_text(b)
    if field in DATE_FIELDS:
        ca, cb = _canon_date(ta), _canon_date(tb)
        if ca and cb:
            return ca[0] == cb[0] and ca[1] == cb[1] and (ca[2] is None or cb[2] is None or ca[2] == cb[2])
    if field in NUMERIC_FIELDS:
        na = set(re.findall(r"\d+", ta.replace(" ", "")))
        nb = set(re.findall(r"\d+", tb.replace(" ", "")))
        if na and nb:
            return bool(na & nb)
    sa = set(re.findall(r"[a-z0-9]+", _ascii(ta)))
    sb = set(re.findall(r"[a-z0-9]+", _ascii(tb)))
    if not sa or not sb:
        return False
    inter = len(sa & sb)
    return inter / len(sa | sb) >= 0.5 or inter / min(len(sa), len(sb)) >= 0.8


def _populated(rec):
    return {f: v for f, v in (rec.get("fields") or {}).items() if has_value(v.get("value"))}


def _div(a, b):
    return a / b if b else None


def compute(results):
    order = [a for a in results["order"] if results["approaches"].get(a, {}).get("records")]
    sizes = {d["name"]: d.get("chars") for d in results["docs"]}
    threshold = results["meta"].get("size_threshold", 10000)
    pop = {a: {r["doc"]: _populated(r) for r in results["approaches"][a]["records"] if r["ok"]} for a in order}

    kpis = {}
    for a in order:
        recs = results["approaches"][a]["records"]
        n = len(recs)
        ok = [r for r in recs if r["ok"]]
        found = [len(_populated(r)) for r in recs]
        confs = [v["confidence"] for r in recs for v in _populated(r).values()
                 if isinstance(v.get("confidence"), (int, float))]
        sus = [(r["doc"], f, why) for r in ok for f, v in _populated(r).items() for why in suspects(f, v["value"])]
        wall = sum(r["seconds"] for r in recs)
        llm = sum(r.get("llm_seconds", 0.0) for r in recs)
        ptok = sum(r.get("prompt_tokens", 0) for r in recs)
        ctok = sum(r.get("completion_tokens", 0) for r in recs)
        req_est = sum(r.get("req_prompt_chars", 0) for r in recs) / 4
        tokens = ptok + ctok
        source = "Ollama-reported"
        if tokens == 0:
            tokens, source = req_est, "chars/4 estimate"
        crit_docs = [all(f in _populated(r) for f in CRITICAL) for r in recs]

        unique = supported = contestable = 0
        for r in ok:
            others = [pop[o][r["doc"]] for o in order if o != a and r["doc"] in pop[o]]
            for f, v in _populated(r).items():
                peers = [p[f]["value"] for p in others if f in p]
                if not peers:
                    if others:
                        unique += 1
                    continue
                contestable += 1
                if any(values_agree(f, v["value"], pv) for pv in peers):
                    supported += 1

        buckets = {}
        for name in ("short", "long"):
            sel = [r for r in recs if sizes.get(r["doc"]) is not None
                   and ((sizes[r["doc"]] < threshold) == (name == "short"))]
            buckets[name] = {
                "n": len(sel),
                "avg_found": _div(sum(len(_populated(r)) for r in sel), len(sel)),
                "avg_seconds": _div(sum(r["seconds"] for r in sel), len(sel)),
                "avg_tokens": _div(sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in sel), len(sel)),
            }

        total_found = sum(found)
        kpis[a] = {
            "n_docs": n, "n_ok": len(ok), "n_fail": n - len(ok),
            "found_total": total_found,
            "avg_found": _div(total_found, n),
            "fill_pct": _div(total_found, n * len(FIELDS)) if n else None,
            "mandatory_avg": _div(sum(len([f for f in _populated(r) if f in MANDATORY]) for r in recs), n),
            "critical_complete_pct": _div(sum(crit_docs), n),
            "critical_rates": {f: _div(sum(1 for r in recs if f in _populated(r)), n) for f in CRITICAL},
            "avg_conf": _div(sum(confs), len(confs)),
            "high_conf_share": _div(sum(1 for c in confs if c >= 0.9), len(confs)),
            "wall_total": wall, "wall_avg": _div(wall, n),
            "llm_seconds": llm, "overhead_seconds": wall - llm,
            "init_seconds": results["approaches"][a].get("init_seconds"),
            "calls_total": sum(r.get("calls", 0) for r in recs),
            "calls_avg": _div(sum(r.get("calls", 0) for r in recs), n),
            "failed_calls": sum(r.get("failed_calls", 0) for r in recs),
            "prompt_tokens": ptok, "completion_tokens": ctok,
            "tokens_total": tokens, "tokens_avg": _div(tokens, n), "tokens_source": source,
            "f_per_1k": _div(total_found, tokens / 1000) if tokens else None,
            "f_per_min": _div(total_found, wall / 60) if wall else None,
            "sec_per_field": _div(wall, total_found),
            "tokens_per_field": _div(tokens, total_found),
            "suspect_total": len(sus), "suspect_rate": _div(len(sus), total_found),
            "suspect_list": sus,
            "unique_found": unique if len(order) > 1 else None,
            "consensus_support": _div(supported, contestable) if len(order) > 1 else None,
            "peak_rss_mb": max([r.get("peak_rss_mb", 0) for r in recs] or [0]),
            "baseline_rss_mb": results["approaches"][a].get("baseline_rss_mb"),
            "buckets": buckets,
        }

    agree = {}
    for a, b in combinations(order, 2):
        both = same = 0
        for doc, pa in pop[a].items():
            pb = pop[b].get(doc)
            if pb is None:
                continue
            for f in pa:
                if f in pb:
                    both += 1
                    same += values_agree(f, pa[f]["value"], pb[f]["value"])
        agree[(a, b)] = (same, both)
    return order, kpis, agree


def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def _pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def _table(header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]

    def line(r):
        return "  ".join(str(c).ljust(widths[i]) if i == 0 else str(c).rjust(widths[i]) for i, c in enumerate(r))

    return "\n".join([line(header), "  ".join("-" * w for w in widths)] + [line(r) for r in rows])


_LEADERS = [
    ("Most fields found per document", "avg_found", True, lambda v: f"{v:.1f}/35"),
    ("Most mandatory fields found", "mandatory_avg", True, lambda v: f"{v:.1f}/11"),
    ("Best critical-field completeness", "critical_complete_pct", True, lambda v: f"{100 * v:.0f}% of docs"),
    ("Most reliable (fewest failed documents)", "n_fail", False, lambda v: f"{v:.0f} failed"),
    ("Fastest per document", "wall_avg", False, lambda v: f"{v:.1f}s"),
    ("Fewest tokens in total", "tokens_total", False, lambda v: f"{v:,.0f}"),
    ("Fewest LLM calls per document", "calls_avg", False, lambda v: f"{v:.1f}"),
    ("Most fields per 1,000 tokens", "f_per_1k", True, lambda v: f"{v:.1f}"),
    ("Most fields per minute", "f_per_min", True, lambda v: f"{v:.1f}"),
    ("Lowest suspect-value rate", "suspect_rate", False, lambda v: f"{100 * v:.0f}%"),
    ("Best cross-approach support", "consensus_support", True, lambda v: f"{100 * v:.0f}%"),
    ("Lowest peak memory of the worker", "peak_rss_mb", False, lambda v: f"{v:,.0f} MB"),
]


def build_report(results):
    order, k, agree = compute(results)
    if not order:
        return "No benchmark data."
    meta = results["meta"]
    lab = LABELS
    head = ["KPI"] + [lab.get(a, a) for a in order]

    def row(label, fn):
        return [label] + [fn(k[a]) for a in order]

    def group(title):
        return [title] + [""] * len(order)

    rows = [
        group("FIELDS FOUND"),
        row("Avg fields found per doc (of 35)", lambda x: _f(x["avg_found"])),
        row("Fill rate", lambda x: _pct(x["fill_pct"])),
        row("Avg mandatory fields found (of 11)", lambda x: _f(x["mandatory_avg"])),
        row("Docs with all 4 critical fields", lambda x: _pct(x["critical_complete_pct"])),
        row("Avg confidence of found fields", lambda x: _f(x["avg_conf"], 2)),
        row("Share of high-confidence fields", lambda x: _pct(x["high_conf_share"])),
        row("Fields found by this approach only", lambda x: "-" if x["unique_found"] is None else str(x["unique_found"])),
        group("COST"),
        row("Wall time, total (s)", lambda x: _f(x["wall_total"])),
        row("Wall time per doc (s)", lambda x: _f(x["wall_avg"])),
        row("  of which LLM inference (s)", lambda x: _f(x["llm_seconds"])),
        row("  of which pre/post-processing (s)", lambda x: _f(x["overhead_seconds"])),
        row("One-time init: model loading (s)", lambda x: _f(x["init_seconds"])),
        row("LLM calls per doc", lambda x: _f(x["calls_avg"])),
        row("Prompt tokens, total", lambda x: f"{x['prompt_tokens']:,}"),
        row("Completion tokens, total", lambda x: f"{x['completion_tokens']:,}"),
        row("Tokens per doc", lambda x: _f(x["tokens_avg"], 0)),
        row("Peak worker RAM (MB)", lambda x: _f(x["peak_rss_mb"], 0)),
        group("EFFICIENCY"),
        row("Fields per 1,000 tokens", lambda x: _f(x["f_per_1k"])),
        row("Fields per minute", lambda x: _f(x["f_per_min"])),
        row("Seconds per field", lambda x: _f(x["sec_per_field"])),
        row("Tokens per field", lambda x: _f(x["tokens_per_field"], 0)),
        group("RELIABILITY AND QUALITY PROXIES"),
        row("Documents failed / total", lambda x: f"{x['n_fail']}/{x['n_docs']}"),
        row("Failed LLM calls (incl. retries)", lambda x: str(x["failed_calls"])),
        row("Suspect values / found fields", lambda x: f"{x['suspect_total']}/{x['found_total']}"),
        row("Suspect-value rate", lambda x: _pct(x["suspect_rate"])),
        row("Found fields corroborated by another approach", lambda x: _pct(x["consensus_support"])),
    ]

    out = []
    out.append("TENDER EXTRACTION BENCHMARK")
    out.append("=" * 27)
    out.append(
        f"Run: {meta.get('started', '?')} | documents: {len(results['docs'])} | model: {meta.get('model')} | "
        f"num_ctx: {meta.get('num_ctx')} | min_confidence: {meta.get('min_confidence')}"
    )
    if any(a == "staged2_judge" for a in order):
        out.append(f"Judge model (Staged 2+judge only): {meta.get('judge_model')}")
    out.append(f"Documents: {', '.join(d['name'] for d in results['docs'])}")
    for a in order:
        cfgs = results["approaches"][a].get("conditions") or []
        out.append(f"Observed LLM settings, {lab.get(a, a)}: {'; '.join(cfgs) if cfgs else 'none recorded'}")
    out.append("")
    out.append("1. SCOREBOARD")
    out.append(_table(head, rows))
    out.append("")

    out.append("2. LEADER PER KPI")
    for label, key, higher, fmt in _LEADERS:
        vals = {a: k[a][key] for a in order if k[a][key] is not None}
        if len(vals) < 1:
            continue
        best_v = max(vals.values()) if higher else min(vals.values())
        winners = [lab.get(a, a) for a, v in vals.items() if v == best_v]
        out.append(f"  {label}: {' / '.join(winners)} ({fmt(best_v)})")
    out.append("")

    out.append("3. CRITICAL FIELDS FOUND (share of documents)")
    out.append(_table(["Field"] + [lab.get(a, a) for a in order],
                      [[f] + [_pct(k[a]["critical_rates"][f]) for a in order] for f in CRITICAL]))
    out.append("")

    sizes_present = any(k[a]["buckets"][b]["n"] for a in order for b in ("short", "long"))
    if sizes_present:
        thr = meta.get("size_threshold", 10000)
        out.append(f"4. BY DOCUMENT SIZE (short < {thr:,} characters <= long, the Staged 2 router threshold)")
        brow = []
        for b in ("short", "long"):
            n = max(k[a]["buckets"][b]["n"] for a in order)
            brow.append([f"{b} docs (n={n}): avg fields"] + [_f(k[a]["buckets"][b]["avg_found"]) for a in order])
            brow.append([f"{b} docs: avg seconds"] + [_f(k[a]["buckets"][b]["avg_seconds"]) for a in order])
            brow.append([f"{b} docs: avg tokens"] + [_f(k[a]["buckets"][b]["avg_tokens"], 0) for a in order])
        out.append(_table(head, brow))
        out.append("")

    out.append("5. PER DOCUMENT (fields found | seconds | tokens)")
    drows = []
    for d in results["docs"]:
        r = [d["name"][:38], "-" if d.get("chars") is None else f"{d['chars']:,}"]
        for a in order:
            rec = next((x for x in results["approaches"][a]["records"] if x["doc"] == d["name"]), None)
            if rec is None:
                r.append("-")
            elif not rec["ok"]:
                r.append("FAILED")
            else:
                tk = rec.get("prompt_tokens", 0) + rec.get("completion_tokens", 0)
                r.append(f"{len(_populated(rec))} | {rec['seconds']:.0f}s | {tk:,}")
        drows.append(r)
    out.append(_table(["Document", "Chars"] + [lab.get(a, a) for a in order], drows))
    out.append("")

    if len(order) > 1:
        out.append("6. AGREEMENT BETWEEN APPROACHES (same field found by both, values match)")
        for (a, b), (same, both) in agree.items():
            out.append(f"  {lab.get(a, a)} vs {lab.get(b, b)}: {_pct(_div(same, both))} ({same}/{both} fields found by both)")
        out.append("")

    s2 = [a for a in order if a.startswith("staged2")]
    for a in s2:
        out.append(f"7. {lab.get(a, a).upper()} INTERNALS")
        srows = []
        for rec in results["approaches"][a]["records"]:
            e = rec.get("extras") or {}
            srows.append([
                rec["doc"][:38], e.get("route", "-"),
                "-" if e.get("reduction_pct") is None else f"{e['reduction_pct']:.0f}%",
                "yes" if e.get("judge_invoked") else "no",
                "-" if e.get("judge_valid") is None else str(e["judge_valid"]),
                str(e.get("revision_rounds", "-")),
            ])
        out.append(_table(["Document", "Route", "Context reduction", "Judge ran", "Judge valid", "Revisions"], srows))
        out.append("")

    out.append("8. FAILURES AND SUSPECT VALUES")
    any_line = False
    for a in order:
        for r in results["approaches"][a]["records"]:
            if not r["ok"]:
                out.append(f"  [{lab.get(a, a)}] {r['doc']}: {r.get('error')}")
                any_line = True
        for doc, f, why in k[a]["suspect_list"][:15]:
            out.append(f"  [{lab.get(a, a)}] {doc} / {f}: {why}")
            any_line = True
    if not any_line:
        out.append("  none")
    out.append("")

    out.append("NOTES")
    for line in [
        "Fields found counts values that exist, not values that are correct. The suspect-value rate and the cross-approach support are cheap proxies; true accuracy needs a gold file.",
        f"Tokens are {', '.join(sorted({k[a]['tokens_source'] for a in order}))}. Ollama does not re-count tokens for a prompt prefix it already cached, so repeated system prompts can make prompt tokens look lower than the request size.",
        "Peak RAM is the Python worker only; the Ollama process and its model memory are not included.",
        "Failed documents count as zero fields in the averages. The one-time init cost (loading NER/embedding models) is reported separately and excluded from per-document time.",
        "Small local models and a small document set limit how far these numbers generalize.",
    ]:
        out.append(f"  - {line}")
    return "\n".join(out) + "\n"


def write_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["approach", "document", "chars", "ok", "fields_found", "mandatory_found", "avg_confidence",
                    "seconds", "llm_seconds", "calls", "prompt_tokens", "completion_tokens", "peak_rss_mb", "error"])
        sizes = {d["name"]: d.get("chars") for d in results["docs"]}
        for a in results["order"]:
            for r in results["approaches"].get(a, {}).get("records", []):
                p = _populated(r)
                confs = [v["confidence"] for v in p.values() if isinstance(v.get("confidence"), (int, float))]
                w.writerow([
                    a, r["doc"], sizes.get(r["doc"]), r["ok"], len(p), len([f for f in p if f in MANDATORY]),
                    round(sum(confs) / len(confs), 3) if confs else "", round(r["seconds"], 2),
                    round(r.get("llm_seconds", 0), 2), r.get("calls", 0), r.get("prompt_tokens", 0),
                    r.get("completion_tokens", 0), round(r.get("peak_rss_mb", 0)), r.get("error") or "",
                ])
