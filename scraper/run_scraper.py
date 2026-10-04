"""Scrape open tenders from public procurement portals and download their
documents into a folder ready for one of the tendersToStructured extraction
pipelines to consume.

Usage:
    python run_scraper.py --output-dir ../tender_docs_new
    python run_scraper.py --source tuneps --keywords informatique logiciel
    python run_scraper.py --limit 20   # stop after 20 new tenders total
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.dedup import SeenIndex
from src.downloader import save_tender
from src.keywords import DEFAULT_KEYWORDS
from src.sources.marchespublics import MarchesPublicsSource
from src.sources.tuneps import TunepsSource

SOURCES = {
    "marchespublics": MarchesPublicsSource,
    "tuneps": TunepsSource,
}

log = logging.getLogger("run_scraper")


class LimitReached(Exception):
    """Raised internally to break out of scraping once --limit is hit."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=[*SOURCES.keys(), "all"],
        default="all",
        help="Which portal(s) to scrape (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="./tender_docs_new",
        help="Where to write downloaded documents + <source>_<reference>.metadata.json sidecars",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="Only download tenders whose title/reference matches one of these keywords",
    )
    parser.add_argument("--max-pages", type=int, default=10, help="Max result pages per source")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many NEW tenders have been scraped in total across all sources "
             "(useful for testing or throttling a run). Default: no limit.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Re-download tenders even if already seen in a previous run",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Write a JSON manifest of this run's newly-scraped tenders/documents to this path "
             "(e.g. for an orchestrator to scope a downstream extraction step to just these files)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen = SeenIndex(output_dir / ".scraper_state" / "seen.json")

    source_names = list(SOURCES.keys()) if args.source == "all" else [args.source]

    manifest_tenders = []
    total_new = 0
    try:
        for source_name in source_names:
            source = SOURCES[source_name]()
            log.info("Scraping %s (%s) for keywords: %s", source.name, source.country, args.keywords)
            for tender in source.list_tenders(args.keywords, max_pages=args.max_pages):
                if args.limit is not None and total_new >= args.limit:
                    log.info("Reached --limit=%d new tenders, stopping.", args.limit)
                    raise LimitReached()

                if not args.no_dedup and seen.has(tender.dedup_key):
                    log.debug("Already seen %s, skipping", tender.dedup_key)
                    continue

                log.info("New tender: %s — %s", tender.reference, tender.title[:80])
                try:
                    attachments = source.fetch_documents(tender)
                except Exception:
                    log.exception("Failed to fetch documents for %s", tender.reference)
                    attachments = []

                doc_paths = save_tender(output_dir, tender, attachments)
                log.info("Saved %d attachment(s) to %s", len(doc_paths), output_dir)
                seen.mark(tender.dedup_key, tender.detail_url)
                total_new += 1
                manifest_tenders.append(
                    {
                        "source": tender.source,
                        "reference": tender.reference,
                        "detail_url": tender.detail_url,
                        "documents": [str(p) for p in doc_paths],
                    }
                )

                if args.limit is not None and total_new >= args.limit:
                    log.info("Reached --limit=%d new tenders, stopping.", args.limit)
                    raise LimitReached()
    except LimitReached:
        pass

    log.info("Done. %d new tender(s) scraped into %s", total_new, output_dir)

    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "output_dir": str(output_dir),
                    "tenders": manifest_tenders,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log.info("Wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()