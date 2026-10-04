"""Scraper for tuneps.tn (Tunisia's national e-procurement portal).

The public site is an Angular SPA with no server-rendered fallback, but it's
backed by a plain public JSON REST API (no auth needed for listing) that was
found by inspecting the app's network traffic on 2026-09-04:

    POST /api2/portail/bid/master/data   — search/list open tenders
    GET  /api2/ged/vAttachFile/getByBidNo?bidNo=...  — list a tender's attachments

Document *download* (e.g. /api2/ged/vAttachFile/download/{id}) returns 401
without a logged-in supplier session — TUNEPS gates DCE downloads behind
account login, unlike Morocco's anonymous download. fetch_documents() below
therefore cannot fetch bytes; it records the available document names in the
tender's metadata instead, so a human can log in and grab them manually via
detail_url.

The site also has a broken TLS chain (fails standard cert verification), so
requests here disable verification — acceptable for scraping public,
unauthenticated tender listings.
"""

import logging
from typing import Iterator

import requests
import urllib3

from ..keywords import matches_keywords
from ..models import TenderListing
from .base import TenderSource

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://www.tuneps.tn"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TenderExtractorBot/1.0)",
    "Content-Type": "application/json",
}
PAGE_SIZE = 20

log = logging.getLogger(__name__)


class TunepsSource(TenderSource):
    name = "tuneps"
    country = "Tunisia"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.session.verify = False

    def list_tenders(self, keywords: list[str], max_pages: int = 10) -> Iterator[TenderListing]:
        seen_bid_nos: set[str] = set()
        for keyword in keywords:
            offset = 0
            for _ in range(max_pages):
                body = {
                    "listSort": [],
                    "dataSearch": [
                        {"key": "publicYn", "value": "Y", "specificSearch": "="},
                        {"key": "bidNmFr", "value": keyword, "specificSearch": "like"},
                    ],
                    "listCol": [],
                    "pagination": {"offSet": offset, "limit": PAGE_SIZE},
                    "sort": {"nameCol": "publicDt", "direction": "desc nulls last"},
                }
                r = self.session.post(f"{BASE_URL}/api2/portail/bid/master/data", json=body, timeout=30)
                r.raise_for_status()
                payload = r.json().get("payload", {})
                rows = payload.get("data", [])
                if not rows:
                    break

                for row in rows:
                    bid_no = row.get("bidNo")
                    if not bid_no or bid_no in seen_bid_nos:
                        continue
                    title = row.get("bidNmFr") or row.get("bidNmAr") or ""
                    if not matches_keywords(title, keywords):
                        continue
                    seen_bid_nos.add(bid_no)

                    epbid_id = row.get("epBidMasterId")
                    yield TenderListing(
                        source=self.name,
                        country=self.country,
                        reference=bid_no,
                        title=title.strip(),
                        buyer=row.get("bidInstNm"),
                        category=None,
                        publish_date=row.get("publicDt"),
                        deadline=row.get("bdRecvEndDt"),
                        detail_url=f"{BASE_URL}/portail/offres/details/{epbid_id}/{bid_no}",
                        source_ids={"bidNo": bid_no, "epBidMasterId": str(epbid_id)},
                    )

                if len(rows) < PAGE_SIZE:
                    break
                offset += PAGE_SIZE

    def fetch_documents(self, tender: TenderListing) -> list[tuple[str, bytes]]:
        bid_no = tender.source_ids.get("bidNo")
        if not bid_no:
            return []

        r = self.session.get(
            f"{BASE_URL}/api2/ged/vAttachFile/getByBidNo", params={"bidNo": bid_no}, timeout=30
        )
        r.raise_for_status()
        attachments = r.json().get("payload", []) or []

        if attachments:
            names = [a.get("fileNm", "") for a in attachments]
            tender.source_ids["available_documents"] = "; ".join(names)
            tender.source_ids["requires_auth"] = "true"
            log.info(
                "%s has %d document(s) but TUNEPS requires a supplier login to download them: %s",
                bid_no,
                len(names),
                names,
            )

        return []
