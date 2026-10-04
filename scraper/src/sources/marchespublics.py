"""Scraper for marchespublics.gov.ma (Morocco's national procurement portal).

The site runs on the Atexo/PRADO PHP framework: pages are plain server-rendered
HTML but results and downloads are driven by postbacks — every "click" is really
a POST of the whole form, with a hidden ``PRADO_PAGESTATE`` field carrying server
state and (for non-plain-submit controls) ``PRADO_POSTBACK_TARGET`` naming which
link/button was "clicked". Reverse-engineered against the live site on 2026-09-04;
selectors may need adjustment if the site's markup changes.

Document download (the 3-step "anonymous DCE" flow in fetch_documents) is
confirmed correct — it fetched a real signed zip during development — but the
site appears to rate-limit or challenge repeated automated postbacks: under
sustained scraping it starts returning the same form back with an empty
validation error instead of the file. fetch_documents retries with backoff,
but a run can still legitimately end with 0 documents downloaded while
listings/metadata are unaffected. When that happens the tender's metadata
(with detail_url) is still saved, so documents can be fetched by hand.
"""

import io
import logging
import time
import zipfile
from typing import Iterator, Optional

import requests
from bs4 import BeautifulSoup

from ..keywords import matches_keywords
from ..models import TenderListing
from .base import TenderSource

BASE_URL = "https://www.marchespublics.gov.ma"
SEARCH_PATH = "/index.php?page=entreprise.EntrepriseAdvancedSearch&searchAnnCons"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TenderExtractorBot/1.0)"}

log = logging.getLogger(__name__)


def _form_fields(form) -> dict:
    """Snapshot a PRADO form's current hidden/text/select values, ready to POST back."""
    data: dict[str, str] = {}
    for inp in form.find_all("input"):
        name = inp.get("name")
        if not name:
            continue
        itype = (inp.get("type") or "text").lower()
        if itype in ("submit", "image"):
            continue
        if itype in ("checkbox", "radio"):
            continue  # opt-in explicitly where needed
        data[name] = inp.get("value", "")
    for sel in form.find_all("select"):
        name = sel.get("name")
        if not name:
            continue
        opt = sel.find("option", selected=True) or sel.find("option")
        data[name] = opt.get("value", "") if opt else ""
    return data


class MarchesPublicsSource(TenderSource):
    name = "marchespublics"
    country = "Morocco"

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)

    def list_tenders(self, keywords: list[str], max_pages: int = 20) -> Iterator[TenderListing]:
        keyword_query = " ".join(keywords[:1])  # the site's keyword box takes one phrase
        r = self.session.get(BASE_URL + SEARCH_PATH, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        form = soup.find("form", id="ctl0_ctl4")
        data = _form_fields(form)
        data["ctl0$CONTENU_PAGE$AdvancedSearch$keywordSearch"] = keyword_query
        data["ctl0$CONTENU_PAGE$AdvancedSearch$lancerRecherche"] = "Lancer la recherche"

        r = self.session.post(BASE_URL + form.get("action"), data=data, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        page = 1
        while page <= max_pages:
            yield from self._parse_results(soup, keywords)

            next_link = soup.find(
                "a", id=lambda i: i and i.endswith("PagerTop_ctl2")
            )
            if not next_link:
                break
            form = soup.find("form", id="ctl0_ctl4")
            data = _form_fields(form)
            data["PRADO_POSTBACK_TARGET"] = next_link["id"].replace("_", "$")
            data["PRADO_POSTBACK_PARAMETER"] = ""
            r = self.session.post(BASE_URL + form.get("action"), data=data, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            page += 1

    def _parse_results(self, soup: BeautifulSoup, keywords: list[str]) -> Iterator[TenderListing]:
        ref_spans = soup.find_all(id=lambda i: i and i.endswith("_reference"))
        for ref_span in ref_spans:
            tr = ref_span.find_parent("tr")
            if tr is None:
                continue
            reference = ref_span.get_text(strip=True)

            objet_div = tr.find(id=lambda i: i and i.endswith("panelBlocObjet"))
            title = ""
            buyer = None
            if objet_div:
                full_text = objet_div.get_text(" ", strip=True)
                if "Acheteur public" in full_text:
                    objet_part, buyer_part = full_text.split("Acheteur public", 1)
                    title = objet_part.replace("Objet", "", 1).strip(" :")
                    buyer = buyer_part.strip(" :")
                else:
                    title = full_text.replace("Objet", "", 1).strip(" :")

            if not matches_keywords(f"{title} {reference}", keywords):
                continue

            detail_link = tr.find(
                "a", href=lambda h: h and "EntrepriseDetailConsultation" in h
            )
            if not detail_link:
                continue
            detail_href = detail_link["href"]

            category = None
            publish_date = None
            first_td = tr.find("td", headers="cons_ref")
            if first_td:
                cat_div = first_td.find(id=lambda i: i and i.endswith("panelBlocCategorie"))
                if cat_div:
                    category = cat_div.get_text(strip=True)
                dates = [d.get_text(strip=True) for d in first_td.find_all("div") if d.get_text(strip=True)]
                for d in dates:
                    if "/" in d and len(d) <= 10:
                        publish_date = d
                        break

            deadline = None
            deadline_td = tr.find("td", headers="cons_dateEnd")
            if deadline_td:
                deadline = deadline_td.get_text(strip=True)

            ref_cons = org_acronyme = None
            for inp in tr.find_all("input", type="hidden"):
                if inp.get("name", "").endswith("$refCons"):
                    ref_cons = inp.get("value")
                elif inp.get("name", "").endswith("$orgCons"):
                    org_acronyme = inp.get("value")

            yield TenderListing(
                source=self.name,
                country=self.country,
                reference=reference,
                title=title,
                buyer=buyer,
                category=category,
                publish_date=publish_date,
                deadline=deadline,
                detail_url=BASE_URL + "/" + detail_href.lstrip("/"),
                source_ids={"refConsultation": ref_cons or "", "orgAcronyme": org_acronyme or ""},
            )

    def fetch_documents(self, tender: TenderListing) -> list[tuple[str, bytes]]:
        ref = tender.source_ids.get("refConsultation")
        org = tender.source_ids.get("orgAcronyme")
        if not ref or not org:
            log.warning("Missing refConsultation/orgAcronyme for %s, skipping download", tender.reference)
            return []

        dce_url = (
            f"{BASE_URL}/index.php?page=entreprise.EntrepriseDemandeTelechargementDce"
            f"&refConsultation={ref}&orgAcronyme={org}"
        )

        # The postback chain below occasionally comes back with a validation
        # error instead of the file — seen in practice under request bursts,
        # so each attempt redoes the whole 3-step flow from a clean GET rather
        # than resuming mid-flow.
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                time.sleep(3 * attempt)

            r = self.session.get(dce_url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            form = soup.find("form", id="ctl0_ctl4")
            if form is None:
                return []

            data = _form_fields(form)
            anon_radio = form.find(
                "input", id="ctl0_CONTENU_PAGE_EntrepriseFormulaireDemande_choixAnonyme"
            )
            if anon_radio:
                data[anon_radio["name"]] = anon_radio.get("value", "")
            accept_terms = form.find(
                "input", id="ctl0_CONTENU_PAGE_EntrepriseFormulaireDemande_accepterConditions"
            )
            if accept_terms:
                data[accept_terms["name"]] = accept_terms.get("value") or "on"
            data["ctl0$CONTENU_PAGE$validateButton"] = "Valider"
            r = self.session.post(BASE_URL + form.get("action"), data=data, timeout=30)
            r.raise_for_status()

            soup = BeautifulSoup(r.text, "html.parser")
            form = soup.find("form", id="ctl0_ctl4")
            if form is None:
                continue
            download_link = soup.find(id=lambda i: i and i.endswith("completeDownload"))
            if download_link is None:
                log.warning("No 'complete download' link for %s (attempt %d)", tender.reference, attempt)
                continue

            data = _form_fields(form)
            data["PRADO_POSTBACK_TARGET"] = download_link["id"].replace("_", "$")
            data["PRADO_POSTBACK_PARAMETER"] = ""
            r = self.session.post(BASE_URL + form.get("action"), data=data, timeout=60)
            r.raise_for_status()

            content_type = r.headers.get("Content-Type", "")
            if "text/html" in content_type:
                log.warning(
                    "Download for %s returned HTML instead of a file (attempt %d/%d)",
                    tender.reference,
                    attempt,
                    max_attempts,
                )
                continue

            return self._extract_zip(r.content, tender.reference)

        log.warning("Giving up on downloading DCE for %s after %d attempts", tender.reference, max_attempts)
        return []

    @staticmethod
    def _extract_zip(content: bytes, reference: str) -> list[tuple[str, bytes]]:
        """The DCE downloads as a single zip; unpack it so the extraction
        pipelines (which only read *.pdf/*.docx/*.doc/*.txt) can see its
        contents directly."""
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                return [
                    (info.filename, zf.read(info))
                    for info in zf.infolist()
                    if not info.is_dir()
                ]
        except zipfile.BadZipFile:
            log.warning("DCE for %s wasn't a valid zip, saving as-is", reference)
            return [(f"{reference}.zip", content)]
