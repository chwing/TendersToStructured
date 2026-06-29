"""Stage 2 — Classical Information Extraction.

Deterministic extraction using regex and rules — avoids unnecessary AI usage.
Extracts: deadlines, budgets, tender references, contact info.
Covers French, Arabic, and English patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


def _log(msg: str):
    print(f"  [stage2] {msg}", flush=True)


@dataclass
class ClassicalFacts:
    """Facts extracted deterministically before any AI involvement."""
    tender_references: list[str] = field(default_factory=list)
    deadlines: list[str] = field(default_factory=list)
    budgets: list[str] = field(default_factory=list)
    contacts: list[str] = field(default_factory=list)
    publication_dates: list[str] = field(default_factory=list)
    currencies: list[str] = field(default_factory=list)

    def best_deadline(self) -> Optional[str]:
        return self.deadlines[0] if self.deadlines else None

    def best_budget(self) -> Optional[str]:
        return self.budgets[0] if self.budgets else None

    def best_reference(self) -> Optional[str]:
        return self.tender_references[0] if self.tender_references else None

    def to_hint_text(self) -> str:
        """Format facts as a hint block to prepend to LLM context."""
        lines = ["=== PRE-EXTRACTED FACTS (high confidence, from classical extraction) ==="]
        if self.tender_references:
            lines.append(f"Tender references found: {', '.join(self.tender_references[:3])}")
        if self.deadlines:
            lines.append(f"Submission deadline(s) found: {', '.join(self.deadlines[:3])}")
        if self.budgets:
            lines.append(f"Budget/amount found: {', '.join(self.budgets[:3])}")
        if self.contacts:
            lines.append(f"Contact info found: {'; '.join(self.contacts[:3])}")
        if self.publication_dates:
            lines.append(f"Publication date(s) found: {', '.join(self.publication_dates[:3])}")
        if self.currencies:
            lines.append(f"Currency detected: {', '.join(set(self.currencies))}")
        lines.append("=== END PRE-EXTRACTED FACTS ===")
        return "\n".join(lines)


# ── Tender reference patterns ──────────────────────────────────────────────────

_REF_PATTERNS = [
    # AO-2026-001, AO/2026/001
    re.compile(r"\b(AO|DAO|AAO|AMI|RFP|RFQ|ITB|APPEL)[\s/\-]?\w{1,6}[\s/\-]\d{2,6}[\w/\-]*", re.IGNORECASE),
    # N°123/45/2026, Réf: 45892
    re.compile(r"(?:n[°o]|ref(?:erence)?|réf(?:érence)?)[:\s.]*([A-Z0-9]{2,}(?:[/\-][A-Z0-9]{1,}){1,})", re.IGNORECASE),
    # Standalone codes like AO-2026-001 or REF-45892
    re.compile(r"\b([A-Z]{2,5}[\-/][0-9]{2,6}(?:[\-/][A-Z0-9]{1,10})*)\b"),
    # Arabic reference patterns: مرجع، رقم
    re.compile(r"(?:مرجع|رقم)[:\s]*([A-Z0-9/\-]{3,20})", re.UNICODE),
]

# ── Date patterns ──────────────────────────────────────────────────────────────

_DEADLINE_KEYWORDS_FR = [
    r"date\s+limite", r"délai\s+de\s+(?:remise|dépôt|soumission)",
    r"dépôt\s+des?\s+offres?", r"remise\s+des?\s+offres?",
    r"clôture", r"avant\s+le",
]
_DEADLINE_KEYWORDS_EN = [
    r"submission\s+deadline", r"closing\s+date", r"due\s+date",
    r"deadline", r"submit\s+by", r"proposals?\s+due",
]
_DEADLINE_KEYWORDS_AR = [
    r"آخر\s+أجل", r"تاريخ\s+الإيداع", r"موعد\s+تقديم",
]

_DATE_PATTERNS = [
    # DD/MM/YYYY or DD-MM-YYYY
    re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b"),
    # YYYY-MM-DD
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    # DD Month YYYY (French)
    re.compile(
        r"\b(\d{1,2})\s+"
        r"(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)"
        r"\s+(\d{4})\b",
        re.IGNORECASE,
    ),
    # DD Month YYYY (English)
    re.compile(
        r"\b(\d{1,2})\s+"
        r"(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+(\d{4})\b",
        re.IGNORECASE,
    ),
    # Month DD, YYYY (English)
    re.compile(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+(\d{1,2}),?\s+(\d{4})\b",
        re.IGNORECASE,
    ),
]

_FR_MONTH_MAP = {
    "janvier": "01", "février": "02", "mars": "03", "avril": "04",
    "mai": "05", "juin": "06", "juillet": "07", "août": "08",
    "septembre": "09", "octobre": "10", "novembre": "11", "décembre": "12",
}
_EN_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

# ── Budget patterns ────────────────────────────────────────────────────────────

_BUDGET_KEYWORDS = [
    r"budget", r"montant", r"prix\s+maximum", r"enveloppe",
    r"coût\s+(?:total|estimé|prévisionnel)",
    r"amount", r"value", r"contract\s+value",
    r"المبلغ", r"الميزانية", r"التكلفة",
]

_BUDGET_PATTERNS = [
    # 500,000.00 DT / TND / EUR / MAD / DZD / USD
    re.compile(
        r"\b([\d\s,.']+(?:\.\d{1,2})?)\s*"
        r"(DT|TND|EUR|MAD|DZD|USD|FCFA|XOF|GBP|SAR|AED|KWD)"
        r"\b",
        re.IGNORECASE,
    ),
    # 2 million / 1,5 million EUR
    re.compile(
        r"\b(\d+(?:[,\.]\d+)?)\s*(?:million|milliard|billion|thousand|mille)\s*"
        r"(DT|TND|EUR|MAD|DZD|USD|FCFA|XOF|GBP|SAR|AED)?\b",
        re.IGNORECASE,
    ),
    # Amount near keyword
    re.compile(
        r"(?:" + "|".join(_BUDGET_KEYWORDS) + r")\s*[:\s]*"
        r"([\d\s,.']+(?:\.\d{1,2})?)\s*"
        r"(DT|TND|EUR|MAD|DZD|USD|FCFA|XOF|GBP)?",
        re.IGNORECASE,
    ),
]

# ── Currency patterns ──────────────────────────────────────────────────────────

_CURRENCY_RE = re.compile(
    r"\b(DT|TND|EUR|MAD|DZD|USD|FCFA|XOF|GBP|SAR|AED|KWD)\b", re.IGNORECASE
)

# ── Contact patterns ───────────────────────────────────────────────────────────

_CONTACT_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE),
    re.compile(r"(?:tél|tel|fax|phone|téléphone)[.\s:]*(\+?[\d\s()\-./]{7,20})", re.IGNORECASE),
    re.compile(r"\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{1,4}[\s\-]?\d{1,9}"),
]


def extract_classical(text: str) -> ClassicalFacts:
    """Run all classical extractors and return structured facts."""
    facts = ClassicalFacts()

    facts.tender_references = _extract_references(text)
    facts.deadlines = _extract_deadlines(text)
    facts.budgets = _extract_budgets(text)
    facts.contacts = _extract_contacts(text)
    facts.publication_dates = _extract_publication_dates(text)
    facts.currencies = _CURRENCY_RE.findall(text)

    _log(
        f"Classical extraction — refs={len(facts.tender_references)}, "
        f"deadlines={len(facts.deadlines)}, budgets={len(facts.budgets)}, "
        f"contacts={len(facts.contacts)}"
    )
    return facts


def _extract_references(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pattern in _REF_PATTERNS:
        for m in pattern.finditer(text):
            ref = m.group(0).strip()
            if ref.upper() not in seen and len(ref) >= 4:
                seen.add(ref.upper())
                found.append(ref)
    return found[:10]


def _extract_deadlines(text: str) -> list[str]:
    """Find dates that appear near deadline-related keywords."""
    found: list[str] = []
    seen: set[str] = set()

    all_keywords = _DEADLINE_KEYWORDS_FR + _DEADLINE_KEYWORDS_EN + _DEADLINE_KEYWORDS_AR
    keyword_re = re.compile(
        r"(?:" + "|".join(all_keywords) + r")\s*[:\s]*(.{0,120})",
        re.IGNORECASE | re.UNICODE,
    )

    for kw_match in keyword_re.finditer(text):
        window = kw_match.group(0)
        for date_pat in _DATE_PATTERNS:
            for dm in date_pat.finditer(window):
                normalized = _normalize_date(dm)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    found.append(normalized)

    # Also scan all dates in the document and return top ones
    all_dates = _all_dates_in_text(text)
    for d in all_dates:
        if d not in seen:
            seen.add(d)
            found.append(d)

    return found[:5]


def _extract_publication_dates(text: str) -> list[str]:
    """Find dates near publication-related keywords."""
    keywords = [
        r"date\s+de\s+publication", r"publié\s+le", r"émis\s+le",
        r"publication\s+date", r"issued\s+on", r"date\s+d[''']émission",
    ]
    kw_re = re.compile(
        r"(?:" + "|".join(keywords) + r")\s*[:\s]*(.{0,80})",
        re.IGNORECASE,
    )
    found: list[str] = []
    seen: set[str] = set()
    for kw_match in kw_re.finditer(text):
        window = kw_match.group(0)
        for date_pat in _DATE_PATTERNS:
            for dm in date_pat.finditer(window):
                normalized = _normalize_date(dm)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    found.append(normalized)
    return found[:3]


def _all_dates_in_text(text: str) -> list[str]:
    """Extract all dates from the document (for context)."""
    found: list[str] = []
    seen: set[str] = set()
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            normalized = _normalize_date(m)
            if normalized and normalized not in seen:
                seen.add(normalized)
                found.append(normalized)
    return found


def _normalize_date(match: re.Match) -> Optional[str]:
    """Try to convert a regex match into YYYY-MM-DD format."""
    try:
        groups = match.groups()
        full = match.group(0)

        # YYYY-MM-DD
        if re.match(r"^\d{4}-\d{2}-\d{2}$", full):
            return full

        # DD/MM/YYYY or DD-MM-YYYY
        if re.match(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4}$", full):
            parts = re.split(r"[/\-.]", full)
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"

        # DD Month YYYY (French or English)
        if len(groups) == 3:
            g1, g2, g3 = groups
            if g2.lower() in _FR_MONTH_MAP:
                return f"{g3}-{_FR_MONTH_MAP[g2.lower()]}-{g1.zfill(2)}"
            if g2.lower() in _EN_MONTH_MAP:
                return f"{g3}-{_EN_MONTH_MAP[g2.lower()]}-{g1.zfill(2)}"
            # Month DD, YYYY (English)
            if g1.lower() in _EN_MONTH_MAP:
                return f"{g3}-{_EN_MONTH_MAP[g1.lower()]}-{g2.zfill(2)}"
    except Exception:
        pass
    return None


def _extract_budgets(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pat in _BUDGET_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(0).strip()
            key = re.sub(r"\s+", " ", raw)
            if key not in seen and len(key) >= 3:
                seen.add(key)
                found.append(key)
    return found[:5]


def _extract_contacts(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pat in _CONTACT_PATTERNS:
        for m in pat.finditer(text):
            contact = m.group(0).strip()
            if contact not in seen:
                seen.add(contact)
                found.append(contact)
    return found[:10]
