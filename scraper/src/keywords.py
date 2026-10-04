import re

DEFAULT_KEYWORDS = [
    "informatique",
    "logiciel",
    "digital",
    "application",
    "systeme d'information",
    "système d'information",
    "developpement",
    "développement",
    "numerique",
    "numérique",
    "plateforme",
    "cloud",
    "cybersecurite",
    "cybersécurité",
]


def _pattern(keyword: str) -> re.Pattern:
    # Multi-word keywords match as a substring; single words match on a
    # word boundary so short/ambiguous tokens (e.g. an "IT" acronym) don't
    # false-positive inside unrelated French words.
    escaped = re.escape(keyword.lower())
    if " " in keyword:
        return re.compile(escaped)
    return re.compile(rf"\b{escaped}\b")


def matches_keywords(text: str, keywords: list[str] = DEFAULT_KEYWORDS) -> bool:
    haystack = text.lower()
    return any(_pattern(kw).search(haystack) for kw in keywords)
