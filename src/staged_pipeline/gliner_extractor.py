from typing import Optional

try:
    from gliner import GLiNER
    _GLINER_AVAILABLE = True
except ImportError:
    _GLINER_AVAILABLE = False

_GLINER_MODEL = "urchade/gliner_multi-v2.1"
_GLINER_LABELS = [
    "organization", "date", "reference number", "budget", "duration",
    "technology", "role", "certification", "city", "country",
]


class GlinerExtractor:
    def __init__(self):
        if not _GLINER_AVAILABLE:
            raise ImportError("gliner is not installed")
        self._model = GLiNER.from_pretrained(_GLINER_MODEL)

    def extract_candidates(self, text: str) -> dict[str, list[str]]:
        """Return a mapping of label → list of candidate strings."""
        if not text.strip():
            return {}

        max_len = 4000
        sample = text[:max_len]

        try:
            entities = self._model.predict_entities(sample, _GLINER_LABELS, threshold=0.4)
        except Exception:
            return {}

        result: dict[str, list[str]] = {}
        for ent in entities:
            label = ent.get("label", "")
            value = ent.get("text", "").strip()
            if label and value:
                result.setdefault(label, [])
                if value not in result[label]:
                    result[label].append(value)

        return result

    def format_hints(self, candidates: dict[str, list[str]]) -> str:
        if not candidates:
            return ""
        lines = ["## NER Candidate Hints"]
        for label, values in candidates.items():
            lines.append(f"- {label}: {', '.join(values[:5])}")
        return "\n".join(lines)
