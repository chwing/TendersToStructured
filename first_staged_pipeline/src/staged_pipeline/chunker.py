import re
from dataclasses import dataclass
from typing import Optional

try:
    from langdetect import detect as langdetect_detect
except ImportError:
    langdetect_detect = None

_HEADING_PATTERN = re.compile(r"^#{1,4}\s+.+$", re.MULTILINE)
_SECTION_SPLIT = re.compile(r"\n{2,}")


@dataclass
class Chunk:
    index: int
    text: str
    section_title: Optional[str] = None
    language: Optional[str] = None


_MIN_CHARS = 80  # paragraphs shorter than this are merged into the previous chunk


def chunk_document(text: str, max_chars: int = 1200) -> list[Chunk]:
    """One chunk per paragraph, with section tracking.

    Short orphan paragraphs (< _MIN_CHARS) are appended to the previous chunk
    rather than becoming standalone chunks that give BM25 no context.
    Long paragraphs (> max_chars) are emitted as-is — splitting mid-paragraph
    would break sentences and hurt retrieval more than oversized chunks do.
    """
    paragraphs = _split_paragraphs(text)
    chunks: list[Chunk] = []
    current_section: Optional[str] = None
    index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if _is_heading(para):
            current_section = para.lstrip("#").strip()
            # Headings are not emitted as standalone chunks — they become the
            # section_title label on the next real paragraph.
            continue

        # Merge short orphans into the previous chunk for context
        if len(para) < _MIN_CHARS and chunks:
            chunks[-1] = Chunk(
                index=chunks[-1].index,
                text=chunks[-1].text + "\n\n" + para,
                section_title=chunks[-1].section_title,
                language=chunks[-1].language,
            )
            continue

        chunks.append(_make_chunk(index, para, current_section))
        index += 1

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    return _SECTION_SPLIT.split(text)


def _is_heading(para: str) -> bool:
    stripped = para.strip()
    return bool(_HEADING_PATTERN.match(stripped)) or (
        len(stripped) < 120 and stripped.isupper() and len(stripped.split()) <= 10
    )


def _make_chunk(index: int, text: str, section_title: Optional[str]) -> Chunk:
    lang = _detect_lang(text)
    return Chunk(index=index, text=text, section_title=section_title, language=lang)


def _detect_lang(text: str) -> Optional[str]:
    if langdetect_detect is None:
        return None
    try:
        return langdetect_detect(text[:500])
    except Exception:
        return None
