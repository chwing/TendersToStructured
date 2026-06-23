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


def chunk_document(text: str, max_chars: int = 1200) -> list[Chunk]:
    paragraphs = _split_paragraphs(text)
    chunks: list[Chunk] = []
    current_section: Optional[str] = None
    index = 0
    buffer = ""

    for para in paragraphs:
        if not para.strip():
            continue

        if _is_heading(para):
            if buffer.strip():
                chunks.append(_make_chunk(index, buffer.strip(), current_section))
                index += 1
                buffer = ""
            current_section = para.strip().lstrip("#").strip()
            buffer = para + "\n\n"
            continue

        if len(buffer) + len(para) > max_chars and buffer.strip():
            chunks.append(_make_chunk(index, buffer.strip(), current_section))
            index += 1
            buffer = ""

        buffer += para + "\n\n"

    if buffer.strip():
        chunks.append(_make_chunk(index, buffer.strip(), current_section))

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
