import pathlib
from typing import Optional

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from langdetect import detect as langdetect_detect
except ImportError:
    langdetect_detect = None


def read_document(path: str) -> tuple[str, Optional[str]]:
    """Return (text, language_code) for a PDF or DOCX file."""
    p = pathlib.Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix in (".docx", ".doc"):
        text = _read_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    language = _detect_language(text)
    return text, language


def _read_pdf(path: str) -> str:
    if pdfplumber is None:
        raise ImportError("pdfplumber is not installed")
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _read_docx(path: str) -> str:
    if DocxDocument is None:
        raise ImportError("python-docx is not installed")
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _detect_language(text: str) -> Optional[str]:
    if langdetect_detect is None or not text.strip():
        return None
    try:
        sample = text[:2000]
        return langdetect_detect(sample)
    except Exception:
        return None
