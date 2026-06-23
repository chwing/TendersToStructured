import sys
from dataclasses import dataclass, field
from typing import Optional

try:
    from docling.document_converter import DocumentConverter as DoclingConverter
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


def _log(msg: str):
    print(f"  [ingest] {msg}", flush=True)


@dataclass
class IngestedDocument:
    text: str
    pages: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    parser: str = "unknown"


def ingest(path: str) -> IngestedDocument:
    suffix = path.lower()
    if suffix.endswith(".pdf"):
        return _ingest_pdf(path)
    elif suffix.endswith(".docx") or suffix.endswith(".doc"):
        return _ingest_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def _ingest_pdf(path: str) -> IngestedDocument:
    if _DOCLING_AVAILABLE:
        _log("Trying Docling parser (downloads models on first run — may take a few minutes) ...")
        try:
            result = _try_docling(path)
            if result and result.text.strip():
                _log(f"Docling OK — {len(result.text)} chars extracted")
                return result
            _log("Docling returned empty content, falling back to pdfplumber")
        except Exception as e:
            _log(f"Docling failed ({e}), falling back to pdfplumber")
    return _ingest_pdf_pdfplumber(path)


def _try_docling(path: str) -> Optional[IngestedDocument]:
    _log("Initialising Docling converter ...")
    converter = DoclingConverter()
    _log(f"Converting {path} ...")
    result = converter.convert(path)
    _log("Export to markdown ...")
    md_text = result.document.export_to_markdown()
    if not md_text or not md_text.strip():
        return None
    return IngestedDocument(text=md_text, pages=[], tables=[], parser="docling")


def _ingest_pdf_pdfplumber(path: str) -> IngestedDocument:
    if not _PDFPLUMBER_AVAILABLE:
        raise ImportError("pdfplumber is not installed")
    _log("Using pdfplumber ...")
    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            print(f"\r  [ingest] pdfplumber: page {i}/{total}", end="", flush=True)
            text = page.extract_text()
            if text:
                pages.append(text)
    print()  # newline after progress
    full_text = "\n\n".join(pages)
    _log(f"pdfplumber OK — {len(pages)} pages, {len(full_text)} chars")
    return IngestedDocument(text=full_text, pages=pages, tables=[], parser="pdfplumber")


def _ingest_docx(path: str) -> IngestedDocument:
    if not _DOCX_AVAILABLE:
        raise ImportError("python-docx is not installed")
    _log("Using python-docx ...")
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    _log(f"python-docx OK — {len(paragraphs)} paragraphs, {len(text)} chars")
    return IngestedDocument(text=text, pages=[text], tables=[], parser="python-docx")
