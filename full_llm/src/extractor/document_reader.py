import pathlib
import re
import shutil
import subprocess
import tempfile
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
    """Return (text, language_code) for a PDF, DOCX, DOC, or TXT file."""
    p = pathlib.Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    elif suffix == ".doc":
        text = _read_doc_legacy(path)
    elif suffix == ".txt":
        text = pathlib.Path(path).read_text(encoding="utf-8")
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
    """Read .docx (modern XML/ZIP format) with python-docx."""
    if DocxDocument is None:
        raise ImportError("python-docx is not installed")
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _read_doc_legacy(path: str) -> str:
    """Read old .doc binary format (Office 97-2003).

    .doc is a proprietary binary — python-docx cannot open it.
    Strategy:
      1. LibreOffice headless conversion to .docx (reliable, free).
      2. Raw binary string extraction (no dependencies, best-effort fallback).
    """
    # ── Option 1: LibreOffice headless ────────────────────────────────────────
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    [soffice, "--headless", "--convert-to", "docx",
                     "--outdir", tmpdir, path],
                    capture_output=True,
                    timeout=60,
                )
                converted = list(pathlib.Path(tmpdir).glob("*.docx"))
                if result.returncode == 0 and converted:
                    return _read_docx(str(converted[0]))
        except Exception:
            pass

    # ── Option 2: Raw binary text extraction ──────────────────────────────────
    with open(path, "rb") as f:
        data = f.read()

    # UTF-16LE runs (Word Unicode content): char + 0x00 pairs, min 5 pairs
    unicode_runs = re.findall(rb"(?:[\x20-\x7e\xc0-\xff]\x00){5,}", data)
    unicode_text = ""
    if unicode_runs:
        raw = b"".join(unicode_runs)
        unicode_text = raw.decode("utf-16-le", errors="ignore")

    # ASCII printable runs, min 6 chars
    ascii_runs = re.findall(rb"[\x20-\x7e\t]{6,}", data)
    ascii_text = b"\n".join(ascii_runs).decode("ascii", errors="ignore")

    text = unicode_text if len(unicode_text) >= len(ascii_text) * 0.8 else ascii_text
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_language(text: str) -> Optional[str]:
    if langdetect_detect is None or not text.strip():
        return None
    try:
        sample = text[:2000]
        return langdetect_detect(sample)
    except Exception:
        return None
