import json
import re
from pathlib import Path

from .models import TenderListing


def _slugify(reference: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", reference.strip())
    return slug.strip("_") or "tender"


def save_tender(
    output_dir: Path,
    tender: TenderListing,
    attachments: list[tuple[str, bytes]],
) -> list[Path]:
    """Write a tender's documents directly into output_dir (flat — this is the
    same folder an extraction pipeline's --input-dir points at, and those
    pipelines only glob *.pdf/*.docx/*.doc/*.txt, so the sidecar .json here is
    ignored by them) plus one <source>_<reference>.metadata.json sidecar per
    tender. Returns the paths of the written document files (not the sidecar)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{tender.source}_{_slugify(tender.reference)}"

    metadata = tender.model_dump()
    metadata["attachments"] = [name for name, _ in attachments]
    (output_dir / f"{prefix}.metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    written = []
    for filename, content in attachments:
        safe_name = re.sub(r"[\\/:*?\"<>|]+", "_", filename)
        doc_path = output_dir / f"{prefix}__{safe_name}"
        doc_path.write_bytes(content)
        written.append(doc_path)

    return written
