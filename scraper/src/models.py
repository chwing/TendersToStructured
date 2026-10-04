from typing import Optional

from pydantic import BaseModel


class TenderListing(BaseModel):
    """A single tender found by a source scraper's listing search."""

    source: str  # "tuneps" | "marchespublics"
    country: str  # "Tunisia" | "Morocco"
    reference: str
    title: str
    buyer: Optional[str] = None
    category: Optional[str] = None
    publish_date: Optional[str] = None
    deadline: Optional[str] = None
    detail_url: str
    # Free-form identifiers a source needs to re-locate this tender for
    # document download (e.g. Morocco's refConsultation/orgAcronyme pair).
    source_ids: dict[str, str] = {}

    @property
    def dedup_key(self) -> str:
        return f"{self.source}:{self.reference}"
