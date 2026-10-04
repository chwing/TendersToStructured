from abc import ABC, abstractmethod
from typing import Iterator

from ..models import TenderListing


class TenderSource(ABC):
    """A scraper for one procurement portal."""

    name: str
    country: str

    @abstractmethod
    def list_tenders(self, keywords: list[str], max_pages: int) -> Iterator[TenderListing]:
        """Yield tenders matching any of the given keywords."""

    @abstractmethod
    def fetch_documents(self, tender: TenderListing) -> list[tuple[str, bytes]]:
        """Return (filename, content) pairs for the tender's downloadable documents."""
