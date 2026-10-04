import json
from datetime import datetime, timezone
from pathlib import Path


class SeenIndex:
    """Tracks tender references already scraped, so re-runs only fetch new ones."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._data: dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._data = {}

    def has(self, dedup_key: str) -> bool:
        return dedup_key in self._data

    def mark(self, dedup_key: str, detail_url: str) -> None:
        self._data[dedup_key] = {
            "detail_url": detail_url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
