from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Minimal normalized schema we aim for after parsing (Phase 1)
# You can extend it later (sensor modality, room mapping, etc.)
@dataclass(frozen=True)
class EventRow:
    timestamp: str
    sensor: str
    value: str
    activity: Optional[str] = None
