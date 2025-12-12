from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional


@contextmanager
def timed(section: str, timings: Optional[Dict[str, float]] = None) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        dur = time.perf_counter() - start
        if timings is not None:
            timings[section] = dur
