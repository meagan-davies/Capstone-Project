"""
timing.py
---------
Context manager for per-stage wall-clock timing.
Used by session_runner.py to log how long each pipeline stage takes.
"""

import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def timed(label: str):
    """
    Log elapsed time for a named pipeline stage.

    Usage:
        with timed("ingest"):
            data = load_something()

    Logs:  INFO [timing] ingest: 0.142s
    """
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    logger.info("[timing] %s: %.3fs", label, elapsed)