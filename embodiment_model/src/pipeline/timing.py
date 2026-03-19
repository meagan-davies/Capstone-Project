import time, logging
from contextlib import contextmanager

@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    logging.info(f"[timing] {label}: {elapsed:.3f}s")