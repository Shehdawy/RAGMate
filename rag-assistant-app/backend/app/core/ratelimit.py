"""In-memory sliding-window rate limiter (per key, e.g. client IP)."""
import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, limit_per_minute: int, window_seconds: float = 60.0, clock=time.monotonic):
        self.limit = limit_per_minute
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds). A limit <= 0 disables limiting."""
        if self.limit <= 0:
            return True, 0
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, int(self.window - (now - hits[0])) + 1)
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory: drop idle keys
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    del self._hits[k]
            return True, 0
