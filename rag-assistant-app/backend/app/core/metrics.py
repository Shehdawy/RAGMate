"""Tiny dependency-free Prometheus metrics (text exposition format)."""
import threading
from collections import defaultdict


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], int] = defaultdict(int)
        self._duration_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._duration_count: dict[tuple[str, str], int] = defaultdict(int)
        self._counters: dict[str, int] = defaultdict(int)

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        with self._lock:
            self._requests[(method, path, str(status))] += 1
            self._duration_sum[(method, path)] += seconds
            self._duration_count[(method, path)] += 1

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def render(self) -> str:
        with self._lock:
            lines = ["# HELP http_requests_total Total HTTP requests.", "# TYPE http_requests_total counter"]
            for (method, path, status), value in sorted(self._requests.items()):
                lines.append(
                    f'http_requests_total{{method="{method}",path="{_escape(path)}",status="{status}"}} {value}'
                )
            lines += [
                "# HELP http_request_duration_seconds HTTP request latency.",
                "# TYPE http_request_duration_seconds summary",
            ]
            for (method, path), total in sorted(self._duration_sum.items()):
                labels = f'method="{method}",path="{_escape(path)}"'
                lines.append(f"http_request_duration_seconds_sum{{{labels}}} {total:.6f}")
                lines.append(f"http_request_duration_seconds_count{{{labels}}} {self._duration_count[(method, path)]}")
            for name, value in sorted(self._counters.items()):
                lines += [f"# TYPE {name} counter", f"{name} {value}"]
            return "\n".join(lines) + "\n"


metrics = Metrics()
