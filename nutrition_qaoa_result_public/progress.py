"""Durable, human-readable progress records for long simulation runs."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def linux_rss_mb() -> float | None:
    """Read this process's resident memory without an extra dependency."""
    status_path = Path("/proc/self/status")
    if not status_path.is_file():
        return None
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


class ProgressLog:
    """Write event history and an atomic latest-state snapshot.

    A separate heartbeat thread leaves evidence that the process was alive even
    when a Qiskit call produces no output for a long time.
    """

    def __init__(self, output_dir: Path, heartbeat_seconds: float = 30.0):
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "progress.jsonl"
        self.status_path = self.output_dir / "status.json"
        self.heartbeat_seconds = heartbeat_seconds
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase = "starting"
        self._state = "running"
        self._last_progress_at = utc_now()
        self._last_progress_elapsed = 0.0
        self._current: dict = {}
        self._last_active_phase: str | None = None
        self._last_active_operation: dict | None = None
        self._rss_mb: float | None = None

    def start(self, **details) -> None:
        self.event("run_start", phase="starting", **details)
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self.event("heartbeat", console=True)

    def event(
        self, name: str, *, phase: str | None = None,
        state: str | None = None, console: bool = True, **details,
    ) -> None:
        elapsed = round(time.monotonic() - self.started, 1)
        timestamp = utc_now()
        with self._lock:
            if name == "heartbeat":
                self._rss_mb = linux_rss_mb()
            if name in {"run_failed", "run_interrupted"}:
                self._last_active_phase = self._phase
                self._last_active_operation = self._current.copy()
            if phase is not None:
                self._phase = phase
            if state is not None:
                self._state = state
            if name != "heartbeat":
                self._last_progress_at = timestamp
                self._last_progress_elapsed = elapsed
                self._current = {"event": name, **details}
            record = {
                "time_utc": timestamp,
                "elapsed_seconds": elapsed,
                "event": name,
                "phase": self._phase,
                "state": self._state,
                "process_rss_mb": self._rss_mb,
                **details,
            }
            status = {
                "state": self._state,
                "phase": self._phase,
                "updated_at_utc": timestamp,
                "elapsed_seconds": elapsed,
                "last_progress_at_utc": self._last_progress_at,
                "seconds_since_progress": round(elapsed - self._last_progress_elapsed, 1),
                "current_operation": self._current,
                "process_rss_mb": self._rss_mb,
                "last_active_phase": self._last_active_phase,
                "last_active_operation": self._last_active_operation,
            }
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary = self.status_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(self.status_path)
        if console:
            if name == "heartbeat":
                print(
                    f"[{timestamp}] +{elapsed:.0f}s heartbeat: phase={self._phase}, "
                    f"waiting {status['seconds_since_progress']:.0f}s, "
                    f"operation={self._current.get('event', 'unknown')}, "
                    f"rss_mb={self._rss_mb}",
                    flush=True,
                )
            else:
                brief = json.dumps(details, ensure_ascii=False, default=str)
                print(f"[{timestamp}] +{elapsed:.0f}s {name}: {brief}", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
