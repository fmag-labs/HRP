"""
Timing utilities for performance monitoring.

Provides context managers and decorators for measuring execution time
of code sections and functions.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from loguru import logger


@dataclass
class TimingMetrics:
    """Container for timing metrics of a code section."""

    name: str
    elapsed_seconds: float = 0.0
    sub_timings: dict[str, float] = field(default_factory=dict)

    def log(self, level: str = "info") -> None:
        """
        Log timing metrics.

        Args:
            level: Log level ('debug', 'info', 'warning')
        """
        log_fn = getattr(logger, level)
        log_fn(f"Timing [{self.name}]: {self.elapsed_seconds:.3f}s")

        for sub_name, sub_time in self.sub_timings.items():
            pct = (sub_time / self.elapsed_seconds * 100) if self.elapsed_seconds > 0 else 0
            log_fn(f"  - {sub_name}: {sub_time:.3f}s ({pct:.1f}%)")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "total_seconds": self.elapsed_seconds,
            **{f"sub_{k}": v for k, v in self.sub_timings.items()},
        }


@contextmanager
def timed_section(name: str) -> Generator[TimingMetrics, None, None]:
    """
    Context manager for timing code sections.

    Usage:
        with timed_section("walk_forward") as metrics:
            result = walk_forward_validate(...)
            metrics.sub_timings["data_fetch"] = data_fetch_time

        metrics.log()

    Args:
        name: Name of the code section being timed

    Yields:
        TimingMetrics instance that can be updated with sub-timings
    """
    metrics = TimingMetrics(name=name)
    start = time.perf_counter()
    try:
        yield metrics
    finally:
        metrics.elapsed_seconds = time.perf_counter() - start


class Timer:
    """
    Simple timer class for measuring elapsed time.

    Usage:
        timer = Timer()
        # do work
        elapsed = timer.elapsed()  # Get current elapsed time
        # do more work
        total = timer.stop()  # Get total elapsed time
    """

    def __init__(self, auto_start: bool = True):
        """
        Initialize timer.

        Args:
            auto_start: Whether to start the timer immediately
        """
        self._start_time: float | None = None
        self._stop_time: float | None = None

        if auto_start:
            self.start()

    def start(self) -> "Timer":
        """Start the timer. Returns self for chaining."""
        self._start_time = time.perf_counter()
        self._stop_time = None
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds."""
        if self._start_time is None:
            raise RuntimeError("Timer was never started")
        self._stop_time = time.perf_counter()
        return self._stop_time - self._start_time

    def elapsed(self) -> float:
        """Get current elapsed time without stopping the timer."""
        if self._start_time is None:
            raise RuntimeError("Timer was never started")
        end = self._stop_time if self._stop_time is not None else time.perf_counter()
        return end - self._start_time

    def reset(self) -> "Timer":
        """Reset the timer. Returns self for chaining."""
        self._start_time = None
        self._stop_time = None
        return self


def add_timing_to_result(result: dict[str, Any], metrics: TimingMetrics) -> dict[str, Any]:
    """
    Add timing metrics to a result dictionary.

    Args:
        result: Result dictionary to update
        metrics: Timing metrics to add

    Returns:
        Updated result dictionary with timing info
    """
    result["timing"] = metrics.to_dict()
    return result
