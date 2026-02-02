"""Centralized output path helpers for agent research notes."""

from datetime import datetime
from pathlib import Path

from hrp.utils.config import get_config


def research_note_path(slug: str) -> Path:
    """Build research note path with date subfolder and timestamp.

    Args:
        slug: e.g. "02-alpha-researcher", "05-quant-developer"

    Returns:
        Path like ~/hrp-data/output/research/2026-02-02/2026-02-02T093015-02-alpha-researcher.md
    """
    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%dT%H%M%S")

    dir_path = get_config().data.research_dir / date_dir
    dir_path.mkdir(parents=True, exist_ok=True)

    return dir_path / f"{timestamp}-{slug}.md"
