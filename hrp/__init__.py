"""
HRP - Hedgefund Research Platform

Personal quantitative research platform for systematic trading strategy development.
"""

# Prefer the literal pyproject version (CalVer YYYY.MMDD.MICRO) so the displayed
# version keeps its leading zero (e.g. 2026.0628.0); installed metadata is PEP 440
# -normalized (2026.628.0). Fall back to metadata for packaged installs without
# pyproject on disk.
try:
    import tomllib
    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        __version__ = tomllib.load(f)["project"]["version"]
except Exception:
    try:
        from importlib.metadata import version

        __version__ = version("hrp")
    except Exception:
        __version__ = "0.0.0"  # Final fallback

__author__ = "Fernando"
