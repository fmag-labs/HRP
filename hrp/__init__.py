"""
HRP - Hedgefund Research Platform

Personal quantitative research platform for systematic trading strategy development.
"""

# Read the version from pyproject (CalVer YYYY.MMDD.MICRO, e.g. 2026.628.0) so
# __version__ always matches the source in editable/dev installs without a
# reinstall. Fall back to installed metadata for packaged installs without
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
