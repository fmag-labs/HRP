"""
HRP - Hedgefund Research Platform

Personal quantitative research platform for systematic trading strategy development.
"""

try:
    from importlib.metadata import version

    __version__ = version("hrp")
except Exception:
    __version__ = "0.0.0"  # Fallback for development

__author__ = "Fernando"
