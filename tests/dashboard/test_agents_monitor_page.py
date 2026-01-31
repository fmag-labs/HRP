"""Tests for agents monitor Streamlit page."""

from __future__ import annotations

from pathlib import Path


def test_agents_monitor_page_exists():
    """Agents monitor page file should exist."""
    page_path = Path("hrp/dashboard/pages/agents_monitor_page.py")
    assert page_path.exists()
    # Page should contain basic Streamlit elements
    content = page_path.read_text()
    assert "import streamlit as st" in content
    assert 'st.title("🤖 Agents Monitor")' in content
