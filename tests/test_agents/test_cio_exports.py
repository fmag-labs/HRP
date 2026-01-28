"""Tests for CIO Agent package exports."""

import pytest


def test_cio_agent_importable():
    """Test CIOAgent can be imported from hrp.agents."""
    from hrp.agents import CIOAgent

    assert CIOAgent is not None
    assert CIOAgent.agent_name == "cio"


def test_cio_dataclasses_importable():
    """Test CIO dataclasses can be imported."""
    from hrp.agents import CIOScore, CIODecision, CIOReport

    assert CIOScore is not None
    assert CIODecision is not None
    assert CIOReport is not None


def test_cio_agent_in_all():
    """Test CIOAgent is in hrp.agents.__all__."""
    from hrp.agents import __all__

    assert "CIOAgent" in __all__
