"""
Tests for Code Materializer Agent.

Tests cover:
- Agent initialization with config
- Strategy spec to code materialization
- Syntax validation
- Code file generation
"""

import os
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def code_materializer_test_db():
    """Create a temporary database with schema for Code Materializer tests."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()
    create_tables(db_path)
    os.environ["HRP_DB_PATH"] = db_path

    from hrp.data.db import get_db

    db = get_db(db_path)
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO data_sources (source_id, source_type, status)
            VALUES ('test_code_materializer', 'code_materializer', 'active')
            """
        )

    yield db_path

    # Cleanup
    try:
        os.remove(db_path)
    except FileNotFoundError:
        pass
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]


# =============================================================================
# Tests
# ==============================================================================


def test_code_materializer_initialization():
    """Code Materializer initializes with default config."""
    from hrp.agents.code_materializer import CodeMaterializer, CodeMaterializerConfig

    config = CodeMaterializerConfig()
    agent = CodeMaterializer(config=config)
    assert agent.ACTOR == "agent:code-materializer"
    assert agent.config.validate_syntax is True


def test_materialize_simple_momentum_strategy(code_materializer_test_db):
    """Materialize simple momentum strategy spec."""
    from hrp.agents.code_materializer import CodeMaterializer

    # Mock hypothesis with strategy spec
    hypothesis = {
        "hypothesis_id": "HYP-TEST-001",
        "title": "Momentum Strategy",
        "metadata": {
            "strategy_spec": {
                "signal_logic": "long top decile of momentum_20d",
                "universe": "sp500",
                "holding_period_days": 20,
                "rebalance_cadence": "weekly",
            }
        },
    }

    agent = CodeMaterializer()
    result = agent._materialize_hypothesis(hypothesis)

    assert result["code_generated"] is True
    assert "momentum_20d" in result["code"]
    assert result["syntax_valid"] is True


def test_syntax_validation():
    """Syntax validation catches invalid code."""
    from hrp.agents.code_materializer import CodeMaterializer

    agent = CodeMaterializer()
    assert agent._validate_syntax("def foo(): return 1") is True
    assert agent._validate_syntax("def foo(: return 1") is False
