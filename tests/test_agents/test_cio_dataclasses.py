"""Tests for CIO Agent dataclasses."""

import pytest
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

# Import what we're about to create
from hrp.agents.cio import CIOScore, CIODecision, CIOReport


class TestCIOScore:
    """Test CIOScore dataclass."""

    def test_create_cio_score(self):
        """Test creating a CIOScore with all dimensions."""
        score = CIOScore(
            hypothesis_id="HYP-2026-001",
            statistical=0.85,
            risk=0.78,
            economic=0.88,
            cost=0.75,
        )
        assert score.hypothesis_id == "HYP-2026-001"
        assert score.statistical == 0.85
        assert score.risk == 0.78
        assert score.economic == 0.88
        assert score.cost == 0.75

    def test_total_score_calculation(self):
        """Test that total_score is the average of 4 dimensions."""
        score = CIOScore(
            hypothesis_id="HYP-2026-001",
            statistical=0.8,
            risk=0.6,
            economic=0.9,
            cost=0.7,
        )
        # (0.8 + 0.6 + 0.9 + 0.7) / 4 = 0.75
        assert score.total == 0.75

    def test_decision_continue(self):
        """Test CONTINUE decision for score >= 0.75."""
        score = CIOScore(
            hypothesis_id="HYP-2026-001",
            statistical=0.85,
            risk=0.78,
            economic=0.88,
            cost=0.75,
        )
        assert score.total >= 0.75
        assert score.decision == "CONTINUE"

    def test_decision_conditional(self):
        """Test CONDITIONAL decision for score 0.50-0.74."""
        score = CIOScore(
            hypothesis_id="HYP-2026-002",
            statistical=0.6,
            risk=0.55,
            economic=0.62,
            cost=0.58,
        )
        assert 0.50 <= score.total < 0.75
        assert score.decision == "CONDITIONAL"

    def test_decision_kill(self):
        """Test KILL decision for score < 0.50."""
        score = CIOScore(
            hypothesis_id="HYP-2026-003",
            statistical=0.3,
            risk=0.35,
            economic=0.28,
            cost=0.31,
        )
        assert score.total < 0.50
        assert score.decision == "KILL"

    def test_critical_failure_auto_pivot(self):
        """Test PIVOT decision when critical_failure is True."""
        score = CIOScore(
            hypothesis_id="HYP-2026-004",
            statistical=0.85,
            risk=0.78,
            economic=0.88,
            cost=0.75,
            critical_failure=True,  # Override score
        )
        assert score.total >= 0.75  # Would be CONTINUE
        assert score.decision == "PIVOT"  # But critical failure overrides


class TestCIODecision:
    """Test CIODecision dataclass."""

    def test_create_decision_with_continue(self):
        """Test creating a CONTINUE decision."""
        score = CIOScore(
            hypothesis_id="HYP-2026-001",
            statistical=0.85,
            risk=0.78,
            economic=0.88,
            cost=0.75,
        )
        decision = CIODecision(
            hypothesis_id="HYP-2026-001",
            decision="CONTINUE",
            score=score,
            rationale="Strong candidate across all dimensions",
            evidence={"mlflow_run": "abc123"},
            paper_allocation=0.042,  # 4.2% allocation
        )
        assert decision.decision == "CONTINUE"
        assert decision.paper_allocation == 0.042
        assert decision.pivot_direction is None

    def test_create_decision_with_pivot(self):
        """Test creating a PIVOT decision with direction."""
        score = CIOScore(
            hypothesis_id="HYP-2026-005",
            statistical=0.5,
            risk=0.4,
            economic=0.6,
            cost=0.5,
            critical_failure=True,
        )
        decision = CIODecision(
            hypothesis_id="HYP-2026-005",
            decision="PIVOT",
            score=score,
            rationale="Target leakage detected - use lagged features",
            evidence={"leakage_correlation": 0.97},
            pivot_direction="Investigate lagged RSI signals (t-1, t-2)",
        )
        assert decision.decision == "PIVOT"
        assert decision.pivot_direction == "Investigate lagged RSI signals (t-1, t-2)"
        assert decision.paper_allocation is None


class TestCIOReport:
    """Test CIOReport dataclass."""

    def test_create_report(self):
        """Test creating a complete CIO report."""
        score1 = CIOScore(
            hypothesis_id="HYP-2026-001",
            statistical=0.85,
            risk=0.78,
            economic=0.88,
            cost=0.75,
        )
        decision1 = CIODecision(
            hypothesis_id="HYP-2026-001",
            decision="CONTINUE",
            score=score1,
            rationale="Strong candidate",
            evidence={"run_id": "abc123"},
            paper_allocation=0.042,
        )

        score2 = CIOScore(
            hypothesis_id="HYP-2026-003",
            statistical=0.3,
            risk=0.35,
            economic=0.28,
            cost=0.31,
        )
        decision2 = CIODecision(
            hypothesis_id="HYP-2026-003",
            decision="KILL",
            score=score2,
            rationale="Insufficient statistical evidence",
            evidence={},
        )

        report = CIOReport(
            report_date=date(2026, 1, 26),
            decisions=[decision1, decision2],
            portfolio_state={
                "nav": 1023456.78,
                "cash": 50000.00,
                "positions_count": 5,
            },
            market_regime="Bull Market",
            next_actions=[
                {"priority": 1, "action": "Approve CONTINUE decisions", "deadline": "2026-01-27"},
            ],
            report_path="docs/reports/2026-01-26/09-00-cio-decision.md",
        )

        assert len(report.decisions) == 2
        assert report.portfolio_state["nav"] == 1023456.78
        assert report.market_regime == "Bull Market"
        assert len(report.next_actions) == 1
