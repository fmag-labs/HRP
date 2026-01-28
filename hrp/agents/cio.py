"""
CIO Agent - Chief Investment Officer Agent.

Makes strategic decisions about research lines and manages paper portfolio.
Advisory mode: presents recommendations, awaits user approval.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

from hrp.agents.sdk_agent import SDKAgent
from hrp.api.platform import PlatformAPI


@dataclass
class CIOScore:
    """
    Balanced score across 4 dimensions for a hypothesis.

    Attributes:
        hypothesis_id: The hypothesis being scored
        statistical: Statistical quality score (0-1)
        risk: Risk profile score (0-1)
        economic: Economic rationale score (0-1)
        cost: Cost realism score (0-1)
        critical_failure: Whether a critical failure was detected
    """

    hypothesis_id: str
    statistical: float
    risk: float
    economic: float
    cost: float
    critical_failure: bool = False

    @property
    def total(self) -> float:
        """Calculate total score as average of 4 dimensions."""
        return (self.statistical + self.risk + self.economic + self.cost) / 4

    @property
    def decision(self) -> Literal["CONTINUE", "CONDITIONAL", "KILL", "PIVOT"]:
        """
        Map score to decision.

        Returns:
            CONTINUE: Score >= 0.75, no critical failure
            CONDITIONAL: Score 0.50-0.74, no critical failure
            KILL: Score < 0.50, no critical failure
            PIVOT: Critical failure detected (overrides score)
        """
        if self.critical_failure:
            return "PIVOT"

        if self.total >= 0.75:
            return "CONTINUE"
        if self.total >= 0.50:
            return "CONDITIONAL"
        return "KILL"


@dataclass
class CIODecision:
    """
    Single decision for a hypothesis.

    Attributes:
        hypothesis_id: The hypothesis this decision is for
        decision: One of CONTINUE, CONDITIONAL, KILL, PIVOT
        score: The CIOScore that led to this decision
        rationale: Human-readable explanation
        evidence: Supporting data (MLflow runs, reports, metrics)
        paper_allocation: For CONTINUE decisions, portfolio weight (0-1)
        pivot_direction: For PIVOT decisions, suggested redirect
    """

    hypothesis_id: str
    decision: Literal["CONTINUE", "CONDITIONAL", "KILL", "PIVOT"]
    score: CIOScore
    rationale: str
    evidence: dict
    paper_allocation: Optional[float] = None
    pivot_direction: Optional[str] = None


@dataclass
class CIOReport:
    """
    Complete weekly CIO report.

    Attributes:
        report_date: When the report was generated
        decisions: All decisions made in this review cycle
        portfolio_state: Current paper portfolio state
        market_regime: Current market regime context
        next_actions: Prioritized action items
        report_path: Path to the generated markdown report
    """

    report_date: date
    decisions: list[CIODecision]
    portfolio_state: dict
    market_regime: str
    next_actions: list[dict]
    report_path: str


class CIOAgent(SDKAgent):
    """
    Chief Investment Officer Agent.

    Makes strategic decisions about research lines and manages paper portfolio.
    Advisory mode: presents recommendations, awaits user approval.
    """

    agent_name = "cio"
    agent_version = "1.0.0"

    DEFAULT_THRESHOLDS = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.20,
        "sharpe_decay_limit": 0.50,
        "min_ic": 0.03,
        "max_turnover": 0.50,
        "critical_sharpe_decay": 0.75,
        "critical_target_leakage": 0.95,
        "critical_max_drawdown": 0.35,
        "min_profitable_regimes": 2,
    }

    def __init__(
        self,
        job_id: str,
        actor: str,
        api: PlatformAPI | None = None,
        thresholds: dict | None = None,
        dependencies: list[str] | None = None,
    ):
        """
        Initialize CIO Agent.

        Args:
            job_id: Unique job identifier
            actor: Actor identity (e.g., "agent:cio")
            api: PlatformAPI instance (created if None)
            thresholds: Custom decision thresholds
            dependencies: List of data requirements
        """
        super().__init__(
            job_id=job_id,
            actor=actor,
            dependencies=dependencies or [],
        )
        self.api = api or PlatformAPI()
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    def execute(self) -> dict[str, any]:
        """
        Execute CIO Agent logic.

        This is a placeholder implementation to satisfy the abstract base class.
        The actual weekly review logic will be implemented in later tasks.

        Returns:
            Empty dict for now
        """
        return {}
