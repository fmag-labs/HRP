"""
Portfolio risk limits for pre-trade validation.

Defines limit configurations and validation reporting structures.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LimitViolation:
    """Record of a single limit violation or clip."""

    limit_name: str
    symbol: str | None
    limit_value: float
    actual_value: float
    action: Literal["clipped", "rejected", "warned"]
    details: str | None = None


@dataclass
class ValidationReport:
    """Report from pre-trade validation."""

    violations: list[LimitViolation] = field(default_factory=list)
    clips: list[LimitViolation] = field(default_factory=list)
    warnings: list[LimitViolation] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no hard violations (clips and warnings are ok)."""
        return len(self.violations) == 0


@dataclass
class RiskLimits:
    """
    Portfolio risk limits for pre-trade validation.

    Conservative institutional defaults for long-only equity.
    """

    # Position limits
    max_position_pct: float = 0.05      # Max 5% in any single position
    min_position_pct: float = 0.01      # Min 1% (avoid tiny positions)
    max_position_adv_pct: float = 0.10  # Max 10% of daily volume

    # Sector limits
    max_sector_pct: float = 0.25        # Max 25% in any sector
    max_unknown_sector_pct: float = 0.10  # Max 10% in unknown sectors

    # Portfolio limits
    max_gross_exposure: float = 1.00    # 100% = no leverage
    min_gross_exposure: float = 0.80    # Stay 80%+ invested
    max_net_exposure: float = 1.00      # Long-only: net = gross

    # Turnover limits
    max_turnover_pct: float = 0.20      # Max 20% turnover per rebalance

    # Concentration limits
    max_top_n_concentration: float = 0.40  # Top 5 holdings < 40%
    top_n_for_concentration: int = 5

    # Liquidity
    min_adv_dollars: float = 1_000_000  # Min $1M daily volume
