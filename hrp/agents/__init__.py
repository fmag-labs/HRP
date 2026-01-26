"""
HRP Agent module.

Research agents for automated hypothesis discovery and signal analysis.
"""

from hrp.agents.jobs import (
    FeatureComputationJob,
    FundamentalsIngestionJob,
    IngestionJob,
    JobStatus,
    PriceIngestionJob,
    SnapshotFundamentalsJob,
    UniverseUpdateJob,
)
from hrp.agents.research_agents import (
    MLScientist,
    MLScientistReport,
    ModelExperimentResult,
    ResearchAgent,
    SignalScanReport,
    SignalScanResult,
    SignalScientist,
)
from hrp.agents.scheduler import IngestionScheduler

__all__ = [
    # Jobs
    "IngestionJob",
    "JobStatus",
    "PriceIngestionJob",
    "FeatureComputationJob",
    "UniverseUpdateJob",
    "FundamentalsIngestionJob",
    "SnapshotFundamentalsJob",
    # Research Agents
    "ResearchAgent",
    "SignalScientist",
    "SignalScanResult",
    "SignalScanReport",
    "MLScientist",
    "MLScientistReport",
    "ModelExperimentResult",
    # Scheduler
    "IngestionScheduler",
]
