"""
HRP Agent module.

Research agents for automated hypothesis discovery and signal analysis.

Names are exported lazily (PEP 562): ``from hrp.agents import CIOAgent`` works,
but importing the package itself does not pull in the full ML/data stack. This
keeps lightweight entrypoints (e.g. the ``hrp`` service CLI) fast.
"""

# The TYPE_CHECKING block re-imports names purely so static checkers/IDEs resolve
# the lazily-exported symbols; they are intentionally "unused" at runtime.
# ruff: noqa: F401

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Map exported name -> submodule it lives in.
_LAZY_EXPORTS: dict[str, str] = {
    # alpha_researcher
    "AlphaResearcher": "alpha_researcher",
    "AlphaResearcherConfig": "alpha_researcher",
    "AlphaResearcherReport": "alpha_researcher",
    "HypothesisAnalysis": "alpha_researcher",
    "StrategySpec": "alpha_researcher",
    # cio
    "CIOAgent": "cio",
    "CIODecision": "cio",
    "CIOReport": "cio",
    "CIOScore": "cio",
    # jobs
    "FeatureComputationJob": "jobs",
    "FundamentalsIngestionJob": "jobs",
    "IngestionJob": "jobs",
    "JobStatus": "jobs",
    "PriceIngestionJob": "jobs",
    "SnapshotFundamentalsJob": "jobs",
    "UniverseUpdateJob": "jobs",
    # report_generator
    "ReportGenerator": "report_generator",
    "ReportGeneratorConfig": "report_generator",
    # research_agents
    "AuditCheck": "research_agents",
    "AuditSeverity": "research_agents",
    "ExperimentAudit": "research_agents",
    "HypothesisValidation": "research_agents",
    "MLQualitySentinel": "research_agents",
    "MLScientist": "research_agents",
    "MLScientistReport": "research_agents",
    "ModelExperimentResult": "research_agents",
    "MonitoringAlert": "research_agents",
    "ParameterVariation": "research_agents",
    "PortfolioRiskAssessment": "research_agents",
    "QualitySentinelReport": "research_agents",
    "QuantDeveloper": "research_agents",
    "QuantDeveloperReport": "research_agents",
    "ResearchAgent": "research_agents",
    "RiskManager": "research_agents",
    "RiskManagerReport": "research_agents",
    "RiskVeto": "research_agents",
    "SignalScanReport": "research_agents",
    "SignalScanResult": "research_agents",
    "SignalScientist": "research_agents",
    "ValidationAnalyst": "research_agents",
    "ValidationAnalystReport": "research_agents",
    "ValidationCheck": "research_agents",
    "ValidationSeverity": "research_agents",
    # kill_gate_enforcer
    "BaselineResult": "kill_gate_enforcer",
    "BaselineType": "kill_gate_enforcer",
    "ExperimentConfig": "kill_gate_enforcer",
    "ExperimentResult": "kill_gate_enforcer",
    "KillGateEnforcer": "kill_gate_enforcer",
    "KillGateEnforcerConfig": "kill_gate_enforcer",
    "KillGateEnforcerReport": "kill_gate_enforcer",
    "KillGateReason": "kill_gate_enforcer",
    "KillGateResult": "kill_gate_enforcer",
    # scheduler
    "IngestionScheduler": "scheduler",
    "LineageEventWatcher": "scheduler",
    "LineageTrigger": "scheduler",
    # sdk_agent
    "AgentCheckpoint": "sdk_agent",
    "SDKAgent": "sdk_agent",
    "SDKAgentConfig": "sdk_agent",
    "TokenUsage": "sdk_agent",
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent access
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


if TYPE_CHECKING:  # eager imports for type-checkers / IDEs only
    from hrp.agents.alpha_researcher import (
        AlphaResearcher,
        AlphaResearcherConfig,
        AlphaResearcherReport,
        HypothesisAnalysis,
        StrategySpec,
    )
    from hrp.agents.cio import CIOAgent, CIODecision, CIOReport, CIOScore
    from hrp.agents.jobs import (
        FeatureComputationJob,
        FundamentalsIngestionJob,
        IngestionJob,
        JobStatus,
        PriceIngestionJob,
        SnapshotFundamentalsJob,
        UniverseUpdateJob,
    )
    from hrp.agents.kill_gate_enforcer import (
        BaselineResult,
        BaselineType,
        ExperimentConfig,
        ExperimentResult,
        KillGateEnforcer,
        KillGateEnforcerConfig,
        KillGateEnforcerReport,
        KillGateReason,
        KillGateResult,
    )
    from hrp.agents.report_generator import ReportGenerator, ReportGeneratorConfig
    from hrp.agents.research_agents import (
        AuditCheck,
        AuditSeverity,
        ExperimentAudit,
        HypothesisValidation,
        MLQualitySentinel,
        MLScientist,
        MLScientistReport,
        ModelExperimentResult,
        MonitoringAlert,
        ParameterVariation,
        PortfolioRiskAssessment,
        QualitySentinelReport,
        QuantDeveloper,
        QuantDeveloperReport,
        ResearchAgent,
        RiskManager,
        RiskManagerReport,
        RiskVeto,
        SignalScanReport,
        SignalScanResult,
        SignalScientist,
        ValidationAnalyst,
        ValidationAnalystReport,
        ValidationCheck,
        ValidationSeverity,
    )
    from hrp.agents.scheduler import (
        IngestionScheduler,
        LineageEventWatcher,
        LineageTrigger,
    )
    from hrp.agents.sdk_agent import (
        AgentCheckpoint,
        SDKAgent,
        SDKAgentConfig,
        TokenUsage,
    )

__all__ = list(_LAZY_EXPORTS)
