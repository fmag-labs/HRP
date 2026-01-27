"""
Data lineage tracking for HRP.

Provides comprehensive lineage tracking for feature computation and data provenance.

Usage:
    from hrp.data.lineage import FeatureLineage, DataProvenance

    # Track feature computation
    lineage = FeatureLineage.track_computation(
        feature_name="momentum_20d",
        symbols=['AAPL'],
        computation_date=date.today(),
        inputs={'prices': ['AAPL']},
        params={'window': 20},
    )

    # Track data provenance
    provenance = DataProvenance.track_source(
        data_type="prices",
        record_id="AAPL_2024-01-15",
        source_system="yfinance",
        source_timestamp=datetime.now(),
    )
"""

from hrp.data.lineage.features import FeatureLineage
from hrp.data.lineage.provenance import DataProvenance
from hrp.data.lineage.queries import (
    get_data_flow,
    get_feature_dependencies,
    get_impact_analysis,
)

__all__ = [
    "FeatureLineage",
    "DataProvenance",
    "get_data_flow",
    "get_feature_dependencies",
    "get_impact_analysis",
]
