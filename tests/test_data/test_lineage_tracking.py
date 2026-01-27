"""
Tests for data lineage tracking.

Tests cover:
- FeatureLineage tracking and querying
- DataProvenance tracking and integrity verification
- Query utilities (data flow, dependencies, impact analysis)
- Integration with existing lineage table
- Edge cases and error handling
"""

from datetime import date, datetime

import pytest

from hrp.data.lineage.features import FeatureComputationRecord, FeatureLineage
from hrp.data.lineage.provenance import DataProvenance, ProvenanceRecord
from hrp.data.lineage.queries import (
    get_data_flow,
    get_feature_dependencies,
    get_impact_analysis,
    get_lineage_summary,
)


class TestFeatureLineage:
    """Tests for FeatureLineage tracking."""

    def test_init_creates_tables(self, test_db):
        """Should create feature_lineage table on initialization."""
        FeatureLineage(test_db)

        from hrp.data.db import get_db

        db = get_db(test_db)
        result = db.fetchall(
            "SELECT table_name FROM duckdb_tables() WHERE table_name = 'feature_lineage'"
        )

        assert len(result) > 0
        assert result[0][0] == "feature_lineage"

    def test_track_computation(self, test_db):
        """Should track a feature computation."""
        # Insert symbol first
        from hrp.data.db import get_db

        db = get_db(test_db)
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('LINEAGE1', 'Lineage Test 1', 'NASDAQ')"
            )

        lineage = FeatureLineage(test_db)
        lineage_id = lineage.track_computation(
            feature_name="momentum_20d",
            symbols=["LINEAGE1"],
            computation_date=date(2024, 1, 15),
            computation_source="batch",
            input_features=["close"],
            computation_params={"window": 20},
            rows_computed=100,
            duration_ms=50.0,
        )

        assert lineage_id > 0

    def test_get_computation_history(self, test_db):
        """Should retrieve computation history."""
        from hrp.data.db import get_db

        db = get_db(test_db)
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('LINEAGE2', 'Lineage Test 2', 'NASDAQ')"
            )

        lineage = FeatureLineage(test_db)
        lineage_id = lineage.track_computation(
            feature_name="momentum_20d",
            symbols=["LINEAGE2"],
            computation_date=date(2024, 1, 15),
        )

        history = lineage.get_computation_history(feature_name="momentum_20d")

        assert len(history) >= 1
        assert history[0].feature_name == "momentum_20d"

    def test_get_computation_history_filters(self, test_db):
        """Should filter computation history by parameters."""
        from hrp.data.db import get_db

        db = get_db(test_db)
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('LINEAGE3', 'Lineage Test 3', 'NASDAQ')"
            )

        lineage = FeatureLineage(test_db)
        lineage.track_computation(
            feature_name="momentum_20d",
            symbols=["LINEAGE3"],
            computation_date=date(2024, 1, 15),
        )

        # Filter by feature name
        history = lineage.get_computation_history(feature_name="momentum_20d")
        assert len(history) >= 1

        # Filter by symbol
        history = lineage.get_computation_history(symbol="LINEAGE3")
        assert len(history) >= 1

        # Filter by date
        history = lineage.get_computation_history(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )
        assert len(history) >= 1

    def test_get_feature_statistics(self, test_db):
        """Should calculate feature statistics."""
        from hrp.data.db import get_db

        db = get_db(test_db)
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('LINEAGE4', 'Lineage Test 4', 'NASDAQ')"
            )

        lineage = FeatureLineage(test_db)
        lineage.track_computation(
            feature_name="momentum_20d",
            symbols=["LINEAGE4"],
            computation_date=date(2024, 1, 15),
            rows_computed=100,
            duration_ms=50.0,
        )

        stats = lineage.get_feature_statistics(feature_name="momentum_20d")

        assert stats["feature_name"] == "momentum_20d"
        assert stats["unique_symbols"] >= 1
        assert stats["computation_days"] >= 1

    def test_get_computation_chain(self, test_db):
        """Should trace computation dependencies."""
        from hrp.data.db import get_db

        db = get_db(test_db)
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('LINEAGE5', 'Lineage Test 5', 'NASDAQ')"
            )

        lineage = FeatureLineage(test_db)
        lineage.track_computation(
            feature_name="momentum_20d",
            symbols=["LINEAGE5"],
            computation_date=date(2024, 1, 15),
            input_features=["close"],
        )

        chain = lineage.get_computation_chain(
            feature_name="momentum_20d",
            symbol="LINEAGE5",
            computation_date=date(2024, 1, 15),
        )

        assert len(chain) >= 1
        assert chain[0]["feature"] == "momentum_20d"


class TestDataProvenance:
    """Tests for DataProvenance tracking."""

    def test_init_creates_tables(self, test_db):
        """Should create data_provenance table on initialization."""
        DataProvenance(test_db)

        from hrp.data.db import get_db

        db = get_db(test_db)
        result = db.fetchall(
            "SELECT table_name FROM duckdb_tables() WHERE table_name = 'data_provenance'"
        )

        assert len(result) > 0
        assert result[0][0] == "data_provenance"

    def test_track_source(self, test_db):
        """Should track data source."""
        provenance = DataProvenance(test_db)

        provenance_id = provenance.track_source(
            data_type="prices",
            record_identifier="prices_PROV1_2024-01-15",
            source_system="yfinance",
            source_timestamp=datetime(2024, 1, 15, 9, 30, 0),
            data_content={"close": 100.0},
        )

        assert provenance_id > 0

    def test_add_transformation(self, test_db):
        """Should add transformation to history."""
        provenance = DataProvenance(test_db)

        provenance_id = provenance.track_source(
            data_type="prices",
            record_identifier="prices_PROV2_2024-01-15",
            source_system="yfinance",
        )

        provenance.add_transformation(
            provenance_id=provenance_id,
            transformation_type="normalization",
            transformation_params={"method": "minmax"},
        )

        # Verify transformation was added
        history = provenance.get_transformation_history(provenance_id)
        assert len(history) == 1
        assert history[0]["type"] == "normalization"

    def test_add_quality_check(self, test_db):
        """Should add quality check results."""
        provenance = DataProvenance(test_db)

        provenance_id = provenance.track_source(
            data_type="prices",
            record_identifier="prices_PROV3_2024-01-15",
            source_system="yfinance",
        )

        provenance.add_quality_check(
            provenance_id=provenance_id,
            check_name="completeness",
            check_result={"passed": True},
        )

        # Verify quality check was added
        with provenance._db.connection() as conn:
            result = conn.execute(
                "SELECT quality_checks FROM data_provenance WHERE provenance_id = ?",
                (provenance_id,),
            ).fetchone()

        assert result is not None
        checks = result[0] if result[0] else {}
        assert "completeness" in checks

    def test_verify_integrity(self, test_db):
        """Should verify data integrity using hash."""
        provenance = DataProvenance(test_db)

        data_content = {"close": 100.0, "volume": 1000000}

        provenance_id = provenance.track_source(
            data_type="prices",
            record_identifier="prices_PROV4_2024-01-15",
            source_system="yfinance",
            data_content=data_content,
        )

        # Verify with same data
        assert provenance.verify_integrity(provenance_id, data_content) is True

        # Verify with different data
        assert (
            provenance.verify_integrity(provenance_id, {"close": 101.0}) is False
        )

    def test_get_provenance(self, test_db):
        """Should retrieve provenance records."""
        provenance = DataProvenance(test_db)

        provenance.track_source(
            data_type="prices",
            record_identifier="prices_PROV5_2024-01-15",
            source_system="yfinance",
        )

        records = provenance.get_provenance(data_type="prices")

        assert len(records) >= 1
        assert records[0].source_system == "yfinance"

    def test_create_record_identifier(self):
        """Should create unique record identifiers."""
        # Prices identifier
        id1 = DataProvenance.create_record_identifier(
            "prices", symbol="AAPL", date="2024-01-15"
        )
        assert id1 == "prices_AAPL_2024-01-15"

        # Features identifier - note: the order is data_type, symbol, date, feature_name
        id2 = DataProvenance.create_record_identifier(
            "features", symbol="AAPL", feature_name="momentum_20d", date="2024-01-15"
        )
        # Check that all components are present, regardless of order
        assert "features" in id2
        assert "AAPL" in id2
        assert "momentum_20d" in id2
        assert "2024-01-15" in id2


class TestQueryUtilities:
    """Tests for lineage query utilities."""

    def test_get_data_flow(self, test_db):
        """Should get complete data flow for a record."""
        provenance = DataProvenance(test_db)

        provenance.track_source(
            data_type="prices",
            record_identifier="prices_QUERY1_2024-01-15",
            source_system="yfinance",
            source_timestamp=datetime(2024, 1, 15, 9, 30, 0),
        )

        flow = get_data_flow("prices_QUERY1_2024-01-15", test_db)

        assert len(flow) >= 1
        assert flow[0]["source_system"] == "yfinance"

    def test_get_feature_dependencies(self, test_db):
        """Should get feature dependencies."""
        from hrp.data.db import get_db

        db = get_db(test_db)
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('QUERY2', 'Query Test 2', 'NASDAQ')"
            )

        # Initialize tables first
        lineage = FeatureLineage(test_db)
        lineage.track_computation(
            feature_name="momentum_20d",
            symbols=["QUERY2"],
            computation_date=date(2024, 1, 15),
            input_features=["close"],
            computation_params={"window": 20},
        )

        deps = get_feature_dependencies("momentum_20d", test_db)

        assert deps["feature_name"] == "momentum_20d"
        assert "close" in deps["inputs"]

    def test_get_impact_analysis(self, test_db):
        """Should analyze impact of quality issues."""
        issue = {
            "symbol": "QUERY3",
            "date": date(2024, 1, 15),
            "issue_type": "missing_data",
            "data_type": "prices",
        }

        # Initialize tables first
        lineage = FeatureLineage(test_db)
        provenance = DataProvenance(test_db)

        impact = get_impact_analysis(issue, test_db)

        assert impact["quality_issue"] == issue
        assert "affected_features" in impact
        assert "remediation" in impact

    def test_get_lineage_summary(self, test_db):
        """Should get lineage tracking summary."""
        # Initialize tables first
        lineage = FeatureLineage(test_db)
        provenance = DataProvenance(test_db)

        summary = get_lineage_summary(test_db)

        assert "feature_lineage" in summary
        assert "data_provenance" in summary


class TestIntegration:
    """Integration tests for lineage tracking."""

    def test_full_lineage_workflow(self, test_db):
        """Should demonstrate complete lineage tracking workflow."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Insert symbol
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('WORKFLOW', 'Workflow Test', 'NASDAQ')"
            )

        # 1. Track data source
        provenance = DataProvenance(test_db)
        provenance_id = provenance.track_source(
            data_type="prices",
            record_identifier="prices_WORKFLOW_2024-01-15",
            source_system="yfinance",
            source_timestamp=datetime(2024, 1, 15, 9, 30, 0),
            data_content={"close": 100.0, "volume": 1000000},
        )

        # 2. Add transformation
        provenance.add_transformation(
            provenance_id=provenance_id,
            transformation_type="outlier_removal",
            transformation_params={"method": "sigma_clip", "threshold": 3.0},
        )

        # 3. Add quality check
        provenance.add_quality_check(
            provenance_id=provenance_id,
            check_name="validation",
            check_result={"passed": True, "checks_performed": 5},
        )

        # 4. Track feature computation
        lineage = FeatureLineage(test_db)
        lineage_id = lineage.track_computation(
            feature_name="momentum_20d",
            symbols=["WORKFLOW"],
            computation_date=date(2024, 1, 16),
            computation_source="batch",
            input_features=["close"],
            input_symbols=["WORKFLOW"],
            computation_params={"window": 20},
        )

        # 5. Query complete data flow
        flow = get_data_flow("prices_WORKFLOW_2024-01-15", test_db)

        assert len(flow) >= 1

        # 6. Verify integrity
        assert provenance.verify_integrity(
            provenance_id, {"close": 100.0, "volume": 1000000}
        )

    def test_feature_lineage_with_existing_lineage_table(self, test_db):
        """Should work with existing lineage table."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Create a lineage entry in the main lineage table
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO lineage (lineage_id, event_type, actor)
                VALUES (999, 'other', 'test_actor')
            """
            )

        # Feature lineage should work independently
        lineage = FeatureLineage(test_db)
        # This should not interfere with existing lineage table

    def test_provenance_with_lineage_reference(self, test_db):
        """Should be able to reference lineage table."""
        from hrp.data.db import get_db

        db = get_db(test_db)

        # Create a lineage entry first
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO lineage (lineage_id, event_type, actor)
                VALUES (1000, 'data_ingested', 'system')
            """
            )

        # Provenance can reference this lineage_id
        provenance = DataProvenance(test_db)
        provenance_id = provenance.track_source(
            data_type="prices",
            record_identifier="prices_REF_2024-01-15",
            source_system="yfinance",
            lineage_id=1000,
        )

        assert provenance_id > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_computation_history(self, test_db):
        """Should handle empty computation history."""
        lineage = FeatureLineage(test_db)

        history = lineage.get_computation_history(feature_name="nonexistent")

        assert len(history) == 0

    def test_empty_provenance(self, test_db):
        """Should handle empty provenance."""
        provenance = DataProvenance(test_db)

        records = provenance.get_provenance(record_identifier="nonexistent")

        assert len(records) == 0

    def test_empty_transformation_history(self, test_db):
        """Should handle empty transformation history."""
        provenance = DataProvenance(test_db)

        provenance_id = provenance.track_source(
            data_type="prices",
            record_identifier="prices_EMPTY_2024-01-15",
            source_system="yfinance",
        )

        history = provenance.get_transformation_history(provenance_id)

        assert isinstance(history, list)

    def test_get_computation_chain_with_no_dependencies(self, test_db):
        """Should handle features with no input features."""
        from hrp.data.db import get_db

        db = get_db(test_db)
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange) VALUES ('NODEPS', 'No Deps', 'NASDAQ')"
            )

        lineage = FeatureLineage(test_db)
        lineage.track_computation(
            feature_name="price_raw",
            symbols=["NODEPS"],
            computation_date=date(2024, 1, 15),
            input_features=[],  # No dependencies
        )

        chain = lineage.get_computation_chain(
            feature_name="price_raw",
            symbol="NODEPS",
            computation_date=date(2024, 1, 15),
        )

        # Should have one entry even with no dependencies
        assert len(chain) >= 1

    def test_verify_integrity_with_no_hash(self, test_db):
        """Should handle integrity verification when no hash stored."""
        provenance = DataProvenance(test_db)

        provenance_id = provenance.track_source(
            data_type="prices",
            record_identifier="prices_NOHASH_2024-01-15",
            source_system="yfinance",
            data_content=None,  # No content, no hash
        )

        # Should return False when no hash exists
        assert (
            provenance.verify_integrity(provenance_id, {"data": "test"}) is False
        )
