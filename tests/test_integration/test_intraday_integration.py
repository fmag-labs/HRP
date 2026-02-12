"""
Integration test for the intraday data pipeline.

Exercises the full flow: bars -> buffer -> DB write -> feature compute -> persist.
Does NOT test the WebSocket connection (that's a unit test concern).
"""

import os
import tempfile
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from hrp.data.connection_pool import ConnectionPool
from hrp.data.db import DatabaseManager
from hrp.data.features.intraday_features import IntradayBar as FeatureBar
from hrp.data.features.intraday_features import IntradayFeatureEngine
from hrp.data.ingestion.intraday import IntradayBarBuffer, _batch_upsert_intraday
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def intraday_db():
    """Create a temporary DB with intraday schema for integration testing."""
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
            INSERT INTO symbols (symbol, name, exchange)
            VALUES
                ('AAPL', 'Apple Inc.', 'NASDAQ'),
                ('MSFT', 'Microsoft Corporation', 'NASDAQ')
            ON CONFLICT DO NOTHING
            """
        )

    yield db_path

    DatabaseManager.reset()
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in [".wal", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


# =============================================================================
# Tests
# =============================================================================


class TestIntradayPipelineIntegration:
    """End-to-end intraday pipeline tests."""

    def test_buffer_to_db_write(self, intraday_db):
        """Test: bars added to buffer -> flush -> batch upsert to DB."""
        buffer = IntradayBarBuffer(max_size=1000)
        for i in range(10):
            buffer.add_bar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                open_=150.0 + i,
                high=151.0 + i,
                low=149.0 + i,
                close=150.5 + i,
                volume=1000 * (i + 1),
                vwap=150.25 + i,
                trade_count=100 + i,
            )

        bars = buffer.flush()
        assert len(bars) == 10

        pool = ConnectionPool(database=intraday_db, max_connections=2)
        rows = _batch_upsert_intraday(bars, pool)
        assert rows == 10

        from hrp.data.db import get_db

        db = get_db(intraday_db)
        with db.connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM intraday_bars WHERE symbol = 'AAPL'"
            ).fetchone()
            assert result[0] == 10

    def test_upsert_preserves_ingested_at(self, intraday_db):
        """Test that re-upserting bars preserves ingested_at for existing rows."""
        pool = ConnectionPool(database=intraday_db, max_connections=2)
        ts = datetime(2024, 1, 15, 9, 30, tzinfo=UTC)

        bars_v1 = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": Decimal("150.0"),
                "high": Decimal("151.0"),
                "low": Decimal("149.0"),
                "close": Decimal("150.5"),
                "volume": 1000,
                "vwap": Decimal("150.25"),
                "trade_count": 100,
                "source": "polygon_ws",
            }
        ]
        _batch_upsert_intraday(bars_v1, pool)

        from hrp.data.db import get_db

        db = get_db(intraday_db)
        with db.connection() as conn:
            row = conn.execute(
                "SELECT ingested_at FROM intraday_bars "
                "WHERE symbol='AAPL' AND timestamp=?",
                [ts],
            ).fetchone()
            original_ingested_at = row[0]

        # Re-upsert with updated close
        bars_v2 = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": Decimal("150.0"),
                "high": Decimal("152.0"),
                "low": Decimal("149.0"),
                "close": Decimal("151.5"),
                "volume": 1200,
                "vwap": Decimal("150.75"),
                "trade_count": 120,
                "source": "polygon_ws",
            }
        ]
        _batch_upsert_intraday(bars_v2, pool)

        with db.connection() as conn:
            row = conn.execute(
                "SELECT close, ingested_at FROM intraday_bars "
                "WHERE symbol='AAPL' AND timestamp=?",
                [ts],
            ).fetchone()
            assert float(row[0]) == pytest.approx(151.5)
            # ingested_at should be preserved (not reset)
            assert row[1] == original_ingested_at

    def test_full_pipeline_bars_to_features(self, intraday_db):
        """Test full pipeline: buffer -> DB -> feature engine -> persist."""
        pool = ConnectionPool(database=intraday_db, max_connections=2)

        # Create 25 bars (enough for 20-bar features)
        buffer = IntradayBarBuffer(max_size=1000)
        for i in range(25):
            buffer.add_bar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                open_=100.0 + i * 0.5,
                high=101.0 + i * 0.5,
                low=99.0 + i * 0.5,
                close=100.0 + i * 0.5,
                volume=1000,
                vwap=100.25 + i * 0.5,
            )

        # Flush and write bars
        bars = buffer.flush()
        _batch_upsert_intraday(bars, pool)

        # Compute features
        engine = IntradayFeatureEngine(db_path=intraday_db)
        feature_bars = [
            FeatureBar(
                symbol=b["symbol"],
                timestamp=b["timestamp"],
                open=float(b["open"]),
                high=float(b["high"]),
                low=float(b["low"]),
                close=float(b["close"]),
                volume=b["volume"],
                vwap=float(b["vwap"]) if b["vwap"] else None,
            )
            for b in bars
        ]
        features_df = engine.compute_features(feature_bars)
        assert not features_df.empty

        # Persist features with conn_pool
        count = engine.persist_features(features_df, conn_pool=pool)
        assert count > 0

        # Verify features in DB
        from hrp.data.db import get_db

        db = get_db(intraday_db)
        with db.connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM intraday_features WHERE symbol = 'AAPL'"
            ).fetchone()
            assert result[0] > 0

            names = conn.execute(
                "SELECT DISTINCT feature_name FROM intraday_features"
            ).fetchall()
            feature_names = [n[0] for n in names]
            assert "intraday_vwap" in feature_names

    def test_day_transition_in_pipeline(self, intraday_db):
        """Test that day transitions are handled correctly in feature computation."""
        engine = IntradayFeatureEngine()

        # Day 1 bars
        day1_bars = [
            FeatureBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 15, 55 + i, tzinfo=UTC),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1000,
            )
            for i in range(5)
        ]
        engine.add_bars(day1_bars)
        day1_open = engine._day_opens["AAPL"]

        # Day 2 bars
        day2_bars = [
            FeatureBar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 16, 9, 30 + i, tzinfo=UTC),
                open=105.0,
                high=106.0,
                low=104.0,
                close=105.0,
                volume=1000,
            )
            for i in range(5)
        ]
        engine.add_bars(day2_bars)

        # Day open should have reset to day 2's first bar
        assert engine._day_opens["AAPL"] == 105.0
        assert engine._day_opens["AAPL"] != day1_open

    def test_multi_symbol_pipeline(self, intraday_db):
        """Test pipeline with multiple symbols simultaneously."""
        pool = ConnectionPool(database=intraday_db, max_connections=2)
        buffer = IntradayBarBuffer(max_size=1000)

        for i in range(10):
            for symbol, base_price in [("AAPL", 150.0), ("MSFT", 400.0)]:
                buffer.add_bar(
                    symbol=symbol,
                    timestamp=datetime(2024, 1, 15, 9, 30 + i, tzinfo=UTC),
                    open_=base_price + i,
                    high=base_price + i + 1,
                    low=base_price + i - 1,
                    close=base_price + i + 0.5,
                    volume=1000,
                )

        bars = buffer.flush()
        assert len(bars) == 20

        rows = _batch_upsert_intraday(bars, pool)
        assert rows == 20

        from hrp.data.db import get_db

        db = get_db(intraday_db)
        with db.connection() as conn:
            aapl_count = conn.execute(
                "SELECT COUNT(*) FROM intraday_bars WHERE symbol = 'AAPL'"
            ).fetchone()[0]
            msft_count = conn.execute(
                "SELECT COUNT(*) FROM intraday_bars WHERE symbol = 'MSFT'"
            ).fetchone()[0]
            assert aapl_count == 10
            assert msft_count == 10
