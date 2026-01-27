# F-020: Transaction Cost & Risk Limits Design

**Date:** 2026-01-26
**Status:** Approved
**Feature ID:** F-020

## Overview

Complete the partial implementation of F-020 by adding:
1. **Realistic transaction cost modeling** with square-root market impact
2. **Full portfolio risk limits** with pre-trade validation
3. **Sector data integration** for sector exposure limits

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Market impact model | Simple square-root | Industry standard, captures key dynamics |
| Risk limits scope | Full portfolio constraints | Position, sector, exposure, turnover, concentration |
| Enforcement point | Pre-trade validation | Clean separation, easy to debug |
| Sector data source | Polygon.io (primary) | Already integrated, reliable |
| Default limits | Conservative institutional | Safe for long-only equity |

---

## 1. Transaction Cost Model

### Cost Components

```
Total Cost = Commission + Spread + Market Impact

Where:
- Commission: min($1, max(0.5% of trade, $0.005/share))  [IBKR tiered]
- Spread: half_spread_bps / 10000 × trade_value         [bid-ask]
- Market Impact: k × σ × sqrt(shares / ADV)             [square-root law]
```

### Market Impact Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` (eta) | 0.1 | Market impact coefficient (calibrated to US large-cap) |
| `σ` | Daily volatility | From `volatility_20d` feature |
| `ADV` | 20-day avg volume | From prices table |

### Implementation

```python
@dataclass
class MarketImpactModel:
    """Square-root market impact cost model."""

    eta: float = 0.1              # Impact coefficient
    spread_bps: float = 5.0       # Half bid-ask spread
    commission_per_share: float = 0.005
    commission_min: float = 1.00
    commission_max_pct: float = 0.005

    def estimate_cost(
        self,
        shares: int,
        price: float,
        adv: float,
        volatility: float,
    ) -> CostBreakdown:
        """Pre-trade cost estimation."""
        trade_value = shares * price

        # Commission (IBKR tiered)
        commission = max(
            self.commission_min,
            min(shares * self.commission_per_share, trade_value * self.commission_max_pct)
        )

        # Spread cost
        spread_cost = (self.spread_bps / 10000) * trade_value

        # Market impact (square-root law)
        participation_rate = shares / adv if adv > 0 else 1.0
        impact_cost = self.eta * volatility * np.sqrt(participation_rate) * trade_value

        return CostBreakdown(
            commission=commission,
            spread=spread_cost,
            market_impact=impact_cost,
            total=commission + spread_cost + impact_cost,
            total_pct=(commission + spread_cost + impact_cost) / trade_value,
        )
```

**Location:** `hrp/risk/costs.py`

---

## 2. Risk Limits Configuration

### Configuration Class

```python
@dataclass
class RiskLimits:
    """Portfolio risk limits for pre-trade validation."""

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
```

### Validation Modes

| Mode | Behavior |
|------|----------|
| `strict` | Reject entire signal set if any limit violated |
| `clip` | Adjust weights to satisfy limits (default) |
| `warn` | Log warnings but allow violations |

**Location:** `hrp/risk/limits.py`

---

## 3. Sector Data Integration

### Schema Change

```sql
ALTER TABLE symbols ADD COLUMN sector VARCHAR(50);
ALTER TABLE symbols ADD COLUMN industry VARCHAR(100);
```

| Column | Example Values |
|--------|----------------|
| `sector` | Technology, Healthcare, Consumer Discretionary, Industrials, etc. (11 GICS sectors) |
| `industry` | Software, Semiconductors, Biotechnology, etc. (finer-grained) |

### Data Source

**Polygon.io** Ticker Details endpoint (primary):
- `GET /v3/reference/tickers/{ticker}`
- Returns `sic_code`, `sic_description`, `primary_exchange`

**SIC to GICS Mapping:** Convert 4-digit SIC codes to 11 GICS sectors via lookup table.

**Fallback:** Yahoo Finance `Ticker.info` if Polygon fails.

### Ingestion Job

```python
class SectorIngestionJob(IngestionJob):
    """Weekly sector data refresh."""

    def run(self) -> JobResult:
        symbols = self._get_universe_symbols()

        for symbol in symbols:
            # Try Polygon first
            sector_data = self._fetch_from_polygon(symbol)

            if not sector_data:
                # Fallback to Yahoo Finance
                sector_data = self._fetch_from_yahoo(symbol)

            if sector_data:
                self._update_symbol_sector(symbol, sector_data)
            else:
                self._mark_sector_unknown(symbol)

        return JobResult(...)
```

### Schedule

- **Weekly:** Saturday at 10:15 AM ET (after fundamentals job)
- **On-demand:** When new symbols added to universe

**Location:** `hrp/data/ingestion/sectors.py`

---

## 4. Pre-Trade Validation

### Validation Flow

```
Signals → PreTradeValidator → Validated Signals → Backtest
                ↓
         ValidationReport (warnings, clips, rejections)
```

### Core Class

```python
class PreTradeValidator:
    """Validates and adjusts signals against risk limits."""

    def __init__(
        self,
        limits: RiskLimits,
        cost_model: MarketImpactModel,
        mode: Literal["strict", "clip", "warn"] = "clip",
    ):
        self.limits = limits
        self.cost_model = cost_model
        self.mode = mode

    def validate(
        self,
        signals: pd.DataFrame,      # Raw signals (weights per symbol per date)
        prices: pd.DataFrame,       # For position sizing
        sectors: pd.Series,         # Symbol → sector mapping
        adv: pd.DataFrame,          # Average daily volume
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """
        Returns:
            validated_signals: Adjusted signal weights
            report: Details on clips, warnings, violations
        """
        ...
```

### Validation Checks (in order)

1. **Liquidity filter** — Remove symbols below `min_adv_dollars`
2. **Position sizing** — Clip weights to `[min_position_pct, max_position_pct]`
3. **ADV constraint** — Reduce if trade > `max_position_adv_pct` of volume
4. **Sector exposure** — Pro-rata reduce if sector > `max_sector_pct`
5. **Concentration** — Reduce top N if exceeds `max_top_n_concentration`
6. **Gross exposure** — Scale all weights if total > `max_gross_exposure`
7. **Turnover** — Limit changes from prior weights if > `max_turnover_pct`

**Location:** `hrp/risk/limits.py` (PreTradeValidator class)

---

## 5. Backtest Integration

### Updated BacktestConfig

```python
@dataclass
class BacktestConfig:
    # ... existing fields ...

    # New fields (backward compatible defaults)
    cost_model: MarketImpactModel = field(default_factory=MarketImpactModel)
    risk_limits: RiskLimits | None = None  # None = no limits
    validation_mode: Literal["strict", "clip", "warn"] = "clip"
```

### Updated run_backtest() Flow

```python
def run_backtest(signals, config, prices=None) -> BacktestResult:
    # 1. Load prices (existing)
    prices = prices or get_price_data(...)

    # 2. Load ADV data (new)
    adv = _compute_adv(prices, window=20)

    # 3. Load sector data (new)
    sectors = _load_sector_mapping(config.symbols)

    # 4. Load volatility (new)
    volatility = _load_volatility(config.symbols)

    # 5. PRE-TRADE VALIDATION (new)
    validation_report = None
    if config.risk_limits:
        validator = PreTradeValidator(
            limits=config.risk_limits,
            cost_model=config.cost_model,
            mode=config.validation_mode,
        )
        signals, validation_report = validator.validate(
            signals, prices, sectors, adv
        )

    # 6. COST ESTIMATION (new)
    estimated_costs = _estimate_backtest_costs(
        signals, prices, adv, volatility, config.cost_model
    )

    # 7. Run VectorBT (updated fees)
    portfolio = vbt.Portfolio.from_signals(
        fees=estimated_costs,
        ...
    )

    # 8. Return results (updated)
    return BacktestResult(
        ...,
        validation_report=validation_report,
        estimated_costs=estimated_costs,
    )
```

### Backward Compatibility

- Default `risk_limits=None` skips validation (current behavior)
- Default `MarketImpactModel()` uses sensible defaults
- Existing tests continue to pass unchanged

---

## 6. File Structure

### New Files

```
hrp/risk/
├── costs.py          # MarketImpactModel, CostBreakdown, cost estimation
├── limits.py         # RiskLimits, PreTradeValidator, ValidationReport

hrp/data/ingestion/
└── sectors.py        # SectorIngestionJob, Polygon API, SIC→GICS mapping
```

### Modified Files

```
hrp/research/config.py      # Add cost_model, risk_limits to BacktestConfig
hrp/research/backtest.py    # Wire in validation + cost estimation
hrp/data/schema.py          # Add sector, industry columns to symbols
hrp/agents/scheduler.py     # Add weekly sector refresh job
hrp/agents/jobs.py          # Add SectorIngestionJob
hrp/risk/__init__.py        # Export new classes
```

---

## 7. Testing Strategy

### Test Files

| Test File | Coverage |
|-----------|----------|
| `tests/test_risk/test_costs.py` | MarketImpactModel, cost calculations, edge cases |
| `tests/test_risk/test_limits.py` | RiskLimits validation, clipping logic, all 7 checks |
| `tests/test_data/test_sectors.py` | Sector ingestion, SIC→GICS mapping, fallback |
| `tests/test_research/test_backtest_costs.py` | End-to-end backtest with costs + limits |

### Key Test Cases

**Cost Model:**
- Small trade (commission minimum applies)
- Large trade (market impact dominates)
- Zero ADV edge case
- Cost breakdown accuracy

**Risk Limits:**
- Position clipping (above max, below min)
- Sector exposure pro-rata reduction
- Concentration limit enforcement
- Turnover constraint
- Strict mode rejection
- Warn mode passthrough

**Integration:**
- Backtest with no limits (backward compat)
- Backtest with conservative limits
- Validation report accuracy

### Verification Commands

```bash
# Run new tests
pytest tests/test_risk/test_costs.py tests/test_risk/test_limits.py -v

# Run full suite to ensure no regressions
pytest tests/ -v
```

---

## 8. Migration Notes

### Database Migration

```sql
-- Add sector columns to symbols table
ALTER TABLE symbols ADD COLUMN sector VARCHAR(50);
ALTER TABLE symbols ADD COLUMN industry VARCHAR(100);

-- Create index for sector queries
CREATE INDEX idx_symbols_sector ON symbols(sector);
```

### Initial Data Load

After deployment:
1. Run `SectorIngestionJob` manually to populate sector data
2. Verify sector coverage: `SELECT sector, COUNT(*) FROM symbols GROUP BY sector`
3. Address any `Unknown` sectors

---

## 9. Success Criteria

- [ ] Market impact model matches expected cost curves for various trade sizes
- [ ] All 7 validation checks correctly clip/reject signals
- [ ] Sector data populated for >95% of universe
- [ ] Existing backtests produce identical results (backward compat)
- [ ] New backtests with limits show realistic cost drag
- [ ] Test coverage >90% for new code
- [ ] Full test suite passes (2,115+ tests)
