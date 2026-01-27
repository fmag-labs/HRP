# Production-Grade Historical Universe Tracking

**Date:** January 27, 2026
**Status:** Planned - Production Tier
**Priority:** Medium (for research rigor) / High (for live trading)

## Problem Statement

The current universe backfill (`scripts/backfill_universe.py`) uses a **simplification**:
- Copies current S&P 500 constituents to ALL historical dates
- Does not track when stocks were added/removed from the index
- Introduces **look-ahead bias** in backtests

**Example of the Problem:**
```python
# Universe on 2010-01-01 shows META (Facebook) - WRONG!
# Facebook IPO'd in 2012, wasn't in S&P 500 until 2013
# But simplified backfill puts META in universe for all dates
```

This leads to:
- **Inflated backtest results** - trading stocks that didn't exist
- **Survivorship bias** - only current winners, no failed companies
- **Sector composition changes** ignored (e.g., tech weight changes over time)

## Production Requirements

For **rigorous research** and **live trading**, we need:

1. **Point-in-Time Accuracy**
   - Only stocks actually in S&P 500 on each date
   - Correct handling of additions, removals, reorganizations

2. **Change Tracking**
   - Historical S&P 500 constituent changes
   - Stock splits, mergers, acquisitions, spinoffs
   - Sector reclassifications (GICS updates)

3. **Exclusion Accuracy**
   - Financials excluded at the right time (sector changes)
   - REITs properly identified
   - Penny stock thresholds adjusted for inflation

## Implementation Options

### Option A: Purchase Historical Data (Recommended for Production)

**Data Sources:**

| Provider | URL | Cost | Coverage |
|----------|-----|------|----------|
| **S&P Capital IQ** | https://www.spglobal.com/capitaliq/ | $$$$ | Most accurate |
| **Refinitiv (LSEG)** | https://www.refinitiv.com/ | $$$$ | Institutional |
| **Xpressfeed** | https://www.xpressfeed.com/ | $$ | Good quality |
| **QuantShare** | https://www.quantshare.com/ | $ | Community data |

**Expected Investment:** $1,000-$5,000/year for institutional quality

**Implementation Steps:**
1. Purchase historical constituent data
2. Load into `universe_historical` table
3. Modify `UniverseManager` to query historical data first
4. Backfill with accurate point-in-time data

---

### Option B: Scrape Wikipedia Historical Changes (Free - Labor Intensive)

**Data Source:** Wikipedia "List of S&P 500 companies" page

**Approach:**
```python
# Manual one-time process:
1. Download Wikipedia page locally
2. Parse "Changes" section for additions/removals
3. Build timeline: {date: {added: [...], removed: [...]}}
4. Backfill universe using timeline
```

**Pros:**
- Free
- Publicly verifiable
- Comprehensive (goes back decades)

**Cons:**
- Manual maintenance required
- Wikipedia page structure may change
- Labor intensive to parse initially

**Implementation Steps:**
```python
# scripts/build_universe_timeline.py
def scrape_wikipedia_changes():
    """
    Scrape historical S&P 500 changes from Wikipedia.

    Returns:
        DataFrame with columns: date, symbol, action, reason
    """
    # Download page manually first to avoid blocking
    # Then parse locally
    from bs4 import BeautifulSoup

    with open('sp500_wikipedia.html') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Find changes section
    changes_table = soup.find('table', {'class': 'wikitable'})
    # Parse rows...
    # Return timeline DataFrame
```

**File Structure:**
```
data/
└── universe_historical/
    ├── sp500_additions.csv      # When stocks joined S&P 500
    ├── sp500_removals.csv       # When stocks left S&P 500
    ├── sp500_changes.csv         # Combined timeline
    └── README.md                # Source documentation
```

---

### Option C: Hybrid Approach (Practical Compromise)

**Combine multiple free sources:**

1. **Current Constituents** - SlickCharts (scraped daily)
2. **Historical Snapshots** - Archive.org snapshots of Wikipedia
3. **GitHub Datasets** - Community-maintained historical data

**Implementation:**
```python
def get_universe_for_date(target_date: date) -> set[str]:
    """
    Get accurate S&P 500 universe for a specific date.

    Strategy:
    1. Check universe table first (cached)
    2. If not found, build from:
       - Most recent universe before target_date
       - Apply additions up to target_date
       - Apply removals up to target_date
    """
    # Check cache first
    cached = universe_at_date(target_date)
    if cached:
        return cached

    # Build from timeline
    base_universe = get_most_recent_universe_before(target_date)

    # Apply changes chronologically
    changes = get_changes_up_to(target_date)

    for change in changes:
        if change.action == 'add':
            base_universe.add(change.symbol)
        elif change.action == 'remove':
            base_universe.discard(change.symbol)

    return base_universe
```

---

### Option D: Price Data Proxy (Least Accurate, Current Implementation)

**Current approach:** Assume all stocks with price data were tradeable

**Problem:** Includes stocks that:
- Weren't public yet
- Weren't in S&P 500
- Should have been excluded (financials, REITs)

**Use Case:** Exploratory analysis only, not for publication or live trading

---

## Recommended Implementation Plan

### Phase 1: Data Acquisition (Week 1-2)

**Tasks:**
1. Research and evaluate data providers
2. Download sample datasets
3. Verify data quality against known benchmarks
4. Make purchase decision or commit to scraping approach

**Deliverables:**
- Data provider evaluation report
- Sample historical data loaded
- Cost/benefit analysis

---

### Phase 2: Enhanced Schema (Week 3)

**New Tables:**

```sql
-- Historical S&P 500 changes
CREATE TABLE universe_changes (
    change_id INTEGER PRIMARY KEY,
    effective_date DATE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL,  -- 'added', 'removed', 'readded'
    reason VARCHAR(100),
    source VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_universe_changes_date ON universe_changes(effective_date);
CREATE INDEX idx_universe_changes_symbol ON universe_changes(symbol);

-- Point-in-time universe cache (pre-computed for performance)
CREATE TABLE universe_cache (
    date DATE PRIMARY KEY,
    symbols VARCHAR(5000)[] NOT NULL,  -- Array of symbols
    count INTEGER NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Schema Changes:**
```sql
-- Add to existing universe table
ALTER TABLE universe ADD COLUMN source VARCHAR(50);
ALTER TABLE universe ADD COLUMN data_quality VARCHAR(20);  -- 'actual', 'proxy', 'estimated'
```

---

### Phase 3: Enhanced Universe Manager (Week 4)

**New Methods in `UniverseManager`:**

```python
class UniverseManager:
    def get_universe_at_date_accurate(self, as_of_date: date) -> list[str]:
        """
        Get accurate point-in-time universe.

        Uses historical change data to reconstruct exact S&P 500
        membership for any date.
        """
        # 1. Check cache first
        cached = self._get_from_cache(as_of_date)
        if cached:
            return cached

        # 2. Build from changes
        base_date = self._get_base_date(as_of_date)
        base_symbols = set(self.get_universe_at_date(base_date))

        # 3. Apply changes chronologically
        changes = self._get_changes_between(base_date, as_of_date)

        for change in changes:
            if change.action == 'added':
                base_symbols.add(change.symbol)
            elif change.action == 'removed':
                base_symbols.discard(change.symbol)

        return list(base_symbols)

    def import_historical_changes(self, csv_path: str) -> dict:
        """
        Import historical S&P 500 changes from CSV.

        Expected CSV format:
        effective_date,symbol,action,reason,source
        2010-01-05,META,added,,"S&P Wikipedia"
        2013-03-01,XYZ,removed,"Acquired by ABC","S&P Wikipedia"
        """
        # Parse CSV, insert into universe_changes table
        # Update universe table with point-in-time data
        pass
```

---

### Phase 4: Backfill with Accurate Data (Week 5)

**New Script:** `scripts/backfill_universe_production.py`

```python
def backfill_universe_production():
    """
    Production-grade universe backfill using historical data.

    Process:
    1. Load historical changes from CSV
    2. For each date with price data:
       - Start with most recent universe before that date
       - Apply all additions/removals chronologically
       - Store in universe table with data_quality='actual'
    3. Cache computed universes for performance
    """
    # Load timeline
    changes = load_historical_changes('data/universe_historical/sp500_changes.csv')

    # Get all dates to backfill
    dates = get_all_price_dates()

    # Process each date
    for target_date in dates:
        universe = compute_universe_at_date(target_date, changes)

        # Insert with actual data quality
        for symbol in universe:
            insert_universe_record(
                symbol=symbol,
                date=target_date,
                in_universe=True,
                data_quality='actual',
                source='historical_changes'
            )
```

---

### Phase 5: Validation & Quality Assurance (Week 6)

**Validation Checks:**

1. **Known Benchmark Tests:**
   - Verify META not in universe before 2013
   - Verify Tesla not in universe before 2020
   - Verify Facebook→Meta rebrand handling

2. **Cross-Reference Checks:**
   - Compare against Wikipedia for specific dates
   - Check sector composition over time makes sense

3. **Sanity Checks:**
   - Universe size ~500 (±50 for reorgs)
   - No sudden unexplained changes
   - Sector weights evolve gradually

**Validation Script:**
```python
def validate_universe_accuracy():
    """
    Validate universe backfill against known benchmarks.
    """
    tests = [
        {
            'date': '2012-01-01',
            'not_in_universe': ['META', 'TSLA', 'GOOGL'],  # Not public yet
            'in_universe': ['AAPL', 'MSFT'],
            'max_size': 510
        },
        {
            'date': '2020-01-01',
            'not_in_universe': ['TSLA'],  # Added Dec 2020
            'in_universe': ['META', 'AAPL'],
            'max_size': 505
        },
        {
            'date': '2024-01-01',
            'in_universe': ['META', 'TSLA', 'AAPL', 'MSFT', 'GOOGL'],
            'max_size': 503
        }
    ]

    for test in tests:
        result = validate_test_case(test)
        print(f"{test['date']}: {'PASS' if result else 'FAIL'}")
```

---

## Data Quality Tiers

| Tier | Description | Backtest Use | Live Trading |
|------|-------------|--------------|--------------|
| **Production** | Actual historical S&P 500 constituents | ✅ Publication | ✅ Required |
| **Hybrid** | Mix of actual + proxy data with documented gaps | ⚠️ Internal only | ❌ No |
| **Proxy** | Current universe applied to all dates (current impl) | ❌ Exploratory only | ❌ No |

**Recommendation:**
- Use **Proxy** for early-stage exploration
- Upgrade to **Hybrid** for internal research
- Must have **Production** for live trading

---

## Alternative: External Universe Data Feed

**Instead of maintaining locally, subscribe to updates:**

```python
# services/universe_feed.py
class UniverseDataFeed:
    """
    Integration with external universe data provider.

    Pulls latest S&P 500 constituents daily and tracks changes.
    """

    def fetch_current_constituents(self) -> set[str]:
        """Fetch current S&P 500 from data provider API."""
        pass

    def fetch_historical_changes(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch historical changes from provider."""
        pass
```

**Providers:**
- **Spotify** (Kafka) for streaming updates
- **S&P Global** API for official data
- **Cloud-based** feeds (AWS Data Exchange, etc.)

---

## Quick Win: Manual Historical Snapshots

**For immediate improvement without full implementation:**

1. **Download 5 key historical snapshots:**
   - 2010-01-01 (pre-Facebook)
   - 2015-01-01 (tech boom beginning)
   - 2020-01-01 (pre-Tesla)
   - 2023-01-01 (post-COVID)
   - Current

2. **Store in `universe_snapshots` table:**
```sql
CREATE TABLE universe_snapshots (
    snapshot_date DATE PRIMARY KEY,
    symbols TEXT NOT NULL,  -- Comma-separated
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

3. **Use nearest snapshot:**
```python
def get_universe_approximate(target_date: date) -> set[str]:
    """
    Get approximate universe using nearest snapshot.

    Good enough for research, not for production.
    """
    snapshot = conn.execute("""
        SELECT symbols FROM universe_snapshots
        WHERE snapshot_date <= ?
        ORDER BY snapshot_date DESC
        LIMIT 1
    """).fetchone()

    return set(snapshot[0].split(','))
```

---

## Timeline & Effort Estimate

| Phase | Duration | Effort | Priority |
|-------|----------|--------|----------|
| Data acquisition | 1-2 weeks | Medium | High |
| Schema enhancement | 1 week | Low | High |
| Enhanced manager | 1 week | Medium | High |
| Production backfill | 1 week | Low | High |
| Validation | 1 week | Low | High |
| Documentation | 3 days | Low | Medium |
| **Total** | **5-7 weeks** | **Medium** | **High** |

---

## Success Criteria

- [ ] Historical S&P 500 changes loaded into database
- [ ] Universe table has `data_quality` column populated
- [ ] `get_universe_at_date()` returns point-in-time accurate data
- [ ] Validation tests pass against known benchmarks
- [ ] Documentation updated with data sources
- [ ] Backtest results show measurable difference from proxy approach

---

## Open Questions

1. **Data Source:** Purchase vs. scrape vs. hybrid?
2. **Maintenance:** Who updates historical data when S&P makes corrections?
3. **Cost Budget:** What's the budget for historical data?
4. **Timeline:** When is this needed? (Pre-production vs. pre-launch)
5. **Granularity:** Daily snapshots vs. continuous tracking?

---

## Next Steps

1. **Decision:** Choose data acquisition approach (purchase/scrape/hybrid)
2. **Research:** Evaluate 2-3 data providers
3. **Design:** Finalize schema for historical changes
4. **Implement:** Build enhanced UniverseManager
5. **Validate:** Run quality assurance tests
6. **Document:** Update CLAUDE.md with new approach

---

**Dependencies:**
- None blocking (can run in parallel with other Production Tier work)

**Blocks:**
- Nothing critical, but improves research rigor significantly

**Risk:**
- Low - Can always use current proxy approach for development
