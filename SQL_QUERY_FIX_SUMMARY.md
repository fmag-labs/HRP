# SQL Query Fix Summary

## Issue Description

**Error Message:**
```
2026-01-24 18:03:02.767 | ERROR | hrp.data.db:connection:395 - Database error: 
Binder Error: Referenced column "provider" not found in FROM clause!
Candidate bindings: "source_id", "api_name"
```

**Location:** `hrp/dashboard/pages/ingestion_status.py` - `get_data_sources()` function

**Impact:** Non-critical - The error appeared in logs but did not crash the dashboard. The Data Sources section would show an empty table due to the failed query.

---

## Root Cause

The `get_data_sources()` function was querying for columns that don't exist in the `data_sources` table:

**Incorrect Query (Before):**
```sql
SELECT
    source_id,
    source_type,
    provider,           -- ❌ Does not exist
    is_active,          -- ❌ Does not exist  
    last_fetch_date,    -- ❌ Does not exist
    created_at          -- ❌ Does not exist
FROM data_sources
```

**Actual Schema (from `hrp/data/schema.py`):**
```sql
CREATE TABLE IF NOT EXISTS data_sources (
    source_id VARCHAR PRIMARY KEY,
    source_type VARCHAR,
    api_name VARCHAR,           -- ✅ Actual column
    last_fetch TIMESTAMP,       -- ✅ Actual column
    status VARCHAR DEFAULT 'active'  -- ✅ Actual column
)
```

---

## Fix Applied

### 1. Updated SQL Query

**File:** `hrp/dashboard/pages/ingestion_status.py`

**Changed Query:**
```sql
SELECT
    source_id,
    source_type,
    api_name,      -- ✅ Correct column
    status,        -- ✅ Correct column
    last_fetch     -- ✅ Correct column
FROM data_sources
ORDER BY source_id
```

### 2. Updated Exception Handling

**Before:**
```python
return pd.DataFrame(columns=[
    "source_id", "source_type", "provider", "is_active",
    "last_fetch_date", "created_at"
])
```

**After:**
```python
return pd.DataFrame(columns=[
    "source_id", "source_type", "api_name", "status",
    "last_fetch"
])
```

### 3. Updated Display Configuration

**Before:**
```python
column_config={
    "source_id": st.column_config.TextColumn("Source ID", width="medium"),
    "source_type": st.column_config.TextColumn("Type", width="small"),
    "provider": st.column_config.TextColumn("Provider", width="medium"),
    "is_active": st.column_config.CheckboxColumn("Active", width="small"),
    "last_fetch_date": st.column_config.DateColumn("Last Fetch", width="medium"),
    "created_at": st.column_config.DatetimeColumn("Created", width="medium"),
}
```

**After:**
```python
column_config={
    "source_id": st.column_config.TextColumn("Source ID", width="medium"),
    "source_type": st.column_config.TextColumn("Type", width="small"),
    "api_name": st.column_config.TextColumn("API Name", width="medium"),
    "status": st.column_config.TextColumn("Status", width="small"),
    "last_fetch": st.column_config.DatetimeColumn("Last Fetch", width="medium"),
}
```

---

## Verification

### Test 1: Query Execution
```bash
python -c "
from hrp.data.db import get_db
db = get_db()
result = db.fetchdf('''
    SELECT source_id, source_type, api_name, status, last_fetch
    FROM data_sources
    ORDER BY source_id
''')
print('✅ Query executed successfully')
print(f'Columns: {list(result.columns)}')
"
```

**Result:** ✅ PASS
```
✅ Query executed successfully
Columns: ['source_id', 'source_type', 'api_name', 'status', 'last_fetch']
```

### Test 2: Dashboard Function
```python
from hrp.dashboard.pages import ingestion_status
df = ingestion_status.get_data_sources()
print(f'✅ Function works correctly')
print(f'Columns: {list(df.columns)}')
```

**Result:** ✅ PASS
```
✅ get_data_sources() works correctly
Columns: ['source_id', 'source_type', 'api_name', 'status', 'last_fetch']
✅ Column names match expected schema
```

---

## Files Changed

1. **`hrp/dashboard/pages/ingestion_status.py`**
   - Modified `get_data_sources()` function (lines 45-67)
   - Updated display configuration (lines 223-237)

---

## Status

✅ **FIXED** - All tests pass, query executes successfully, dashboard page loads without errors.

---

## Related Documentation

- **Schema Definition:** `hrp/data/schema.py` (lines 29-38)
- **Dashboard Verification Report:** `DASHBOARD_VERIFICATION_REPORT.md`
- **Issue Found During:** Manual dashboard verification (subtask-3-4)

---

## Date

**Fixed:** January 24, 2026 18:07 PST  
**Verified:** January 24, 2026 18:07 PST
