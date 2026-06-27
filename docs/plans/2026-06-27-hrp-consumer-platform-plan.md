# HRP Consumer Platform Plan: Streamlit Retirement & Advisory Front-End

**Date:** 2026-06-27
**Context:** Streamline HRP into a product a non-technical owner can trust and use daily. Two directions are locked: (1) replace Streamlit with a React/Next.js consumer app, (2) a *moderate* scope cut — keep the engine fully intact, trim the surface. This plan also folds in a competitive teardown of **The Assembly** (`assemblyprivate.com`), a premium retail research terminal, which effectively spec's the consumer app for us.

---

## Locked Decisions

| Decision | Choice | Implication |
|----------|--------|-------------|
| Consumer UI tech | **React / Next.js SPA** | Needs a real HTTP/JSON API (does not exist yet — see Gap below) |
| Scope posture | **Moderate — keep engine, trim surface** | Nothing capable is removed. Only the Streamlit UI, dead scripts, and launcher sprawl go. Backend stays callable via API/CLI. |

## Strategic Frame: Keep the Engine, Replace the Cockpit

HRP today is an **institution-grade engine under a contributor-grade surface**. The ~80,700-line `hrp/` engine (DuckDB data layer, 45 features, walk-forward validation, VaR/CVaR, 10-agent pipeline, advisory service) is genuinely good. The ~16,000-line, 21-page Streamlit dashboard is a *researcher's workbench* — agents, kill gates, MLflow, drift, ops thresholds — none of which a non-technical owner wants to see.

The product a non-technical owner wants answers three questions: **What should I do today? Why? How am I doing vs. just holding the S&P?** That product already exists at ~80% (`hrp/advisory/`) but is buried as page 13 of 21 behind a login.

**Therefore the work is mostly subtraction + repackaging, not new capability.** Split into two surfaces:

- **Surface 1 — The Product (new, small):** a clean Next.js app, ~5 views, on a new HTTP API over the existing `PlatformAPI`.
- **Surface 2 — The Operator cockpit (you):** everything else stays **headless** — CLI, log files, MLflow, and the already-coded email digest (`hrp/advisory/digest.py`). No web UI needed.

---

## Competitive Teardown: The Assembly

A live walkthrough of `assemblyprivate.com/research` (logged in as a Member, data feed `FMP`). It is the clearest reference for what HRP's consumer surface should feel like — and where HRP can beat it.

### What it is

A **premium retail research terminal**: dark editorial "private members" aesthetic (serif headlines, gold-on-black, monospace data, live UTC clock, "EXECUTE the ticket" trading-desk language). Under the hood it is a **presentation + curation layer over commodity data** — the data feed is Financial Modeling Prep (FMP), a low-cost API.

### Three pillars

1. **A curated "Conviction List"** (their `Positions` page, labeled *"Manually curated Assembly positions and watchlist ideas with live prices and current performance"*). Each pick is tagged `Entry · Signal Date · Signaled By`, where *Signaled By* is a **named human creator** (e.g. "NoLimit", "Manu Invests", "Leap Trader"). Track-record stats: **Open / Watching / Average returns**.
2. **A research suite** — ~46 pages across 7 modules, all FMP data presented well.
3. **An AI assistant ("Vault Assistant")** — a RAG chatbot grounded on *"vault data only"*, rate-limited to 30 messages/day, with a feedback loop ("Rate this response").

### Module taxonomy (reference for our phased research suite)

| Module | Pages |
|--------|-------|
| Market Pulse | Macro, Market Sentiment, Custom Screener, Valuation Map, Commodities, Crypto, Sectors, Movers |
| Stock Screens | Undervalued, Quality, Dividends, Multiples, Momentum, Breakouts, Unusual Volume, Buybacks/Dilution, Squeeze Radar, Compare |
| Smart Money | Superinvestors (13F), Insider Buying, Congress Trading, 13D Activism, Analyst Ratings, Conviction Dashboard, Insider Track-Record, ETF Flows |
| Federal Intel | Live Filings, FRED Rates, COT (CFTC), Short Volume (FINRA), Recession Risk, Rulemaking, Federal Contracts |
| Options & Derivatives | (3 pages — flow/derivatives) |
| Calendar | Earnings, Earnings Intel, IPOs, Economic |
| Tools | Profit Calculator, Watchlist, My Portfolio (live-priced personal book), Vault Assistant |

### The strategic read

**The Assembly is most of what HRP wants to be — except the part HRP already does better.** Their weakness is HRP's moat:

- **They curate by hand → HRP validates systematically.** Their conviction list is one person's taste. HRP's can be generated from walk-forward-validated hypotheses, kill gates, and the 10-agent pipeline. That is a *defensible* edge.
- **They wrap FMP → HRP has a real engine.** HRP does not need to catch up on data; it needs to catch up on **presentation and breadth of context**.
- **Their assistant is the template for ours.** HRP's `advisory/explainer.py` already produces thesis/risk text. Wrap it in the same output contract and ground a chatbot on the existing DuckDB. `ANTHROPIC_API_KEY` plumbing already exists.

### Design language to adopt

The editorial/terminal aesthetic is a large part of why The Assembly reads as "premium" rather than "dashboard." Worth copying: dark background, serif headlines, monospace numerics, restrained single accent color, live clock, and an always-visible compliance footer (`// NOT INVESTMENT ADVICE`). This alone is the biggest perceived-quality jump over Streamlit.

---

## The Recommendation Output Contract

Every pick HRP surfaces — in the Conviction List, the Assistant, and the email digest — should serialize to one structure. This is the single most important spec in this document; it is the product. (Modeled on The Assembly's assistant output, upgraded with HRP's validation provenance.)

```
Recommendation
  ticker, company, action            # BUY / HOLD / SELL
  confidence                         # HIGH / MEDIUM / LOW (from signal strength + stability)
  thesis                             # named, plain-English narrative
  trade_params: { entry, target, stop }
  key_risks: [ plain-English bullets ]
  supporting_data: [ valuation / capital-returns / technicals / factor exposure ]
  provenance:                        # HRP's differentiator vs. "Signaled By: some guy"
    hypothesis_id, validation_status, walk_forward_stability, backtest_sharpe
  disclaimer
  feedback: rate_this(rec_id)        # closes the loop, feeds kill-gate calibrator
```

`hrp/advisory/explainer.py`, `recommendation_engine.py`, and `track_record.py` already produce most of these fields. The work is to (a) standardize the schema, (b) expose it over HTTP, (c) render it identically in all three channels.

---

## Consumer App: View Set

The Assembly walkthrough effectively defines the view list. Five views, each mapped to existing HRP modules.

| # | View | Mirrors (Assembly) | Backed by (HRP today) | Differentiator |
|---|------|--------------------|------------------------|----------------|
| 1 | **Conviction List** | Positions | `advisory/recommendation_engine.py`, `track_record.py` | Picks show *why they passed validation*, not who tipped them |
| 2 | **My Portfolio** | My Portfolio | `execution/positions.py`, live prices via data layer | Copy their ticket UX: Type/Symbol/Qty/Cost/Note → live NAV, allocation donut, P/L |
| 3 | **Assistant** | Vault Assistant | `advisory/explainer.py` + Claude + DuckDB RAG | Grounded on HRP's validated data; same output contract; rate-limited |
| 4 | **Track Record** | (Positions stats) | `advisory/track_record.py` | Open / Watching / Avg-return, alpha vs SPY |
| 5 | **Settings** | (account) | `advisory` user profile / safeguards | Risk level, exclusions, notification prefs |

A slim **Research** module (a few screens — Momentum, Quality, Unusual Volume — powered by the existing feature store) is deferred to a later phase; we do not need all 46 Assembly pages.

---

## Phased Roadmap

### Phase 0 — Cleanup & Consolidation *(low risk, do first, ~1 session)*
- Delete duplicate scripts after confirming the canonical one: `scripts/backfill_universe{,_from_source,_improved}.py`, redundant `load_*universe*.py`.
- Collapse `setup.sh` / `startup.sh` / `go_live.sh` / `run_full_cycle.sh` / `.command` launchers into a single CLI: **`hrp start` / `status` / `stop` / `doctor`**.
- Fix the broken `/doctor` permission rule (a heredoc pasted into `settings.local.json › permissions.allow`).
- *No new features — de-risks everything after.*

### Phase 1 — HTTP/JSON API *(the unlock for React)*
- **Gap:** the only HTTP server today is `hrp/ops/server.py` (health/metrics). `PlatformAPI` is 74 Python methods with no JSON layer. A React SPA cannot consume Python objects.
- Add `hrp/api/http/` (FastAPI routers) exposing ~12 endpoints: conviction list, recommendation detail, approve/reject, portfolio CRUD + live valuation, track record, settings, assistant query.
- Pydantic response models implementing the Recommendation Output Contract. Token/session auth. Tests.

### Phase 2 — Next.js Consumer App *(the visible product)*
- `web/` Next.js app, the 5 views above, on the Phase 1 API.
- Adopt the editorial/terminal design language. Compliance footer everywhere.
- Copy The Assembly's portfolio ticket UX and live-priced holdings table as the reference pattern.

### Phase 3 — Retire Streamlit
- Move the 2–3 researcher views actually used into CLI commands / a notebook; confirm the email digest covers operator monitoring.
- Delete `hrp/dashboard/` (~16k lines), `streamlit*` + `streamlit-authenticator` deps, dashboard auth.

### Phase 4 — Reliability Polish *(trust for non-technical users)*
- `hrp doctor`: data staleness, gaps, service health.
- "Data is N days stale — refresh" banners in-app (never a silent empty chart).
- One onboarding path; one front door.

### Phase 5+ — Research Modules *(HRP's breadth, phased)*
Build only screens HRP can power from its own feature store, mapped to existing features:

| Screen | Powered by (existing) | Status |
|--------|------------------------|--------|
| Momentum | `momentum_20d/60d/252d` | Have |
| Quality / Multiples | fundamental features (`pe_ratio`, `pb_ratio`, `ev_ebitda`) | Have |
| Unusual Volume | `volume_ratio`, `obv` | Have |
| Undervalued / Valuation Map | fundamentals + sector aggregation | Partial |
| Smart Money (13F / insider / congress) | — | Build (new data sources) |
| Macro / FRED / Calendar | — | Build (new data sources) |

---

## What Gets Deleted

- `hrp/dashboard/` (entire Streamlit app, ~16k lines) — Phase 3.
- `streamlit`, `plotly` (if unused elsewhere), `streamlit-authenticator` from dependencies — Phase 3.
- Duplicate `scripts/backfill_universe*` and `load_*universe*` — Phase 0 (keep canonical).
- Launcher sprawl (`.command` files, overlapping shell scripts) folded into `hrp` CLI — Phase 0.

**Explicitly NOT deleted (moderate-cut boundary):** `hrp/data`, `research`, `ml`, `risk`, `advisory`, `execution`, `agents`, `ops`, `mcp`. Every backend capability remains callable via API or CLI.

---

## Open Decisions & Risks

- **Assistant rate-limit & cost.** Claude-backed RAG needs a per-day cap (Assembly uses 30/day) and caching to control `ANTHROPIC_API_KEY` spend.
- **Live pricing source for the consumer portfolio.** Confirm whether intraday valuation uses the existing daily DuckDB close or a live quote source; The Assembly uses live FMP quotes.
- **Auth for the SPA.** New session/token scheme replaces `streamlit-authenticator`; decide cookie vs JWT.
- **Hosting.** Local-first on the Mac vs. a small deploy; affects the Next.js build/run story and `hrp start`.
- **Compliance.** Keep "not investment advice" framing throughout; HRP already restricts agents from approving deployments — preserve that human-in-the-loop boundary in the consumer approve/reject flow.

---

## Appendix: The Assembly URL Map (reference)

```
/research/home               feed (watchlist + portfolio driven)
/research/assistant          RAG chatbot, 30/day, output contract above
/research/portfolio          "Positions" — curated Conviction List + track record
/research/my-portfolio       live-priced personal book (ticket UX)
/research/watchlist          watchlist
/research/profit-calculator  position P/L calculator
Market Pulse:   /macro /sentiment /screener /sectors-pe /commodities /crypto /sectors /movers
Stock Screens:  /undervalued /quality /dividends /pe /momentum /breakouts /unusual-volume /buybacks /short-interest /compare
Smart Money:    /superinvestors /insider-buying /congress /activism /analyst-changes /conviction /insider-track-record /etf-flows
Federal Intel:  /filings /fred /cot /short-volume /treasury /regulations /contracts
Calendar:       /calendar /earnings /ipos /economic
Education:      /research /education
```
