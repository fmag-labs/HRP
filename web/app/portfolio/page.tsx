"use client";

import { api } from "@/lib/api";
import { useApi } from "@/components/useApi";
import {
  PageHeader,
  ErrorBanner,
  Loading,
  EmptyState,
  fmtMoney,
  fmtPct,
  pnlClass,
  signed,
} from "@/components/ui";

const ALLOC_COLORS = [
  "#c9a24b",
  "#8c6f33",
  "#a98b3f",
  "#6f5826",
  "#d8b765",
  "#5c4a20",
];

export default function PortfolioPage() {
  const { data, error, loading } = useApi(() => api.getPortfolio(), []);

  return (
    <div>
      <PageHeader label="Members · Holdings" title="My Portfolio" />

      {error ? (
        <ErrorBanner error={error} />
      ) : loading || !data ? (
        <Loading rows={6} />
      ) : (
        <div className="space-y-8">
          {/* NAV summary */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="border border-[var(--border)] bg-[var(--surface)] px-6 py-6">
              <div className="label mb-2">Net Asset Value</div>
              <div className="mono text-4xl font-semibold text-gold">
                {fmtMoney(data.total_value)}
              </div>
            </div>
            <div className="border border-[var(--border)] bg-[var(--surface)] px-6 py-6">
              <div className="label mb-2">Open Positions</div>
              <div className="mono text-4xl font-semibold text-foreground">
                {data.position_count}
              </div>
            </div>
          </div>

          {/* Allocation bars */}
          {data.positions.length > 0 ? (
            <section>
              <div className="label mb-3">Allocation by Market Value</div>
              <div className="flex h-3 w-full overflow-hidden border border-[var(--border)]">
                {data.positions.map((p, i) => {
                  const share =
                    data.total_value > 0 ? p.market_value / data.total_value : 0;
                  return (
                    <div
                      key={p.symbol}
                      title={`${p.symbol} ${(share * 100).toFixed(1)}%`}
                      style={{
                        width: `${share * 100}%`,
                        background: ALLOC_COLORS[i % ALLOC_COLORS.length],
                      }}
                    />
                  );
                })}
              </div>
              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
                {data.positions.map((p, i) => (
                  <span key={p.symbol} className="mono flex items-center gap-1.5 text-[11px] text-muted">
                    <span
                      className="inline-block h-2 w-2"
                      style={{ background: ALLOC_COLORS[i % ALLOC_COLORS.length] }}
                    />
                    {p.symbol}
                  </span>
                ))}
              </div>
            </section>
          ) : null}

          {/* Positions table */}
          {data.positions.length === 0 ? (
            <EmptyState message="No open positions" />
          ) : (
            <div className="overflow-x-auto border border-[var(--border)]">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="border-b border-[var(--border)] bg-[var(--surface)] text-left">
                    {["Symbol", "Qty", "Avg Cost", "Price", "Market Value", "Unrealized P/L"].map(
                      (h) => (
                        <th key={h} className="label px-4 py-3 font-normal">
                          {h}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody>
                  {data.positions.map((p) => (
                    <tr
                      key={p.symbol}
                      className="border-b border-[var(--border)] last:border-0 hover:bg-[var(--surface)]"
                    >
                      <td className="px-4 py-3 mono text-base font-semibold text-gold">
                        {p.symbol}
                      </td>
                      <td className="px-4 py-3 mono text-muted">{p.quantity}</td>
                      <td className="px-4 py-3 mono">{fmtMoney(p.avg_cost)}</td>
                      <td className="px-4 py-3 mono">{fmtMoney(p.current_price)}</td>
                      <td className="px-4 py-3 mono">{fmtMoney(p.market_value)}</td>
                      <td className={`px-4 py-3 mono font-semibold ${pnlClass(p.unrealized_pnl)}`}>
                        {signed(p.unrealized_pnl, (n) => fmtMoney(n).replace("$", "$"))}
                        <span className="ml-2 text-xs font-normal text-muted">
                          {signed(p.unrealized_pnl_pct, (n) => fmtPct(n))}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
