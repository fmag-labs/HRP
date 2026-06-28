"use client";

import { api } from "@/lib/api";
import { useApi } from "@/components/useApi";
import {
  PageHeader,
  ErrorBanner,
  Loading,
  EmptyState,
  fmtPct,
  fmtDate,
  pnlClass,
  signed,
} from "@/components/ui";

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="border border-[var(--border)] bg-[var(--surface)] px-6 py-6">
      <div className="label mb-2">{label}</div>
      <div className={`mono text-3xl font-semibold ${tone || "text-foreground"}`}>
        {value}
      </div>
    </div>
  );
}

export default function TrackRecordPage() {
  const { data, error, loading } = useApi(() => api.getTrackRecord(), []);

  // Headline figures: aggregate from the most recent period when available.
  const latest = data && data.length > 0 ? data[data.length - 1] : null;

  return (
    <div>
      <PageHeader label="Members · Performance Ledger" title="Track Record" />

      {error ? (
        <ErrorBanner error={error} />
      ) : loading || !data ? (
        <Loading rows={6} />
      ) : data.length === 0 ? (
        <EmptyState message="No performance periods recorded" />
      ) : (
        <div className="space-y-8">
          {latest ? (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <Stat
                label="Win Rate"
                value={fmtPct(latest.win_rate, 1)}
                tone="text-gold"
              />
              <Stat
                label="Avg Return"
                value={signed(latest.avg_return, (n) => fmtPct(n))}
                tone={pnlClass(latest.avg_return)}
              />
              <Stat
                label="Excess vs Benchmark"
                value={signed(latest.excess_return, (n) => fmtPct(n))}
                tone={pnlClass(latest.excess_return)}
              />
            </div>
          ) : null}

          <div className="overflow-x-auto border border-[var(--border)]">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-[var(--border)] bg-[var(--surface)] text-left">
                  {[
                    "Period",
                    "Recs",
                    "Closed",
                    "W / L",
                    "Win Rate",
                    "Avg Ret",
                    "Best",
                    "Worst",
                    "Benchmark",
                    "Excess",
                  ].map((h) => (
                    <th key={h} className="label px-4 py-3 font-normal">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.map((p, i) => (
                  <tr
                    key={`${p.period_start}-${i}`}
                    className="border-b border-[var(--border)] last:border-0 hover:bg-[var(--surface)]"
                  >
                    <td className="px-4 py-3 mono text-xs text-muted">
                      {fmtDate(p.period_start)} → {fmtDate(p.period_end)}
                    </td>
                    <td className="px-4 py-3 mono">{p.total_recommendations}</td>
                    <td className="px-4 py-3 mono">{p.closed_recommendations}</td>
                    <td className="px-4 py-3 mono text-xs">
                      <span className="text-pos">{p.profitable}</span>
                      <span className="text-muted"> / </span>
                      <span className="text-neg">{p.unprofitable}</span>
                    </td>
                    <td className="px-4 py-3 mono text-gold">{fmtPct(p.win_rate, 1)}</td>
                    <td className={`px-4 py-3 mono ${pnlClass(p.avg_return)}`}>
                      {signed(p.avg_return, (n) => fmtPct(n))}
                    </td>
                    <td className="px-4 py-3 mono text-pos">{p.best_pick || "—"}</td>
                    <td className="px-4 py-3 mono text-neg">{p.worst_pick || "—"}</td>
                    <td className="px-4 py-3 mono text-muted">
                      {signed(p.benchmark_return, (n) => fmtPct(n))}
                    </td>
                    <td className={`px-4 py-3 mono ${pnlClass(p.excess_return)}`}>
                      {signed(p.excess_return, (n) => fmtPct(n))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
