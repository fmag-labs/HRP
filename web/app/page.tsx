"use client";

import Link from "next/link";
import { useState } from "react";
import { api } from "@/lib/api";
import { useApi } from "@/components/useApi";
import {
  PageHeader,
  ErrorBanner,
  Loading,
  EmptyState,
  ActionBadge,
  StatusPill,
  StrengthBar,
  fmtMoney,
  fmtPct,
  pnlClass,
} from "@/components/ui";

const STATUS_FILTERS = ["", "active", "closed", "approved", "rejected"];

export default function ConvictionListPage() {
  const [status, setStatus] = useState("");
  const { data, error, loading } = useApi(
    () => api.getRecommendations({ status: status || undefined, limit: 100 }),
    [status]
  );

  return (
    <div>
      <PageHeader label="Members · Conviction List" title="The Conviction List">
        <select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="mono border border-[var(--border-strong)] bg-[var(--surface)] px-3 py-1.5 text-xs uppercase tracking-widest text-foreground outline-none focus:border-gold"
        >
          {STATUS_FILTERS.map((s) => (
            <option key={s} value={s}>
              {s || "all"}
            </option>
          ))}
        </select>
      </PageHeader>

      {error ? (
        <ErrorBanner error={error} />
      ) : loading ? (
        <Loading rows={8} />
      ) : !data || data.length === 0 ? (
        <EmptyState message="No recommendations on the list" />
      ) : (
        <div className="overflow-x-auto border border-[var(--border)]">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-[var(--border)] bg-[var(--surface)] text-left">
                {["Ticker", "Action", "Confidence", "Entry", "Signal", "Return", "Status"].map(
                  (h) => (
                    <th key={h} className="label px-4 py-3 font-normal">
                      {h}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody>
              {data.map((r) => (
                <tr
                  key={r.recommendation_id}
                  className="group border-b border-[var(--border)] last:border-0 hover:bg-[var(--surface)]"
                >
                  <td className="px-4 py-3">
                    <Link
                      href={`/recommendations/${r.recommendation_id}`}
                      className="mono text-base font-semibold text-gold group-hover:underline"
                    >
                      {r.symbol}
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <ActionBadge action={r.action} />
                  </td>
                  <td className="px-4 py-3 mono text-muted">{r.confidence}</td>
                  <td className="px-4 py-3 mono">{fmtMoney(r.entry_price)}</td>
                  <td className="px-4 py-3">
                    <StrengthBar value={r.signal_strength} />
                  </td>
                  <td className={`px-4 py-3 mono ${pnlClass(r.realized_return)}`}>
                    {r.realized_return === null || r.realized_return === undefined
                      ? "—"
                      : fmtPct(r.realized_return)}
                  </td>
                  <td className="px-4 py-3">
                    <StatusPill status={r.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data && data.length > 0 ? (
        <p className="label mt-4">
          {data.length} {data.length === 1 ? "position" : "positions"} · click a ticker for
          the full dossier
        </p>
      ) : null}
    </div>
  );
}
