"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import { useApi } from "@/components/useApi";
import {
  ErrorBanner,
  Loading,
  EmptyState,
  isComingSoon,
  fmtNum,
  fmtDate,
} from "@/components/ui";

function ScreenNotFound() {
  return (
    <div className="flex flex-col items-center justify-center border border-dashed border-[var(--border-strong)] bg-[var(--surface)] px-6 py-16 text-center">
      <div className="label mb-3">Not Found</div>
      <h2 className="serif text-2xl text-gold">Screen not found</h2>
      <p className="mt-2 max-w-md text-sm text-muted">
        This screen does not exist. Return to research to browse the available
        screens.
      </p>
    </div>
  );
}

export default function ScreenViewPage() {
  const params = useParams<{ key: string }>();
  const key = params.key;
  const { data, error, loading } = useApi(() => api.getScreen(key), [key]);

  return (
    <div>
      <Link
        href="/research"
        className="label mb-6 inline-block transition-colors hover:text-gold"
      >
        ← Back to Research
      </Link>

      {error ? (
        isComingSoon(error) ? (
          <ScreenNotFound />
        ) : (
          <ErrorBanner error={error} />
        )
      ) : loading || !data ? (
        <Loading rows={8} />
      ) : (
        <div>
          <div className="mb-7 border-b border-[var(--border)] pb-5">
            <div className="label mb-2">Members · Research Screen</div>
            <h1 className="serif text-3xl font-semibold leading-none text-foreground sm:text-4xl">
              {data.title}
            </h1>
            <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted">
              {data.subtitle}
            </p>
            <p className="label mt-3">as of {fmtDate(data.as_of)}</p>
          </div>

          {data.rows.length === 0 ? (
            <EmptyState message="No results for this screen yet." />
          ) : (
            <div className="overflow-x-auto border border-[var(--border)]">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="border-b border-[var(--border)] bg-[var(--surface)] text-left">
                    <th className="label px-4 py-3 font-normal">Rank</th>
                    <th className="label px-4 py-3 font-normal">Ticker</th>
                    <th className="label px-4 py-3 font-normal">Company</th>
                    <th className="label px-4 py-3 font-normal">Sector</th>
                    <th className="label px-4 py-3 text-right font-normal">
                      {data.value_label}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((row) => (
                    <tr
                      key={`${row.rank}-${row.symbol}`}
                      className="border-b border-[var(--border)] last:border-0 hover:bg-[var(--surface)]"
                    >
                      <td className="px-4 py-3 mono text-muted">{row.rank}</td>
                      <td className="px-4 py-3 mono text-base font-semibold text-gold">
                        {row.symbol}
                      </td>
                      <td className="px-4 py-3 text-foreground">
                        {row.name || "—"}
                      </td>
                      <td className="px-4 py-3 text-muted">{row.sector || "—"}</td>
                      <td className="px-4 py-3 mono text-right text-foreground">
                        {fmtNum(row.value, 2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {data.rows.length > 0 ? (
            <p className="label mt-4">
              {data.rows.length} {data.rows.length === 1 ? "name" : "names"} ·
              ranked by {data.value_label.toLowerCase()}
            </p>
          ) : null}
        </div>
      )}
    </div>
  );
}
