"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";
import { api, ApprovalResult, RejectionResult } from "@/lib/api";
import { useApi } from "@/components/useApi";
import {
  ErrorBanner,
  Loading,
  ActionBadge,
  StatusPill,
  StrengthBar,
  fmtMoney,
  fmtPct,
} from "@/components/ui";

function Param({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="border border-[var(--border)] bg-[var(--surface)] px-4 py-3">
      <div className="label mb-1.5">{label}</div>
      <div className="mono text-lg text-foreground">{value}</div>
    </div>
  );
}

export default function RecommendationDetailPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const { data, error, loading } = useApi(() => api.getRecommendation(id), [id]);

  const [actionResult, setActionResult] = useState<
    ApprovalResult | RejectionResult | null
  >(null);
  const [actionError, setActionError] = useState<unknown>(null);
  const [busy, setBusy] = useState<"approve" | "reject" | null>(null);

  async function onApprove() {
    setBusy("approve");
    setActionError(null);
    setActionResult(null);
    try {
      const res = await api.approveRecommendation(id, { actor: "member", dry_run: true });
      setActionResult(res);
    } catch (e) {
      setActionError(e);
    } finally {
      setBusy(null);
    }
  }

  async function onReject() {
    setBusy("reject");
    setActionError(null);
    setActionResult(null);
    try {
      const res = await api.rejectRecommendation(id, {
        actor: "member",
        reason: "Declined by member",
      });
      setActionResult(res);
    } catch (e) {
      setActionError(e);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div>
      <Link
        href="/"
        className="label mb-6 inline-block transition-colors hover:text-gold"
      >
        ← Back to Conviction List
      </Link>

      {error ? (
        <ErrorBanner error={error} />
      ) : loading || !data ? (
        <Loading rows={6} />
      ) : (
        <div className="space-y-8">
          {/* Header */}
          <div className="flex flex-wrap items-end justify-between gap-4 border-b border-[var(--border)] pb-6">
            <div>
              <div className="label mb-2">Members · Conviction Dossier</div>
              <div className="flex items-center gap-4">
                <h1 className="serif text-5xl font-bold leading-none text-gold">
                  {data.symbol}
                </h1>
                <ActionBadge action={data.action} />
                <StatusPill status={data.status} />
              </div>
            </div>
            <div className="text-right">
              <div className="label mb-1">Confidence</div>
              <div className="mono text-2xl text-foreground">
                {data.confidence}
              </div>
              <div className="mt-2">
                <StrengthBar value={data.signal_strength} />
              </div>
            </div>
          </div>

          {/* Thesis */}
          <section>
            <div className="label mb-3">Investment Thesis</div>
            <p className="serif max-w-3xl text-xl leading-relaxed text-foreground">
              {data.thesis || "No thesis recorded for this recommendation."}
            </p>
          </section>

          {/* Parameter block */}
          <section>
            <div className="label mb-3">Trade Parameters</div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Param label="Entry" value={fmtMoney(data.entry_price)} />
              <Param label="Target" value={fmtMoney(data.target_price)} />
              <Param label="Stop" value={fmtMoney(data.stop_price)} />
              <Param
                label="Position"
                value={
                  data.position_pct === null || data.position_pct === undefined
                    ? "—"
                    : fmtPct(data.position_pct, 1)
                }
              />
              <Param
                label="Horizon"
                value={
                  data.time_horizon_days === null || data.time_horizon_days === undefined
                    ? "—"
                    : `${data.time_horizon_days}d`
                }
              />
            </div>
          </section>

          {/* Risks */}
          <section>
            <div className="label mb-3">Key Risks</div>
            <p className="max-w-3xl text-sm leading-relaxed text-muted">
              {data.risks || "No specific risks recorded."}
            </p>
          </section>

          {/* Provenance */}
          <section className="border border-[var(--gold-dim)] bg-[rgba(201,162,75,0.05)] px-5 py-5">
            <div className="label mb-3 text-gold">Provenance · Validated, Not a Hot Tip</div>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div>
                <div className="label mb-1">Hypothesis</div>
                <div className="mono text-sm text-foreground">
                  {data.provenance?.hypothesis_id || "—"}
                </div>
              </div>
              <div>
                <div className="label mb-1">Model</div>
                <div className="mono text-sm text-foreground">
                  {data.provenance?.model_name || "—"}
                </div>
              </div>
              <div>
                <div className="label mb-1">Validation Status</div>
                <div className="mono text-sm font-semibold text-pos">
                  {data.provenance?.validation_status || "—"}
                </div>
              </div>
            </div>
          </section>

          {/* Actions */}
          <section className="border-t border-[var(--border)] pt-6">
            <div className="label mb-3">Member Action</div>
            <div className="flex flex-wrap gap-3">
              <button
                onClick={onApprove}
                disabled={busy !== null}
                className="mono border border-[var(--pos)] px-5 py-2 text-xs font-semibold uppercase tracking-widest text-pos transition-colors hover:bg-[rgba(74,222,128,0.1)] disabled:opacity-40"
              >
                {busy === "approve" ? "Approving…" : "Approve (dry run)"}
              </button>
              <button
                onClick={onReject}
                disabled={busy !== null}
                className="mono border border-[var(--neg)] px-5 py-2 text-xs font-semibold uppercase tracking-widest text-neg transition-colors hover:bg-[rgba(248,113,113,0.1)] disabled:opacity-40"
              >
                {busy === "reject" ? "Rejecting…" : "Reject"}
              </button>
            </div>

            {actionError ? (
              <div className="mt-4">
                <ErrorBanner error={actionError} />
              </div>
            ) : null}

            {actionResult ? (
              <div className="mt-4 border border-[var(--border-strong)] bg-[var(--surface)] px-4 py-3">
                <div className="label mb-1 text-gold">
                  {actionResult.action} · {actionResult.actor}
                </div>
                <p className="text-sm text-foreground">{actionResult.message}</p>
                {"order_id" in actionResult && actionResult.order_id ? (
                  <p className="mono mt-1 text-xs text-muted">
                    order_id: {actionResult.order_id}
                  </p>
                ) : null}
                {"reason" in actionResult && actionResult.reason ? (
                  <p className="mono mt-1 text-xs text-muted">
                    reason: {actionResult.reason}
                  </p>
                ) : null}
              </div>
            ) : null}
          </section>
        </div>
      )}
    </div>
  );
}
