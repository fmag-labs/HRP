"use client";

import { ApiError } from "@/lib/api";

// ---- Page header ----

export function PageHeader({
  label,
  title,
  children,
}: {
  label: string;
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="mb-7 flex flex-wrap items-end justify-between gap-4 border-b border-[var(--border)] pb-5">
      <div>
        <div className="label mb-2">{label}</div>
        <h1 className="serif text-3xl font-semibold leading-none text-foreground sm:text-4xl">
          {title}
        </h1>
      </div>
      {children ? <div className="flex items-center gap-2">{children}</div> : null}
    </div>
  );
}

// ---- Error banner ----

export function ErrorBanner({ error }: { error: unknown }) {
  const message =
    error instanceof Error ? error.message : "An unexpected error occurred.";
  return (
    <div
      role="alert"
      className="flex items-start gap-3 border border-[var(--neg)] bg-[rgba(248,113,113,0.07)] px-4 py-3"
    >
      <span className="mono mt-0.5 text-xs font-semibold text-neg">ERR</span>
      <div className="text-sm text-foreground">
        <p className="mono text-xs uppercase tracking-widest text-neg">
          Terminal connection fault
        </p>
        <p className="mt-1 text-muted">{message}</p>
      </div>
    </div>
  );
}

// Distinguish a 404/501 (feature coming soon) from a hard failure.
export function isComingSoon(error: unknown): boolean {
  return error instanceof ApiError && (error.status === 404 || error.status === 501);
}

export function ComingSoon({ title }: { title: string }) {
  return (
    <div className="flex flex-col items-center justify-center border border-dashed border-[var(--border-strong)] bg-[var(--surface)] px-6 py-16 text-center">
      <div className="label mb-3">Provisioning</div>
      <h2 className="serif text-2xl text-gold">{title}</h2>
      <p className="mt-2 max-w-md text-sm text-muted">
        This module is being wired into the terminal. Check back shortly.
      </p>
    </div>
  );
}

// ---- Loading skeleton ----

export function Loading({ rows = 6 }: { rows?: number }) {
  return (
    <div className="space-y-2" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="h-12 w-full animate-pulse border border-[var(--border)] bg-[var(--surface)]"
        />
      ))}
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="border border-[var(--border)] bg-[var(--surface)] px-6 py-12 text-center">
      <p className="mono text-xs uppercase tracking-widest text-muted">{message}</p>
    </div>
  );
}

// ---- Action badge ----

export function ActionBadge({ action }: { action: string }) {
  const a = action.toUpperCase();
  const color =
    a === "BUY"
      ? "border-[var(--pos)] text-pos"
      : a === "SELL"
        ? "border-[var(--neg)] text-neg"
        : "border-[var(--gold-dim)] text-gold";
  return (
    <span
      className={`mono inline-block border px-2 py-0.5 text-[11px] font-semibold tracking-widest ${color}`}
    >
      {a}
    </span>
  );
}

// ---- Status pill ----

export function StatusPill({ status }: { status: string }) {
  return (
    <span className="mono inline-block border border-[var(--border-strong)] bg-[var(--surface-2)] px-2 py-0.5 text-[10px] uppercase tracking-widest text-muted">
      {status}
    </span>
  );
}

// ---- Signal strength bar ----

export function StrengthBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-24 bg-[var(--surface-2)]">
        <div className="h-full bg-gold" style={{ width: `${pct}%` }} />
      </div>
      <span className="mono text-[11px] text-muted">{pct.toFixed(0)}</span>
    </div>
  );
}

// ---- Number formatting helpers ----

export function fmtMoney(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

export function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function fmtDate(v: string | null | undefined): string {
  if (!v) return "—";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return v;
  return d.toISOString().slice(0, 10);
}

export function pnlClass(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "text-muted";
  if (v > 0) return "text-pos";
  if (v < 0) return "text-neg";
  return "text-muted";
}

export function signed(v: number | null | undefined, fn: (n: number) => string): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const s = fn(Math.abs(v));
  return v < 0 ? `-${s}` : v > 0 ? `+${s}` : s;
}
