"use client";

import { useEffect, useState } from "react";

import { api, type Status } from "@/lib/api";

/**
 * Global notice bar. Fetches /api/status and warns when market data is stale or
 * missing, so a view is never silently empty without an explanation. Renders
 * nothing when data is fresh, or when the status call itself fails (the page's
 * own error states handle an unreachable API).
 */
export function StatusBanner() {
  const [status, setStatus] = useState<Status | null>(null);

  useEffect(() => {
    let active = true;
    api
      .getStatus()
      .then((s) => active && setStatus(s))
      .catch(() => active && setStatus(null));
    return () => {
      active = false;
    };
  }, []);

  if (!status || status.ok) return null;

  return (
    <div
      role="status"
      className="border-b border-[var(--border)] bg-[rgba(201,162,75,0.10)] px-6 py-2 text-sm text-gold"
    >
      <span className="label mr-2">⚠ Notice</span>
      {status.message}
    </div>
  );
}
