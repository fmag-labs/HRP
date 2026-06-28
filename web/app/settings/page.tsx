"use client";

import { useEffect, useState } from "react";
import { api, Settings, SettingsUpdate, Horizon } from "@/lib/api";
import { useApi } from "@/components/useApi";
import {
  PageHeader,
  ErrorBanner,
  Loading,
  ComingSoon,
  isComingSoon,
} from "@/components/ui";

const RISK_LABELS: Record<number, string> = {
  1: "1 · Conservative",
  2: "2 · Cautious",
  3: "3 · Balanced",
  4: "4 · Assertive",
  5: "5 · Aggressive",
};

const HORIZONS: Horizon[] = ["short", "medium", "long"];

function toList(text: string): string[] {
  return text
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

export default function SettingsPage() {
  const { data, error, loading } = useApi(() => api.getSettings(), []);

  const [form, setForm] = useState<Settings | null>(null);
  const [excludedSymbolsText, setExcludedSymbolsText] = useState("");
  const [excludedSectorsText, setExcludedSectorsText] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<unknown>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (data) {
      setForm(data);
      setExcludedSymbolsText((data.excluded_symbols || []).join(", "));
      setExcludedSectorsText((data.excluded_sectors || []).join(", "));
    }
  }, [data]);

  async function onSave(e: React.FormEvent) {
    e.preventDefault();
    if (!form) return;
    setSaving(true);
    setSaveError(null);
    setSaved(false);
    const payload: SettingsUpdate = {
      name: form.name,
      risk_tolerance: form.risk_tolerance,
      portfolio_value: form.portfolio_value,
      max_positions: form.max_positions,
      max_position_pct: form.max_position_pct,
      preferred_horizon: form.preferred_horizon,
      excluded_symbols: toList(excludedSymbolsText).map((s) => s.toUpperCase()),
      excluded_sectors: toList(excludedSectorsText),
    };
    try {
      const res = await api.updateSettings(payload);
      setForm(res);
      setExcludedSymbolsText((res.excluded_symbols || []).join(", "));
      setExcludedSectorsText((res.excluded_sectors || []).join(", "));
      setSaved(true);
    } catch (err) {
      setSaveError(err);
    } finally {
      setSaving(false);
    }
  }

  if (error && isComingSoon(error)) {
    return (
      <div>
        <PageHeader label="Members · Configuration" title="Settings" />
        <ComingSoon title="Settings are being provisioned" />
      </div>
    );
  }

  return (
    <div>
      <PageHeader label="Members · Configuration" title="Settings" />

      {error ? (
        <ErrorBanner error={error} />
      ) : loading || !form ? (
        <Loading rows={5} />
      ) : (
        <form onSubmit={onSave} className="max-w-xl space-y-6">
          <div>
            <label className="label mb-2 block">Profile Name</label>
            <input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder="Member"
              className="mono w-full border border-[var(--border-strong)] bg-[var(--surface)] px-4 py-2.5 text-sm text-foreground outline-none placeholder:text-muted focus:border-gold"
            />
          </div>

          <div>
            <label className="label mb-2 block">Risk Tolerance</label>
            <select
              value={form.risk_tolerance}
              onChange={(e) =>
                setForm({ ...form, risk_tolerance: Number(e.target.value) })
              }
              className="mono w-full border border-[var(--border-strong)] bg-[var(--surface)] px-4 py-2.5 text-sm text-foreground outline-none focus:border-gold"
            >
              {[1, 2, 3, 4, 5].map((r) => (
                <option key={r} value={r}>
                  {RISK_LABELS[r]}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="label mb-2 block">Preferred Horizon</label>
            <select
              value={form.preferred_horizon}
              onChange={(e) =>
                setForm({ ...form, preferred_horizon: e.target.value as Horizon })
              }
              className="mono w-full border border-[var(--border-strong)] bg-[var(--surface)] px-4 py-2.5 text-sm text-foreground outline-none focus:border-gold"
            >
              {HORIZONS.map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="label mb-2 block">Portfolio Value (USD)</label>
            <input
              type="number"
              min={0}
              step="any"
              value={form.portfolio_value}
              onChange={(e) =>
                setForm({ ...form, portfolio_value: Number(e.target.value) })
              }
              className="mono w-full border border-[var(--border-strong)] bg-[var(--surface)] px-4 py-2.5 text-sm text-foreground outline-none focus:border-gold"
            />
          </div>

          <div>
            <label className="label mb-2 block">Max Positions</label>
            <input
              type="number"
              min={0}
              value={form.max_positions}
              onChange={(e) =>
                setForm({ ...form, max_positions: Number(e.target.value) })
              }
              className="mono w-full border border-[var(--border-strong)] bg-[var(--surface)] px-4 py-2.5 text-sm text-foreground outline-none focus:border-gold"
            />
          </div>

          <div>
            <label className="label mb-2 block">
              Max Position Size (fraction, 0–1)
            </label>
            <input
              type="number"
              min={0}
              max={1}
              step="0.01"
              value={form.max_position_pct}
              onChange={(e) =>
                setForm({ ...form, max_position_pct: Number(e.target.value) })
              }
              className="mono w-full border border-[var(--border-strong)] bg-[var(--surface)] px-4 py-2.5 text-sm text-foreground outline-none focus:border-gold"
            />
          </div>

          <div>
            <label className="label mb-2 block">Excluded Symbols (comma separated)</label>
            <input
              value={excludedSymbolsText}
              onChange={(e) => setExcludedSymbolsText(e.target.value)}
              placeholder="TSLA, GME, AMC"
              className="mono w-full border border-[var(--border-strong)] bg-[var(--surface)] px-4 py-2.5 text-sm text-foreground outline-none placeholder:text-muted focus:border-gold"
            />
          </div>

          <div>
            <label className="label mb-2 block">Excluded Sectors (comma separated)</label>
            <input
              value={excludedSectorsText}
              onChange={(e) => setExcludedSectorsText(e.target.value)}
              placeholder="Financials, Energy"
              className="mono w-full border border-[var(--border-strong)] bg-[var(--surface)] px-4 py-2.5 text-sm text-foreground outline-none placeholder:text-muted focus:border-gold"
            />
          </div>

          {saveError ? <ErrorBanner error={saveError} /> : null}
          {saved ? (
            <p className="mono text-xs uppercase tracking-widest text-pos">
              Settings saved
            </p>
          ) : null}

          <button
            type="submit"
            disabled={saving}
            className="mono border border-gold px-6 py-2.5 text-xs font-semibold uppercase tracking-widest text-gold transition-colors hover:bg-[rgba(201,162,75,0.1)] disabled:opacity-40"
          >
            {saving ? "Saving…" : "Save Settings"}
          </button>
        </form>
      )}
    </div>
  );
}
