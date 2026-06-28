"use client";

import Link from "next/link";
import { api } from "@/lib/api";
import { useApi } from "@/components/useApi";
import { PageHeader, ErrorBanner, Loading, EmptyState } from "@/components/ui";

export default function ResearchPage() {
  const { data, error, loading } = useApi(() => api.getScreens(), []);

  return (
    <div>
      <PageHeader label="Members · Research Screens" title="Research" />

      {error ? (
        <ErrorBanner error={error} />
      ) : loading || !data ? (
        <Loading rows={4} />
      ) : data.length === 0 ? (
        <EmptyState message="No research screens available yet" />
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {data.map((screen) => (
            <Link
              key={screen.key}
              href={`/research/${screen.key}`}
              className="group flex flex-col border border-[var(--border)] bg-[var(--surface)] px-5 py-5 transition-colors hover:border-[var(--gold-dim)]"
            >
              <h2 className="serif text-2xl font-semibold text-foreground transition-colors group-hover:text-gold">
                {screen.title}
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-muted">
                {screen.subtitle}
              </p>
              <span className="label mt-4 transition-colors group-hover:text-gold">
                View screen →
              </span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
