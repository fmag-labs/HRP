"use client";

import { useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { PageHeader, ErrorBanner, ComingSoon, isComingSoon } from "@/components/ui";

interface Message {
  role: "user" | "assistant";
  text: string;
  grounded_on?: string[];
}

export default function AssistantPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [comingSoon, setComingSoon] = useState(false);
  const [notConfigured, setNotConfigured] = useState(false);
  const [limitReached, setLimitReached] = useState(false);
  const [remaining, setRemaining] = useState<number | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || busy) return;

    setMessages((m) => [...m, { role: "user", text: question }]);
    setInput("");
    setBusy(true);
    setError(null);
    setLimitReached(false);

    try {
      const res = await api.askAssistant(question);
      setRemaining(res.remaining_today);
      setMessages((m) => [
        ...m,
        { role: "assistant", text: res.answer, grounded_on: res.grounded_on },
      ]);
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) {
        setNotConfigured(true);
      } else if (err instanceof ApiError && err.status === 429) {
        setRemaining(0);
        setLimitReached(true);
      } else if (isComingSoon(err)) {
        setComingSoon(true);
      } else {
        setError(err);
      }
    } finally {
      setBusy(false);
      requestAnimationFrame(() => {
        listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
      });
    }
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader label="Members · Private Research Desk" title="Vault Assistant">
        {remaining !== null ? (
          <span className="mono text-xs text-muted">{remaining} queries left today</span>
        ) : null}
      </PageHeader>

      {comingSoon ? (
        <ComingSoon title="The Vault Assistant is almost ready" />
      ) : notConfigured ? (
        <ComingSoon title="Assistant is not configured yet" />
      ) : (
        <div className="flex flex-1 flex-col">
          <div
            ref={listRef}
            className="mb-4 min-h-[320px] flex-1 space-y-4 overflow-y-auto border border-[var(--border)] bg-[var(--surface)] p-5"
          >
            {messages.length === 0 ? (
              <p className="serif text-lg text-muted">
                Ask the desk about a ticker, the conviction list, or the research process.
                Answers are informational only.
              </p>
            ) : (
              messages.map((m, i) => (
                <div
                  key={i}
                  className={m.role === "user" ? "flex justify-end" : "flex justify-start"}
                >
                  <div
                    className={[
                      "max-w-[80%] px-4 py-3 text-sm leading-relaxed",
                      m.role === "user"
                        ? "border border-[var(--gold-dim)] bg-[rgba(201,162,75,0.08)] text-foreground"
                        : "border border-[var(--border)] bg-[var(--surface-2)] text-foreground",
                    ].join(" ")}
                  >
                    <div className="label mb-1">
                      {m.role === "user" ? "You" : "Desk"}
                    </div>
                    <p className="whitespace-pre-wrap">{m.text}</p>
                    {m.grounded_on && m.grounded_on.length > 0 ? (
                      <p className="mono mt-2 text-[11px] text-muted">
                        grounded on: {m.grounded_on.join(", ")}
                      </p>
                    ) : null}
                  </div>
                </div>
              ))
            )}
            {busy ? (
              <div className="label animate-pulse">Desk is thinking…</div>
            ) : null}
          </div>

          {limitReached ? (
            <div className="mb-4 border border-[var(--gold-dim)] bg-[rgba(201,162,75,0.08)] px-4 py-3">
              <p className="mono text-xs uppercase tracking-widest text-gold">
                Daily question limit reached
              </p>
              <p className="mt-1 text-sm text-muted">
                You have used all of today&apos;s questions. Check back tomorrow.
              </p>
            </div>
          ) : null}

          {error ? (
            <div className="mb-4">
              <ErrorBanner error={error} />
            </div>
          ) : null}

          <form onSubmit={send} className="flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask the research desk…"
              className="flex-1 border border-[var(--border-strong)] bg-[var(--surface)] px-4 py-3 text-sm text-foreground outline-none placeholder:text-muted focus:border-gold"
            />
            <button
              type="submit"
              disabled={busy || !input.trim() || limitReached}
              className="mono border border-gold px-5 py-3 text-xs font-semibold uppercase tracking-widest text-gold transition-colors hover:bg-[rgba(201,162,75,0.1)] disabled:opacity-40"
            >
              Send
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
