"use client";

import { useEffect, useRef, useState } from "react";
import { api, ApiError, ModelInfo } from "@/lib/api";
import { PageHeader, ErrorBanner, ComingSoon, isComingSoon } from "@/components/ui";

interface Message {
  role: "user" | "assistant";
  text: string;
  grounded_on?: string[];
  model?: string;
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
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    api
      .getAssistantModels()
      .then((list) => {
        if (!active) return;
        setModels(list);
        const def = list.find((m) => m.available) ?? list[0];
        if (def) setSelectedModel(def.key);
      })
      .catch(() => {
        // Selector is optional; a failed model fetch falls back to the
        // server's default model (empty selection sends no override).
      });
    return () => {
      active = false;
    };
  }, []);

  const modelLabel = (key?: string): string | null => {
    if (!key) return null;
    return models.find((m) => m.key === key)?.label ?? key;
  };

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
      const res = await api.askAssistant(question, selectedModel || undefined);
      setRemaining(res.remaining_today);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: res.answer,
          grounded_on: res.grounded_on,
          model: res.model,
        },
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
                    {m.role === "assistant" && m.model ? (
                      <p className="mono mt-2 text-[11px] text-muted">
                        via {modelLabel(m.model)}
                      </p>
                    ) : null}
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
            {models.length > 0 ? (
              <label className="flex items-center gap-2 border border-[var(--border-strong)] bg-[var(--surface)] px-3">
                <span className="label">Model</span>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="mono bg-transparent py-3 text-xs text-foreground outline-none"
                >
                  {models.map((m) => (
                    <option
                      key={m.key}
                      value={m.key}
                      disabled={!m.available}
                      className="bg-[var(--surface)] text-foreground"
                    >
                      {m.label}
                      {m.available ? "" : " (no key)"}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
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
