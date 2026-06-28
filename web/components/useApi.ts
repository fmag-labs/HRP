"use client";

import { useCallback, useEffect, useState } from "react";

interface ApiState<T> {
  data: T | null;
  error: unknown;
  loading: boolean;
  reload: () => void;
}

// Client-side fetch hook with loading + error states. Keeps the build free of
// any live-API dependency (data is fetched in the browser, never at build time).
export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: ReadonlyArray<unknown> = []
): ApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fetcher()
      .then((d) => {
        if (active) setData(d);
      })
      .catch((e) => {
        if (active) setError(e);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  return { data, error, loading, reload };
}
