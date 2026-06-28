"use client";

import { useEffect, useState } from "react";

// Live UTC clock + date, updates every second. Renders nothing until mounted
// to avoid a server/client hydration mismatch on the timestamp.
export function Clock() {
  const [now, setNow] = useState<Date | null>(null);

  useEffect(() => {
    setNow(new Date());
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const time = now
    ? now.toISOString().slice(11, 19)
    : "--:--:--";
  const date = now
    ? now.toISOString().slice(0, 10)
    : "----/--/--";

  return (
    <div className="flex items-center gap-3 mono text-xs">
      <span className="hidden text-muted sm:inline" suppressHydrationWarning>
        {date}
      </span>
      <span className="inline-flex items-center gap-2 text-gold" suppressHydrationWarning>
        <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-pos" />
        {time} UTC
      </span>
    </div>
  );
}
