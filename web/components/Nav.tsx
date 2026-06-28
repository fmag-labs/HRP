"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const ITEMS: { href: string; label: string; code: string }[] = [
  { href: "/", label: "Conviction List", code: "01" },
  { href: "/portfolio", label: "My Portfolio", code: "02" },
  { href: "/track-record", label: "Track Record", code: "03" },
  { href: "/research", label: "Research", code: "04" },
  { href: "/assistant", label: "Vault Assistant", code: "05" },
  { href: "/settings", label: "Settings", code: "06" },
];

function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/" || pathname.startsWith("/recommendations");
  return pathname === href || pathname.startsWith(href + "/");
}

export function Nav({ horizontal = false }: { horizontal?: boolean }) {
  const pathname = usePathname();

  return (
    <nav
      className={
        horizontal
          ? "flex flex-wrap gap-2"
          : "sticky top-[57px] flex flex-col gap-0.5 p-3"
      }
    >
      {ITEMS.map((item) => {
        const active = isActive(pathname, item.href);
        return (
          <Link
            key={item.href}
            href={item.href}
            className={[
              "group flex items-center gap-3 border px-3 py-2 transition-colors",
              horizontal ? "text-xs" : "",
              active
                ? "border-[var(--gold-dim)] bg-[var(--surface)] text-gold"
                : "border-transparent text-muted hover:border-[var(--border)] hover:bg-[var(--surface)] hover:text-foreground",
            ].join(" ")}
          >
            <span
              className={[
                "mono text-[10px]",
                active ? "text-gold" : "text-[var(--border-strong)] group-hover:text-muted",
              ].join(" ")}
            >
              {item.code}
            </span>
            <span className="mono text-[11px] uppercase tracking-[0.14em]">
              {item.label}
            </span>
          </Link>
        );
      })}
    </nav>
  );
}
