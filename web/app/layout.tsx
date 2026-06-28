import type { Metadata } from "next";
import { Playfair_Display, IBM_Plex_Mono, Inter } from "next/font/google";
import "./globals.css";
import { Clock } from "@/components/Clock";
import { Nav } from "@/components/Nav";

const serif = Playfair_Display({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-serif",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
});

const sans = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "HRP · Members Research Terminal",
  description: "Private members research terminal. Not investment advice.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${serif.variable} ${mono.variable} ${sans.variable}`}>
      <body className="min-h-screen bg-background text-foreground antialiased">
        <div className="flex min-h-screen flex-col">
          {/* Top bar */}
          <header className="sticky top-0 z-30 flex items-center justify-between border-b border-[var(--border)] bg-[rgba(10,10,10,0.95)] px-6 py-3 backdrop-blur">
            <div className="flex items-baseline gap-3">
              <span className="serif text-2xl font-bold tracking-tight text-gold">HRP</span>
              <span className="label hidden sm:inline">Members · Research Terminal</span>
            </div>
            <Clock />
          </header>

          <div className="flex flex-1">
            {/* Left nav */}
            <aside className="hidden w-56 shrink-0 border-r border-[var(--border)] md:block">
              <Nav />
            </aside>

            {/* Main */}
            <main className="min-w-0 flex-1 px-5 py-6 sm:px-8 sm:py-8">
              {/* Mobile nav */}
              <div className="mb-6 md:hidden">
                <Nav horizontal />
              </div>
              {children}
            </main>
          </div>

          {/* Footer disclaimer */}
          <footer className="border-t border-[var(--border)] px-6 py-3">
            <span className="mono text-xs tracking-widest text-gold-dim">
              // NOT INVESTMENT ADVICE
            </span>
          </footer>
        </div>
      </body>
    </html>
  );
}
