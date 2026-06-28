# HRP — Members Research Terminal

A consumer front-end for the HRP advisory platform. Dark, editorial, gold-on-black —
a private members research terminal. Built with Next.js (App Router), TypeScript, and
Tailwind CSS. All data comes from the frozen HRP advisory HTTP/JSON API.

> **// NOT INVESTMENT ADVICE**

## Quick start

```bash
cd web
npm install
cp .env.example .env.local   # then edit if your API is not on localhost:8090
npm run dev                  # http://localhost:3000
```

Production build / preview:

```bash
npm run build
npm run start
```

## Environment variables

| Variable                 | Default                  | Purpose                                                        |
| ------------------------ | ------------------------ | ------------------------------------------------------------- |
| `NEXT_PUBLIC_API_BASE`   | `http://localhost:8090`  | Base URL of the HRP advisory API.                             |
| `NEXT_PUBLIC_API_TOKEN`  | _(unset)_                | If set, sent as `Authorization: Bearer <token>` on requests.  |

Both are `NEXT_PUBLIC_*` because the app fetches client-side. Data is fetched in the
browser at runtime, so the build never depends on a live API.

## Views

| Route                    | View             | Source endpoint(s)                                       |
| ------------------------ | ---------------- | -------------------------------------------------------- |
| `/`                      | Conviction List  | `GET /api/recommendations`                               |
| `/recommendations/[id]`  | Detail / Dossier | `GET/POST /api/recommendations/{id}[/approve\|reject]`   |
| `/portfolio`             | My Portfolio     | `GET /api/portfolio`                                     |
| `/track-record`          | Track Record     | `GET /api/track-record`                                  |
| `/assistant`             | Vault Assistant  | `POST /api/assistant/query`                              |
| `/settings`              | Settings         | `GET/PUT /api/settings`                                  |

The Vault Assistant and Settings views tolerate `404`/`501` gracefully with a
"coming soon" state, since those endpoints are being built in parallel.

## Architecture

- `lib/api.ts` — typed API client (frozen contract), bearer-token aware, throws
  `ApiError` with a status code so views can distinguish failures from "coming soon".
- `components/useApi.ts` — client-side fetch hook with loading + error states.
- `components/ui.tsx` — shared terminal primitives (badges, bars, formatters, banners).
- `components/Clock.tsx` — live UTC clock (updates every second).
- `components/Nav.tsx` — left navigation with active-route highlighting.
- `app/*` — App Router pages, all client components so the build has no API dependency.

Fonts via `next/font`: Playfair Display (serif headings), IBM Plex Mono (numbers /
tickers / labels), Inter (body).
