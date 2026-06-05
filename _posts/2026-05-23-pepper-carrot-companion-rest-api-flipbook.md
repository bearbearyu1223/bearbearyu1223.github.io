---
title: "Pepper & Carrot AI-powered flipbook · Part 7 of 16 — From Database to JSON: A Typed REST API"
date: 2026-05-23 00:00:00 -0800
categories: [Full-Stack, RAG, Local AI]
tags: [fastapi, pydantic, openapi, rest, sqlalchemy, cors, peppercarrot, portfolio]
description: >-
  Post 7 of the Pepper & Carrot AI flipbook series. With one episode
  sitting in Postgres + LocalStorage from Post 5, it's time to surface
  it. Build two typed FastAPI routes — list episodes and episode
  detail — that resolve relative storage keys into absolute URLs at
  response time via the Storage Protocol, with the OpenAPI spec as the
  wire-format contract. By the end you have two endpoints returning
  episode JSON your browser can consume.
pin: true
---

Post 7 of the [*Pepper & Carrot AI-powered flipbook*]({% post_url 2026-05-09-pepper-carrot-companion-trailer %}) series. [Post 5]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}) put one full episode into Postgres + ChromaDB + `LocalStorage`. It's sitting there, fully described, queryable from `psql`, three image variants per page on disk — and nothing speaks JSON over HTTP yet. This post builds the seam that does. We design two REST endpoints, ship the route handlers, and make the OpenAPI spec the written-down contract between the database and whatever consumes it. The most load-bearing piece: every `pages.image_url` relative key turns into an absolute URL *at response time*, through the `Storage` Protocol from [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) — which is what makes the local→R2 storage swap a config change instead of a migration. The browser half — a real page-flipping flipbook that renders this JSON — is [Post 8]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}).

> **What you'll build in this post.**
> - Two FastAPI route handlers in `backend/app/api/episodes.py`:
>   - `GET /api/episodes` — list episodes with cover image, plot summary, and page count.
>   - `GET /api/episodes/{slug}` — full episode detail, including every page with its display image and metadata.
> - Pydantic v2 response models (`EpisodeListItem`, `EpisodeListResponse`, `EpisodeDetail`, `PageDetail`) that resolve `pages.image_url` relative keys into absolute URLs at response time via the `Storage` Protocol from [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}).
> - The OpenAPI spec FastAPI publishes for free at `/openapi.json` (and renders at `/docs`) as the machine-readable wire-format contract.
> - The CORS middleware that lets a browser on a different origin call the API.
> - Three smoke tests that prove `GET /api/episodes/{slug}` returns absolute image URLs and 404s on unknown slugs.
>
> **Prerequisites.**
> - The workshop starter at [the state Post 5 left it in](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop): Postgres up (`docker compose ps` shows healthy), `alembic upgrade head` applied, `seed.py` run, and Episode 1 ingested by the `ingest-from-images` skill. Verify with `docker exec peppercarrot-postgres psql -U peppercarrot -d peppercarrot -c "SELECT COUNT(*) FROM pages;"` — three rows means you're ready.
> - [Node.js 20+](https://nodejs.org/) installed (we set this up in [Post 2]({% post_url 2026-05-10-pepper-carrot-companion-workshop %}) step 1) if you also plan to follow [Post 8]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %})'s frontend. Confirm with `node --version`.
> - No new external services. Everything in this post runs against the same local stack you already have.

> **About the repo URL.** The backend additions (`backend/app/api/episodes.py`, the response models, the smoke tests) live in the same workshop starter that backed [Posts 2–5](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop). Pull the latest to pick up the full-stack additions. The full project repository — chat orchestrator, world-graph overlay, cloud deploy — still goes up alongside the deploy guide near the end of the series.
>
> **Checking out the code.** This post and [Post 8]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}) (the flipbook UI) share one checkpoint: `git checkout post-07-08-fullstack` gives you both the typed REST API in this post and the React reader in the next. Each later post adds its own tag (`post-09-rag`, `post-10-streaming`, …); see the README's [Following along with the blog series](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop#following-along-with-the-blog-series).

---

## Table of Contents

1. [The Code in Front of You: Tour + Quick Start](#tour)
2. [Two Halves, One Wire Format](#two-halves)
3. [What a REST API Actually Is](#rest-api)
4. [Designing the API Surface Backwards From the UI](#api-design)
5. [Pydantic v2 in Two Minutes](#pydantic-aside)
6. [Where the Relative Key Becomes an Absolute URL](#url-composition)
7. [The Two Route Handlers, End to End](#route-handlers)

---

## The Code in Front of You: Tour + Quick Start {#tour}

Before any concepts, let's get the running app in front of you and orient around the files this post adds. Skim this section even if you plan to read the rest carefully — the rest is easier to follow if you've seen the app render once.

### What's new in the workshop starter

Two directories are new compared to where [Post 5]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}) left things. Everything else carries forward unchanged.

```
pepper-carrot-companion-workshop/
├── backend/
│   └── app/
│       ├── main.py            ← updated: lifespan + /api/episodes mounted
│       └── api/               ← NEW (Post 7)
│           ├── __init__.py
│           └── episodes.py    ← two route handlers + Pydantic response models
│   └── tests/
│       └── test_episodes_api.py  ← NEW: 3 hermetic smoke tests
│
└── frontend/                  ← NEW (Post 8) — the entire React app
    ├── package.json           ←   react, react-dom, page-flip, vite, typescript
    ├── vite.config.ts         ←   dev proxy for /api and /images
    ├── tsconfig.json
    ├── index.html
    └── src/
        ├── main.tsx           ←   4-line bootstrap
        ├── App.tsx            ←   picker ↔ reader view-switch
        ├── api/
        │   ├── client.ts      ←   listEpisodes + getEpisode (plain fetch)
        │   └── types.ts       ←   TS mirror of the Pydantic models
        ├── components/
        │   ├── EpisodePicker.tsx
        │   └── Flipbook.tsx   ←   StPageFlip wrapped via a ref
        └── styles/global.css
```

That's the whole surface. **Two route handlers on the backend, six small files on the frontend, one wire format between them.** The rest of this post unpacks each of those pieces — why it's shaped the way it is, and what the alternatives were.

### Three terminals, three commands

You need three things running at once: Postgres (already up from Post 2), the FastAPI backend, and the Vite dev server. Each owns one terminal.

```bash
# Terminal 1 — Postgres (carried over from Post 2; skip if it's already up)
cd path/to/pepper-carrot-companion-workshop
docker compose up -d
docker compose ps                          # postgres should show (healthy)

# Terminal 2 — FastAPI backend on :8000
cd backend
uv sync                                    # picks up any new deps
uv run uvicorn app.main:app --reload
#   INFO:     Uvicorn running on http://127.0.0.1:8000

# Terminal 3 — Vite dev server on :5173
cd frontend
npm install                                # ~30 s the first time
npm run dev
#   ➜  Local:   http://localhost:5173/
```

Open <http://localhost:5173/> in a browser. If Episode 1 was ingested in [Post 5]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}), you should see one episode card; click it and a real page-flipping flipbook renders. Drag a page corner to flip; resize the window to portrait and the spread collapses to a single page.

Optional sanity check from a fourth terminal — the backend handlers in isolation:

```bash
curl -s http://localhost:8000/api/episodes | jq '.episodes[0]'
# { "id": "...", "slug": "ep01-potion-of-flight",
#   "title": "Potion of Flight", "episode_number": 1,
#   "cover_image_url": "http://localhost:8000/images/...",
#   "page_count": 3, "plot_summary": "..." }

curl -s http://localhost:8000/api/episodes/ep01-potion-of-flight | jq '.pages | length'
# 3
```

Or open <http://localhost:8000/docs> and click "Try it out" on either endpoint — FastAPI's Swagger UI renders the same JSON in a clickable form, generated automatically from the Pydantic models.

### What fires when the page loads

Open <http://localhost:5173/>, open the browser DevTools → **Network** tab, hit refresh. Two kinds of requests fire on first load, in order:

1. **One JSON call to FastAPI** — `GET /api/episodes`. Triggered by a `useEffect` in `EpisodePicker.tsx` that calls `api.listEpisodes()` when the component mounts. In dev this goes to Vite at `:5173`, which proxies it to `http://localhost:8000/api/episodes`; in prod it goes cross-origin and CORS lets it through. Status `200`, response body is the `{ "episodes": [...] }` payload from `list_episodes()` in `backend/app/api/episodes.py`.
2. **One image call per cover** — `GET /images/episodes/.../cover.webp`, repeated once per episode that came back with a non-null `cover_image_url`. These are fired automatically by the *browser*, not by your JavaScript — it sees `<img src="...">` in the DOM and goes to fetch the bytes. They land on FastAPI's `/images/*` StaticFiles mount; no Python handler runs, the bytes stream straight off disk.

Filter the Network tab by **Fetch/XHR** for the first kind, **Img** for the second. The "Initiator" column shows you why each request fired — `episode-picker.tsx:14` (a `useEffect`) for the JSON call, `img` (the browser, on behalf of the DOM) for the image calls.

Two gotchas worth knowing on day one:

- **React StrictMode double-fires effects in dev.** `main.tsx` wraps `<App />` in `<React.StrictMode>`, which intentionally re-invokes every effect once to surface buggy side effects. You'll see **two `GET /api/episodes` rows** in DevTools, both `200`. The second response is discarded by React; this is dev-only and goes away in `npm run build`. ([React docs on StrictMode](https://react.dev/reference/react/StrictMode#fixing-bugs-found-by-double-rendering-in-development).)
- **No `GET /api/episodes/{slug}` yet.** The detail endpoint only fires *after* you click an episode card — a separate `useEffect` inside `Flipbook.tsx` watching `episode.slug`. This is deliberate: the detail response is 5–20× larger than the list, and there's no reason to fetch every episode's pages before the reader picks one. **Network volume tracks user intent**, not arrival on the page.

(If `cover_image_url` came back `null` — which it currently does in the workshop starter, because the `peppercarrot.com` `metadata.yaml` we ingest doesn't carry a cover URL — there are no `/images/.../cover.webp` rows, and the card renders the parchment-gradient `.episode-cover-placeholder` instead. That's correctness, not a bug: missing data → fallback UI.)

### Closing the orientation

If all four of those produce output, **every concept in this post is now live in front of you.** From here on, when the text references a file like `backend/app/api/episodes.py` or `frontend/src/components/Flipbook.tsx`, you have it open in your editor and running in your browser.

---

## Two Halves, One Wire Format {#two-halves}

This post has two halves — a Python half and a TypeScript half — and they connect through exactly one thing: a [REST API](https://en.wikipedia.org/wiki/REST) speaking JSON.

The temptation, especially for a portfolio project, is to start with the half you find more fun and figure out the contract along the way. **Don't.** When the contract is an afterthought, the two halves end up subtly misaligned — the backend returns `episode_number`, the frontend expects `number`, and you discover the gap when something silently renders blank in production.

The fix is to make the contract a real artifact, written down, with both sides agreeing on it. FastAPI's automatically-generated [OpenAPI spec](https://www.openapis.org/) — published at `/openapi.json` and rendered interactively at `/docs` — *is* that artifact. The backend's Pydantic response models are the source of truth; the spec is generated from them; the frontend's TypeScript interfaces mirror the same shapes. We'll see in [§ The Frontend's Type Contract]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}#type-contract) in [Post 8]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}) why the frontend keeps the mirror in sync **by hand** at two endpoints and where the line is for graduating to a generator.

Everything else in this post is a consequence of that ordering: design the wire format first, then implement each side against it. Before we get there, the next section unpacks what "REST API" actually means in concrete terms — feel free to skip it if you've built web services before.

---

## What a REST API Actually Is {#rest-api}

If you've never built one, "REST API" can feel like a phrase people use without explaining. The good news: there is very little hidden depth. **It's a convention for letting two programs talk to each other over HTTP, where URLs name *things* and HTTP methods name *what you want to do with them*.** Most of the rest is mechanics. This section lays out the mechanics in concrete terms, using the two endpoints we're about to build as the running example.

### The three pieces of every REST call

Every HTTP request — and every REST call is just an HTTP request — has three things on the way out, and a fourth on the way back:

1. **A method.** One of `GET`, `POST`, `PUT`, `PATCH`, `DELETE` (and a few rarer ones). The method is your *verb* — it says what kind of operation you intend.
2. **A URL.** A path like `/api/episodes/ep01-potion-of-flight`. The URL is your *noun* — it identifies the thing you're operating on.
3. **An optional body.** JSON, usually. Present for `POST` / `PUT` / `PATCH` (you're sending something); absent for `GET` / `DELETE` (the URL alone identifies what you want).

Then the server sends back:

4. **A status code + a body.** A three-digit number (`200`, `404`, `422`, …) saying how it went, plus an optional JSON payload.

That's all. Every "API call" anywhere in this post — and the ten thousand calls a real web app makes per session — fits this shape.

### HTTP methods, in plain English

Five methods cover essentially all of REST. The convention is so consistent across the industry that the meaning of each is roughly the same in every framework, language, and tutorial you'll ever see.

| Method | Means | This post uses it? |
|---|---|---|
| `GET` | Read. Fetch a resource. Doesn't change anything on the server. | ✅ both endpoints |
| `POST` | Create. Make a new resource. | Not yet — Post 9 uses it for chat sessions |
| `PUT` | Replace. Overwrite a resource with a new version. | Not in this project |
| `PATCH` | Update. Modify part of a resource. | Not yet — Post 9 uses it for "user moved to page N" |
| `DELETE` | Remove. Delete a resource. | Not in this project (read-only demo) |

Two things to internalize:

**`GET` requests must be safe.** Calling `GET /api/episodes` ten times in a row should produce the same answer ten times and change nothing on the server. This is more than etiquette — browsers, [CDNs](https://en.wikipedia.org/wiki/Content_delivery_network), and tools like `curl` all assume `GET` is safe and will happily retry or cache it. If your `GET` handler secretly increments a counter or sends an email, you'll get phantom emails the first time a search engine indexes your site.

**Method + URL is the API.** When somebody says *"the episodes endpoint,"* they usually mean *"`GET /api/episodes`"* — a specific (method, URL) pair, not just a URL. The same URL can have a `GET` (read) and a `DELETE` (delete) handler that do completely different things. FastAPI's `@router.get("")` vs `@router.delete("/{id}")` decorators is how you declare which is which.

### URLs as nouns: collections and items

The convention is **collection / item**:

- `/api/episodes` is *the collection of episodes* — plural, an index. `GET` on it returns a list.
- `/api/episodes/{slug}` is *one specific episode* — singular, an item. `GET` on it returns just that one.

You can stack the pattern: `/api/episodes/{slug}/pages` would be "the collection of pages within one episode," `/api/episodes/{slug}/pages/{page_number}` would be one specific page. We could have built it that way — but Post 7 returns *all* pages of an episode inside the episode detail response (the flipbook needs them all at once anyway), so the nested URLs aren't worth the extra route handlers. **Pick the URL shape that matches how the client actually loads data**, not the shape that maximizes RESTful purity.

A few other URL conventions worth knowing on first contact:

- **Query parameters** (`?page=4&mode=focus`) are for filtering, paging, and option flags — modifiers on a `GET`, not the identity of the thing you're fetching.
- **Path parameters** (`/episodes/{slug}`) are the identity. `{slug}` here is just FastAPI's syntax for "this segment is a variable; capture it and pass it to the handler."
- **The `/api/` prefix** isn't required by REST — it's just a convention that keeps API URLs separate from the URLs the frontend itself serves (e.g., `/read/ep01-...`). Helps with CORS configuration and CDN routing later.

### Status codes: 200, 404, 422, 500

The response code is the first thing the client reads. The five-hundreds and four-hundreds are how the server tells the client what happened without making the client parse the body. Six codes cover 95% of cases:

| Code | Meaning | When this post returns it |
|---|---|---|
| `200 OK` | Success, here's the response body. | Both endpoints on the happy path. |
| `404 Not Found` | The thing you asked for doesn't exist. | `GET /api/episodes/does-not-exist`. |
| `422 Unprocessable Entity` | Your request was syntactically valid but semantically wrong (missing field, wrong type). | FastAPI auto-returns this when Pydantic validation fails on a request body. |
| `500 Internal Server Error` | The server crashed. Bug to fix. | When your code raises an unhandled exception. |
| `401 Unauthorized` | **You're not authenticated** — the server doesn't know who you are. Send credentials and try again. | Not in this project — no auth. |
| `403 Forbidden` | The server knows who you are, but you're not allowed to do this. | Not in this project — no auth. |

> *Worth flagging: `401 Unauthorized` is misleadingly named.* Despite the word "unauthorized," the code semantically means **un-authenticated** — i.e., *"I don't know who you are, please identify yourself."* The "you're identified but not allowed" case is `403 Forbidden`. The misnaming is baked into [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110#name-401-unauthorized) and isn't going to change, but every full-stack developer trips over it eventually, so it's worth knowing the actual semantics. Authentication = *who are you?*; authorization = *what can you do?*; `401` is the first, `403` is the second.

The mental model: **status code first, body second.** A client that ignores the status code and tries to parse `{"detail": "Episode '...' not found"}` as if it were an `EpisodeDetail` will explode on the missing fields. The two-line `if (!res.ok) throw ...` guard in `frontend/src/api/client.ts` is exactly this discipline written in TypeScript.

### Why JSON

JSON is the wire format because both sides parse it natively. Python's `json` module reads it into dicts; JavaScript reads it directly with `JSON.parse` (a browser built-in). Pydantic adds *typed* parsing on top, refusing dicts that don't match its schema. The alternatives — XML, [Protocol Buffers](https://protobuf.dev/), [MessagePack](https://msgpack.org/) — are sometimes the right call (protobuf for high-throughput RPC, MessagePack when bandwidth dominates), but at the scale of a portfolio web app, the cost of JSON's redundant text is invisible and the readability win is real. **You can `curl` any of this post's endpoints, pipe to `jq`, and see exactly what the frontend is seeing.** That property is worth a lot when you're debugging.

### Cross-origin requests: the same-origin policy and CORS

There's one piece of web mechanics that catches every full-stack developer at least once, and it's worth understanding *before* we get to the frontend half: **browsers don't let one origin freely call another origin's API.** This post's frontend at `http://localhost:5173` and backend at `http://localhost:8000` are two different origins, and that single fact shapes a non-trivial amount of the wire setup.

> *Plain-English aside: what's an "origin"?* An **origin** is the tuple **(scheme, host, port)** of a URL — for example `(http, localhost, 5173)`. Two URLs share an origin only if all three match exactly. So `http://localhost:5173` and `http://localhost:8000` are *different* origins (different port), and `http://example.com` and `https://example.com` are *different* origins (different scheme).

#### The same-origin policy

The browser enforces a security rule called the [**same-origin policy**](https://developer.mozilla.org/en-US/docs/Web/Security/Same-origin_policy): by default, JavaScript running on a page from one origin **cannot read responses from a different origin**. Without this rule, any random tab in your browser could fire off `fetch('https://yourbank.example/api/transfer', ...)` and silently move your money — your bank's cookies would be sent along automatically. The same-origin policy is one of the load-bearing security boundaries of the web.

The catch: legitimate cross-origin calls are *everywhere*. A static frontend on `https://app.example.com` calling an API at `https://api.example.com`. A page embedding fonts from Google's CDN. A SPA at `http://localhost:5173` calling its dev backend at `http://localhost:8000`. The web would be useless if every cross-origin call were forbidden, so there has to be an opt-in.

#### CORS is the opt-in

**CORS** (**Cross-Origin Resource Sharing**, [MDN reference](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)) is that opt-in. It's a small protocol where the **server** being called explicitly tells the browser *"yes, requests from this list of other origins are allowed."* The browser checks the server's response headers (`Access-Control-Allow-Origin`, etc.) before delivering the response to the JavaScript that made the call; if the server didn't include the right headers for this origin, the browser blocks the response and the `fetch()` promise rejects.

Two non-obvious properties of CORS:

- **CORS is browser-enforced, not server-enforced.** The server receives and processes the request just fine — it's the browser that refuses to hand the response to JavaScript. That's why `curl http://localhost:8000/api/episodes` from your terminal works even if CORS isn't configured: `curl` doesn't enforce CORS. Browsers do.
- **For some requests, the browser asks first.** Simple `GET` / `HEAD` / `POST` requests (with allowlisted content types) go straight through. Anything fancier — `PATCH`, `DELETE`, custom headers like `Authorization`, JSON request bodies — triggers a **preflight**: an `OPTIONS` request the browser sends *before* the real one to ask *"can I make this request?"* The server answers with what's allowed; if it matches, the real request goes. You can see this in Chrome DevTools → Network tab: the `OPTIONS` row shows up just before any `PATCH` / `POST` that needs it. Browsers cache the preflight for a few minutes so it's not per-request.

#### What we configure in this project

FastAPI ships a CORS middleware. Look at `backend/app/main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,   # ["http://localhost:5173"] by default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

What is `settings.cors_origins`? Step by step:

1. It's a Python list of origin strings — the **allowlist** of websites the browser is permitted to make API calls from.
2. Its value comes from your `.env` file. Put a line like `CORS_ORIGINS=["http://localhost:5173"]` there, and pydantic-settings (set up in [Post 2]({% post_url 2026-05-10-pepper-carrot-companion-workshop %})) loads it into `Settings.cors_origins` on startup.
3. If `.env` doesn't set it, the default declared in `backend/app/config.py` kicks in — and that default is `["http://localhost:5173"]`, the URL where Vite serves the frontend in dev.

When a request comes in, `CORSMiddleware` looks at the request's `Origin` header (the browser sets this automatically to whichever site the page came from) and checks whether that origin is in the list. **If yes, FastAPI adds an `Access-Control-Allow-Origin: <that origin>` header to the response — the green light the browser is waiting for.** If no, the header is absent, and the browser blocks the JavaScript from reading the response.

So in one sentence: **the list above is the set of websites your API will *talk to*; everything else is silently muted by the browser.** In production you swap `http://localhost:5173` for your real frontend's URL (e.g. `https://flipbook.example.com`) by editing `.env` — no code changes, no redeploy of any Python.

*If you're wondering how `CORS_ORIGINS=...` in `.env` actually ends up as a Python list on `Settings` — the pydantic-settings machinery that does the loading is walked through in [§ Appendix: How `Settings` Reads Your `.env`]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}#appendix-settings) at the end of [Post 8]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}).*

But there's also a second mechanism in play, which the [Frontend Stack section]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}#frontend-stack) in [Post 8]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}) covers in code: **Vite's dev-server proxy.** With `proxy: { '/api': 'http://localhost:8000' }` in `vite.config.ts`, the browser's `fetch('/api/episodes')` actually hits Vite at `:5173` first — same origin, no CORS check — and Vite then makes a *server-to-server* request to the FastAPI backend and returns the result. The browser sees a same-origin response and never even consults the CORS headers.

So why both?

- **In dev**, the proxy keeps the network tab clean (no preflight `OPTIONS` rows cluttering it) and dodges the rare case where misconfigured CORS would block a fetch you're trying to debug.
- **In production**, the dev proxy isn't there — the frontend is served as static files from a CDN (Cloudflare Pages in the eventual cloud deploy). The backend at a different origin is the *only* way the API gets called, and the CORS middleware is what makes that work.

Belt-and-suspenders. If you see a `"blocked by CORS policy"` error in your browser's console down the road, you now know what file to look in (`backend/app/main.py`, `cors_origins`) and what to check (does the origin in the error message match what's allowed?).

### REST is a convention, not a standard

One important thing for anyone digging into the history: REST was [described in a PhD dissertation](https://www.ics.uci.edu/~fielding/pubs/dissertation/top.htm) by Roy Fielding in 2000, and what people call "REST" today is much looser than what the dissertation defined. Strict REST requires things like [HATEOAS](https://en.wikipedia.org/wiki/HATEOAS) (responses contain links to related actions) that almost no production API actually implements. The pragmatic shape most of the industry has settled on — sometimes called **RESTish** or **REST-style** — is what you've just read: HTTP methods + collection/item URLs + JSON + status codes. **That's what this post's API is, and what 95% of "REST APIs" you'll encounter are.** Don't let a purist tell you otherwise.

### The contract idea

The reason REST matters for *this* post is that the two halves of the app — Python backend, TypeScript frontend — can be developed and reasoned about independently *because* both sides agree on the same (method, URL, request shape, response shape) tuples. Neither side imports the other's code; neither side knows what framework the other uses. **The contract is the only shared surface.** That's why we spend the rest of the post making the contract a real, written-down, machine-readable artifact (the OpenAPI spec) instead of an oral tradition.

### Seeing the contract: three URLs every FastAPI app exposes for free

The convenient part: FastAPI generates that contract automatically and serves three views of it on every running app, with no config at all. Boot the backend (`uv run uvicorn app.main:app --reload`) and open these in a browser:

| URL | What you see |
|---|---|
| <http://localhost:8000/docs> | **Swagger UI** — interactive. Click any endpoint, expand "Try it out", hit "Execute", see the live response inline. |
| <http://localhost:8000/redoc> | **ReDoc** — same content, different layout. Three-column reference-style. Read-only, no "try it" button. |
| <http://localhost:8000/openapi.json> | **The raw OpenAPI spec** as JSON. This is the machine-readable file that `openapi-typescript`, [Postman](https://www.postman.com/), [Stoplight](https://stoplight.io/), and the rest of the OpenAPI tool ecosystem consume. |

You can `curl` the JSON one to see what's actually in it:

```bash
# All registered routes
curl -s http://localhost:8000/openapi.json | jq '.paths | keys'
# [
#   "/api/episodes",
#   "/api/episodes/{slug}",
#   "/health"
# ]

# All response/request models
curl -s http://localhost:8000/openapi.json | jq '.components.schemas | keys'
# ["CharacterSummary", "EpisodeDetail", "EpisodeListItem",
#  "EpisodeListResponse", "HTTPValidationError",
#  "PageCharacterRef", "PageDetail", "ValidationError"]
```

**Every Pydantic model in `backend/app/api/episodes.py` shows up under `components.schemas`; every `@router.get(...)` shows up under `paths`.** FastAPI rebuilds the file from your route handlers on each request — there is no separate generation step, no source-of-truth drift between the code and the spec. **The spec can't get out of sync with the routes, because it *is* the routes, serialized.** That property is what lets us treat it as the contract instead of as documentation that someone has to remember to update.

A few small details worth knowing:

- **You don't have to enable any of this.** It's on by default. To turn it off (e.g., for a hardened production deployment), pass `docs_url=None`, `redoc_url=None`, `openapi_url=None` to the `FastAPI(...)` constructor. Most projects leave docs/redoc on in dev and auth-gate or disable them in prod.
- **Which view to use when:** `/docs` for daily dev — it's the one you'll have open in a side tab. `/redoc` for sharing with reviewers or printing a PDF. `/openapi.json` for feeding into tools.
- **The Pydantic field constraints become spec constraints.** `episode_number: int = Field(ge=1)` in Python turns into `"minimum": 1` in the JSON spec, which Swagger UI renders as a hint and which generated TypeScript types could (in principle) enforce. The contract isn't just *names and types* — it carries semantic constraints too.

The [FastAPI features overview](https://fastapi.tiangolo.com/features/#automatic-docs) has the canonical write-up. **Open `/docs` once before reading the next section** — it'll make the "design the API surface" conversation easier when you can see the interactive form for the endpoints we're about to build.

---

## Designing the API Surface Backwards From the UI {#api-design}

Before writing route handlers, it's worth asking *what shapes does the UI actually need?* Designing the API surface forward from the database — *"well, we have an `episodes` table, so I guess `GET /api/episodes` returns episode rows"* — almost always produces too many round-trips. Designing it backwards from screens produces fewer, fatter responses that match how the UI loads.

There are exactly two screens in the read-only flipbook MVP:

1. **Episode picker.** A grid of episode cards: cover image, title, episode number, the plot summary written at ingestion time, and a page count. One request when the user lands on the page.
2. **Reader.** The flipbook, which needs *every page of one episode* — page number, display image URL, the per-page metadata (width, height, blurhash, dominant color) used for loading placeholders, and the per-page character list — all at once, because StPageFlip wants the full page list upfront. One request when the user clicks an episode.

That's it. No chat session yet (Post 9), no chat panel yet (Post 10), no world graph yet (Post 12). Two screens → two endpoints. Every additional endpoint added in this post would be one I'd have to maintain and document without anything calling it.

The two endpoints fall out of the two screens:

| Endpoint | Returns | Used by |
|---|---|---|
| `GET /api/episodes` | `EpisodeListResponse` — `{ episodes: EpisodeListItem[] }`, each with id, slug, title, episode number, cover URL, page count, plot summary | Episode picker |
| `GET /api/episodes/{slug}` | One `EpisodeDetail` — same fields plus credits URL, an episode-level character roster, and every page in reading order | Reader / flipbook |

The slug is what the frontend uses as the lookup key — it's the natural noun for the URL once we add deep linking later. Using the UUID `id` instead would mean the frontend has to do two requests just to render a bookmarkable URL.

Here are the Pydantic v2 response models. They live alongside the routes in `backend/app/api/episodes.py` — the project keeps response shapes adjacent to the handlers that produce them, since they're always edited together. Excerpt:

```python
# backend/app/api/episodes.py
from uuid import UUID
from typing import Any
from pydantic import BaseModel


class EpisodeListItem(BaseModel):
    id: UUID
    slug: str
    title: str
    episode_number: int
    cover_image_url: str | None        # ← absolute URL or None, already resolved
    page_count: int
    plot_summary: str | None


class EpisodeListResponse(BaseModel):
    episodes: list[EpisodeListItem]


class CharacterSummary(BaseModel):
    id: UUID
    name: str
    image_url: str | None              # ← absolute URL, resolved


class PageCharacterRef(BaseModel):
    id: UUID
    name: str


class PageDetail(BaseModel):
    id: UUID
    page_number: int
    image_url: str                     # ← absolute URL, ready to drop into <img src>
    thumbnail_url: str | None
    image_metadata: dict[str, Any]     # {width, height, blurhash, dominant_color}
    characters: list[PageCharacterRef]


class EpisodeDetail(BaseModel):
    id: UUID
    slug: str
    title: str
    episode_number: int
    plot_summary: str | None
    credits_url: str | None
    characters: list[CharacterSummary]
    pages: list[PageDetail]
```

A few small design choices worth naming, because each one was deliberate:

- **The list response is wrapped in `{ episodes: [...] }`.** Plain top-level arrays are valid JSON, but wrapping in an object leaves room for siblings (a `next_cursor`, a `total_count`) without breaking existing clients. At two episodes it doesn't matter; at forty it will. Cheap insurance.
- **`image_metadata` stays a `dict[str, Any]`.** The DB column is JSONB holding `{width, height, blurhash, dominant_color}` and the frontend reads them all together. Flattening to typed sub-fields would require a Pydantic sub-model and a per-field assignment loop in the handler. The JSONB blob travels intact through Pydantic with one line; if the contents drift (a future `palette` field, say), no migration needed.
- **`PageDetail.characters` uses `PageCharacterRef`, not the full character record.** The episode-level `characters: list[CharacterSummary]` carries the full info (id, name, image URL). Each `PageDetail.characters` entry is just an id-and-name reference back to that list. This shaves bytes off the wire when a character appears on multiple pages — `Pepper` is in nearly every page of nearly every episode — and the frontend joins on `id` if it needs an image.
- **`EpisodeDetail` is *not* `EpisodeListItem` extended.** They overlap a lot, but the picker doesn't need `credits_url` or character lists, and the reader needs more than the picker has. Two flat models is clearer to read than an inheritance chain when each is consumed in exactly one place.
- **No request models in this post.** Both endpoints are pure GETs with no body. The request side of the API surface gets interesting in Post 9 (chat session creation, page-update PATCH).

The corresponding OpenAPI spec — accessible at `http://localhost:8000/openapi.json` once the backend boots — is what makes these models the contract instead of just the implementation. The Swagger UI at `/docs` renders the same schemas in a clickable form. **That spec is what the frontend's types mirror by hand**, and what a generator like [openapi-typescript](https://github.com/openapi-ts/openapi-typescript) would consume if we add one later (see [§ The Frontend's Type Contract]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}#type-contract) in [Post 8]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %})).

---

## Pydantic v2 in Two Minutes {#pydantic-aside}

> *Plain-English aside: what's Pydantic?* **[Pydantic](https://docs.pydantic.dev/latest/)** is a Python library that turns *"a class with type hints"* into *"a runtime validator that parses, coerces, and rejects malformed data."* You declare a class with typed fields; Pydantic gives you back a class that knows how to construct itself from a JSON string, a dict, or another ORM object — and that raises a structured error if any field is missing or the wrong type. Version 2 (the one we're on) is roughly 10× faster than v1 because the core validator is implemented in Rust.

In FastAPI specifically, Pydantic plays three roles at once:

1. **Request validation.** If you type a route handler's parameter as `payload: SomeModel`, FastAPI parses the JSON request body through `SomeModel.model_validate(...)` before your code runs. Malformed inputs become structured `422 Unprocessable Entity` responses, with field-level error messages.
2. **Response serialization.** If you declare the route's `response_model=SomeOtherModel`, FastAPI uses that model to *re-validate and serialize* whatever you return. This is a quiet superpower: even if your handler accidentally returns an ORM object with extra fields, only the fields declared on the response model end up in the JSON.
3. **OpenAPI generation.** Both of the above get reflected automatically into the [OpenAPI spec](https://swagger.io/specification/) FastAPI publishes at `/openapi.json`. The spec is what generates the interactive [Swagger UI](https://fastapi.tiangolo.com/features/#automatic-docs) at `/docs` — and it's the canonical wire-format document any future generated client (or hand-rolled TypeScript file) needs to stay in sync with.

If you've worked with [Java's Jackson](https://github.com/FasterXML/jackson) or [Go's `encoding/json` struct tags](https://pkg.go.dev/encoding/json), Pydantic occupies roughly the same niche — but it doubles as a request-validation layer and as a spec generator, which the others don't. **For a project where the JSON wire format is the seam between two languages, having all three of those in one library is the cheapest possible glue.**

---

## Where the Relative Key Becomes an Absolute URL {#url-composition}

This is the most important section of the post, and the one that makes the [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) abstraction pay off in a way you can see.

Recall the project's [`CLAUDE.md`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/CLAUDE.md) rule 5:

> `pages.image_url` stores a relative key like `episodes/ep01-pollution/pages/001-display.webp`, not a full URL. The full URL is composed at API response time using the configured storage backend (`LocalStorage.url_for()` or `R2Storage.url_for()`). This way, swapping storage backends (local → R2) is a config change, not a migration.

The motivation is simple: a database migration that rewrites every `image_url` from `http://localhost:8000/images/...` to `https://images.peppercarrot.example.com/...` is a real cost — slow on big tables, scary on a running app, and *guaranteed* to happen the first time you move from dev to prod if URLs are stored absolute. Keeping the DB value relative and composing the absolute URL at response time costs *nothing*, and the swap-to-R2 day becomes a one-line change in `.env`.

Here's how it lands in code. The route handler takes two FastAPI dependencies — an async DB session and a `Storage` — and the per-row resolution happens inline. No `Service` class wrapping it; for an API surface this small the indirection would cost more than it pays.

```python
# backend/app/api/episodes.py (excerpt)
from typing import Annotated
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients import Storage, get_storage
from app.config import Settings, get_settings
from app.db.models import Episode, Page
from app.db.session import get_session

router = APIRouter()


def get_storage_client(settings: Annotated[Settings, Depends(get_settings)]) -> Storage:
    """Build the configured storage client.

    Wrapped as a FastAPI dependency so route handlers never import storage
    backends directly and tests can override it via app.dependency_overrides.
    """
    return get_storage(settings)


SessionDep = Annotated[AsyncSession, Depends(get_session)]
StorageDep = Annotated[Storage, Depends(get_storage_client)]


@router.get("", response_model=EpisodeListResponse)
async def list_episodes(db: SessionDep, storage: StorageDep) -> EpisodeListResponse:
    page_counts = (
        select(Page.episode_id, func.count(Page.id).label("page_count"))
        .group_by(Page.episode_id)
        .subquery()
    )
    stmt = (
        select(Episode, func.coalesce(page_counts.c.page_count, 0).label("page_count"))
        .outerjoin(page_counts, Episode.id == page_counts.c.episode_id)
        .order_by(Episode.episode_number.asc())
    )
    rows = (await db.execute(stmt)).all()

    items: list[EpisodeListItem] = []
    for episode, page_count in rows:
        cover_url = (
            await storage.url_for(episode.cover_image_url)
            if episode.cover_image_url
            else None
        )
        items.append(
            EpisodeListItem(
                id=episode.id,
                slug=episode.slug,
                title=episode.title,
                episode_number=episode.episode_number,
                cover_image_url=cover_url,
                page_count=int(page_count),
                plot_summary=_summary_for_card(episode.plot_summary),
            )
        )
    return EpisodeListResponse(episodes=items)
```

A few things to notice — and a couple are foundational enough that they deserve a plain-English aside before we keep going.

**The `@router.get("", response_model=EpisodeListResponse)` decorator.** Three concepts wrapped in one line. Let's unpack each.

> *Plain-English aside: what's a decorator?* A **decorator** in Python is a function that wraps another function to add behavior, written with the `@` syntax just above the wrapped function. `@router.get(...)` is shorthand for *"after this `def`, take the function I just defined and pass it to `router.get(...)`"* — FastAPI then stores it in an internal table that maps `(method, URL)` pairs to handlers. The decorator doesn't *call* your function; it just *registers* it so the framework can call it later, at request time, when an HTTP request arrives with a matching path. (Python decorators in general: [docs](https://docs.python.org/3/glossary.html#term-decorator).)
>
> *Plain-English aside: what's a router?* The `router` variable comes from `APIRouter()` at the top of the file — it's a **container for related routes**. Think of it as a sub-application: it has its own `.get` / `.post` / `.patch` / `.delete` methods, the routes you register on it sit dormant until the top-level FastAPI app *mounts* the router with a path prefix. In our case, `app/main.py` does `app.include_router(episodes.router, prefix="/api/episodes", tags=["episodes"])`, which means the `@router.get("")` here actually serves `GET /api/episodes` once mounted, and `@router.get("/{slug}")` serves `GET /api/episodes/{slug}`. The path on the decorator is **relative to the router's prefix**. ([FastAPI APIRouter docs](https://fastapi.tiangolo.com/tutorial/bigger-applications/#apirouter).)
>
> *And the `tags=["episodes"]` argument?* Pure OpenAPI-spec metadata — **no runtime effect at all**. FastAPI attaches the tag to every operation registered through this router, so each entry under `/openapi.json` → `paths.<route>.get.tags` ends up as `["episodes"]`. Swagger UI at `/docs` then groups all operations sharing a tag into one collapsible section with the tag name as its header, and ReDoc uses tags for left-sidebar navigation. With one router and two routes the value is small, but in the full project (five routers — episodes, sessions, messages, pages, world-graph), it's what keeps `/docs` readable instead of a flat list of 20+ endpoints. Some generated-client tools (`openapi-typescript`, `openapi-fetch`, Postman) also emit one namespace per tag, so the value carries through to the frontend if you wire that up.
>
> *And `response_model=EpisodeListResponse`?* This argument tells FastAPI **what shape the response will have**. At runtime, FastAPI will take whatever the handler returns and re-validate it through `EpisodeListResponse` before serializing it to JSON — so even if the handler accidentally returns extra fields, only the ones declared on the response model end up on the wire. At spec-generation time, FastAPI uses the same model to populate the response schema in OpenAPI, which is what `/docs` renders and what generated frontend types would mirror. **One declaration, three jobs**: routing, response validation, spec generation.

**`async def` instead of plain `def`.**

> *Plain-English aside: what's `async`?* An `async def` function is a **coroutine** — a function that can pause its execution at `await` points, hand control back to the event loop, and resume later when whatever it was waiting on finishes. The practical effect: while *this* request is waiting on the database to respond, the same Python process can serve *another* request that just came in, instead of blocking on the first one. (The classic explainer is [PEP 492](https://peps.python.org/pep-0492/); FastAPI's [async page](https://fastapi.tiangolo.com/async/) walks the same idea in app-specific terms.)
>
> Why is it the right shape for this code? Every external thing we touch in a route handler is I/O — Postgres queries, image-bytes lookups, eventually model API calls. While Python is waiting on a Postgres reply, there's no CPU work for it to do; it should be servicing the next request, not sitting idle. `async def` + `await` is what makes that possible without spawning a separate thread per request. You can also write plain `def` route handlers and FastAPI will run them in a thread pool — that works, but you give up the cheap concurrency.
>
> The discipline: **if you write `async def`, every external call inside has to be `await`-able**. That's why our session uses `from sqlalchemy.ext.asyncio import AsyncSession`, our storage uses `aiofiles`, etc. The async-everywhere discipline from CLAUDE.md rule 6 is the same rule, written in Python.

**The `Storage` Protocol is injected, not constructed.** The route's `storage` parameter is typed as `Storage` — the Protocol from `backend/app/clients/storage.py` — not as `LocalStorage`. The implementation is wired up by `get_storage_client`, which is itself a FastAPI dependency that reads `Settings` and asks the factory from [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) for the configured backend. The handler never imports either concrete class. **The flip from local to R2 is, structurally, a single environment variable.** No code in this file changes.

**The single SQL query computes `page_count` in the database.** A naive implementation would fetch episodes first, then loop `len(episode.pages)` — but that's an N+1 (one query for the episode list, one per episode to load pages). The `outerjoin + group_by + func.count(Page.id)` subquery produces one SQL statement that returns `(Episode, page_count)` tuples. For three episodes this is invisible; for 40 it's the difference between one query and 41. The full step-by-step of how the subquery, outer join, and `COALESCE` work together — and why a single direct join would have been worse — is in [§ Appendix: The `list_episodes` Query, Built Up Step by Step]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}#appendix-list-query) at the end of [Post 8]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}).

**The `Annotated[..., Depends(...)]` alias pattern.** This one needs unpacking if you haven't seen `Annotated` before, because it's load-bearing for the rest of the file.

> *Plain-English aside: what's `Annotated`?* [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated) is a typing construct (introduced by [PEP 593](https://peps.python.org/pep-0593/) in Python 3.9) that lets you **attach extra metadata to a type without changing what the type means to a type-checker**. The shape is `Annotated[ActualType, metadata1, metadata2, ...]`. The first slot is the real type; everything after is metadata that frameworks can read at runtime via Python's introspection. To `mypy`, `Annotated[AsyncSession, Depends(get_session)]` *is just* `AsyncSession` — the `Depends(...)` part is invisible to the type system. To FastAPI, the metadata is a hint that says *"don't expect the caller to pass this argument; instead, call `get_session()` and inject its return value here."* So you get one declaration that satisfies two readers — the type-checker sees the type, the framework sees the wiring.
>
> Why does FastAPI care? Before `Annotated` was widely available, FastAPI used **default arguments** for dependency injection: `db: AsyncSession = Depends(get_session)`. That worked, but it was a slight semantic abuse — default arguments are for actual default values, and using them for "this gets injected at call time" muddied the meaning. `Annotated` is the cleaner pattern the FastAPI team [now recommends](https://fastapi.tiangolo.com/tutorial/dependencies/#share-annotated-dependencies). Both still work; new code should use `Annotated`.

In our file, the line `SessionDep = Annotated[AsyncSession, Depends(get_session)]` reads as *"`SessionDep` is the type `AsyncSession`, plus a `Depends` instruction that FastAPI will use to fill it in."* Aliasing the whole expression once at module scope lets every route handler write `db: SessionDep` instead of repeating `db: Annotated[AsyncSession, Depends(get_session)]` — saves repeating ~30 characters per handler, and makes the dependency relationship obvious from the name (`SessionDep` reads better than the `Annotated[...]` form when you're scanning a file). Pure ergonomic win, same runtime behavior.

The detail endpoint is structurally the same — fetch one `Episode` with its `pages` and per-page `characters` eagerly loaded, resolve every relative URL through `storage.url_for`, then return an `EpisodeDetail`. The full source is in [`backend/app/api/episodes.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/backend/app/api/episodes.py).

---

## The Two Route Handlers, End to End {#route-handlers}

We've seen the list handler. Here's the detail handler in full, because it's the one the flipbook actually depends on, and the resolution loop is the load-bearing part:

```python
@router.get("/{slug}", response_model=EpisodeDetail)
async def get_episode(slug: str, db: SessionDep, storage: StorageDep) -> EpisodeDetail:
    stmt = (
        select(Episode)
        .where(Episode.slug == slug)
        .options(selectinload(Episode.pages).selectinload(Page.characters))
    )
    episode = (await db.execute(stmt)).scalar_one_or_none()
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode '{slug}' not found")

    # Episode-level character roster: union across all pages, deduped by id,
    # sorted by name for stable output.
    seen_char_ids: set[UUID] = set()
    episode_characters: list[CharacterSummary] = []
    for page in episode.pages:
        for char in page.characters:
            if char.id in seen_char_ids:
                continue
            seen_char_ids.add(char.id)
            char_image = (
                await storage.url_for(char.image_url) if char.image_url else None
            )
            episode_characters.append(
                CharacterSummary(id=char.id, name=char.name, image_url=char_image)
            )
    episode_characters.sort(key=lambda c: c.name)

    pages: list[PageDetail] = []
    for page in episode.pages:
        image_url = await storage.url_for(page.image_url)
        thumbnail_url = (
            await storage.url_for(page.thumbnail_url) if page.thumbnail_url else None
        )
        pages.append(
            PageDetail(
                id=page.id,
                page_number=page.page_number,
                image_url=image_url,
                thumbnail_url=thumbnail_url,
                image_metadata=page.image_metadata or {},
                characters=[
                    PageCharacterRef(id=c.id, name=c.name) for c in page.characters
                ],
            )
        )

    return EpisodeDetail(
        id=episode.id,
        slug=episode.slug,
        title=episode.title,
        episode_number=episode.episode_number,
        plot_summary=episode.plot_summary,
        credits_url=episode.credits_url,
        characters=episode_characters,
        pages=pages,
    )
```

A few last points worth pulling out before we leave the backend:

- **`selectinload(Episode.pages).selectinload(Page.characters)` is intentional.** Without it, SQLAlchemy's default async session would lazy-load `episode.pages` and `page.characters` mid-serialization, raising `MissingGreenlet` on the async path. Chained `selectinload` issues two extra `IN (...)` queries — one for the pages of the matched episode, one for the characters of those pages — eagerly, in the same database round-trip cycle as the parent. Three queries total instead of one-plus-N. The [SQLAlchemy docs](https://docs.sqlalchemy.org/en/20/orm/queryguide/relationships.html#sqlalchemy.orm.selectinload) cover the trade-offs against `joinedload` (which uses `LEFT OUTER JOIN`s and can balloon the result set on deep relationships); for one-to-many like this, `selectinload` is usually cheaper. The whole N+1-problem story plus the `scalar_one_or_none()` companion call below is walked through from first principles in [§ Appendix: A Tour of `selectinload` and `scalar_one_or_none()`]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}#appendix-sqlalchemy) at the end of [Post 8]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}).
- **The 404 path is explicit.** FastAPI doesn't infer "this might 404" from your code; you have to raise it. The error message is intentionally specific — passing the unknown slug back to the client makes debugging in the browser console less of a guessing game.
- **The character roster is computed in Python, not SQL.** It would be marginally faster as a single CTE-with-DISTINCT — but at ≤10 characters per episode, the Python dedup loop is unambiguous to read in a code review and produces stable, alphabetized output for free.
- **The mount path is `/api/episodes`, set in `main.py`** via `app.include_router(episodes.router, prefix="/api/episodes", tags=["episodes"])`. Inside the router file, route paths are written as `""` (list) and `"/{slug}"` (detail) — the prefix lives in one place, not duplicated on every decorator.

Mount the router and start the server:

```bash
cd backend && uv run uvicorn app.main:app --reload
```

Smoke-test from a second terminal:

```bash
curl -s http://localhost:8000/api/episodes | jq '.episodes | length'
# 1

curl -s http://localhost:8000/api/episodes/ep01-potion-of-flight | jq '.pages[0]'
# {
#   "id": "...",
#   "page_number": 1,
#   "image_url": "http://localhost:8000/images/episodes/ep01-potion-of-flight/pages/001-display.webp",
#   "thumbnail_url": "http://localhost:8000/images/episodes/ep01-potion-of-flight/pages/001-thumbnail.webp",
#   "image_metadata": {"width": 1600, "height": 1131, "blurhash": "...", "dominant_color": "#e7d3a8"},
#   "characters": [{"id": "...", "name": "Carrot"}, {"id": "...", "name": "Pepper"}]
# }
```

The `image_url` is absolute and points at the FastAPI static-files mount from [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}#seam-storage). A browser hitting that URL gets the WebP back. The seam works end-to-end.

Open <http://localhost:8000/docs> in a browser — Swagger UI renders the two endpoints, the response schemas, the example responses, the 404 path, all from the same Python code with zero extra annotation work. That spec is the wire-format contract the frontend will mirror in TypeScript.

---

Next up: **Post 8 — A Real Flipbook in the Browser: React + StPageFlip.** The two endpoints you just built return episode JSON with absolute image URLs — but none of it is visible yet. The next post crosses the wire: a Vite + React + TypeScript frontend that hand-mirrors the Pydantic models in a `types.ts`, a tiny `fetch` client, an episode picker, and a `<Flipbook>` component wrapping [StPageFlip](https://github.com/Nodlik/StPageFlip) so a reader can flip through a real page-turning book — single page in portrait, two-page spread in landscape. It also carries the appendices that go deeper on the two SQLAlchemy idioms behind these handlers (`selectinload` and `scalar_one_or_none()`) and the `list_episodes` query, plus how `Settings` reads your `.env`.

The **workshop starter** that backs this post is at <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop>, tagged `post-07-08-fullstack` (shared with the [Post 8]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}) flipbook post) — pull the latest to pick up `backend/app/api/episodes.py` and the smoke tests. The **full source repository** and the public live-demo URL go up alongside the final post — the deploy guide — once it's published.

*Pepper & Carrot* is © [David Revoy](https://www.davidrevoy.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All credit to him for the source material that made this project possible.

**All opinions expressed are my own.**
