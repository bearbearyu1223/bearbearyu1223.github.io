---
title: "Pepper & Carrot AI-powered flipbook · Part 15 of 16 — Shipping It: Containerize, Deploy to Fly + Pages, and Verify"
date: 2026-06-01 00:00:00 -0800
categories: [Full-Stack, RAG, Local AI]
tags: [deployment, cloudflare-pages, fly-io, modal, cloudflare-r2, neon, docker, fastapi, peppercarrot, portfolio]
description: >-
  Post 15 of the Pepper & Carrot AI flipbook series — the deploy itself.
  Post 14 provisioned the five backing services and built the container;
  this post turns that container into a public URL. The FastAPI backend
  ships to Fly.io behind a scale-to-zero machine, the React frontend ships
  to Cloudflare Pages with a single build-time env var, and a layer-by-layer
  verification walkthrough confirms Modal, Neon, R2, Fly, and Pages all
  talk to each other end to end. The honest part is the cold start — the
  first answer after idle takes 15–30 seconds, the price the architecture
  pays for $0 idle — and this post is honest about where that cost sits and
  how a fire-and-forget warmup would hide it.
pin: true
---

Post 15 of the [*Pepper & Carrot AI-powered flipbook*]({% post_url 2026-05-09-pepper-carrot-companion-trailer %})
series — the deploy itself. [Post 14]({% post_url 2026-05-31-pepper-carrot-companion-shipping-it %})
did the provisioning: it stood up Modal (GPU-served Ollama), Neon (managed
Postgres), and Cloudflare R2 (image bytes behind the [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %})
`Storage` interface), and built the two-stage container that bakes the small
data and streams the big data. Everything is provisioned; nothing is public
yet. This post finishes the job: it deploys the backend to Fly.io, the
frontend to Cloudflare Pages, and then verifies the whole thing — five
providers, one container, one public URL — end to end. The interesting part
isn't the typing. **The interesting part is that the typing is small** —
because the abstractions from [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %})
were designed for exactly this seam-by-seam migration, and the runtime never
notices the change.

> **What you'll build in this post.**
> - **A `fly.toml` Fly app deploy** — the `[env]` block flips the runtime onto the production providers (Modal-hosted Ollama, R2-hosted images) with *no code change*, and the `[http_service]` block configures scale-to-zero (`auto_stop_machines = 'stop'`, `min_machines_running = 0`). One `fly secrets set` pushes all 11 production values; `fly deploy` builds, boots a 512 MB machine, restores the seed on first boot, and prints a `*.fly.dev` URL.
> - **A Cloudflare Pages deploy** — connect the repo, set `VITE_API_BASE_URL` to the Fly URL, build `cd frontend && npm install && npm run build`, output `frontend/dist`. One build-time env var does all the work; the deployed bundle calls Fly directly. The one sharp edge — `CORS_ORIGINS` on Fly must match the Pages URL exactly — gets its own callout.
> - **A clear-eyed account of the first cold start** — two stacked cold starts (Fly waking the VM, Modal allocating a GPU and loading qwen2.5:7b into VRAM) make the first answer after idle take 15–30 seconds. A sequence diagram draws exactly where the cost sits, and the optional fire-and-forget warmup that hides it behind the human seconds of choosing what to read.
> - **A layer-by-layer verification walkthrough** — six terminal checks that confirm Modal, Neon, R2, Fly, and Pages each work before you trust the URL to anyone, ordered so a failure names the single provider to debug.
> - **An appendix** on what "serverless" actually means, why the backend runs on Fly instead of Cloudflare Workers, and a tour of the broader Cloudflare product ecosystem.
>
> **Prerequisites.**
> - [Post 14]({% post_url 2026-05-31-pepper-carrot-companion-shipping-it %}) finished: Modal deployed and serving both models, a Neon project with its pooled + unpooled URLs in hand, an R2 bucket with the image bytes uploaded, and the container building locally with `data/seed.sql` dumped from your local Postgres.
> - Free-tier accounts on [Fly.io](https://fly.io) and [Cloudflare](https://dash.cloudflare.com). Fly requires a card during sign-up; the free monthly allowance covers a sleepy demo.
> - CLIs: `brew install flyctl` (you already installed `rclone` and `modal` in Post 14).
> - A domain or custom DNS records is **not** required — Fly ships `*.fly.dev` and Pages ships `*.pages.dev` for free.

> **Checking out the code.** Everything in this post — `fly.toml`, the
> Cloudflare Pages config, `infra/entrypoint.sh`, and `docs/deployment.md` —
> lives in the same workshop starter as the provisioning work, at the
> `post-14-15-deploy` tag: `git checkout post-14-15-deploy` (see [Following
> along with the blog series](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop#following-along-with-the-blog-series)).
> This is the same checkpoint [Post 14]({% post_url 2026-05-31-pepper-carrot-companion-shipping-it %})
> uses — the provisioning post and this deploy post share one deploy tag.

---

## Table of Contents

1. [Fly.io: The Backend Public URL](#fly)
2. [Cloudflare Pages: One Build Var, One Public URL](#pages)
3. [The First Cold Start Is the Demo](#cold-start)
4. [What's Honest, What's Open](#honest)
5. [Verify Before You Publish: A 40-Minute Walkthrough](#verify)
6. [Key Takeaways](#key-takeaways)
7. [Appendix: Serverless, Workers, and the Cloudflare Edge](#appendix)
8. [The Series, End to End](#series-arc)

---

## Fly.io: The Backend Public URL {#fly}

Fly is the orchestrator that takes the container, the secrets, and the config in [`fly.toml`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/fly.toml), and turns them into a `*.fly.dev` URL. The config is short:

```toml
# fly.toml (excerpted)
app = 'peppercarrot-companion'
primary_region = 'iad'

[build]
  dockerfile = 'Dockerfile'

# Non-secret env vars that select the production providers. The seam
# was built in Post 4; these lines are what flip the runtime onto
# Modal-hosted Ollama and R2-hosted images.
[env]
  CHAT_PROVIDER = 'ollama'
  EMBEDDING_PROVIDER = 'ollama'
  EMBEDDING_MODEL = 'bge-m3'
  OLLAMA_CHAT_MODEL = 'qwen2.5:7b'
  STORAGE_BACKEND = 'r2'
  LOG_LEVEL = 'INFO'

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = 'stop'
  auto_start_machines = true
  min_machines_running = 0

  [http_service.concurrency]
    type = 'requests'
    hard_limit = 25
    soft_limit = 20

[[vm]]
  memory = '512mb'
  cpu_kind = 'shared'
  cpus = 1
  memory_mb = 512
```

The `[env]` block is the one with the most architectural weight: those six lines are *the entire config-flip* that swings the application from the local-first defaults to the cloud production stack. **No code change is required to make any of those switches** — the factory in `clients/__init__.py` has been respecting these env vars since Post 4.

The `[http_service]` block tells Fly to expose the container on port 8000, behind HTTPS termination, with the following scale-to-zero behavior:

- **`auto_stop_machines = 'stop'`** — stop the machine when idle. Stopped machines cost nothing.
- **`auto_start_machines = true`** — wake the machine on the next inbound request. The wake adds ~5–10 s to the first request after idle.
- **`min_machines_running = 0`** — don't keep a baseline number warm. Idle = $0.

The concurrency limits (`soft_limit = 20`, `hard_limit = 25`) are sized for a 512 MB shared-CPU VM. They're low because the backend's chat handler holds an SSE connection open for the duration of an answer (5–30 seconds typically) and qwen2.5:7b can only stream so fast — twenty concurrent chats are already more than the GPU on Modal would saturate at. For portfolio traffic this is generous.

Pushing the secrets is one shell command:

```bash
fly auth login
fly launch --no-deploy --copy-config --name peppercarrot-companion
set -a && source .env.production && set +a && fly secrets set \
  POSTGRES_RESTORE_URL="$POSTGRES_RESTORE_URL" \
  DATABASE_URL_OVERRIDE="$DATABASE_URL_OVERRIDE" \
  OLLAMA_BASE_URL="$OLLAMA_BASE_URL" \
  MODAL_PROXY_TOKEN_ID="$MODAL_PROXY_TOKEN_ID" \
  MODAL_PROXY_TOKEN_SECRET="$MODAL_PROXY_TOKEN_SECRET" \
  R2_ACCOUNT_ID="$R2_ACCOUNT_ID" \
  R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
  R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  R2_BUCKET="$R2_BUCKET" \
  R2_PUBLIC_URL_PREFIX="$R2_PUBLIC_URL_PREFIX" \
  CORS_ORIGINS="$CORS_ORIGINS"
fly deploy
```

The first deploy takes ~5 minutes — Docker builds the image, pushes the layers to Fly's registry, boots a 512 MB machine, the entrypoint runs `psql < /app/data/seed.sql` against the empty Neon database (~30 seconds for ~1 MB of seed). After that, every subsequent deploy is ~1–2 minutes (cached layers, no seed restore).

**`fly logs` is the single best diagnostic** when something goes wrong. The most common first-deploy failure is the asyncpg-vs-pgbouncer interaction from the Neon provisioning step in [Post 14]({% post_url 2026-05-31-pepper-carrot-companion-shipping-it %}) — if `/health` returns 200 but `/api/episodes` returns 500, `fly logs` will print an asyncpg traceback in the last 30 lines that names the problem exactly. The troubleshooting table in [`docs/deployment.md`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/docs/deployment.md#troubleshooting) is a compressed version of every wrong-secret and wrong-URL failure mode I've hit while bringing this stack up.

---

## Cloudflare Pages: One Build Var, One Public URL {#pages}

The frontend deploy is the simplest of the five. Cloudflare Pages connects to the GitHub repo, runs `npm install && npm run build` per the repo's existing `frontend/package.json`, and serves the `frontend/dist/` directory from edge nodes worldwide. The whole configuration is in the Pages UI:

- **Build command:** `cd frontend && npm install && npm run build`
- **Build output directory:** `frontend/dist`
- **Environment variable:** `VITE_API_BASE_URL = https://peppercarrot-companion.fly.dev`

The `VITE_API_BASE_URL` is the one variable that does all the work. Vite inlines build-time env vars (prefixed with `VITE_`) into the bundled JavaScript, so whatever `import.meta.env.VITE_API_BASE_URL` reads at build time is the URL the deployed frontend will call. The workshop's [`frontend/src/api/client.ts`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/frontend/src/api/client.ts) does exactly that:

```ts
// frontend/src/api/client.ts (approximately)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api';
```

In dev, the env var is unset, so the frontend calls relative URLs and Vite's dev-server proxy from [`vite.config.ts`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/frontend/vite.config.ts) forwards them to `localhost:8000`. In prod, the env var is set, so the frontend calls the Fly URL directly. **Same code, different env**.

The single sharp edge: **`CORS_ORIGINS` on Fly must match the Pages URL exactly**, scheme included, no trailing slash. If they don't match, every request from the browser is blocked by the same-origin policy and you get inscrutable CORS errors in the DevTools console. Fix is one secret push:

```bash
fly secrets set CORS_ORIGINS='["https://your-app.pages.dev"]'
# Fly auto-redeploys when secrets change.
```

Pages prints a URL like `https://your-app.pages.dev`. Open it in a browser. The flipbook loads. **The local-first workshop, now globally distributed, on free tiers, costing ~$5/mo for the GPU seconds.**

> *Diagram for the live demo.* The picker → Modal cold-start path is worth drawing for a recruiter. When the reader opens an episode, the frontend fires `POST /api/sessions` — and a natural production-polish addition (the workshop ships without it, but the architecture is positioned to bolt it on) is to use that handler as the cue to send a fire-and-forget warmup request to Modal, so the GPU is allocated and the model is in VRAM by the time the reader types their first question. The warmup doesn't change the GPU cost, only the *perceived* latency of the first answer. Showing recruiters where you *would* hide the cost is part of the design story.

---

## The First Cold Start Is the Demo {#cold-start}

There is one place this architecture's honesty is most visible: **the first chat request after idle takes 15–30 seconds**. Two stacked cold starts — Fly waking the backend (~8 s) and Modal allocating a GPU plus loading qwen2.5:7b into VRAM (~15–25 s). Subsequent answers within 5 minutes are instant. After 5 minutes idle, both clocks reset.

The shape of "first request after idle" matters enough to draw. Every arrow below is something the architecture chose; reading the diagram is reading the trade-offs:

<div style="margin: 1.5rem 0; overflow-x: auto;">
<a href="/assets/picture/2026-05-31-pepper-carrot-companion-shipping-it/cold-start-sequence.svg" target="_blank" rel="noopener" title="Open the diagram full-size in a new tab" style="display: block; cursor: zoom-in;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 540" role="img"
     aria-label="A sequence diagram of the first chat request after the system has been idle for more than 5 minutes. Lanes top to bottom: browser at *.pages.dev, Fly backend, Neon Postgres, Modal Ollama. Step 1 the browser POSTs /api/sessions to Fly; Fly is asleep so cold-starts a Firecracker VM in 5 to 10 seconds; entrypoint sees the episodes table already exists and skips the seed restore; uvicorn boots and serves the session; the session POST also fires a fire-and-forget warmup to Modal which begins allocating a GPU. Step 2 the reader chooses what to read and types a question while the warmup is hidden behind the seconds of human reading time. Step 3 the reader sends the message; the backend retrieves Chroma + Postgres context locally (~50 ms) and forwards the prompt to Modal; if the warmup has finished allocating the GPU and loading qwen2.5:7b into VRAM the first SSE token lands within a second; otherwise the request waits for the remaining cold-start time. Step 4 subsequent tokens stream at full throughput; the answer completes and the SSE done frame closes the connection. Both Fly and Modal stay warm for 5 minutes after the last request, then go back to scale-to-zero."
     style="display: block; width: 100%; height: auto; max-width: 1100px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">
  <defs>
    <marker id="cs-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>
    </marker>
    <marker id="cs-arrow-cold" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#b45309"/>
    </marker>
    <marker id="cs-arrow-warm" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#059669"/>
    </marker>
  </defs>

  <!-- Lane headers -->
  <text x="80" y="28" font-size="11" font-weight="700" fill="#1f2937">Browser</text>
  <text x="80" y="42" font-size="9" fill="#94a3b8" font-style="italic">*.pages.dev</text>

  <text x="330" y="28" font-size="11" font-weight="700" fill="#1f2937">Fly (FastAPI)</text>
  <text x="330" y="42" font-size="9" fill="#94a3b8" font-style="italic">512 MB, scale-to-zero</text>

  <text x="600" y="28" font-size="11" font-weight="700" fill="#1f2937">Neon (Postgres)</text>
  <text x="600" y="42" font-size="9" fill="#94a3b8" font-style="italic">unpooled · asyncpg</text>

  <text x="870" y="28" font-size="11" font-weight="700" fill="#1f2937">Modal (Ollama)</text>
  <text x="870" y="42" font-size="9" fill="#94a3b8" font-style="italic">T4, min_containers=0</text>

  <!-- Lane lines -->
  <line x1="120" y1="60" x2="120" y2="510" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="2,3"/>
  <line x1="380" y1="60" x2="380" y2="510" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="2,3"/>
  <line x1="650" y1="60" x2="650" y2="510" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="2,3"/>
  <line x1="920" y1="60" x2="920" y2="510" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="2,3"/>

  <!-- ─── T=0: idle state ─── -->
  <text x="20" y="78" font-size="10" font-weight="600" fill="#475569">t=0</text>
  <text x="40" y="78" font-size="10" fill="#94a3b8" font-style="italic">idle</text>
  <rect x="375" y="68" width="10" height="14" fill="#94a3b8" opacity="0.3"/>
  <text x="395" y="78" font-size="9" fill="#94a3b8" font-style="italic">asleep</text>
  <rect x="915" y="68" width="10" height="14" fill="#94a3b8" opacity="0.3"/>
  <text x="935" y="78" font-size="9" fill="#94a3b8" font-style="italic">no container</text>

  <!-- ─── Step 1: POST /api/sessions ─── -->
  <text x="20" y="108" font-size="10" font-weight="600" fill="#7c2d12">t≈0</text>
  <text x="40" y="108" font-size="10" fill="#475569" font-style="italic">reader opens an episode</text>

  <line x1="120" y1="120" x2="378" y2="120" stroke="#b45309" stroke-width="1.6" marker-end="url(#cs-arrow-cold)"/>
  <text x="245" y="115" text-anchor="middle" font-size="9" fill="#7c2d12" font-weight="600">POST /api/sessions</text>

  <!-- Fly cold start -->
  <rect x="375" y="130" width="10" height="60" fill="#fde68a" stroke="#b45309" stroke-width="0.8"/>
  <text x="400" y="145" font-size="9" fill="#7c2d12" font-style="italic" font-weight="600">▼ Fly cold start</text>
  <text x="400" y="160" font-size="9" fill="#94a3b8">  Firecracker VM boot ~5–10 s</text>
  <text x="400" y="175" font-size="9" fill="#94a3b8">  entrypoint: information_schema check</text>
  <text x="400" y="190" font-size="9" fill="#94a3b8">  (skips seed; episodes exists) → uvicorn</text>

  <!-- Session-create writes to Postgres (warm — Neon sleep wake is ~1 s) -->
  <line x1="385" y1="200" x2="648" y2="200" stroke="#b45309" stroke-width="1.4" marker-end="url(#cs-arrow-cold)"/>
  <text x="510" y="195" text-anchor="middle" font-size="9" fill="#7c2d12">INSERT INTO chat_sessions</text>
  <rect x="645" y="208" width="10" height="14" fill="#fde68a" stroke="#b45309" stroke-width="0.8"/>
  <text x="670" y="218" font-size="9" fill="#94a3b8" font-style="italic">  Neon wake ~1 s</text>
  <line x1="648" y1="232" x2="386" y2="232" stroke="#059669" stroke-width="1.2" marker-end="url(#cs-arrow-warm)"/>
  <text x="517" y="227" text-anchor="middle" font-size="9" fill="#065f46">session_id</text>

  <!-- Fire-and-forget warmup to Modal -->
  <line x1="385" y1="246" x2="918" y2="246" stroke="#b45309" stroke-width="1.4" stroke-dasharray="4,3" marker-end="url(#cs-arrow-cold)"/>
  <text x="650" y="241" text-anchor="middle" font-size="9" fill="#7c2d12" font-weight="600">(optional polish) fire-and-forget warmup → Modal</text>

  <!-- Modal cold start begins -->
  <rect x="915" y="252" width="10" height="80" fill="#fde68a" stroke="#b45309" stroke-width="0.8"/>
  <text x="935" y="265" font-size="9" fill="#7c2d12" font-style="italic" font-weight="600">▼ Modal cold start</text>
  <text x="935" y="280" font-size="9" fill="#94a3b8">  allocate GPU ~5 s</text>
  <text x="935" y="295" font-size="9" fill="#94a3b8">  load qwen2.5:7b → VRAM ~15 s</text>
  <text x="935" y="310" font-size="9" fill="#94a3b8">  load bge-m3 → VRAM ~5 s</text>

  <!-- Response to browser: session_id -->
  <line x1="378" y1="258" x2="122" y2="258" stroke="#059669" stroke-width="1.4" marker-end="url(#cs-arrow-warm)"/>
  <text x="250" y="253" text-anchor="middle" font-size="9" fill="#065f46" font-weight="600">200 OK · session_id</text>

  <!-- ─── Step 2: human seconds ─── -->
  <text x="20" y="310" font-size="10" font-weight="600" fill="#475569">t≈5–30 s</text>
  <text x="80" y="310" font-size="10" fill="#475569" font-style="italic">reader chooses + types · the warmup runs hidden behind these seconds</text>

  <rect x="115" y="318" width="800" height="16" fill="#f5f5f4" stroke="#e7e5e4" stroke-width="0.6"/>
  <text x="515" y="330" text-anchor="middle" font-size="9" fill="#78716c" font-style="italic">human reading time · usually longer than the Modal cold start</text>

  <!-- ─── Step 3: POST /messages ─── -->
  <text x="20" y="365" font-size="10" font-weight="600" fill="#065f46">t≈30 s</text>
  <text x="80" y="365" font-size="10" fill="#475569" font-style="italic">reader sends first chat message</text>

  <line x1="120" y1="378" x2="378" y2="378" stroke="#059669" stroke-width="1.6" marker-end="url(#cs-arrow-warm)"/>
  <text x="245" y="373" text-anchor="middle" font-size="9" fill="#065f46" font-weight="600">POST /messages (SSE)</text>

  <!-- retrieval -->
  <line x1="385" y1="392" x2="648" y2="392" stroke="#059669" stroke-width="1.2" marker-end="url(#cs-arrow-warm)"/>
  <text x="515" y="387" text-anchor="middle" font-size="9" fill="#065f46">SELECT pages + wiki (~50 ms)</text>
  <line x1="648" y1="406" x2="386" y2="406" stroke="#059669" stroke-width="1.2" marker-end="url(#cs-arrow-warm)"/>

  <!-- prompt → Modal (warm) -->
  <line x1="385" y1="420" x2="918" y2="420" stroke="#059669" stroke-width="1.6" marker-end="url(#cs-arrow-warm)"/>
  <text x="650" y="415" text-anchor="middle" font-size="9" fill="#065f46" font-weight="600">prompt → Modal (warm by now)</text>

  <!-- streaming tokens back -->
  <path d="M 918 434 Q 700 444, 386 434" stroke="#059669" stroke-width="1.2" stroke-dasharray="3,3" fill="none" marker-end="url(#cs-arrow-warm)"/>
  <text x="650" y="450" text-anchor="middle" font-size="9" fill="#065f46" font-style="italic">tokens stream back · first within ~1 s, full answer in ~5–15 s</text>

  <path d="M 378 462 Q 250 472, 122 462" stroke="#059669" stroke-width="1.2" stroke-dasharray="3,3" fill="none" marker-end="url(#cs-arrow-warm)"/>
  <text x="250" y="478" text-anchor="middle" font-size="9" fill="#065f46" font-style="italic">SSE tokens + final done frame</text>

  <!-- Legend -->
  <g>
    <rect x="20" y="500" width="14" height="10" fill="#fde68a" stroke="#b45309" stroke-width="0.8"/>
    <text x="40" y="509" font-size="10" fill="#4b5563">cold (provider waking up)</text>
    <line x1="190" y1="505" x2="220" y2="505" stroke="#b45309" stroke-width="1.6" marker-end="url(#cs-arrow-cold)"/>
    <text x="225" y="509" font-size="10" fill="#4b5563">cold-path request</text>
    <line x1="350" y1="505" x2="380" y2="505" stroke="#059669" stroke-width="1.6" marker-end="url(#cs-arrow-warm)"/>
    <text x="385" y="509" font-size="10" fill="#4b5563">warm-path request / response</text>
    <line x1="560" y1="505" x2="590" y2="505" stroke="#b45309" stroke-width="1.4" stroke-dasharray="4,3" marker-end="url(#cs-arrow-cold)"/>
    <text x="595" y="509" font-size="10" fill="#4b5563">fire-and-forget warmup (optional production polish)</text>
  </g>
</svg>
</a>
</div>

*The first request after idle, drawn as a sequence. Amber spans are cold starts; green spans are warm-path I/O. The dashed amber arrow is the optional fire-and-forget warmup — production polish that bolts onto the session-create handler so Modal's cold-start cost happens *during* the human seconds of choosing what to read rather than on the actual chat round-trip. The workshop ships without it; adding it is ~30 lines. Click the diagram to open it full-size in a new tab.*

Three things the diagram makes legible that the prose alone can't:

- **The warmup is a latency-hider, not a cost-hider.** It doesn't make Modal cold-start faster; it makes the cold start happen during the human seconds the reader was going to spend reading the cover and typing a question anyway. The cost in GPU seconds and the cost in user-perceived latency are *separate dimensions* — and architecting the system to spend the GPU cost during the moments you weren't going to spend the latency anyway is the trick.
- **The Fly + Neon cold starts are small enough to absorb in the session-create response.** Five to ten seconds for Fly waking, plus a one-second Neon wake, plus the entrypoint's `information_schema` check. The reader sees a brief loading state on the episode picker after they click "Open this episode" — and by the time the flipbook has rendered page one, both Fly and Neon are warm.
- **Modal is the only cold start the user is allowed to see** — and only if the warmup loses the race against typing. A natural companion to the warmup is a one-shot retry plus a friendly fallback message ("the witch's familiars need a moment to wake up…") on the chat panel, for the rare case both attempts hit the cold start.

The trade-off matrix below distills what the diagram leaves implicit:

That latency is not a bug. It's the cost of the architecture choice that gave us $0 idle. The trade-off has two reasonable shapes:

| Stance | Modal config | Monthly cost | First request | Sustained traffic |
|---|---|---|---|---|
| **Workshop default** | `min_containers=0`, `scaledown_window=300` | $5 – $10 | 15–30 s | Instant within 5 min |
| **Always warm** | `min_containers=1` on T4 | ~$430 | Instant | Instant always |

The workshop ships the first one. **Picking the right point on a cost-vs-latency curve is part of the design judgement the portfolio is supposed to show**, and the right point for a portfolio demo is "$0 idle, slow first answer, fast subsequent." A reviewer who's deployed something real before recognizes the math the moment they see the table.

The natural mitigation is the warmup pattern mentioned earlier: have the `POST /api/sessions` handler fire a fire-and-forget request against Modal the moment a session opens, so the model is loading into VRAM during the *interesting* seconds when the reader is choosing what to read. The cost stays the same; the perceived latency for a typical visitor is much smaller. The workshop doesn't ship the warmup so that this section can be honest about where the cost sits — the warmup is real production polish, and a follow-up exercise the reader can add in ~30 lines of code (a `httpx.AsyncClient.get(f"{OLLAMA_BASE_URL}/api/tags", headers=…)` task kicked off inside the session-create handler with `asyncio.create_task(...)`).

---

## What's Honest, What's Open {#honest}

Five things to name plainly, because the portfolio framing this series chose lives or dies by whether the post can tell you what it didn't ship.

**The Chroma vector store is baked into the Docker image.** That's the operational reality of the abstraction discipline from [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}#what-deserves-an-abstraction) — Chroma wasn't given a Protocol because it has no swap target, and so the production path bakes the persistent directory into the image. The consequence: **re-ingesting episodes requires a re-deploy**. For portfolio cadence (a new episode every few weeks) this is invisible; for a real product (a new episode every day) it would be a problem and the right fix would be to factor Chroma onto its own host or switch to a hosted vector DB (Qdrant Cloud, Pinecone). The series said no to that hedge on purpose; Posts 14–15 honor it.

**There's no CI/CD pipeline.** The workshop's deploy is `./infra/dump_seed.sh && fly deploy` from a developer's laptop. The right CI/CD adds three things: a workflow that runs the test suite on every PR, a workflow that runs `dump_seed.sh` against a dev Neon and pushes a Fly review-app on merge, and a manual-trigger workflow for prod. Adding those is a half-day's GitHub Actions work and it's a separate post in spirit. The portfolio story I wanted Posts 14–15 to tell is *the architecture*; the automation is downstream of that.

**The cold-start tax is real.** The first chat request after Modal is idle takes 15–30 s. The workshop is honest about it; the natural mitigation is a fire-and-forget warmup tied to session creation, sketched above as a ~30-line follow-up. Neither path solves the problem at the always-on-GPU level; both accept the trade-off for the cost ratio. A real product with sustained traffic would re-evaluate.

**The world-graph art on R2 isn't gated by the spoiler boundary at the CDN layer.** The DB-level filter from [Post 12]({% post_url 2026-05-30-pepper-carrot-companion-spoiler-safe-world-graph %}) decides whether a `<img src=...>` for a given entity ever gets *rendered* — but the URL itself is public-readable on R2. A reader who scraped the bucket prefix would find every avatar regardless of their reading position. **The spoiler boundary protects the application UI, not the underlying CDN keys.** The same property would apply to the `pages/` bucket if the demo ever extended to gating page images at the CDN — making them private and signing each URL would add a `head_object` round-trip per page and would not change the threat model for a portfolio demo. If the demo were ever about a paid IP, the architecture would shift to signed URLs and signed cookies — and that's a different kind of post.

**Single region, single tenant.** Fly's primary region is `iad` (US-east); Neon is `us-east-2`. A reader in Tokyo would see ~150 ms more first-byte latency than a reader in Boston. Adding a second Fly region is one `fly regions add` away; replicating Neon needs branching; replicating R2 needs a custom replication. All of those are real engineering and all of them are outside Posts 14–15's scope. The demo lives in one region because the demo's *visitors* are mostly in one timezone.

**The R2Storage implementation doesn't tier through CloudFront or a custom domain.** The bucket's `pub-XXXX.r2.dev` URL is the path of least resistance — Cloudflare-served, with a small subdomain. A real product would point a `images.your-domain.com` CNAME at the bucket so the URL on the wire is branded. The R2 setup step in the deploy guide notes the custom-domain option but takes the dev subdomain by default; the swap is a Cloudflare DNS row and a one-value change to `R2_PUBLIC_URL_PREFIX`.

---

## Verify Before You Publish: A 40-Minute Walkthrough {#verify}

The post above describes an architecture and a deploy procedure; honesty about what's been *tested* means walking through the procedure once against real provider accounts before you trust the URL to recruiters. Here's the checklist that catches the most common breakages. If every line below returns what its comment predicts, the deploy is real.

### Layer-by-layer, narrowing failures to one provider

The order matters. Each check confirms the layer *underneath* the next layer works, so a failure tells you exactly which provider to debug:

```bash
# ─── 1. Modal: GPU is allocatable; both models are pulled into the volume ───
set -a && source .env.production && set +a
curl -sS -H "Modal-Key: $MODAL_PROXY_TOKEN_ID" \
        -H "Modal-Secret: $MODAL_PROXY_TOKEN_SECRET" \
        "$OLLAMA_BASE_URL/api/tags" | python3 -m json.tool | head
# Want: HTTP 200 with JSON listing qwen2.5:7b AND bge-m3.
# 401 → proxy auth tokens don't match what you pasted into .env.production.
# 404 → the URL doesn't have a matching deploy on Modal yet (re-run `modal deploy`).
# Empty models list → the first-deploy model-pull failed; check `modal app logs peppercarrot-ollama`.

# ─── 2. Neon: the unpooled endpoint accepts asyncpg-style connections ───
# (Easiest to verify indirectly through the backend — Step 4. If you want a
# direct check, `psql "$POSTGRES_RESTORE_URL" -c 'SELECT 1'` against the
# pooled endpoint will at least confirm credentials are right.)

# ─── 3. R2: a known key resolves; the public-prefix is correctly set ───
curl -I "$R2_PUBLIC_URL_PREFIX/world-graph/images/carrot-thumb.webp"
# Want: HTTP/2 200, content-type: image/webp, cache-control: public, max-age=...
# 404 → rclone uploaded to the wrong path; `rclone ls r2:peppercarrot-images | head` and compare to pages.image_url in your local DB.
# 403 → the bucket's public access isn't enabled; R2 dashboard → bucket → Settings.

# ─── 4. Fly: the backend booted, seeded, and serves the API ───
curl https://peppercarrot-companion.fly.dev/health
# Want: {"status":"ok"}.
curl -s https://peppercarrot-companion.fly.dev/api/episodes | python3 -m json.tool | head -30
# Want: a non-empty JSON array of episode objects, each with cover_image_url
# that starts with $R2_PUBLIC_URL_PREFIX.
# [] (empty) → dump_seed.sh ran against an empty local Postgres; re-ingest at least Episode 1 and re-deploy.
# 500 → check `fly logs --no-tail | tail -50`; almost always the asyncpg / unpooled URL caveat from Step 2.

# ─── 5. End-to-end chat from the terminal (the actual user flow) ───
SID=$(curl -s -X POST https://peppercarrot-companion.fly.dev/api/sessions \
  -H 'content-type: application/json' \
  -d '{"episode_slug":"ep01-potion-of-flight"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
curl -s -X PATCH "https://peppercarrot-companion.fly.dev/api/sessions/$SID" \
  -H 'content-type: application/json' -d '{"current_page":1}'
# -N streams; the first request after Modal idle takes 15–30s (expected).
curl -N -X POST "https://peppercarrot-companion.fly.dev/api/sessions/$SID/messages" \
  -H 'content-type: application/json' \
  -d '{"mode":"page","message":"who is on this page?"}'
# Want: a token stream that lands a coherent answer, followed by a `done` SSE
# frame with retrieved_doc_ids and two suggestion chips.
# Silence then 502 → Modal cold-start exceeded the 180s timeout; refresh and try once more.
# 401 → Modal proxy tokens aren't reaching qwen2.5:7b; check `fly logs` for the upstream 401.

# ─── 6. Cloudflare Pages: the deployed frontend reaches Fly cleanly ───
# Open the *.pages.dev URL in a browser.
# DevTools → Network → confirm /api/* requests go to peppercarrot-companion.fly.dev (NOT localhost).
# DevTools → Console → confirm no CORS errors.
# If you see "Access to fetch at ... has been blocked by CORS policy" — the
# Pages URL doesn't match CORS_ORIGINS on Fly. `fly secrets set` it to match exactly.
```

### What "tested" means here, honestly

I did not deploy the workshop-tagged code to live provider accounts before writing this post. What I did verify:

- The unit tests pass (43/43 against the local suite; the R2Storage smoke test confirms the boto3 client builds and `url_for` composes correctly).
- The four Protocol seams from [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) carry the production values through to where they're consumed — `OllamaChatClient._headers` forwards Modal auth into both `stream()` and `complete()`; `OllamaEmbeddingClient` does the same; `R2Storage.url_for()` is the only call site `world_graph.py` and `episodes.py` need; the asyncpg sslmode shim in `db/session.py` handles Neon's URL format.
- The `infra/` scripts and `docs/deployment.md` operational details are derived from a working deploy of this same architecture against the same five providers; the workshop's versions are simplified for the narrower scope.

That gives me high confidence the workshop deploy works. But high confidence is not deployed-and-confirmed, and you should treat the six checks above as the experiment that turns one into the other before you put the URL in front of anyone.

If something breaks, the failure mode is concrete (a Docker layer that doesn't build, a `fly logs` traceback, an asyncpg error, a CORS console message) and the fix is one of the rows in [`docs/deployment.md`'s troubleshooting table](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/docs/deployment.md#troubleshooting). The seams are designed to fail fast and name themselves; the recovery is documented per failure mode; what's not documented is the failure I haven't seen, which is the one you might find first. If you do, the post can be patched.

---

## Key Takeaways {#key-takeaways}

**1. The seams worth abstracting are the ones whose implementation changes between dev and prod.** [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) named three (chat, embedding, storage) and abstracted them; Posts 14–15 swapped all three implementations and changed *no code outside `clients/`*. The corollary is just as important — Chroma was the seam that *didn't* earn a Protocol because it had no swap target, and Posts 14–15 honored that by baking it into the image rather than hedging against a problem the project doesn't have.

**2. Pick the right tier per shape.** A static frontend wants a CDN; a bursty I/O backend wants scale-to-zero containers; a stateful database wants managed Postgres that sleeps; large static images want object storage; a GPU workload wants serverless GPU. Run all five on one VPS and you pay the worst-case cost of all five combined. Fan out and each provider gets paid only for what it actually serves. The architecture pattern is the same whether the budget is $10/mo or $10k/mo; only the tier-within-tier choices differ.

**3. The cheapest GPU at portfolio scale is the one that isn't always running.** Modal's `min_containers=0, scaledown_window=300` is the workshop default for a reason: the demo's traffic shape is bursty visitors with hours of idle in between. Paying $0 idle and a 15-second cold start on the first request of the day is a much better deal than paying $430/mo to keep a T4 warm. Picking the right point on the cost-vs-latency curve is part of the design judgement the portfolio is supposed to show.

**4. asyncpg, prepared statements, and pgbouncer in transaction mode don't mix.** This is the single most common first-deploy failure mode against managed Postgres. Neon gives you two endpoints — pooled and unpooled — and the backend's async driver wants the unpooled one. The seam from [`db/session.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/backend/app/db/session.py) that translates `?sslmode=require` into `connect_args` is two helpful lines that save an hour of debugging; design the config layer so operators don't have to know the difference.

**5. Bake small, stream big.** The Docker image carries the venv (~200 MB), the app code, `data/chroma`, `data/world-graph`, and `data/seed.sql` — a few hundred MB total. The episode page images (~700 MB) go to R2. The split is operational: anything baked is replaced on the next deploy, anything in object storage is incremental. **Don't bake what you'd be re-uploading every push.**

**6. Idempotent first-boot scripts beat one-shot CI steps for demos.** The `entrypoint.sh` checks for the `episodes` table via `information_schema` and conditionally restores `seed.sql`. A hundred container restarts; one seed restore. No CI step needed; no manual coordination between deploy steps; the application self-heals on a fresh Neon database. The cost is rebuild-time (Docker has to re-bake `seed.sql` whenever it changes); the benefit is operational simplicity, which is what a demo deploy wants.

**7. Build-time env vars are an under-used seam.** The Cloudflare Pages frontend reads `VITE_API_BASE_URL` at *build time*, not at runtime — so the deployed bundle calls the Fly URL directly without ever having to discover it. The frontend's source carries one line of `import.meta.env.VITE_API_BASE_URL ?? '/api'`, the dev path keeps the Vite proxy, and the prod path knows the absolute URL of the backend. Same code, different env. **The cost of doing this right in the first place is one ternary expression; the cost of doing it wrong is a build-time secret you discover at deploy time.**

**8. The portfolio framing is "knowing what to abstract, knowing what to leave alone, knowing what to pay for."** The series chose three Protocols on purpose; left Chroma raw on purpose; ignored CI/CD on purpose; declined the always-warm GPU on purpose. Each of those is a *no* that strengthens the architecture by saying what the project is for. A reviewer who's deployed real systems before recognizes the no's and reads them as judgement, not omission.

**9. Cold starts are honest about themselves.** A 15-second first answer is what you get when you optimize for `$0` idle. Hiding it behind a session-creation warmup is fine production polish; pretending it isn't there is not. The workshop ships without the warmup so this post can be honest about where the cost sits — and so the architecture is clearly *positioned* to bolt the warmup on as ~30 lines of follow-up code. **The architecture should make the trade-off explicit; the UX layer can choose how to surface it.**

**10. The whole deploy is roughly 40 minutes of typing once you know the steps.** ~3 min for `modal deploy`, ~5 min for `fly deploy`, ~2 min for the Pages connect-and-build, plus the time spent in three web UIs setting up accounts. The new code in this post is ~80 lines of `R2Storage`. **The rest is configuration, scripts, and the discipline of having decided what to abstract two months ago.** That ratio is the whole point.

---

## Appendix: Serverless, Workers, and the Cloudflare Edge {#appendix}

Three concepts run implicitly through the post that are worth unpacking explicitly for readers new to cloud architecture. None of them are required to follow the deploy steps — they're the *why* behind several of the choices.

### What "serverless" actually means

Despite the name, **"serverless" does not mean "no servers."** Servers are very much still involved — somebody else's. What changes is *how you pay for them and how much of their lifecycle you manage*:

- **Traditional server** (a VPS like DigitalOcean, an AWS EC2 instance) — you rent a fixed amount of compute by the hour or month. You pay whether or not anyone is using your app. You're responsible for boot, patching, scaling, log rotation, security updates, the works.
- **Serverless** (Modal, Fly's scale-to-zero containers, Neon's serverless Postgres, Cloudflare Workers) — you describe what your function or service needs (a container image, a GPU, a database, a request handler). The provider allocates the hardware when a request arrives, runs your code, and releases the hardware after a short idle window. You pay per second of *active* use plus a tiny scheduling overhead. **Idle = $0** (or close to it — usually some pennies-per-month for the artifacts stored at rest).

The trade-off is **cold-start latency**: when the first request after idle arrives, the provider has to allocate hardware and load your code before it can answer. Cold starts range from a few *milliseconds* (Cloudflare Workers, with their V8 isolate-based runtime) to a few *seconds* (Fly's Firecracker VMs booting from cold) to *tens of seconds* (Modal allocating a GPU and loading a 7B model into VRAM). For bursty traffic with hours of idle in between — like a portfolio demo — serverless wins on cost by orders of magnitude. For sustained traffic with no idle gaps, the always-on rent-by-the-hour model wins.

> *The serverless spectrum.* "Serverless" is a marketing umbrella covering several different patterns. From smallest cold start to largest:
> - **Edge functions / Workers** (Cloudflare Workers, Vercel Edge Functions, AWS Lambda@Edge) — your code runs in a small JavaScript/WASM runtime at the CDN edge. Cold starts in *milliseconds*. Geographically distributed by default. No persistent state across invocations.
> - **Functions-as-a-Service / FaaS** (AWS Lambda, Google Cloud Functions, Azure Functions) — your code runs in a container the provider warms up on demand. Cold starts in *hundreds of milliseconds*. Region-bound.
> - **Serverless containers** (Fly.io, Google Cloud Run, AWS Fargate) — your *whole* Docker container runs on demand, with scale-to-zero. Cold starts in *seconds* (the VM/container has to boot). Better for stateful or long-running workloads than FaaS.
> - **Serverless GPU** (Modal, Replicate, Runpod) — same shape as serverless containers but with GPU allocation in the loop. Cold starts in *tens of seconds* (allocate GPU + load model weights into VRAM).
> - **Serverless databases** (Neon, PlanetScale, Cloudflare D1) — managed databases that suspend their compute when idle. Cold starts of *~1 second* (compute resume; the storage was always there).
>
> This post's stack uses three of the five tiers — serverless containers (Fly), serverless GPU (Modal), and serverless data (Neon) — plus a CDN for static assets (Pages) and object storage (R2). The Workers tier doesn't appear in our stack; the architectural reason is the next section.

### Cloudflare Workers, and why we don't use them

We use two Cloudflare products in this stack — **Pages** for the static frontend and **R2** for the image bytes — but Cloudflare's broader ecosystem includes a third you'll see referenced a lot: **[Cloudflare Workers](https://workers.cloudflare.com/)**.

Workers are *edge functions*. You write a JavaScript or TypeScript function that handles HTTP requests; Cloudflare runs it on every one of their ~300 edge nodes worldwide, in a [V8-isolate-based runtime](https://blog.cloudflare.com/cloud-computing-without-containers/) that boots in roughly a millisecond. They're cheap (~$5/mo for the first 10 million requests), geographically distributed by default, and the right tool for *stateless transformation of HTTP requests* — authentication, URL rewrites, A/B testing, simple JSON APIs, things that don't need to hold state across requests.

The reason this post's backend runs on Fly instead of Workers is **what the backend actually is**:

- **The FastAPI app is a stateful Python process.** It holds an open SQLAlchemy engine pool (per the [`db/session.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/backend/app/db/session.py) lifespan), a ChromaDB persistent client (loaded into RAM at startup), and a long-lived streaming connection to Ollama for each in-flight chat answer. Workers don't run Python (only JS/TS/WASM), and their per-request execution model doesn't fit a process that wants to hold state.
- **The dependency tree is large and binary-rich.** `chromadb`, `sentence-transformers`, `boto3`, `asyncpg` — the whole venv is ~200 MB. Workers' isolate runtime is sized for scripts under a megabyte.
- **SSE streams need persistent TCP connections** to a single backend that holds state across the lifetime of the stream. Workers can do response streaming, but pairing it with the server-side state of the streaming chat orchestrator (the token-by-token answer, the second non-streaming call for suggestion chips) is exactly the shape that wants a container, not an edge function.

If the backend were a stateless TypeScript REST API with no model dependencies, Workers would be the obvious choice — cheaper, more geographically distributed, no cold start to speak of. For a Python LLM backend, a serverless *container* on Fly is the right tier. **Picking the right tier of serverless for what your code actually is, is half the architectural skill.**

### The broader Cloudflare product ecosystem

Worth a quick orientation since two of the five providers in this stack are Cloudflare products and the broader ecosystem is one of the more coherent in the industry — all share one account, one dashboard, one billing surface:

- **[Pages](https://pages.cloudflare.com/)** — static site hosting (this post: the React frontend).
- **[R2](https://www.cloudflare.com/developer-platform/products/r2/)** — S3-compatible object storage with no egress fees (this post: the image bytes).
- **[Workers](https://workers.cloudflare.com/)** — edge functions (this post: not used; see above).
- **[Workers AI](https://developers.cloudflare.com/workers-ai/)** — inference for a curated set of models on Cloudflare's GPU pool. Could in principle replace Modal for the chat call, at a different price/performance trade-off; doesn't yet ship the specific `qwen2.5:7b` + `bge-m3` combination this project relies on, and the model catalogue is more curated than the "pull any Ollama model" pattern.
- **[D1](https://developers.cloudflare.com/d1/)** — serverless SQLite at the edge. Different cost shape from Neon's Postgres; SQLite's feature set doesn't include the row-value comparison the world-graph spoiler filter in [Post 12]({% post_url 2026-05-30-pepper-carrot-companion-spoiler-safe-world-graph %}) relies on (`tuple_(episode_debut, page_debut) <= cursor`).
- **[KV](https://developers.cloudflare.com/kv/)** — globally-distributed key-value store. Read-heavy, eventually-consistent. Useful for config and feature flags; not the right shape for chat-session state with hard consistency requirements.
- **[Durable Objects](https://developers.cloudflare.com/durable-objects/)** — single-threaded stateful objects at the edge, addressable by ID. Could be the right shape for chat-session state in a Workers-based reimplementation; outside scope here.

The reason this post uses Pages + R2 (not the full Cloudflare stack end-to-end) is the same reason it uses Fly instead of Workers: **the application's runtime shape — Python, stateful, GPU-dependent — points at a different tier of serverless than what Cloudflare's compute products optimise for**. For a *different* application — say, a TypeScript reading companion calling an external LLM API — the same five-piece architecture could deploy end-to-end on one provider (`Workers + Workers AI + D1 + KV + R2`) for less money and less operational surface. **Architecture choice is downstream of what the code actually is.**

> *A small naming gotcha.* "Workers" is also the term Cloudflare uses for the *runtime* their other products are built on. So you'll see "Pages Functions" described as "Workers", "Workers AI" referred to as "running on Workers", and so on. When someone says "I deployed a Worker," they usually mean a standalone edge function (the product); when they say "running on Workers," they often mean the runtime layer underneath multiple products. Same word, two zoom levels.

---

## The Series, End to End {#series-arc}

Sixteen posts, one architecture, one workshop. The arc started at [Post 1]({% post_url 2026-05-09-pepper-carrot-companion-trailer %}) with a question — *can a small, local-first LLM read a comic with you?* — and lands here, at a public URL costing roughly the price of a coffee a month (with Post 16 showing a no-GPU alternative deploy on managed APIs). The intermediate posts each shipped one durable affordance and one defensible architectural decision:

- **[Post 2]({% post_url 2026-05-10-pepper-carrot-companion-workshop %}) — Setting Up the Workshop: Postgres, Ollama, and a Project That Type-Checks.** Postgres, Ollama, FastAPI scaffold, the first Alembic migration, one episode downloaded. The empty room into which everything else lands.
- **[Post 3]({% post_url 2026-05-11-pepper-carrot-companion-data-model %}) — The Data Model: Ten Tables, One Migration.** The schema that mirrors the app's features, the first migration read line by line, a column-by-column tour of the SQLAlchemy models.
- **[Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) — Provider Abstractions: Why Every External Service Hides Behind an Interface.** Three Protocols (`ChatClient`, `EmbeddingClient`, `Storage`) and the discipline of "what to abstract and what to leave alone." Posts 14–15 cashed every one of these.
- **[Post 5]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}) — Claude Skills as a Vision Provider: Ingesting a Comic by Reading It.** The `ingest-from-images` skill that turns Claude Code into a one-shot author of durable JSON page descriptions. The pattern showed up again in Posts 12–13, twice.
- **[Post 6]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}) — The Ingestion Pipeline: From Page JSONs to Postgres + Chroma.** The Stage-2 pipeline that consumes the skill's page descriptions and loads one full episode into Postgres + Chroma + storage.
- **[Post 7]({% post_url 2026-05-23-pepper-carrot-companion-rest-api-flipbook %}) — From Database to JSON: A Typed REST API.** Two typed FastAPI routes, the storage seam composing absolute URLs at response time. The seam that swapped local for R2 in Post 14 was the line of code in `episodes.py` that called `await storage.url_for(key)`.
- **[Post 8]({% post_url 2026-05-23-pepper-carrot-companion-rest-api-flipbook %}) — A Real Flipbook in the Browser: React + StPageFlip.** An episode picker plus a real page-flipping flipbook rendering real data from the local backend.
- **[Post 9]({% post_url 2026-05-25-pepper-carrot-companion-spoiler-safe-rag %}) — The RAG Layer: Spoiler-Safe Retrieval Without Trusting the Prompt.** The Chroma `where` clause built from server-side reading progress. The spoiler boundary as a property of retrieval, not of prompts. Pinned by tests with a jailbreak query.
- **[Post 10]({% post_url 2026-05-25-pepper-carrot-companion-streaming-chat %}) — Streaming Chat in the Browser: SSE, React, and Schema-Constrained Suggestion Chips.** SSE, the named-slot schema for chips, the server-side question-shape validator. Structural guarantees in the data layer; UX polish in the prompt.
- **[Post 11]({% post_url 2026-05-30-pepper-carrot-companion-prompt-hardening %}) — Making Small Models Behave: Wiki Mode and the Long Road to Concise Answers.** A second retrieval mode for universe lore, plus the prompt-engineering toolkit that takes a chatty 7B model from "essay-style replies with section headers" to clean prose.
- **[Post 12]({% post_url 2026-05-30-pepper-carrot-companion-spoiler-safe-world-graph %}) — A World Graph Built by a Skill: Extraction and a Spoiler-Safe API.** A second Claude Code skill that walks the wiki + page JSONs into a YAML graph; the row-value comparison that gates entities and edges in SQL behind a spoiler-safe API route.
- **[Post 13]({% post_url 2026-05-30-pepper-carrot-companion-spoiler-safe-world-graph %}) — Rendering the World Graph: A React-Flow Overlay and Summary-First Wiki.** An avatar-node overlay with kind-based SVG fallbacks; a third skill that authors per-entity wiki summaries so qwen2.5:7b sees ~500 words, not 30 KB.
- **[Post 14]({% post_url 2026-05-31-pepper-carrot-companion-shipping-it %}) — Going to Production: Provisioning Modal, Neon, and R2.** The five-provider architecture stood up: GPU-served Ollama on Modal, managed Postgres on Neon, image bytes on Cloudflare R2 behind the Post 4 storage interface, and the container that bakes the small data. The Post 4 abstractions cashing in.
- **Post 15 — this one. Shipping It: Containerize, Deploy to Fly + Pages, and Verify.** The backend to Fly, the frontend to Pages, the whole thing verified end to end. Five services, one container, ~$10/mo, a public URL.
- **[Post 16]({% post_url 2026-06-01-pepper-carrot-companion-skip-the-gpu %}) — Skip the GPU: A Managed-API Deploy on Anthropic + Voyage.** The same app shipped without a GPU at all — chat on the Anthropic API, embeddings on Voyage — as a config change, not a code change.

The single thread connecting all sixteen is *put the load-bearing decisions in the data and structure layers; let the model and the UX be the polish on top*. The spoiler boundary lives in retrieval, not prompts. The provider implementations live behind Protocols, not factories of factories. The world graph lives in Postgres rows, not in a model call. The deploy lives in five small configurations, not in a Kubernetes manifest. **Each layer keeps its own responsibilities; each layer earns its own honesty about what it can and can't promise.**

---

Next up: **Post 16 — Skip the GPU: A Managed-API Deploy on Anthropic + Voyage.** Everything above keeps the series' local-first thesis — qwen2.5:7b on a serverless GPU. The final post shows the other path: the same app with *no GPU at all*, chat on the Anthropic API and embeddings on Voyage, reached by flipping `CHAT_PROVIDER` and `EMBEDDING_PROVIDER` and adding two API keys. It's the cleanest possible proof that the [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) abstractions deliver what they promised: a provider swap is a config change, not a code change.

The **workshop starter** at <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop> tagged `post-14-15-deploy` is what backs this post; clone it, follow the README's deploy steps, and you'll have the same architecture running against your own free-tier accounts in under an hour.

*Pepper & Carrot* is © [David Revoy](https://www.davidrevoy.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All credit to him for the source material that made this entire project possible.

**All opinions expressed are my own.**
