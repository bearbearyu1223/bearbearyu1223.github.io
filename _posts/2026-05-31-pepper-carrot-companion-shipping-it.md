---
title: "Pepper & Carrot AI-powered flipbook · Part 14 of 16 — Going to Production: Provisioning Modal, Neon, and R2"
date: 2026-05-31 12:00:00 -0800
categories: [Full-Stack, RAG, Local AI]
tags: [deployment, modal, neon, cloudflare-r2, docker, fastapi, peppercarrot, portfolio]
description: >-
  Post 14 of the Pepper & Carrot AI flipbook series — the provisioning
  half of the deploy. The flipbook, the spoiler-safe RAG, the world
  graph all run beautifully on the developer laptop the first twelve
  posts built around. This one stands up the three stateful backing
  services the cloud build needs — Modal for the GPU-served Ollama,
  Neon for managed Postgres, Cloudflare R2 for the image bytes — and
  builds the two-stage container that bakes the small data and streams
  the big data. The provider abstractions from Post 4 finally cash in:
  the backend doesn't notice that Ollama moved off localhost, the
  storage swap is one env var, the database URL is one secret. The new
  code is small (a boto3-backed R2Storage finally lands behind the
  Post 4 Protocol, a Dockerfile, three short infra scripts) — the
  harder work is the architectural judgement about which seams to draw
  and which five services to fan out across. Post 15 takes the
  container public.
pin: true
---

Post 14 of the [*Pepper & Carrot AI-powered flipbook*]({% post_url 2026-05-09-pepper-carrot-companion-trailer %}) series — the first of two deploy posts. The previous twelve built a local-first reading companion on a developer laptop: a flipbook with stPageFlip in [Post 8]({% post_url 2026-05-23-pepper-carrot-companion-rest-api-flipbook %}), a spoiler-safe RAG layer in [Post 9]({% post_url 2026-05-25-pepper-carrot-companion-spoiler-safe-rag %}), a streaming chat panel with suggestion chips in [Post 10]({% post_url 2026-05-25-pepper-carrot-companion-streaming-chat %}), a prompt-hardened answer surface in [Post 11]({% post_url 2026-05-30-pepper-carrot-companion-prompt-hardening %}), and a spoiler-aware world graph overlay in [Posts 12–13]({% post_url 2026-05-30-pepper-carrot-companion-spoiler-safe-world-graph %}). Everything runs against `localhost:11434` (Ollama), `localhost:5432` (Postgres), and the filesystem (images). This post takes the same architecture and starts pushing it onto the public internet at a price point a portfolio demo can sustain: typically $5 to $15 a month, almost all of it Modal GPU seconds, everything else on free tiers. It provisions the three stateful backing services (Modal, Neon, R2) and builds the container; [Post 15]({% post_url 2026-06-01-pepper-carrot-companion-deploy-verify %}) deploys that container to Fly, ships the frontend to Cloudflare Pages, and verifies the whole thing. The interesting part is not the typing. It's that the typing is small, because the abstractions from [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) were designed for exactly this seam-by-seam migration, and the runtime never notices the change.

> **What you'll build in this post.**
> - **A boto3-backed `R2Storage` implementation** in `backend/app/clients/storage.py` that finally fills in the [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) Protocol. boto3 is imported lazily inside the constructor so the workshop's default local path doesn't need it; synchronous calls run through `asyncio.to_thread` so the FastAPI event loop never blocks on a network round-trip. The runtime's read path only ever touches `url_for()` — a string compose — because the image bytes were uploaded by `rclone` during deploy.
> - **A two-stage `Dockerfile`** that builds the venv once (cached layer), copies the app code, and bakes the small data assets (`data/seed.sql`, `data/chroma`, `data/world-graph`) into the image. Episode page images are *not* baked — they ship to R2.
> - **A `fly.toml` Fly app config** plus an `infra/entrypoint.sh` that restores `data/seed.sql` into a fresh Neon database on the first boot (idempotent via an `information_schema` existence check) and then exec's uvicorn. A `512 MB` shared-CPU machine, `auto_stop_machines = 'stop'`, scale-to-zero.
> - **An `infra/modal_ollama.py` Modal deployment** that runs Ollama serving `qwen2.5:7b` and `bge-m3` on a serverless T4 GPU. Persistent volume holds the model weights across cold starts (~6 GB once, then survives forever); `scaledown_window = 300` keeps the container warm for five minutes after the last request. Proxy-auth on by default so the URL alone isn't the secret.
> - **An `infra/dump_seed.sh`** one-liner that `pg_dump`s the local Postgres into `data/seed.sql` with `--no-owner --no-acl --no-privileges` (Neon's role differs from the local one). Re-run before every `fly deploy` whenever ingestion has changed the DB.
> - **An `.env.production.example`** carrying the 11 values every secret on Fly maps to — `DATABASE_URL_OVERRIDE`, `POSTGRES_RESTORE_URL`, `OLLAMA_BASE_URL`, the two Modal proxy tokens, the four R2 creds, `R2_PUBLIC_URL_PREFIX`, `CORS_ORIGINS` — with inline comments explaining the asyncpg-vs-pgbouncer caveat that breaks the unwary.
> - **A `docs/deployment.md`** that is the step-by-step operational reference, and **`docs/decisions/0004-cloud-deployment.md`** that captures the why — including the alternatives weighed (one VPS, Vercel + Supabase + Replicate, Fly's hosted Postgres) and the trade-offs each one made.
>
> **Prerequisites.**
> - The workshop starter at the [`post-14-15-deploy` tag](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/tree/post-14-15-deploy): `git checkout post-14-15-deploy` (see [Following along with the blog series](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop#following-along-with-the-blog-series)). Everything [Posts 12–13]({% post_url 2026-05-30-pepper-carrot-companion-spoiler-safe-world-graph %}) needed — Postgres up, migrations applied, at least Episode 1 ingested, the wiki summaries ingested, the world-graph YAML loaded — running end-to-end locally before you reach for cloud.
> - Free-tier accounts on [Neon](https://neon.tech), [Cloudflare](https://dash.cloudflare.com), and [Modal](https://modal.com). (You'll also want a [Fly.io](https://fly.io) account for Post 15's deploy.)
> - CLIs: `brew install rclone` and `uv tool install modal` (or `pipx install modal`). (Add `brew install flyctl` for Post 15.)
> - A domain or custom DNS records is **not** required — every service ships with a working free subdomain (`*.r2.dev`, `*.modal.run`, and in Post 15 `*.fly.dev` and `*.pages.dev`).

> **About the repo URL.** Everything in this post — `Dockerfile`, `.env.production.example`, the `infra/` directory, the boto3-backed `R2Storage`, `docs/deployment.md`, and `docs/decisions/0004-cloud-deployment.md` — lives in the same workshop starter that backed [Posts 2–13](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop), now tagged `post-14-15-deploy`. File links below point at that tag. This deploy checkpoint is shared with [Post 15]({% post_url 2026-06-01-pepper-carrot-companion-deploy-verify %}), which takes the container public; together the two posts make the workshop end-to-end reproducible — you can clone, ingest, provision, and deploy without leaving this single repository.

---

## Table of Contents

1. [The Code in Front of You: Tour + Quick Start](#tour)
2. [What This Adds, and What It Doesn't](#what-this-adds)
3. [Meet the Five Providers](#providers)
4. [Why Five Services, Not One](#why-five)
5. [The Pipeline, End to End](#diagram)
6. [Five Seams Designed in Post 4, Cashed in Posts 14–15](#seams)
7. [Modal: Serverless GPU for Ollama](#modal)
8. [Neon: The Two Connection Strings](#neon)
9. [Cloudflare R2: The Implementation That Finally Landed](#r2)
10. [The Container: Bake Small Data, Stream Big Data](#dockerfile)

---

## The Code in Front of You: Tour + Quick Start {#tour}

The whole point of the deploy is to put a URL in the hands of a recruiter. Skim this section even if you read the rest carefully: watching the chat answer the same question from a `*.pages.dev` URL that you watched it answer from `localhost:5173` two posts ago is the entire payoff of the abstractions.

### Get the code at this post's tag

```bash
git clone https://github.com/bearbearyu1223/pepper-carrot-companion-workshop
cd pepper-carrot-companion-workshop
git checkout post-14-15-deploy
```

Already cloned from an earlier post? `git fetch --tags && git checkout post-14-15-deploy`.

### What's new in the workshop starter

Three changes to existing files (one of them load-bearing — `R2Storage` finally lands), seven new files, and one new ADR:

```
pepper-carrot-companion-workshop/
├── Dockerfile                       ← NEW (Posts 14–15): two-stage Python build
├── fly.toml                         ← NEW (Posts 14–15): Fly app config + env block
├── .env.production.example          ← NEW (Posts 14–15): 11 values mapping to Fly secrets
├── .dockerignore                    ← NEW (Posts 14–15): keep build context tiny
├── .gitignore                       ← updated: .env.production + data/seed.sql
├── infra/
│   ├── modal_ollama.py              ← NEW (Posts 14–15): serverless Ollama on Modal T4
│   ├── entrypoint.sh                ← NEW (Posts 14–15): psql-restore on first boot, then uvicorn
│   └── dump_seed.sh                 ← NEW (Posts 14–15): pg_dump local → data/seed.sql
├── backend/
│   ├── app/clients/storage.py       ← UPDATED: R2Storage put/exists/url_for finally implemented
│   └── pyproject.toml               ← updated: boto3 mypy override
├── docs/
│   ├── deployment.md                ← NEW (Posts 14–15): step-by-step reference
│   └── decisions/
│       └── 0004-cloud-deployment.md ← NEW (Posts 14–15): ADR for the five-service split
├── README.md                        ← updated: post-14-15-deploy entry, Step 12 deploy block
└── CLAUDE.md                        ← updated: scope expanded to include cloud deploy
```

The diff is roughly 600 lines, of which the only new runtime code is the boto3-backed `R2Storage` — eighty lines of the kind of code Post 4 promised would be local-only. Everything else is configuration, scripts, and documentation. That ratio is intentional. The portfolio signal of these two deploy posts is not "I learned Docker"; it's "the abstractions from Post 4 made deploying a five-service architecture mostly a configuration exercise."

### Deploy it: roughly forty minutes, mostly waiting on builds

The full step-by-step is in [`docs/deployment.md`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/docs/deployment.md). The shape is:

```bash
# 0. One-time tooling.
brew install flyctl rclone
uv tool install modal

# 1. Fill in 11 values.
cp .env.production.example .env.production
$EDITOR .env.production

# 2. Deploy Ollama on Modal.
modal token new
modal deploy infra/modal_ollama.py   # ~3 min: pulls 6 GB of weights into a volume

# 3. Provision Neon (web UI: console.neon.tech → New project),
#    then copy the pooled + unpooled URLs into .env.production.

# 4. Provision R2 (web UI: dash.cloudflare.com → R2 → Create bucket),
#    then upload the image bytes. --exclude flags keep macOS Finder's
#    .DS_Store junk out and skip the 2 MB -original.jpg source files
#    the frontend never reads.
find data/images data/world-graph/images -name .DS_Store -delete
rclone copy data/images r2:peppercarrot-images --progress \
    --exclude ".DS_Store" --exclude "**/.DS_Store" \
    --exclude "**/*-original.jpg"
rclone copy data/world-graph/images r2:peppercarrot-images/world-graph/images --progress \
    --exclude ".DS_Store" --exclude "**/.DS_Store"

# 5. Dump the local Postgres so the container can restore it on first boot.
./infra/dump_seed.sh

# 6. Fly: launch + secrets + deploy.
fly auth login
fly launch --no-deploy --copy-config --name peppercarrot-companion
set -a && source .env.production && set +a && fly secrets set \
  DATABASE_URL_OVERRIDE="$DATABASE_URL_OVERRIDE" \
  POSTGRES_RESTORE_URL="$POSTGRES_RESTORE_URL" \
  OLLAMA_BASE_URL="$OLLAMA_BASE_URL" \
  MODAL_PROXY_TOKEN_ID="$MODAL_PROXY_TOKEN_ID" \
  MODAL_PROXY_TOKEN_SECRET="$MODAL_PROXY_TOKEN_SECRET" \
  R2_ACCOUNT_ID="$R2_ACCOUNT_ID" \
  R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
  R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  R2_BUCKET="$R2_BUCKET" \
  R2_PUBLIC_URL_PREFIX="$R2_PUBLIC_URL_PREFIX" \
  CORS_ORIGINS="$CORS_ORIGINS"
fly deploy                            # ~5 min on first deploy

# 7. Cloudflare Pages: connect repo via the dashboard, set
#    VITE_API_BASE_URL=https://peppercarrot-companion.fly.dev,
#    build = `cd frontend && npm install && npm run build`,
#    output = frontend/dist.
```

Step 7 prints a `*.pages.dev` URL. Open it in a browser. The flipbook loads, you pick an episode, you ask a question. The first answer takes 15–30 seconds because Modal is cold; subsequent ones are immediate. The same UI you were running against `localhost:8000` two minutes ago is now answering from three separate clouds.

### Validate it from the terminal

Belt-and-suspenders: confirm each tier separately before debugging the integration.

```bash
# Modal — endpoint up, both models pulled.
set -a && source .env.production && set +a
curl -sS -H "Modal-Key: $MODAL_PROXY_TOKEN_ID" \
        -H "Modal-Secret: $MODAL_PROXY_TOKEN_SECRET" \
        "$OLLAMA_BASE_URL/api/tags" | python3 -m json.tool | head
# {"models": [{"name": "qwen2.5:7b", ...}, {"name": "bge-m3", ...}]}

# Fly — backend serves the health route + the episodes API.
curl https://peppercarrot-companion.fly.dev/health
# {"status":"ok"}
curl -s https://peppercarrot-companion.fly.dev/api/episodes | head -c 200
# JSON: an array of episodes with absolute R2 cover URLs.

# R2 — at least one image is publicly readable from the bucket prefix.
curl -I "$R2_PUBLIC_URL_PREFIX/world-graph/images/carrot-thumb.webp"
# HTTP/2 200, content-type: image/webp, cache-control: public, max-age=...
```

If all three return what the comments predict, the integration is live. If one of them fails, you've narrowed the problem to a single tier without having to read three log streams. The troubleshooting table at the bottom of [`docs/deployment.md`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/docs/deployment.md#troubleshooting) lists the eight failure modes that account for ~95% of first-deploy issues — most of them about the asyncpg-vs-pgbouncer-vs-Neon-pooler interaction that's [Step 7 of the deploy guide](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/docs/deployment.md#step-2--provision-neon-postgres).

---

## What This Adds, and What It Doesn't {#what-this-adds}

Twelve posts shipped one affordance each. Posts 14–15 are the ones that take everything those twelve built and put a public URL in front of it.

| | Affordance | Built locally in | Shipped publicly by Posts 14–15 |
|---|---|---|---|
| **Post 8** | Episode flipbook | Vite + StPageFlip | Cloudflare Pages |
| **Post 9** | Spoiler-safe page chat | Ollama + Chroma + the spoiler boundary | Modal (Ollama) + image-baked Chroma + the same boundary |
| **Post 10** | Streaming SSE + suggestion chips | FastAPI + Ollama | Fly + Modal — SSE works through Fly's proxy |
| **Post 11** | Prompt hardening | `core/prompts.py` + `react-markdown` | Unchanged at the seam; runs on the Modal-served qwen2.5:7b |
| **Posts 12–13** | World-graph overlay | Postgres + react-flow | Neon (Postgres) + Cloudflare-served avatar art |
| **Posts 14–15** | **A public URL** | n/a | the workshop's `post-14-15-deploy` tag |

Three things this post **isn't**:

- **It isn't a Kubernetes tutorial.** No clusters, no Helm charts, no service meshes. Five providers, one container per provider's idiom. The portfolio framing is "I picked the right tier for each component" — not "I operated a control plane."
- **It isn't a CI/CD walkthrough.** The deploy is `fly deploy` from a developer's laptop. Wiring up GitHub Actions to run `dump_seed.sh` and push on every merge to `main` is a few hours' work, but it's a separate kind of post and the brief's scope is the architecture. Adding it is a one-day follow-up project that consumes nothing from the existing code.
- **It isn't a "make it scale" post.** A 512 MB Fly machine with `min_machines_running = 0` is sized for portfolio traffic — visitors arriving in ones and twos, sometimes hours apart. The cold-start trade-off [Post 15]({% post_url 2026-06-01-pepper-carrot-companion-deploy-verify %}) covers is the *entire scaling story*. Building toward "always warm at any load" needs different numbers (always-on Modal containers cost ~$430/mo on a T4), and the demo wouldn't pay for it.

The architectural through-line of the series, in one sentence: **the seams worth abstracting are the ones whose implementation changes between dev and prod.** Post 4 named three (chat, embedding, storage), abstracted them behind Protocols, and shipped local-only implementations. Posts 5–13 wrote everything else against those Protocols and made spoiler safety a property of retrieval. Posts 14–15 ship the production implementations of the three Protocols and change *no code outside `clients/`* to pick them up. That's the payoff.

---

## Meet the Five Providers {#providers}

If you've deployed a web app before, this section is "skim and continue," since every provider below has a recognisable analog you've worked with. If some of these names are new, the paragraph each is what you actually need to know to follow the rest of the post, and the deep-dive sections later go further on the specific features the architecture uses.

**[Cloudflare Pages](https://pages.cloudflare.com/)** — A free static-site host. You give it a GitHub repo, it builds your frontend on every push, and serves the resulting JS/CSS/HTML from servers worldwide (a *content delivery network*, or CDN — servers placed in many countries so the bytes are physically close to whoever's loading them). Free for unlimited bandwidth, capped at a few hundred builds per month — generous for anything portfolio-shaped. *Closest analogs:* GitHub Pages, Vercel, Netlify.

**[Fly.io](https://fly.io/)** — A container-hosting platform. You hand Fly a Docker image and a small `fly.toml` config; Fly runs that image as a lightweight virtual machine (built on Amazon's open-source Firecracker tech) in regions you pick, and gives you a public `*.fly.dev` URL. The feature that matters for portfolio cost: **scale-to-zero** — the machine sleeps when nobody's using it, wakes on the next request, so a sleepy demo costs roughly $0. The free monthly allowance covers a small backend at portfolio traffic; you only pay if usage exceeds the free tier. *Closest analogs:* Render, Railway, AWS Fargate.

**[Neon](https://neon.tech/)** — A managed Postgres database. Postgres is the world's most-used relational database; "managed" means Neon runs it for you, handles backups, hands you a connection string, and stops at "be a database." Neon's specific innovation is separating storage from compute, so the database can suspend its compute when idle (you stop paying for it) and resume on the next query in about a second — the same cost shape as Fly applied to a database. Because it's just Postgres, every Postgres client (asyncpg, the official `psql` CLI) and every extension works unchanged. Free tier: 0.5 GB of storage. *Closest analogs:* Supabase, AWS RDS Serverless, PlanetScale (which speaks MySQL instead).

**[Cloudflare R2](https://www.cloudflare.com/developer-platform/products/r2/)** — Object storage. "Object storage" means a bucket you throw arbitrary files into and read back over HTTPS — typically used for images, videos, and other large static assets that don't fit cleanly in a database. R2 is API-compatible with [AWS S3](https://aws.amazon.com/s3/) (the original and still-dominant object-storage service) but charges **$0 for egress** — the bytes read out of the bucket. Egress is usually the biggest line on an S3 bill once a bucket gets traffic; for image-heavy portfolio sites it can be the difference between $0/mo and $30/mo. Storage itself is free for the first 10 GB. Because the API is S3-compatible, every S3 client (boto3, rclone, the AWS CLI) works against R2 with a one-line `endpoint_url` override. *Closest analogs:* AWS S3, Backblaze B2, Wasabi.

**[Modal](https://modal.com/)** — Serverless GPU. A GPU is the specialised chip a language model needs to run quickly; renting one by the hour starts around $0.20/hr ($150+/mo always-on) on most clouds. Modal's pitch is to allocate a GPU only while a request needs it, run the function, and release the hardware after a short configurable idle window — *per-second billing* instead of per-hour. You describe what your function needs in a Python file (a Docker image, a GPU tier like T4 or A10G, the idle window) and Modal handles the orchestration. For a portfolio demo where the model runs maybe ten seconds per visitor in bursts hours apart, the cost shape comes out to typically $5–10/month instead of $150+. *Closest analogs:* Replicate, Runpod, Banana, AWS SageMaker Serverless.

A common theme across all five: **revenue scales with usage, and idle usage costs near-zero**. The portfolio shape — bursty visitors arriving in ones and twos, with hours of nothing in between — is exactly the load shape these free tiers were designed around. The architecture this post describes works at ~$10/month not because we negotiated discounts but because the providers were built to make small idle workloads cost nothing. The flip side: at sustained product-scale traffic, the same providers cost the same as their always-on competitors. You pick scale-to-zero when bursty traffic is the design target, and you'd pick differently if it weren't.

Now the architectural argument — why *these specific five*, rather than running everything on one server.

---

## Why Five Services, Not One {#why-five}

The most natural first instinct for "deploy this thing" is one box: rent a VPS, `docker-compose up`, point a domain, done. It would work. It also fails the portfolio framing in a subtle way that's worth naming.

The application doesn't have one shape. It has five shapes, and they conflict:

- **The frontend is static** — built once, served from edge nodes worldwide, no per-request work. The right hosting shape is a CDN.
- **The backend is bursty but I/O-bound** — long idle stretches between requests, each request does ~1 SQL query plus a model call. The right hosting shape is a container that scales to zero.
- **Postgres is stateful** — needs persistence across deploys, idle 99% of the time at portfolio scale. The right hosting shape is *managed* Postgres that itself sleeps when idle.
- **The image bytes are large and static** — never change once authored, but a *lot* of them. The right hosting shape is object storage with a CDN front.
- **The AI models need a GPU** — only when actually answering a question, and even then for ten seconds at a time. The right hosting shape is serverless GPU.

Run all five on one VPS and you pay the worst-case cost of all five combined: the box has to be sized for the *peak* of each component. The minimum useful GPU-equipped instance starts at roughly $0.20/hr ($150/mo always-on), and CPU-only inference at 7B is slow enough that the streaming UX from Post 10 would feel broken, with the first token landing in tens of seconds instead of one.

Fan out instead and each provider gets paid only for what it actually serves. Idle ≈ $0 on every tier except the Modal model-weights volume (~$1/mo). The same code runs; only the URLs change.

> *Plain-English aside: scale-to-zero.* When a service is idle, the provider shuts the machine down and you stop paying. The next request triggers a **cold start** — the time to allocate hardware and become ready to answer. Fly's cold start is a Firecracker VM boot (~5–10 s). Modal's is "allocate a GPU and load the model weights into VRAM" (~15–25 s after the first deploy). For a portfolio demo where visitors arrive in ones and twos, paying $0 idle and a 15-second cold start on the first request of the day is a much better deal than paying $150/mo to keep one GPU warm.

The five-service split also gives the application **five separate failure boundaries**. A Modal cold start doesn't break the picker; an R2 outage doesn't break the chat; a Neon maintenance window doesn't take the frontend down. That's not a design goal for a portfolio demo, but it is a property the architecture inherits for free, and it's the kind of property a recruiter who's deployed a real system once recognizes immediately. The full alternatives-considered analysis is in [`docs/decisions/0004-cloud-deployment.md`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/docs/decisions/0004-cloud-deployment.md).

---

## The Pipeline, End to End {#diagram}

One picture for the whole deploy. Notice that the boxes the request flows through don't change shape between dev (top) and prod (bottom); only the URLs do. The provider abstractions from Post 4 are the seams the colored arrows cross, and the runtime code on either side of the seam is identical.

<div style="margin: 1.5rem 0; overflow-x: auto;">
<a href="/assets/picture/2026-05-31-pepper-carrot-companion-shipping-it/deploy-architecture.svg" target="_blank" rel="noopener" title="Open the diagram full-size in a new tab" style="display: block; cursor: zoom-in;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 800" role="img"
     aria-label="The deploy architecture, two tiers. Dev tier (top): the browser at localhost:5173 calls the Vite dev server which proxies to the FastAPI backend at localhost:8000; the backend fans out to three local resources — Postgres on port 5432, Ollama on port 11434, and the filesystem at data/ (which holds both image bytes and the Chroma persistent directory). The three fan-out arrows from FastAPI to the right column are highlighted as the Post 4 provider-Protocol seams. A dashed seam line separates the tiers. Prod tier (bottom): the browser at your-app.pages.dev calls Cloudflare Pages (static CDN) which proxies to Fly.io for the FastAPI backend, which fans out to Neon (Postgres with asyncpg sslmode shim), Modal (T4 GPU serving Ollama with qwen2.5:7b and bge-m3), and Chroma which is baked into the Docker image in-process. Cloudflare R2 sits separately above the main row and is fetched directly by the browser (the URL is composed at API-response time by R2Storage.url_for and the image bytes never transit Fly). A cost summary box at the bottom shows the per-service monthly cost — Cloudflare Pages, Neon, and R2 are free at portfolio scale; Fly is zero to two dollars; Modal is five to ten dollars; total five to fifteen dollars per month."
     style="display: block; width: 100%; height: auto; max-width: 1100px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">
  <defs>
    <marker id="d-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>
    </marker>
    <marker id="d-arrow-seam" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#b45309"/>
    </marker>
    <marker id="d-arrow-dim" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/>
    </marker>
  </defs>

  <!-- ── DEV TIER ───────────────────────────────────────────────────────── -->
  <text x="20" y="28" font-size="13" font-weight="700" fill="#1e40af" font-style="italic">DEV (Posts 2–13) · one laptop, the same Protocols</text>

  <!-- Left row: Browser → Vite → FastAPI -->
  <g>
    <rect x="40" y="80" width="180" height="60" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/>
    <text x="130" y="106" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">Browser</text>
    <text x="130" y="124" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">localhost:5173</text>
  </g>

  <g>
    <rect x="260" y="80" width="180" height="60" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/>
    <text x="350" y="106" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">Vite dev server</text>
    <text x="350" y="124" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">/api + /images proxy</text>
  </g>

  <g>
    <rect x="480" y="80" width="180" height="60" rx="6" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.3"/>
    <text x="570" y="106" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">FastAPI backend</text>
    <text x="570" y="124" text-anchor="middle" font-size="10" fill="#92400e" font-style="italic">localhost:8000</text>
  </g>

  <!-- Right column: three backing services, generously spaced -->
  <g>
    <rect x="740" y="55" width="220" height="44" rx="6" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.3"/>
    <text x="850" y="74" text-anchor="middle" font-size="11" font-weight="600" fill="#1f2937">Postgres :5432</text>
    <text x="850" y="90" text-anchor="middle" font-size="9" fill="#92400e" font-style="italic">docker-compose</text>
  </g>

  <g>
    <rect x="740" y="109" width="220" height="44" rx="6" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.3"/>
    <text x="850" y="128" text-anchor="middle" font-size="11" font-weight="600" fill="#1f2937">Ollama :11434</text>
    <text x="850" y="144" text-anchor="middle" font-size="9" fill="#92400e" font-style="italic">qwen2.5:7b + bge-m3</text>
  </g>

  <g>
    <rect x="740" y="163" width="220" height="44" rx="6" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.3"/>
    <text x="850" y="182" text-anchor="middle" font-size="11" font-weight="600" fill="#1f2937">Filesystem · data/</text>
    <text x="850" y="198" text-anchor="middle" font-size="9" fill="#92400e" font-style="italic">images + chroma (in-process)</text>
  </g>

  <!-- Left-row arrows -->
  <line x1="220" y1="110" x2="258" y2="110" stroke="#6b7280" stroke-width="1.5" marker-end="url(#d-arrow)"/>
  <line x1="440" y1="110" x2="478" y2="110" stroke="#6b7280" stroke-width="1.5" marker-end="url(#d-arrow)"/>

  <!-- Fan-out from FastAPI to the three backing services (the Post 4 seam) -->
  <line x1="660" y1="95"  x2="738" y2="77"  stroke="#b45309" stroke-width="1.6" marker-end="url(#d-arrow-seam)"/>
  <line x1="660" y1="110" x2="738" y2="131" stroke="#b45309" stroke-width="1.6" marker-end="url(#d-arrow-seam)"/>
  <line x1="660" y1="125" x2="738" y2="185" stroke="#b45309" stroke-width="1.6" marker-end="url(#d-arrow-seam)"/>

  <!-- Single seam label, well clear of the arrows -->
  <text x="370" y="240" font-size="11" fill="#7c2d12" font-style="italic" font-weight="600">★ The three amber arrows cross the Post 4 seam: ChatClient · EmbeddingClient · Storage</text>

  <!-- Post 4 seam dividing line -->
  <line x1="40" y1="265" x2="1060" y2="265" stroke="#b45309" stroke-width="1" stroke-dasharray="6,3"/>
  <text x="20" y="263" font-size="10" fill="#7c2d12" font-style="italic" font-weight="600">★ Post 4 seam (Protocol)</text>
  <text x="550" y="263" text-anchor="middle" font-size="11" fill="#7c2d12" font-style="italic" font-weight="600">— same Protocols, different implementations below —</text>
  <text x="1040" y="263" text-anchor="end" font-size="10" fill="#7c2d12" font-style="italic" font-weight="600">★ Post 4 seam</text>

  <!-- ── PROD TIER ──────────────────────────────────────────────────────── -->
  <text x="20" y="298" font-size="13" font-weight="700" fill="#065f46" font-style="italic">PROD (Posts 14–15) · five providers, one container, ~$10/mo</text>

  <!-- R2 elevated above the main row — accessed directly by the browser -->
  <g>
    <rect x="740" y="325" width="220" height="50" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="1.6"/>
    <text x="850" y="346" text-anchor="middle" font-size="12" font-weight="700" fill="#7c2d12">Cloudflare R2 · images</text>
    <text x="850" y="364" text-anchor="middle" font-size="9" fill="#7c2d12" font-style="italic">pub-XXXX.r2.dev (CDN, public-read)</text>
  </g>

  <!-- Browser → R2 — a short, unambiguous curve up to the elevated box -->
  <path d="M 220 410 C 360 410, 540 350, 738 350"
        stroke="#b45309" stroke-width="1.6" stroke-dasharray="5,3" fill="none"
        marker-end="url(#d-arrow-seam)"/>
  <text x="480" y="365" text-anchor="middle" font-size="10" fill="#7c2d12" font-style="italic">browser fetches images directly · URL composed by R2Storage.url_for()</text>

  <!-- Main row: Browser → Pages → Fly -->
  <g>
    <rect x="40" y="395" width="180" height="60" rx="6" fill="#d1fae5" stroke="#059669" stroke-width="1.3"/>
    <text x="130" y="420" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">Browser</text>
    <text x="130" y="438" text-anchor="middle" font-size="10" fill="#065f46" font-style="italic">your-app.pages.dev</text>
  </g>

  <g>
    <rect x="260" y="395" width="180" height="60" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="1.6"/>
    <text x="350" y="420" text-anchor="middle" font-size="12" font-weight="700" fill="#7c2d12">Cloudflare Pages</text>
    <text x="350" y="438" text-anchor="middle" font-size="10" fill="#7c2d12" font-style="italic">static CDN (free)</text>
  </g>

  <g>
    <rect x="480" y="395" width="180" height="60" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="1.6"/>
    <text x="570" y="420" text-anchor="middle" font-size="12" font-weight="700" fill="#7c2d12">Fly.io · FastAPI</text>
    <text x="570" y="438" text-anchor="middle" font-size="10" fill="#7c2d12" font-style="italic">512 MB · scale-to-zero</text>
  </g>

  <!-- Right column: Neon, Modal, Chroma -->
  <g>
    <rect x="740" y="395" width="220" height="44" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="1.5"/>
    <text x="850" y="414" text-anchor="middle" font-size="11" font-weight="700" fill="#7c2d12">Neon · Postgres</text>
    <text x="850" y="430" text-anchor="middle" font-size="9" fill="#7c2d12" font-style="italic">asyncpg + sslmode shim</text>
  </g>

  <g>
    <rect x="740" y="449" width="220" height="44" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="1.5"/>
    <text x="850" y="468" text-anchor="middle" font-size="11" font-weight="700" fill="#7c2d12">Modal · Ollama (T4 GPU)</text>
    <text x="850" y="484" text-anchor="middle" font-size="9" fill="#7c2d12" font-style="italic">qwen2.5:7b + bge-m3</text>
  </g>

  <g>
    <rect x="740" y="503" width="220" height="44" rx="6" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="850" y="522" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Chroma (baked)</text>
    <text x="850" y="538" text-anchor="middle" font-size="9" fill="#92400e" font-style="italic">in-process in the image</text>
  </g>

  <!-- Main-row arrows: Browser → Pages → Fly -->
  <line x1="220" y1="425" x2="258" y2="425" stroke="#6b7280" stroke-width="1.5" marker-end="url(#d-arrow)"/>
  <line x1="440" y1="425" x2="478" y2="425" stroke="#6b7280" stroke-width="1.5" marker-end="url(#d-arrow)"/>

  <!-- Fan-out from Fly to Neon / Modal / Chroma (same seam as DEV) -->
  <line x1="660" y1="415" x2="738" y2="417" stroke="#b45309" stroke-width="1.6" marker-end="url(#d-arrow-seam)"/>
  <line x1="660" y1="430" x2="738" y2="471" stroke="#b45309" stroke-width="1.6" marker-end="url(#d-arrow-seam)"/>
  <line x1="660" y1="445" x2="738" y2="525" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="3,3" marker-end="url(#d-arrow-dim)"/>

  <text x="550" y="572" text-anchor="middle" font-size="11" fill="#7c2d12" font-style="italic" font-weight="600">★ Same three Protocols — Modal speaks the Ollama API; R2Storage absorbs the bucket URL.</text>

  <!-- ── COST SUMMARY ───────────────────────────────────────────────────── -->
  <g>
    <rect x="40" y="600" width="1020" height="180" rx="8" fill="#fef9c3" stroke="#ca8a04" stroke-width="1"/>
    <text x="550" y="626" text-anchor="middle" font-size="13" font-weight="700" fill="#713f12">Cost at portfolio traffic · typical monthly</text>

    <text x="80" y="654" font-size="11" font-weight="600" fill="#1f2937">Cloudflare Pages</text>
    <text x="260" y="654" font-size="11" fill="#475569">free</text>
    <text x="350" y="654" font-size="9" fill="#94a3b8" font-style="italic">(static CDN, unlimited bandwidth)</text>

    <text x="80" y="676" font-size="11" font-weight="600" fill="#1f2937">Fly.io</text>
    <text x="260" y="676" font-size="11" fill="#475569">$0 – $2</text>
    <text x="350" y="676" font-size="9" fill="#94a3b8" font-style="italic">(free monthly allowance covers a sleepy 512 MB)</text>

    <text x="80" y="698" font-size="11" font-weight="600" fill="#1f2937">Neon</text>
    <text x="260" y="698" font-size="11" fill="#475569">free</text>
    <text x="350" y="698" font-size="9" fill="#94a3b8" font-style="italic">(0.5 GB tier — plenty for the demo)</text>

    <text x="80" y="720" font-size="11" font-weight="600" fill="#1f2937">Cloudflare R2</text>
    <text x="260" y="720" font-size="11" fill="#475569">free</text>
    <text x="350" y="720" font-size="9" fill="#94a3b8" font-style="italic">(10 GB free tier, no egress fees)</text>

    <text x="80" y="742" font-size="11" font-weight="600" fill="#1f2937">Modal</text>
    <text x="260" y="742" font-size="11" fill="#7c2d12" font-weight="600">$5 – $10</text>
    <text x="350" y="742" font-size="9" fill="#94a3b8" font-style="italic">(GPU seconds + ~$1/mo volume — the dominant cost item)</text>

    <line x1="60" y1="758" x2="1040" y2="758" stroke="#a16207" stroke-width="0.8"/>
    <text x="550" y="774" text-anchor="middle" font-size="12" font-weight="700" fill="#7c2d12">Total · $5 to $15 / month</text>
  </g>
</svg>
</a>
</div>

*Two tiers, the same Protocols on each. The amber-bordered boxes below the dashed seam line are what changed; the seams themselves were drawn in Post 4. Click the diagram to open it full-size in a new tab.*

> *Diagram for the live demo.* When walking a recruiter through this, a useful second diagram is a **sequence diagram of the first request after idle**: browser → Pages → Fly → (Fly cold-start ~8 s) → Modal → (Modal cold-start ~20 s) → first SSE token. It makes the cold-start tax legible and turns "the first answer is slow" into a story you control rather than a thing the demo apologizes for.

### How the Pieces Talk: One Chat Question, End to End

The diagram above shows *where the boxes live*. This one shows the conversation between them, and it's simpler than five clouds makes it sound. Almost everything routes through **three wires**, and each wire is a single config value:

- **Browser ↔ Fly** — the frontend talking to the backend.
- **Fly ↔ Neon** — the backend reading the database.
- **Fly ↔ Modal** — the backend calling the AI models.

A fourth wire — **Browser → R2** — sits off to the side: the page images are fetched straight from the bucket and never touch Fly. Here is the order the wires fire in when a reader types a question and hits send.

<div style="margin: 1.5rem 0; overflow-x: auto;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 660" role="img"
     aria-label="A sequence diagram of one chat question, end to end, across four lifelines: Browser (your-app.pages.dev), Fly (FastAPI), Neon (Postgres), and Modal (Ollama GPU). Step 1, the browser POSTs to the Fly backend at /api/sessions/{id}/messages over Server-Sent-Events with the mode and message. Step 2, Fly selects the chat session row from Neon to get the reader's current episode and page; Neon returns the spoiler-boundary integers. Step 3, Fly asks Modal to embed the question with bge-m3, sending the Modal-Key and Modal-Secret proxy-auth headers; Modal returns the query vector. Step 4, Fly queries Chroma, which is baked into the Docker image and runs in-process, for the nearest chunk IDs filtered by the spoiler boundary. Step 5, Fly fetches the chunk text from Neon by those IDs. Step 6, Fly calls Modal for a streaming chat completion with qwen2.5:7b. Step 7, Modal streams tokens back through Fly to the browser as SSE frames and the answer renders live. Step 8, Fly asks Modal for two follow-up suggestion chips. The arrows are colour-coded by wire: green for browser-to-Fly, blue for Fly-to-Neon, amber for Fly-to-Modal, grey for the in-process Chroma call."
     style="display: block; width: 100%; height: auto; max-width: 1080px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">
  <defs>
    <marker id="s-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#059669"/>
    </marker>
    <marker id="s-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
    <marker id="s-amber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#b45309"/>
    </marker>
    <marker id="s-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>
    </marker>
  </defs>

  <!-- ── Lifeline headers ─────────────────────────────────────────────────── -->
  <g>
    <rect x="55" y="18" width="150" height="46" rx="6" fill="#d1fae5" stroke="#059669" stroke-width="1.3"/>
    <text x="130" y="40" text-anchor="middle" font-size="12" font-weight="700" fill="#1f2937">Browser</text>
    <text x="130" y="56" text-anchor="middle" font-size="9" fill="#065f46" font-style="italic">your-app.pages.dev</text>
  </g>
  <g>
    <rect x="345" y="18" width="150" height="46" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="1.3"/>
    <text x="420" y="40" text-anchor="middle" font-size="12" font-weight="700" fill="#7c2d12">Fly · FastAPI</text>
    <text x="420" y="56" text-anchor="middle" font-size="9" fill="#7c2d12" font-style="italic">scale-to-zero</text>
  </g>
  <g>
    <rect x="625" y="18" width="150" height="46" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/>
    <text x="700" y="40" text-anchor="middle" font-size="12" font-weight="700" fill="#1e3a8a">Neon · Postgres</text>
    <text x="700" y="56" text-anchor="middle" font-size="9" fill="#1e40af" font-style="italic">unpooled · asyncpg</text>
  </g>
  <g>
    <rect x="875" y="18" width="150" height="46" rx="6" fill="#fde68a" stroke="#b45309" stroke-width="1.3"/>
    <text x="950" y="40" text-anchor="middle" font-size="12" font-weight="700" fill="#7c2d12">Modal · Ollama</text>
    <text x="950" y="56" text-anchor="middle" font-size="9" fill="#7c2d12" font-style="italic">T4 GPU · proxy-auth</text>
  </g>

  <!-- ── Lifelines ────────────────────────────────────────────────────────── -->
  <line x1="130" y1="64" x2="130" y2="612" stroke="#cbd5e1" stroke-width="1.2"/>
  <line x1="420" y1="64" x2="420" y2="612" stroke="#cbd5e1" stroke-width="1.2"/>
  <line x1="700" y1="64" x2="700" y2="612" stroke="#cbd5e1" stroke-width="1.2"/>
  <line x1="950" y1="64" x2="950" y2="612" stroke="#cbd5e1" stroke-width="1.2"/>

  <!-- ① Browser → Fly -->
  <text x="132" y="96" font-size="10.5" font-weight="600" fill="#065f46">① POST /api/sessions/{id}/messages — SSE · {mode, message}</text>
  <line x1="130" y1="104" x2="418" y2="104" stroke="#059669" stroke-width="1.6" marker-end="url(#s-green)"/>

  <!-- ② Fly → Neon, return -->
  <text x="422" y="140" font-size="10.5" font-weight="600" fill="#1e3a8a">② load the chat-session row → current (episode, page)</text>
  <line x1="420" y1="148" x2="698" y2="148" stroke="#2563eb" stroke-width="1.6" marker-end="url(#s-blue)"/>
  <text x="698" y="174" text-anchor="end" font-size="9.5" fill="#1e40af" font-style="italic">spoiler-boundary integers (server-side, never from the user)</text>
  <line x1="700" y1="182" x2="422" y2="182" stroke="#2563eb" stroke-width="1.2" stroke-dasharray="5,3" marker-end="url(#s-blue)"/>

  <!-- ③ Fly → Modal embed, return -->
  <text x="422" y="216" font-size="10.5" font-weight="600" fill="#7c2d12">③ embed the question · bge-m3  (+ Modal-Key / Modal-Secret headers)</text>
  <line x1="420" y1="224" x2="948" y2="224" stroke="#b45309" stroke-width="1.6" marker-end="url(#s-amber)"/>
  <text x="948" y="250" text-anchor="end" font-size="9.5" fill="#7c2d12" font-style="italic">query vector</text>
  <line x1="950" y1="258" x2="422" y2="258" stroke="#b45309" stroke-width="1.2" stroke-dasharray="5,3" marker-end="url(#s-amber)"/>

  <!-- ④ Fly self-call: Chroma in-process -->
  <path d="M 420 292 L 470 292 L 470 306 L 422 306" fill="none" stroke="#6b7280" stroke-width="1.4" marker-end="url(#s-gray)"/>
  <text x="478" y="300" font-size="10.5" font-weight="600" fill="#374151">④ Chroma (baked into the image, in-process): nearest chunk IDs, filtered by the boundary</text>

  <!-- ⑤ Fly → Neon fetch text, return -->
  <text x="422" y="338" font-size="10.5" font-weight="600" fill="#1e3a8a">⑤ fetch the chunk text · SELECT … WHERE id IN (…)</text>
  <line x1="420" y1="346" x2="698" y2="346" stroke="#2563eb" stroke-width="1.6" marker-end="url(#s-blue)"/>
  <text x="698" y="372" text-anchor="end" font-size="9.5" fill="#1e40af" font-style="italic">grounding text (the canonical copy lives in Postgres)</text>
  <line x1="700" y1="380" x2="422" y2="380" stroke="#2563eb" stroke-width="1.2" stroke-dasharray="5,3" marker-end="url(#s-blue)"/>

  <!-- ⑥ Fly → Modal chat, stream return -->
  <text x="422" y="414" font-size="10.5" font-weight="600" fill="#7c2d12">⑥ chat completion · qwen2.5:7b (streaming, prompt + grounding)</text>
  <line x1="420" y1="422" x2="948" y2="422" stroke="#b45309" stroke-width="1.6" marker-end="url(#s-amber)"/>
  <text x="948" y="448" text-anchor="end" font-size="9.5" fill="#7c2d12" font-style="italic">tokens →  →  →</text>
  <line x1="950" y1="456" x2="422" y2="456" stroke="#b45309" stroke-width="1.2" stroke-dasharray="5,3" marker-end="url(#s-amber)"/>

  <!-- ⑦ Fly → Browser SSE stream -->
  <text x="132" y="490" font-size="10.5" font-weight="600" fill="#065f46">⑦ SSE token frames re-streamed to the browser — the answer renders live as it arrives</text>
  <line x1="420" y1="498" x2="132" y2="498" stroke="#059669" stroke-width="1.6" stroke-dasharray="5,3" marker-end="url(#s-green)"/>

  <!-- ⑧ Fly → Modal suggestions -->
  <text x="422" y="532" font-size="10.5" font-weight="600" fill="#7c2d12">⑧ generate two follow-up suggestion chips → final SSE frame</text>
  <line x1="420" y1="540" x2="948" y2="540" stroke="#b45309" stroke-width="1.6" marker-end="url(#s-amber)"/>

  <!-- ── Legend ───────────────────────────────────────────────────────────── -->
  <line x1="55" y1="572" x2="1025" y2="572" stroke="#e2e8f0" stroke-width="1"/>
  <g font-size="10">
    <line x1="60" y1="592" x2="92" y2="592" stroke="#059669" stroke-width="2" marker-end="url(#s-green)"/>
    <text x="98" y="595" fill="#374151">Browser ↔ Fly — <tspan font-style="italic">VITE_API_BASE_URL</tspan> + CORS</text>
    <line x1="360" y1="592" x2="392" y2="592" stroke="#2563eb" stroke-width="2" marker-end="url(#s-blue)"/>
    <text x="398" y="595" fill="#374151">Fly ↔ Neon — <tspan font-style="italic">DATABASE_URL_OVERRIDE</tspan></text>
    <line x1="660" y1="592" x2="692" y2="592" stroke="#b45309" stroke-width="2" marker-end="url(#s-amber)"/>
    <text x="698" y="595" fill="#374151">Fly ↔ Modal — <tspan font-style="italic">OLLAMA_BASE_URL</tspan> + tokens</text>
  </g>
  <text x="60" y="612" font-size="9.5" fill="#94a3b8" font-style="italic">Page images take a fourth wire not shown here: the browser fetches them straight from R2's public URL — they never transit Fly.</text>
</svg>
</div>

*One question, eight hops, three wires. The grey self-call (④) is the only step that stays inside Fly — Chroma is baked into the container, so the vector search is a function call, not a network round-trip.*

Each wire is exactly one config value, and that's the whole "how does it connect" story:

- **Browser → Fly (frontend ↔ backend).** At build time, Cloudflare Pages inlines `VITE_API_BASE_URL=https://…fly.dev` into the JavaScript, so the shipped bundle calls your Fly URL instead of `localhost:8000`. Fly answers cross-origin requests only because `CORS_ORIGINS` lists the exact `*.pages.dev` URL. The chat request is a `POST` that streams back over Server-Sent-Events — the browser's built-in `EventSource` can't `POST`, so [`streamMessage`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/frontend/src/api/client.ts) reads the response body as a stream and parses the `event:` / `data:` frames by hand (hops ① and ⑦).

- **Fly → Neon (backend ↔ database).** The `DATABASE_URL_OVERRIDE` secret points the async engine at Neon's *unpooled* endpoint (hops ② and ⑤). It has to be unpooled because asyncpg uses prepared statements and Neon's pgbouncer pooler hands each query to a different backend that's never seen them; the [Seam 4](#seams) `sslmode`-to-`ssl` shim lives on this wire too. The load-bearing detail: the reader's position, the integers that become the spoiler boundary, comes from the session **row** (②), never from the user's message, so there is nothing in the prompt for a jailbreak to widen.

- **Fly → Modal (backend ↔ models).** The `OLLAMA_BASE_URL` secret points at the `*.modal.run` endpoint, and every request carries the `Modal-Key` / `Modal-Secret` proxy-auth headers so the URL alone isn't the secret. It's the *same Ollama HTTP API* as `localhost:11434`, which is why this is a URL swap, not a rewrite ([Seams 2 & 3](#seams)). One question hits Modal up to three times: embed (③), chat (⑥), and the suggestion chips (⑧). The first one after idle eats the cold start; the rest land within the 5-minute warm window.

And the **fourth wire** keeps the heavy bytes off the backend entirely: the database stores image *keys* like `episodes/ep01-…/pages/001-display.webp`, [`R2Storage.url_for()`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/backend/app/clients/storage.py) composes them into `https://pub-XXXX.r2.dev/…` at API-response time, and the browser fetches each image directly from R2's CDN. Fly composes a string; R2 serves the megabytes.

---

## Five Seams Designed in Post 4, Cashed in Posts 14–15 {#seams}

[Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) named the abstraction discipline that made this deploy possible: three Protocols (`ChatClient`, `EmbeddingClient`, `Storage`), a factory in `clients/__init__.py`, and a config object that toggles the implementation per env var. The promise was that the rest of the codebase imports the Protocol, the factory chooses the implementation, and swapping local for cloud is a config flip. These two deploy posts are where that promise is tested.

Five concrete seams; each one's "cash in" call is one or two lines.

**Seam 1 — `Storage`: `LocalStorage` → `R2Storage`.** The factory's `if/elif/else` already had the branch ready since [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}). The implementation it pointed at was the [`R2Storage`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/backend/app/clients/storage.py) class with `raise NotImplementedError` in its body. Post 14 fills it in. Eighty lines of boto3 wrapper plus an `asyncio.to_thread` around each network-bound call, and zero changes outside `clients/storage.py`. The route handlers that compose URLs via `await storage.url_for(key)` don't know there's a CDN involved.

**Seam 2 — `ChatClient`: `OllamaChatClient(localhost:11434)` → `OllamaChatClient(*.modal.run)`.** Not even a class swap — *same class, different URL*. Ollama on Modal speaks the same HTTP API Ollama on `localhost:11434` speaks, because it *is* Ollama. The single new wrinkle is the proxy-auth headers Modal adds (the `Modal-Key` / `Modal-Secret` pair) so the URL isn't itself the secret, and even that was anticipated in the [`clients/__init__.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/backend/app/clients/__init__.py) factory back in Post 4, with a `_modal_proxy_headers` helper that translates `MODAL_PROXY_TOKEN_ID` + `MODAL_PROXY_TOKEN_SECRET` env vars into the right header dict if both are set:

```python
# backend/app/clients/__init__.py (excerpted; from Post 4)
def _modal_proxy_headers(settings: Settings) -> dict[str, str]:
    """Modal proxy-auth headers when both tokens are set; empty otherwise.

    Setting only one of the two is a config error — fail loudly so the
    operator notices before requests start 401-ing in production.
    """
    if settings.modal_proxy_token_id and settings.modal_proxy_token_secret:
        return {
            "Modal-Key": settings.modal_proxy_token_id,
            "Modal-Secret": settings.modal_proxy_token_secret,
        }
    if settings.modal_proxy_token_id or settings.modal_proxy_token_secret:
        raise RuntimeError(
            "Modal proxy auth requires BOTH modal_proxy_token_id and "
            "modal_proxy_token_secret to be set."
        )
    return {}
```

That "fail loudly when half-configured" rule is the kind of guardrail that has zero value on the first day and infinite value on the day you accidentally roll-back one of the two secrets and your prod app is silently 401-ing. **Design a config object that knows its own coupling constraints.**

**Seam 3 — `EmbeddingClient`: same shape as Seam 2.** `OllamaEmbeddingClient(localhost:11434)` → `OllamaEmbeddingClient(*.modal.run)` with the same proxy-auth headers. The factory uses the same `_modal_proxy_headers(settings)` call. The `RetrievalService` from Post 9 never notices.

**Seam 4 — Postgres URL: `localhost:5432` → Neon's unpooled endpoint.** The `database_url_override` setting on the `Settings` class lands the full Neon URL straight through. The one subtlety is in [`backend/app/db/session.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/backend/app/db/session.py): SQLAlchemy's asyncpg dialect forwards unknown URL query params as kwargs to `asyncpg.connect()`, which accepts `ssl=` but not `sslmode=`. Neon's connection-string UI gives you `?sslmode=require` (the libpq spelling). The `_extract_ssl_connect_args` helper pops the param off the URL and translates it into the `connect_args` dict asyncpg understands. This is the same shape of seam — Post 4's data model said "the runtime cares about a `database_url`," and the production environment hands us a slightly different dialect of URL, so the seam absorbs the dialect difference.

**Seam 5 — Chroma is the one that *isn't* abstracted.** The series' provider-abstraction discipline ([Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}#what-deserves-an-abstraction)) explicitly excluded Chroma: it's the single vector store, not a provider with a local/cloud alternative to swap between, so it didn't earn a Protocol. Post 14 honors that. Chroma's persistent directory is baked into the Docker image at `data/chroma/` and the `RetrievalService` reads it via the same `chromadb.PersistentClient(path=...)` call it used at `localhost`. The trade-off is operational: re-ingesting episodes means a re-deploy (the `data/chroma/` layer of the image rebuilds, picking up the new vectors), which is fine at portfolio cadence and would not be at real product cadence. The honesty there is that abstracting Chroma to a hosted service would have been a hedge against a problem we don't have, and the Post 4 discipline said no to that hedge on purpose. Post 14 doesn't second-guess it.

The five seams together are roughly 20 lines of code change outside `R2Storage` itself. The rest of the deploy is configuration. That's the abstraction story this post exists to tell, and it's the part recruiters who've deployed real systems recognize immediately.

---

## Modal: Serverless GPU for Ollama {#modal}

The most exotic of the five services is Modal, and it's the one doing the most architectural work: replacing a GPU you'd otherwise have to rent by the hour with one allocated on demand.

> *Plain-English aside: what does "serverless GPU" actually mean?* On a normal cloud GPU (DigitalOcean, Lambda Labs, your favourite VPS), you rent the GPU by the hour or month. It's always running; you always pay; it doesn't care whether anyone's using it. **Serverless GPU** flips that. You hand the provider a container; they allocate a GPU only when a request needs one; you pay for active seconds plus a short idle window after each burst. When nobody's looking at your demo, the bill is approximately $0. The cost is the **cold start** — the time between a request arriving and the GPU being ready to answer (~15–25 s on Modal for `qwen2.5:7b` after the first deploy). For a portfolio demo where visitors arrive in bursts hours apart, this is an excellent trade: zero idle cost, slow first answer, fast subsequent answers within the 5-minute warm window.

The whole Modal deployment is one Python file, [`infra/modal_ollama.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/infra/modal_ollama.py). Modal's discipline is unusual (the deployment description and the runtime entrypoint are the same Python file), and that makes for a very dense ~30 lines:

```python
# infra/modal_ollama.py (abridged)
import modal

OLLAMA_PORT = 11434
CHAT_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "bge-m3"

app = modal.App("peppercarrot-ollama")

# Persistent volume — weights survive across cold starts so we only
# pay the download cost on the first deploy.
models_volume = modal.Volume.from_name(
    "peppercarrot-ollama-models", create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
)

@app.function(
    image=image,
    gpu="T4",                     # 16 GB VRAM; sufficient for 7b + embeddings
    volumes={"/root/.ollama": models_volume},
    scaledown_window=300,         # stay warm 5 min after the last request
    timeout=600,
    min_containers=0,             # scale-to-zero when idle
)
@modal.web_server(
    port=OLLAMA_PORT,
    startup_timeout=600,
    requires_proxy_auth=True,
)
def serve() -> None:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"0.0.0.0:{OLLAMA_PORT}"
    subprocess.Popen(["ollama", "serve"], env=env)
    # wait for /api/tags to respond, then pull both models, then commit the volume
```

Five things in there are worth naming:

- **`gpu="T4"`** is the cheapest Modal GPU. 16 GB of VRAM is enough for a 7B model with room left over for the embeddings model and a small context window. Upgrading to `"L4"` or `"A10G"` doubles or triples throughput but doubles the per-second cost; for a single-user demo, T4 is the right pick. Picking the right GPU tier for the load is half the cost-tuning work; the other half is `scaledown_window`.
- **`scaledown_window=300`** says "keep the container warm for 5 minutes after the last request." Shorter = more cold starts, less idle cost. Longer = fewer cold starts, more idle cost. 300 is the goldilocks number for a portfolio demo: a recruiter who clicks the link, asks two questions over two minutes, and walks away keeps the GPU warm for both questions and costs almost nothing.
- **`min_containers=0`** is scale-to-zero. Setting it to `1` keeps one container *always* warm — no cold starts, but ~$430/mo for the always-on T4. For a portfolio demo with bursty traffic that's a strict loss; for sustained traffic (a real product), it can be worth it.
- **`requires_proxy_auth=True`** turns on Modal's header-pair authentication. Without it, the deployed URL is itself the only secret, and anyone who finds it can run up the bill. The factory in [`backend/app/clients/__init__.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/backend/app/clients/__init__.py) reads `MODAL_PROXY_TOKEN_ID` + `MODAL_PROXY_TOKEN_SECRET` and translates them into a `{"Modal-Key": ..., "Modal-Secret": ...}` header dict that both `OllamaChatClient` and `OllamaEmbeddingClient` accept on construction. This is the seam from Post 4 cashing in.
- **The persistent volume** at `/root/.ollama` is where Ollama caches model weights. The first deploy pulls qwen2.5:7b (~4.7 GB) and bge-m3 (~1.2 GB) into the volume; subsequent cold starts skip the download and only pay the VRAM-load cost (~15–25s). Without the volume, every cold start would re-download 6 GB of weights — which would push cold-start latency over a minute and the deploy would feel broken.

Deploy it once:

```bash
modal token new                         # one-time browser auth
modal deploy infra/modal_ollama.py
```

The first deploy takes ~3 minutes, mostly the model download. The output prints a URL of the form `https://<workspace>--peppercarrot-ollama-serve.modal.run` — that's what goes into `OLLAMA_BASE_URL` in `.env.production`. Generate the proxy-auth token pair from the Modal dashboard (Settings → Proxy Auth Tokens → Create) and paste both into `.env.production` too.

Smoke-test from your shell:

```bash
set -a && source .env.production && set +a
curl -sS -H "Modal-Key: $MODAL_PROXY_TOKEN_ID" \
        -H "Modal-Secret: $MODAL_PROXY_TOKEN_SECRET" \
        "$OLLAMA_BASE_URL/api/tags"
# {"models": [{"name": "qwen2.5:7b", ...}, {"name": "bge-m3", ...}]}
```

If the first request takes a minute and then succeeds, you're watching a cold start in real time. Subsequent requests within five minutes are instant. The chat in your deployed backend is going to feel exactly this way.

> *About the first answer.* A natural production-polish addition is a fire-and-forget warmup the backend issues against Modal the moment a reader opens a session, bolted onto the existing `POST /api/sessions` handler. While the reader is reading the episode cover and typing their first question, qwen2.5:7b is quietly loading into VRAM. By the time they hit Enter, the model is usually ready. The workshop ships without the warmup — partly to keep the code small, partly because the cold start is *the part this post is honest about*. The warmup is the kind of polish that hides a real cost from the user; the cost is still real, and the architecture should be designed to make it small, not to make it invisible.

### Choosing a GPU tier — or skipping the GPU entirely

The workshop ships with `gpu="T4"` because it's Modal's cheapest GPU and qwen2.5:7b + bge-m3 both fit comfortably in 16 GB of VRAM with room for context windows. Two adjacent decisions are worth naming.

**Upgrading the GPU.** Modal also offers L4 (24 GB, ~$0.80/hr, ~1.5× T4 throughput), A10G (24 GB, ~$1.10/hr, ~2× T4), and A100/H100 (40+ GB, $3+/hr). For qwen2.5:7b at portfolio traffic, T4 stays the right pick: per-second cost roughly tracks per-second throughput, so the bigger GPUs don't lower the per-question bill, they just answer faster. The upgrade is worth it only when (a) you switch to a larger model (qwen2.5:14b needs at least an L4), or (b) you have sustained traffic where lowering active GPU time per request actually matters.

**Skipping the GPU entirely** is architecturally more interesting because the [Post 4 provider abstraction]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) was designed for it. The chat call can swap to `AnthropicChatClient` with a single env-var flip, and that class already ships in `backend/app/clients/chat.py`:

```bash
CHAT_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5
```

But here's the part worth flagging: you still need an embedding model. Every chat question gets embedded to do the vector search against ChromaDB (see [Post 9]({% post_url 2026-05-25-pepper-carrot-companion-spoiler-safe-rag %})), regardless of which chat provider you use. So "skip Modal" really means "find an embeddings home that isn't Modal." Three options, in increasing order of work:

- **In-process `sentence-transformers` on Fly.** `EMBEDDING_PROVIDER=sentence-transformers` is already supported and works against the local model files. The catch: bge-m3 is ~1.5 GB resident in RAM, so the workshop's 512 MB Fly machine isn't big enough — you'd bump the VM to 2 GB (~$3/mo) and accept a longer Fly cold start (the model loads into RAM on every container boot).
- **[Voyage AI](https://www.voyageai.com/)** — Anthropic's recommended embeddings partner. `EMBEDDING_PROVIDER=voyage` flips the factory onto the bundled [`VoyageEmbeddingClient`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/backend/app/clients/embedding.py) (~80 lines: thin POST to `api.voyageai.com/v1/embeddings`, defensive index-resort, mocked unit tests). Voyage's `voyage-3-lite` runs around $0.02/M tokens — essentially free at portfolio traffic.
- **Keep Modal for embeddings only.** Run Modal with `gpu=None` (Modal does CPU-only functions), drop the chat model from the served pair, keep bge-m3. Awkward middle option — you still operate a Modal endpoint, but CPU-only is cheap (~$0.10/hr active) and qwen2.5:7b's GPU bill is gone.

The factory in `backend/app/clients/__init__.py` carries one branch per provider; `EMBEDDING_PROVIDER=voyage` plus a `VOYAGE_API_KEY` is the whole config. The Post 4 abstraction was designed for exactly this: provider swaps stay one env var, never a code change.

The cost comparison at portfolio traffic (~100 chat questions/month, bursty visitor sessions):

| Cost component | Modal-hosted Ollama *(workshop default)* | Anthropic Haiku + Voyage AI |
|---|---|---|
| **Chat inference** | T4 GPU at $0.59/hr × ~10 active GPU-minutes/mo + 5-min idle window per burst | $0.25/M input + $1.25/M output tokens × ~100 q/mo |
| **Embeddings** | (same Modal endpoint — included in chat cost) | $0.02/M tokens × ~5K question-tokens/mo |
| **Model-weights storage at rest** | ~$1/mo (Modal volume holding qwen2.5:7b + bge-m3) | $0 |
| **Monthly chat-layer total** | **~$5–10** | **~$0.10** |
| **First-request latency after idle** | 15–30 s (GPU + VRAM load) | ~1 s (always-on API) |
| **Self-hosted / data privacy** | ✓ — prompts and answers never leave your infra | ✗ — every prompt goes to Anthropic, every embed-query to Voyage |
| **Matches the series' local-first thesis** | ✓ | ✗ |

Two operational notes if you switch:

- **Re-indexing.** Chroma's `pages_v1` and `wiki_v1` collections were built with bge-m3 vectors. Voyage's embeddings have different dimensionality and a different vector space — vectors from one embedder don't make sense in the other's coordinate system, so similarity scores would be meaningless. You'd re-embed everything via the ingestion pipeline (`ingest.py` per episode + `ingest_wiki.py` once) before retrieval would work. The data in Postgres + R2 stays put; only the Chroma collections rebuild.
- **The thesis.** The series' framing is "local-first inference on commodity GPU" — the project exists *because* of that constraint, and Post 11's prompt hardening is calibrated against qwen2.5:7b's specific limitations. Reaching for the Anthropic API trades that thesis for cost, latency, and operational simplicity. For a portfolio piece *about local-first*, Modal + T4 is the right pick. For a portfolio piece where chat quality and zero cold start matter more than the framing, the workshop ships ready to flip — `CHAT_PROVIDER=anthropic` plus `EMBEDDING_PROVIDER=voyage` plus two API keys — and demonstrating that the Post 4 abstractions actually deliver that flip is itself a portfolio signal, regardless of which path you ship. See [`docs/deployment.md`'s "Alternative" section](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/docs/deployment.md#alternative-skip-modal-entirely-anthropic--voyage-ai) for the three-step delta from the default flow.

---

## Neon: The Two Connection Strings {#neon}

Neon is hosted Postgres that sleeps when idle. The integration is "give the backend a connection string and walk away," with one wrinkle worth its own section, because the wrinkle is exactly the kind of subtle failure mode that turns a first deploy into a debugging marathon.

The wrinkle: **asyncpg and Neon's connection pooler don't get along in transaction mode.**

> *Plain-English aside: connection pooling and prepared statements.* Neon (like most managed Postgres providers) puts a process called **pgbouncer** in front of the database to multiplex connections. Pgbouncer comes in three modes — session, transaction, and statement — that vary in how aggressively they share backend connections across clients. Neon defaults to **transaction mode**, which is the most efficient (each transaction lands on whichever backend connection is free) but breaks **prepared statements**. Prepared statements are an asyncpg optimization: the client tells the server "remember this query plan as statement `__asyncpg_stmt_42__`" and then says "run statement 42" on subsequent calls. In transaction mode pgbouncer hands each query to a different backend, none of which have seen statement 42, and asyncpg raises `prepared statement "__asyncpg_stmt_42__" does not exist`. The fix is to bypass the pooler: connect to the unpooled endpoint and asyncpg has its own connection to make prepared-statement promises against.

Neon's UI gives you two endpoints — the pooled one (hostname includes `-pooler`) and the unpooled one (no `-pooler`). The `.env.production` template carries both:

```bash
# Used by infra/entrypoint.sh during the one-shot psql seed restore.
# psql doesn't use prepared statements; the pooler is fine.
POSTGRES_RESTORE_URL=postgresql://neondb_owner:PASS@ep-XXXX-pooler.REGION.aws.neon.tech/neondb?sslmode=require

# Used by the FastAPI backend at runtime. asyncpg + prepared statements.
# Drop -pooler from the hostname for direct connections.
DATABASE_URL_OVERRIDE=postgresql+asyncpg://neondb_owner:PASS@ep-XXXX.REGION.aws.neon.tech/neondb?sslmode=require
```

The scheme prefix is the other difference: `postgresql+asyncpg://` tells SQLAlchemy "use the async driver," while `postgresql://` is the libpq scheme `psql` expects. The host is the same minus the `-pooler` suffix. The `?sslmode=require` works for both, and the [`db/session.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/backend/app/db/session.py) shim from earlier translates the URL param into the format asyncpg actually accepts, so the operator never has to know the difference.

> *About the `?sslmode=require` shim.* SQLAlchemy's asyncpg dialect forwards unknown URL query params straight to `asyncpg.connect()`, which accepts `ssl=` but not `sslmode=`. The naive thing is to make the operator rewrite the URL to use `ssl=true` instead of `sslmode=require`, and then *also* discover that asyncpg rejects `ssl=true` as a string and wants the literal `"require"`. Both surprises eat 20 minutes the first time. The `_extract_ssl_connect_args` helper in `db/session.py` accepts whichever form the operator pasted in and translates it. Three lines of code that save an hour of head-scratching are exactly the kind of seam absorbing the operator deserves.

On the Neon side, sleep is a property the application doesn't have to do anything about. After ~5 minutes of no queries, Neon's compute stops; the next query wakes it up (a ~1-second pause, lower than Modal's GPU cold start by orders of magnitude). At portfolio traffic the daily compute usage is small enough that the 0.5 GB free tier covers it forever. Stateful storage that sleeps when idle is a thing Neon does so well it can disappear from the architecture conversation entirely, which is the highest praise a managed service can earn.

---

## Cloudflare R2: The Implementation That Finally Landed {#r2}

R2 is the longest-running unfinished business in the workshop. [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) introduced the [`Storage` Protocol](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/backend/app/clients/storage.py) with three methods (`put`, `url_for`, `exists`), a working `LocalStorage` implementation, and a stub `R2Storage` whose methods all `raise NotImplementedError`. Ten posts later, R2 is the thing that turns the stub into a Storage:

```python
# backend/app/clients/storage.py (the R2Storage that lands in Post 14)
class R2Storage:
    """Cloudflare R2 (S3-compatible) storage. Production target — see Post 14."""

    # Public-read R2 buckets serve every object with these cache headers,
    # so the browser caches them aggressively after the first hit. Comic
    # pages never change once authored; if they do, the ingestion pipeline
    # writes to a new key rather than mutating an existing one.
    _CACHE_CONTROL = "public, max-age=31536000, immutable"

    def __init__(
        self, account_id, access_key_id, secret_access_key, bucket, public_url_prefix
    ) -> None:
        self._bucket = bucket
        self._public_url_prefix = public_url_prefix.rstrip("/")
        # boto3 imported lazily so the workshop's local-only path doesn't need it.
        # The factory in clients/__init__.py validates all four R2_* env vars
        # before reaching this constructor.
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for STORAGE_BACKEND=r2. "
                "Install with `uv sync` — boto3 is pinned in pyproject.toml."
            ) from exc

        self._client: Any = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

    async def put(self, key: str, content: bytes, content_type: str) -> None:
        def _put() -> None:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=content,
                ContentType=content_type,
                CacheControl=self._CACHE_CONTROL,
            )
        await asyncio.to_thread(_put)

    async def url_for(self, key: str) -> str:
        # The runtime hot path. No I/O — just a string compose.
        return f"{self._public_url_prefix}/{key}"

    async def exists(self, key: str) -> bool:
        # head_object returns 200 on hit, 404 on miss. Other errors propagate.
        ...
```

Five details worth surfacing because they encode operational decisions you'd otherwise have to discover yourself:

- **The `_CACHE_CONTROL = "public, max-age=31536000, immutable"` header.** Comic pages never change once authored; the ingestion pipeline writes new keys for new versions rather than mutating existing ones. With these headers, browsers cache aggressively, the R2 CDN caches aggressively, and a repeat visitor pays almost no bandwidth on the second page of any episode. The `immutable` directive in particular is what tells modern browsers "don't even bother to revalidate."
- **boto3 is imported lazily inside the constructor.** The workshop's default `STORAGE_BACKEND=local` path doesn't pull boto3 into the import graph at all. This is the smallest possible respect-the-abstraction discipline — the SDK touches one file, and only when the factory selects this implementation.
- **`asyncio.to_thread` wraps every network call.** boto3 is synchronous. FastAPI is async. Mixing them naively (calling `self._client.put_object(...)` inside an async route handler) blocks the event loop for the duration of the upload, and a slow upload can starve every other inbound request. `await asyncio.to_thread(_put)` parks the blocking call on a worker thread and yields the event loop. The pattern is a one-liner because the abstraction lets it be.
- **`url_for()` is a string compose, no I/O.** The runtime read path — what FastAPI does on every `GET /api/episodes/{slug}` — never hits R2 at all. The DB stores a relative key (`episodes/ep01-potion-of-flight/pages/001-display.webp`), `R2Storage.url_for()` prepends the public prefix, and the browser fetches the image directly from Cloudflare's CDN. The backend's bandwidth bill stays at zero even when the demo gets traffic.
- **`exists()` translates S3's 404 into Python's `False` instead of an exception.** It's the kind of detail nobody hits until they need it (the workshop's `LocalStorage.exists()` follows the same shape), and it's an example of the **one-translation-per-difference** discipline: every place asyncpg-vs-libpq and S3-vs-filesystem differ in shape, the difference gets absorbed inside the implementation that owns it, so the rest of the codebase reads uniform.

The uploads themselves don't go through `R2Storage.put` at portfolio scale. They go through [`rclone`](https://rclone.org/), the open-source S3-compatible copy tool, because the ingestion pipeline runs locally and `rclone copy` is a one-liner that walks the entire `data/images/` tree once. Two `--exclude` flags are doing real work and worth naming:

```bash
rclone copy data/images r2:peppercarrot-images --progress \
    --exclude ".DS_Store" --exclude "**/.DS_Store" \
    --exclude "**/*-original.jpg"
rclone copy data/world-graph/images r2:peppercarrot-images/world-graph/images --progress \
    --exclude ".DS_Store" --exclude "**/.DS_Store"
```

The `.DS_Store` exclusion keeps macOS Finder's per-directory metadata files out of the bucket — without it, every directory you ever opened in Finder leaks one to a publicly-readable URL. The `**/*-original.jpg` exclusion skips the 2 MB source JPEGs that ingestion kept locally as the canonical source-of-truth for re-processing image variants. The runtime only reads `-display.webp` and `-thumbnail.webp`, so the originals are 4× bucket weight with zero user-facing benefit. (Keeping them on R2 is free under the 10 GB tier; excluding them is just cosmetic discipline.)

`put()` exists for the future case of ingestion-jobs-that-run-remotely. For now, it's covered by the smoke test in the repo and exercised by nothing else.

> *About the bucket layout.* The DB stores keys like `episodes/ep01-potion-of-flight/pages/001-display.webp` — slugged, hierarchical, sortable. The R2 bucket layout matches exactly: `rclone lsf r2:peppercarrot-images/episodes/ --dirs-only | sort` should print one line per ingested episode (12 lines if you have ep01–12). The most common first-deploy failure mode here is the "double prefix" — you `rclone copy data/images r2:peppercarrot-images/images` and end up with `images/episodes/.../001-display.webp`, which doesn't match what the DB stores. Fix is to `rclone delete r2:peppercarrot-images/images` and re-copy with the right destination. The smoke test (`curl -I "$R2_PUBLIC_URL_PREFIX/world-graph/images/carrot-thumb.webp"` returning 200) is the cheap check that the keys line up before you go debug the whole frontend.

> *Re-deploying with a smaller / different episode set.* `rclone copy` is **additive — it never deletes**. If you re-ingest with fewer episodes (say, ep01–12 instead of the ep01–39 the bucket already has), the stale episodes stay in R2 forever. The fix is to swap `copy` for `sync` (which mirrors source → dest including deletes) and target the `episodes/` subdirectory so the `world-graph/` prefix isn't touched. Always with `--dry-run` first, because a wrong-shaped source path will happily wipe data you wanted to keep. The full recipe is in [`docs/deployment.md`'s "Pruning stale uploads from R2"](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/docs/deployment.md#pruning-stale-uploads-from-r2) section.

---

## The Container: Bake Small Data, Stream Big Data {#dockerfile}

The Fly side of the deploy needs a container, and the container packs three categories of stuff with very different lifecycles:

| Category | Size | Lifecycle | Where |
|---|---|---|---|
| Backend code (Python venv + `app/` + `alembic/`) | ~200 MB | Changes on every deploy | Baked into the image |
| Small data (Chroma vectors, world-graph YAML, `seed.sql`) | ~5 MB | Changes when ingestion runs | Baked into the image |
| Large data (episode page images) | ~700 MB | Changes when episodes are ingested | R2, *not* baked |

The reason for the split is the deploy round-trip. Anything baked into the image is replaced by the next `fly deploy`; anything in R2 (or Neon) is incremental, uploaded once and served forever. Baking the small data simplifies operations (one command rebuilds the world); baking the large data would inflate every push by 700 MB and break the "fast iterate, slow first deploy" rhythm the demo wants.

The [`Dockerfile`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/Dockerfile) reads top to bottom:

```dockerfile
# ── Stage 1: install deps into a venv (cached layer) ──────────────────────────
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock /app/
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

# psql is needed by infra/entrypoint.sh to restore data/seed.sql on first boot.
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LOCAL_IMAGE_DIR=/app/data/images \
    CHROMA_PERSIST_DIR=/app/data/chroma

COPY backend/app /app/app
COPY backend/alembic /app/alembic
COPY backend/alembic.ini /app/alembic.ini

# Bake small data: chroma vectors + world-graph YAML.
# Episode page images are NOT baked — they go to R2.
COPY data/chroma /app/data/chroma
COPY data/world-graph /app/data/world-graph

# DB seed produced by infra/dump_seed.sh before `fly deploy`.
COPY data/seed.sql /app/data/seed.sql

COPY infra/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
```

Three patterns are worth naming:

- **Two-stage builds reduce the runtime image.** Stage 1 installs `uv` and resolves the venv from `uv.lock`; stage 2 copies the resulting `.venv` over and forgets stage 1 ever existed. The runtime image is a slim Python plus the venv plus `psql`, and that's it. No `uv`, no build toolchain, no dev dependencies.
- **The COPY order is cache-conscious.** Python deps change rarely; app code changes often. Putting `pyproject.toml` + `uv.lock` ahead of `backend/app` means a code-only change skips re-resolving deps. Same shape for the small-data baking: `data/chroma` changes only when ingestion has run, so it sits on its own layer that the build can reuse if nothing's changed.
- **The seed restore happens in [`entrypoint.sh`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/infra/entrypoint.sh), not in the Dockerfile.** Image builds are stateless; the restore needs to happen against a live Neon database that the image doesn't know about at build time. The entrypoint runs once per container start, checks whether the `episodes` table exists, and conditionally invokes `psql < /app/data/seed.sql`. Idempotent by an `information_schema` query — the entrypoint can run a hundred times against the same Neon DB and only does work once:

```bash
# infra/entrypoint.sh
have_episodes="$(psql "$POSTGRES_RESTORE_URL" -tAc \
    "SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name='episodes'")"
if [ "$have_episodes" != "1" ]; then
    echo "[entrypoint] Seeding Postgres from /app/data/seed.sql ..."
    psql "$POSTGRES_RESTORE_URL" < /app/data/seed.sql
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The `dump_seed.sh` script that produces `data/seed.sql` is itself two lines of `pg_dump` with `--no-owner --no-acl --no-privileges` (Neon's role name differs from local; the default dump emits `ALTER OWNER` lines Postgres would reject):

```bash
# infra/dump_seed.sh
pg_dump -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
    --no-owner --no-acl --no-privileges --format=plain > data/seed.sql
```

The whole pattern — "bake the small data, restore on first boot, gitignore the dump" — is one of the smallest end-to-end deploys that's actually defensible. The full version of this project (the public demo URL goes up alongside this post) keeps the same shape; the only difference is that a CI pipeline runs `dump_seed.sh` and `fly deploy` automatically. For the workshop, `./infra/dump_seed.sh && fly deploy` from the developer's laptop is what ships.

> *About `.dockerignore`.* The companion to the Dockerfile, often underappreciated. [`.dockerignore`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-14-15-deploy/.dockerignore) keeps `node_modules` (~300 MB on a fresh `npm install`), `data/postgres` (the Docker bind mount Postgres writes into — would be tens of GB), `data/raw` (the downloaded episode JPEGs), `.venv`, `.git`, and the various test/cache directories out of the build context Docker sends to the daemon. Without it, every `fly deploy` would upload hundreds of MB of irrelevance, slowing the deploy by minutes. The `!.env.production.example` exclusion is deliberate — the *example* template is fine to ship in the image; the real `.env.production` with actual secrets is not.

---

Next up: **Post 15 — Shipping It: Deploy and Verify.** The three backing services are provisioned and the container builds; what's left is to make it public. Post 15 deploys the container to a scale-to-zero Fly machine, ships the React frontend to Cloudflare Pages with a single build-time env var, walks the first cold start (and the warmup that hides it), and runs a layer-by-layer verification so a failure names the single provider to debug — then hands you a `*.pages.dev` URL.

The **workshop starter** that backs this post is at <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop>, tagged `post-14-15-deploy` — the same deploy checkpoint [Post 15]({% post_url 2026-06-01-pepper-carrot-companion-deploy-verify %}) uses. Clone it, provision the three services per the steps above, and the container will build locally before you take it public in the next post.

*Pepper & Carrot* is © [David Revoy](https://www.davidrevoy.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All credit to him for the source material that made this project possible.

**All opinions expressed are my own.**
