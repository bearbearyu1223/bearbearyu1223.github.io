---
title: "Setting Up the Workshop: Postgres, Ollama, and a Project That Type-Checks"
date: 2026-05-10 00:00:00 -0800
categories: [Full-Stack, RAG, Local AI]
tags: [postgres, ollama, fastapi, uv, mypy, ruff, alembic, docker, peppercarrot, portfolio]
description: >-
  Post 2 of the Pepper & Carrot AI flipbook series. Stand up the local
  workshop: Postgres in Docker, Ollama serving qwen2.5:7b and bge-m3, a
  typed FastAPI scaffold that passes mypy strict, the first Alembic
  migration, and one downloaded episode on disk. By the end you have
  the workshop every later post is built on.
pin: true
---

Post 2 of the [*Pepper & Carrot AI-powered flipbook*]({% post_url 2026-05-09-pepper-carrot-companion-trailer %})
series. The unglamorous-but-essential preflight: a database, a local
language-model server, a Python project that type-checks, a real
database schema, and one episode of the comic downloaded into
`data/raw/`. From here every later post assumes a green light on every
verification step below.

> **What you'll build in this post.**
> - A local [PostgreSQL 16](https://www.postgresql.org/) container, reachable on `localhost:5432`.
> - A local [Ollama](https://ollama.com/) instance serving two text-only models — `qwen2.5:7b` (chat) and `bge-m3` (embeddings).
> - A [FastAPI](https://fastapi.tiangolo.com/) backend scaffold that passes [`mypy --strict`](https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-strict) and [`ruff check`](https://docs.astral.sh/ruff/) cleanly.
> - The first [Alembic](https://alembic.sqlalchemy.org/) migration applied, with 10 application tables visible in `\dt`.
> - One episode of *Pepper & Carrot* downloaded into `data/raw/`, with real images and structured metadata.
> - A `.env` file that ties the five pieces above together.
>
> **Prerequisites.**
> - A laptop with ~10 GB of free disk (Ollama models are heavy) and ≥16 GB RAM. (24 GB+ unlocks an optional upgrade to the larger `qwen2.5:14b` chat model, but the 7B default is what we run here and what the cloud deploy in Post 10 uses.)
> - Comfort opening a terminal and running shell commands. You do not need any prior AI / RAG / FastAPI knowledge — every concept gets a plain-language definition on first mention.
> - macOS or Linux (the commands assume a Unix shell; Windows users should use WSL2).

---

## Table of Contents

1. [Why a "Workshop" Phase Deserves a Whole Post](#why-workshop)
2. [The Four Tools You'll Install on Your Machine](#tools-to-install)
3. [The Project Skeleton, at a Glance](#project-skeleton)
4. [Postgres in a Container](#postgres)
5. [Ollama: An LLM Server for Your Laptop](#ollama)
6. [A Typed Python Project with uv, mypy, and ruff](#python-project)
7. [The First Migration: Real Schema in Real Postgres](#first-migration)
8. [Your First Episode in `data/raw/`](#first-episode)
9. [The `.env` File That Ties It Together](#env)
10. [Verification: The Whole Workshop, End to End](#verification)
11. [Key Takeaways](#key-takeaways)
12. [Appendix A: What the SQLAlchemy Models Actually Look Like](#appendix-models)

---

## Why a "Workshop" Phase Deserves a Whole Post {#why-workshop}

If you've followed AI tutorials before, you've seen this pattern: install five things, run a magic script, look at a chatbot. It works on the author's machine. Then a week later you try to debug a retrieval issue and discover the model server was never actually serving the model you thought, the migrations were never applied, and your code is silently importing the wrong embedding library.

A "workshop" phase is the antidote. Each tool gets installed, configured, and **verified to work in isolation** before any application code touches it. You write zero feature code in this post — but you finish it with a green-lit, reproducible base camp.

There's a portfolio judgement here worth naming: real production systems live or die by their setup story. A demo that needs ten paragraphs of caveats to run for the reviewer is — at the level of professional impression — worse than a smaller demo that "just works." Investing one post in setup is the cost of being able to write the next eight posts without "did you remember to also..." sprinkled through each one.

The project encodes this same idea as a hard rule it calls **provider
abstraction**: every external service — database, model server, image
store — sits behind a typed interface, so local Postgres and cloud
Postgres look identical to the code, and local Ollama and serverless
GPU Ollama look identical to the code. Post 3 makes the rule
explicit and writes the first two of those abstractions. The workshop
you build in *this* post is what makes that possible.

---

## The Four Tools You'll Install on Your Machine {#tools-to-install}

Four installs, each with a one-sentence reason:

| Tool | What it does | Install |
|---|---|---|
| [Docker](https://www.docker.com/) | Runs Postgres in an isolated container so you don't pollute your machine with a system-level database. | [Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS/Windows) or `apt install docker.io docker-compose-plugin` (Linux). |
| [`uv`](https://github.com/astral-sh/uv) | Python package manager. ~10–100× faster than `pip`, with a lockfile and `python` version management baked in. | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Node.js + npm](https://nodejs.org/) | JavaScript runtime + package manager for the React frontend (we wire up the frontend in Post 5; install it now so it's done). | `brew install node` on macOS or [nodesource](https://github.com/nodesource/distributions) on Linux. Use Node 20+. |
| [Claude Code CLI](https://docs.claude.com/en/docs/claude-code) | Anthropic's terminal coding assistant. We use it in Post 4 as the vision provider for page descriptions — install it now and log in so the OAuth dance isn't blocking you later. | `npm install -g @anthropic-ai/claude-code`, then run `claude` once and follow the browser prompt. |

> *A plain-English aside on the unfamiliar names.*
>
> **Docker** packages a process plus everything it needs to run (filesystem, libraries, network config) into a sealed bundle called a *container*. Think of it as "ship a program with its whole house attached." It's how we get the same Postgres on every developer's laptop.
>
> **`uv`** is a Python toolchain rewrite in [Rust](https://www.rust-lang.org/), from the same team that builds [`ruff`](https://docs.astral.sh/ruff/). Once you've used it, going back to `pip` and `venv` feels archaic.
>
> **Claude Code** is a command-line agent — you describe a task in plain English in your terminal, and it edits files, runs tests, and iterates. Different category of product from a chatbot; we'll see it earning its keep in Post 4.

Verify each of the four landed:

```bash
docker --version
uv --version
node --version          # should print v20.x.x or newer
claude --version
```

All four should print a version string. If any errors, fix that one before moving on — the rest of this post depends on every one of them.

---

## The Project Skeleton, at a Glance {#project-skeleton}

Before any `cd backend` or `cp .env.example .env`, a one-screen orientation to what's on disk. From the next section onward, the post will start `cd`-ing into specific subdirectories and editing specific files — and that's a lot less confusing once you've seen the whole layout once.

```
peppercarrot-companion/
├── docker-compose.yml            ← Postgres container definition (this post)
├── .env.example                  ← copy to .env, fill in (this post)
├── README.md                     ← human-facing setup guide
├── CLAUDE.md                     ← project conventions for AI/human contributors
│
├── backend/                      ← Python · FastAPI · SQLAlchemy
│   ├── pyproject.toml            ← deps + mypy/ruff config (this post)
│   ├── uv.lock                   ← pinned dependency tree (this post)
│   ├── alembic.ini               ← migration tool config (this post)
│   ├── alembic/
│   │   ├── env.py                ← migration runner
│   │   └── versions/             ← one .py per schema change (this post adds the first)
│   ├── app/
│   │   ├── main.py               ← FastAPI app entrypoint (Post 5)
│   │   ├── config.py             ← typed settings loaded from .env
│   │   ├── api/                  ← REST route handlers (Post 5)
│   │   ├── clients/              ← provider abstractions (Post 3)
│   │   ├── core/prompts.py       ← all system prompts (Post 8)
│   │   ├── db/                   ← SQLAlchemy models (the schema this post creates)
│   │   ├── orchestration/        ← chat pipeline (Posts 6–8)
│   │   └── retrieval/            ← ChromaDB query layer (Post 6)
│   └── tests/
│
├── frontend/                     ← React · TypeScript · Vite (Post 5+)
│   ├── package.json
│   └── src/
│
├── ingestion/                    ← offline data pipeline
│   ├── acquire.py                ← download episodes from peppercarrot.com (this post)
│   └── ingest.py                 ← load images + descriptions into Postgres / Chroma (Post 4)
│
└── data/                         ← gitignored — your workshop fills this in
    ├── postgres/                 ← Docker bind mount for Postgres data (this post)
    ├── chroma/                   ← embedded vector index (Post 4)
    ├── raw/                      ← downloaded episode source (this post)
    └── images/                   ← image variants written by ingestion (Post 4)
```

Two things to take away from that tree:

- **You'll only touch a handful of these in this post.** Top-level `docker-compose.yml` and `.env.example`; `backend/pyproject.toml` and the `backend/alembic/` migration directory; the SQLAlchemy models under `backend/app/db/`; and `ingestion/acquire.py`. Every other path is structural placeholder for code that arrives in later posts — they're listed so you can see where things will land as the series builds out.
- **`data/` is gitignored — it's not part of the source.** It's a working directory each developer creates as they run the workshop. By the end of this post you'll have `./data/postgres/` (live Postgres data files), `./data/raw/ep01-potion-of-flight/` (one downloaded episode of comic source), and a placeholder for `./data/chroma/` that gets populated in Post 4.

> **About the repo URL.** As [Post 1]({% post_url 2026-05-09-pepper-carrot-companion-trailer %}) notes, the public GitHub repository goes up alongside the deploy guide in Post 10 — so all ten posts publish into a finished, runnable project rather than a half-explained skeleton. Until then, read the tree above as a map of where each piece will live once the repo is public; the file paths in every later command (`backend/app/clients/`, `backend/app/db/models.py`, etc.) are the same paths you'll find in the repo when it goes live.

---

## Postgres in a Container {#postgres}

> *What's Postgres?* [**PostgreSQL**](https://www.postgresql.org/about/) is a battle-tested open-source relational database — the same kind of database that powers Instagram, Reddit, and most boring-but-load-bearing internal tools at large companies. "Relational" means data lives in tables with columns of declared types, and you query it with [SQL](https://www.postgresql.org/docs/current/sql.html). We use it to store everything *except* embeddings (those go in a *vector store* — a different kind of database that we'll set up in Post 3 using [ChromaDB](https://www.trychroma.com/)): episodes, pages, characters, chat sessions, and the actual prose descriptions of each comic page.

### The compose file, line by line

Two plain-language definitions first:

- A **container** is a lightweight isolated environment that runs a single process and everything it needs (libraries, config, filesystem layout). Think of it as a tiny self-contained Linux machine that shares your laptop's kernel and starts in milliseconds. Postgres publishes an official container *image* — a frozen snapshot you can launch from — on [Docker Hub](https://hub.docker.com/_/postgres).
- **[Docker Compose](https://docs.docker.com/compose/)** is a tool for describing one or more containers as YAML and launching them together. The conventional filename is `docker-compose.yml` in the project root, run via `docker compose up`.

Here's the whole file:

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16-alpine
    container_name: peppercarrot-postgres
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-peppercarrot}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-peppercarrot_dev}
      POSTGRES_DB: ${POSTGRES_DB:-peppercarrot}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-peppercarrot}"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped
```

A walk through every block:

- **`services:`** is the top-level key. Compose can manage many containers; each named service becomes one container. We have one service (`postgres`); when we deploy to the cloud in Post 10, we'll have several.

- **`postgres:`** is *our* name for this service — used by `docker compose logs postgres`, by `depends_on:` declarations in other services, and as the DNS hostname other containers on the same Compose network use to reach it. (We'll see this matter when we add pgAdmin below.)

- **`image: postgres:16-alpine`** says "launch from the official `postgres` image on [Docker Hub](https://hub.docker.com/_/postgres), tagged `16-alpine`." Two parts:
  - **`16`** pins the major version so today's schema still works tomorrow. Without a pin, you get whatever `latest` happens to be the day you run it, which is a great way to wake up to a broken setup after a Postgres major-version upgrade.
  - **`alpine`** is a [tiny Linux base distribution](https://alpinelinux.org/) — the image weighs ~80 MB instead of the default Debian-based ~400 MB. Faster first-time pull, less disk, fewer packages to keep secure.

- **`container_name: peppercarrot-postgres`** names the *running* container (the live instance, not the image) so it shows up legibly in `docker ps` and you can run `docker exec -it peppercarrot-postgres psql -U peppercarrot` without copying a random hash.

- **`environment:`** is a map of environment variables passed into the container's process. The Postgres image is designed to read three of them on first boot — `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` — and create the initial superuser and database from those values. Set them once at first boot, they're baked into the data directory thereafter.

  The `${VAR:-default}` syntax is straight from [POSIX shells](https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_06_02): "use the value of `$VAR` if it's set in the environment, otherwise use this fallback." Compose reads `.env` automatically, so a `POSTGRES_PASSWORD=hunter2` line in your `.env` propagates here without any further wiring. (This is why we copied `.env.example` to `.env` before running `docker compose up`.)

- **`ports:`** maps `host_port:container_port`. Postgres listens on 5432 inside the container; we expose it on `localhost:5432` on your laptop, so any tool on your machine can connect with `postgresql://...@localhost:5432/...`. If you already have a system Postgres on 5432, override `POSTGRES_PORT=5433` in `.env` — the `${POSTGRES_PORT:-5432}` substitution then maps host port 5433 to container port 5432.

- **`volumes:`** mounts a directory from your host into the container — a [*bind mount*](https://docs.docker.com/storage/bind-mounts/). `./data/postgres` on your laptop becomes `/var/lib/postgresql/data` inside the container, which is where Postgres stores every byte of every table. Without this, `docker compose down` would also delete your database; with it, your data survives container restarts, image upgrades, and laptop reboots.

  Two follow-on points worth knowing:
  - `./data/postgres` is **gitignored** — binary database files belong in backups, not version control.
  - If you ever want to start fresh: `docker compose down && rm -rf data/postgres && docker compose up -d` gets you a virgin database with all the initial-boot setup re-run.

- **`healthcheck:`** tells Docker how to find out whether the container is *actually ready*, not just running. The image starts the moment the process launches, but Postgres takes a couple of seconds before it accepts connections. The check runs [`pg_isready`](https://www.postgresql.org/docs/current/app-pg-isready.html) — Postgres's own "are you up?" command — every 5 seconds, gives up on a single probe after 5 seconds, and tolerates 5 consecutive failures before marking the container *unhealthy*. `docker compose ps` shows `(healthy)` once the first probe succeeds, which is also the signal the FastAPI app uses in Post 3 to know it's safe to open its DB connection pool.

- **`restart: unless-stopped`** auto-restarts the container if it crashes, and brings it back up automatically when you reboot your laptop — but stays out of the way if you explicitly `docker compose stop`. A reasonable default for a dev database you don't want to think about.

### Optional: pgAdmin for browsing the database visually

The project's `docker-compose.yml` also defines a second service, [pgAdmin 4](https://www.pgadmin.org/) — a web UI for Postgres that beginners often appreciate over typing `psql` commands:

```yaml
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: peppercarrot-pgadmin
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@local.dev
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      - postgres
```

Two new ideas here worth a line each:

- **`depends_on: - postgres`** tells Compose to start the `postgres` service *before* `pgadmin`. By default this only waits for the container to start, not for the healthcheck to pass — fine here because pgAdmin happily retries its own database connection until it succeeds.
- **`5050:80`** maps the container's port 80 (where pgAdmin's web server listens) to `localhost:5050` on your laptop.

After `docker compose up -d`, browse to [http://localhost:5050](http://localhost:5050) and log in with `admin@local.dev` / `admin`. The dashboard is empty until you tell pgAdmin where Postgres is. Click **Add New Server** (or right-click **Servers** → **Register** → **Server…**), give it any name on the **General** tab, then on the **Connection** tab:

| Field | Value |
|---|---|
| Host name / address | `postgres` *(the service name — Compose resolves it to the postgres container's IP on the shared internal network)* |
| Port | `5432` |
| Maintenance database | `peppercarrot` |
| Username | `peppercarrot` |
| Password | `peppercarrot_dev` |

> ⚠️ **pgAdmin does not strip whitespace from these fields.** Copy `peppercarrot ` with a trailing space and the connection fails with `FATAL: password authentication failed for user "peppercarrot "` — Postgres echoes the username back verbatim and treats the two as different users. Select-all and retype each field if in doubt.

Once registered, your tables live under **Servers → *(your name)* → Databases → peppercarrot → Schemas → public → Tables**. Right-click any table → **View/Edit Data** → **All Rows** for a spreadsheet view, or **Tools** → **Query Tool** (with `peppercarrot` selected in the tree) for SQL — **F5** runs. If the Tables list is empty, you haven't applied the migration yet — that's the section a few headings down.

The useful "what's actually populated?" check while the schema is fresh — most tables stay empty until ingestion in Post 4:

```sql
SELECT relname AS table_name, n_live_tup AS row_count
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC, relname;
```

Skip the whole pgAdmin section if you're comfortable in `psql`; turn it on if you want to *see* the rows fill in as later posts build on top.

### Start it

From the project root:

```bash
cp .env.example .env       # we'll fill this in shortly
docker compose up -d       # -d = detached (runs in the background)
```

After ~5 seconds, check it's healthy:

```bash
docker compose ps
# NAME                     IMAGE              STATUS                   PORTS
# peppercarrot-postgres    postgres:16-alpine Up 5 seconds (healthy)   0.0.0.0:5432->5432/tcp
```

Now verify you can actually talk to the database. We'll use [`psql`](https://www.postgresql.org/docs/current/app-psql.html) — the official Postgres command-line client. It does **not** come bundled with Docker, so you'll need to install it separately if you don't already have it. You have two options:

**Option A — install `psql` on your host machine.** This is the cleanest path; you can connect to *any* Postgres (local or remote) the same way.

```bash
# macOS (Homebrew)
brew install libpq                         # client-only — doesn't run a server
brew link --force libpq                    # puts psql on your PATH

# Debian / Ubuntu / WSL
sudo apt install postgresql-client

# Fedora / RHEL
sudo dnf install postgresql
```

> *Why `libpq` instead of `postgresql` on macOS?* The full Homebrew `postgresql@16` formula installs both the server and the client. You don't need the server — Docker is running it — so `libpq` (the C client library, which ships `psql`, `pg_dump`, and friends) is the leaner install.

Once installed, run:

```bash
psql "postgresql://peppercarrot:peppercarrot_dev@localhost:5432/peppercarrot" -c "SELECT version();"
```

You should see `PostgreSQL 16.x on x86_64-pc-linux-musl, ...`.

That single line is doing four things, all of which would be the *first* things to fail if your setup is broken. Worth taking apart:

| Piece | What it is |
|---|---|
| `psql` | The Postgres command-line client you just installed. |
| `"postgresql://peppercarrot:peppercarrot_dev@localhost:5432/peppercarrot"` | A [**connection URL**](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING) — the standard way to describe a Postgres connection in one string. Anatomy: `postgresql://<user>:<password>@<host>:<port>/<database>`. So this says "connect as user `peppercarrot` with password `peppercarrot_dev` to the Postgres server at `localhost:5432`, and pick the database named `peppercarrot` once connected." Every key here was set in `docker-compose.yml` from `.env` — the same shape of URL is what the FastAPI app will use in later posts. |
| `-c "..."` | "Run this one SQL statement, print the result, then exit." Without `-c`, `psql` drops you into an interactive shell — handy for exploring, but for a one-shot check `-c` is cleaner. |
| `SELECT version();` | A SQL query that asks Postgres for [its own version string](https://www.postgresql.org/docs/current/functions-info.html). It's the database equivalent of `--version` — and it's a better health check than that, because to answer it Postgres has to (1) accept your TCP connection, (2) authenticate your user/password, (3) open the named database, and (4) run a real SQL query against it. If you see a version string come back, **all four** of those work. If any one of them is broken, you get a specific error: `could not connect to server`, `password authentication failed`, `database "peppercarrot" does not exist`, etc. — each of which points at exactly which piece to fix. |

So this isn't just a "did it install?" check. It's an end-to-end probe of the whole local Postgres setup in one line.

**Option B — skip the install and use the `psql` that's already inside the container.** Slightly more typing, zero new software on your laptop:

```bash
docker exec -it peppercarrot-postgres psql -U peppercarrot -d peppercarrot -c "SELECT version();"
```

Same idea as Option A, but with a different connection mechanism. Worth taking apart in the same way:

| Piece | What it is |
|---|---|
| [`docker exec`](https://docs.docker.com/reference/cli/docker/container/exec/) | "Run a new command inside an already-running container." It does *not* start a new container — it attaches to the one you already booted with `docker compose up`. |
| `-it` | Two flags glued together. `-i` keeps STDIN open (so you can type into the process if it's interactive); `-t` allocates a [pseudo-TTY](https://en.wikipedia.org/wiki/Pseudoterminal) (so `psql`'s output is formatted the way it would be in a real terminal — aligned columns, colored prompts, that sort of thing). Both are technically optional for a one-shot `-c` command, but including them keeps the output looking right. |
| `peppercarrot-postgres` | The container to run inside — this is the `container_name:` value we set in `docker-compose.yml`. Everything after this name is the command to run *inside* that container. |
| `psql` | The Postgres client — but this time the one that ships *inside* the Postgres image. This is why you don't have to install anything on your laptop for Option B: the image already has it. |
| `-U peppercarrot` | The Postgres user to connect as. Equivalent to the `<user>` part of the connection URL in Option A. |
| `-d peppercarrot` | The database to open. Equivalent to the `<database>` part of the URL. |
| `-c "SELECT version();"` | Same as Option A — run one SQL statement and exit. |

Two things you'll notice are *missing* compared to Option A: there's no host, no port, and **no password**. That's deliberate — when `psql` runs *inside* the same container as the Postgres server, it connects over a local [Unix socket](https://en.wikipedia.org/wiki/Unix_domain_socket) instead of over TCP, and the Postgres image is configured to trust local connections from its own user without a password (this is called [*peer authentication*](https://www.postgresql.org/docs/current/auth-peer.html)). Convenient for dev; not how the FastAPI app will talk to Postgres in later posts, which goes over TCP with the full URL.

Both options produce the same result. The rest of the post shows the Option A form (the bare `psql ...` command) since it's the one you'll use most often, but every command works identically if you prefix it with `docker exec -it peppercarrot-postgres` and swap the URL for `-U peppercarrot -d peppercarrot`.

That's it — you now have a real relational database, running locally, with a known username, password, port, and database name. Every later post talks to it through this URL.

Here's a snapshot of what your laptop looks like at the end of this post — three local services on three known ports, each persisting to a known directory. The third box (FastAPI + an embedded ChromaDB) doesn't start until Post 3, but it's drawn in so the picture stays complete:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  YOUR LAPTOP                                                             │
│                                                                          │
│  Process: Postgres 16  (running inside Docker)                           │
│    Listens on  : localhost:5432   (TCP)                                  │
│    Persists to : ./data/postgres/        ← table files, gitignored       │
│                                                                          │
│  Process: Ollama daemon  (`ollama serve`)                                │
│    Listens on  : localhost:11434  (HTTP)                                 │
│    Models      : qwen2.5:7b (chat) + bge-m3 (embeddings)                 │
│    Persists to : ~/.ollama/models/       ← ~6 GB of weights              │
│                                                                          │
│  Process: FastAPI dev server  (uvicorn)        ─── starts in Post 3 ───  │
│    Listens on  : localhost:8000   (HTTP)                                 │
│    Embeds      : ChromaDB  (in the same Python process — not a server)   │
│    Persists to : ./data/chroma/          ← vector index, gitignored      │
│                                                                          │
│  Other data on disk  (no running process — just files):                  │
│    ./data/raw/                           ← one episode of comic source   │
│    ./data/images/                        ← image variants (Post 4)       │
└──────────────────────────────────────────────────────────────────────────┘
```

Three things worth noticing in this picture:

- **Three different ports, three different protocols.** Postgres speaks the [Postgres wire protocol](https://www.postgresql.org/docs/current/protocol.html) on `:5432`, Ollama speaks plain HTTP on `:11434`, and FastAPI will speak HTTP on `:8000`. Nothing overlaps, so all three can run at the same time without conflict.
- **ChromaDB is not a server.** Unlike Postgres or Ollama, ChromaDB is a *library* — it runs inside the FastAPI process, reads/writes a directory on disk, and has no port of its own. This is why it's drawn *inside* the FastAPI box, not next to it. (Post 3 explains why this is the right call for a portfolio-scale project, and Post 10 covers when you'd outgrow it.)
- **Everything that matters persists under `./data/`** (except Ollama's model weights, which Ollama manages in its own home directory). That whole tree is gitignored — `git status` should never show database rows or model files. To start completely fresh, `rm -rf data/` and re-run the workshop.

---

## Ollama: An LLM Server for Your Laptop {#ollama}

> *What's Ollama, in plain English?* [**Ollama**](https://ollama.com/) is a tiny HTTP server that runs [open-weights language models](https://ollama.com/library) locally on your machine. You `ollama pull` a model (downloads the weights, typically a few GB), and `ollama serve` (or the desktop app) makes that model reachable at `http://localhost:11434` with an [HTTP API](https://github.com/ollama/ollama/blob/main/docs/api.md) similar to the [OpenAI API](https://openai.com/api/). The point: a language model becomes a normal local service, no API key, no per-request billing, no rate limits other than your laptop's CPU/GPU.

### Why "local-first" is a real design decision, not just a budget call

You could absolutely build this project against the [Anthropic API](https://docs.claude.com/en/api/getting-started) or the [OpenAI API](https://openai.com/api/). The project's [`anthropic`](https://github.com/anthropics/anthropic-sdk-python) Python SDK is even in the dependencies — switching is a config change (Post 3 covers exactly how). But starting local-first has three benefits worth naming explicitly:

1. **You iterate without watching a meter.** During development, you'll hit the model thousands of times — embedding test corpora, debugging prompts, replaying conversations. A free local model removes the small cognitive tax of "is this experiment worth $0.40?".
2. **You'll discover provider abstractions you would have skipped.** Once your code has to talk to a model at `http://localhost:11434` *and* a hosted API behind a Bearer token, you stop hard-coding either one. That [interface](https://docs.python.org/3/library/typing.html#typing.Protocol) is the whole topic of Post 3.
3. **The cloud port becomes a one-day project.** In Post 10, we move the *same* Ollama process onto a serverless GPU on [Modal](https://modal.com/). It's the identical HTTP API — only the URL changes. No model swap. No prompt rewriting.

### Install and pull two models

[Install Ollama](https://ollama.com/download) for your OS. On macOS:

```bash
brew install ollama
ollama serve &   # or just open the Ollama.app — it runs as a menu-bar daemon
```

Pull two models:

```bash
# Chat. qwen2.5:7b is the default — it matches the cloud deploy in
# Post 10 (Modal's smaller GPU tier) and runs comfortably on a 16GB
# laptop. On 24GB+ machines, you can optionally upgrade to qwen2.5:14b
# for tighter adherence to the strict response-format rules we'll set
# in Post 8:
#   ollama pull qwen2.5:14b      # and set OLLAMA_CHAT_MODEL accordingly
ollama pull qwen2.5:7b

# Embeddings. bge-m3 is a multilingual embedding model from BAAI;
# it produces 1024-dimensional vectors and is a strong default
# for general-purpose RAG.
ollama pull bge-m3
```

> *What's the difference between a chat model and an embedding model?*
>
> A **chat model** like [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) (Alibaba's open-weights LLM) takes a sequence of text — "system prompt + conversation history + new user message" — and generates text in reply.
>
> An **embedding model** like [BGE-M3](https://huggingface.co/BAAI/bge-m3) (from the [Beijing Academy of Artificial Intelligence](https://www.baai.ac.cn/)) takes a piece of text and produces a fixed-length list of numbers — say 1024 of them — that captures the *meaning* of the text. Texts with similar meaning produce vectors that are close together in that 1024-dimensional space. We use that to do similarity search across all the page descriptions: "find me the chunks closest in meaning to the user's question."
>
> Both models live happily inside the same Ollama process. We'll touch each of them with one line of Python in Post 3.

### Verify

```bash
ollama list
# NAME                ID              SIZE      MODIFIED
# qwen2.5:7b          abc123          4.7 GB    1 minute ago
# bge-m3              def456          1.2 GB    30 seconds ago

curl http://localhost:11434/api/tags | jq '.models[].name'
# "qwen2.5:7b"
# "bge-m3"
```

If both names appear in both commands, you have a working local LLM stack.

> **A note on the vision model that *isn't* there.** Ollama also ships a `qwen2.5vl` *vision* variant. We don't use it. Page descriptions in this project come from a Claude Code [skill](https://docs.claude.com/en/docs/claude-code/skills) — Claude itself reads each comic page image and writes a structured JSON description next to it. Post 4 explains why this is a deliberate architectural call (better quality, cost-effective at portfolio scale since it rides on an existing Claude subscription rather than a per-request API bill, auditable artifacts on disk) rather than a workaround. Don't pull `qwen2.5vl` — it's 1–2 GB you don't need.

---

## A Typed Python Project with uv, mypy, and ruff {#python-project}

Now to the backend itself. We're not writing application code yet; we're just bootstrapping a Python project that's strict about what it accepts.

```bash
cd backend
uv sync
```

That single command:

1. Creates a virtual environment in `.venv/`.
2. Reads `pyproject.toml` and the lockfile (`uv.lock`).
3. Installs every dependency at the exact pinned version.

> *What's a lockfile?* A second file (`uv.lock`) that records every transitive dependency at the exact resolved version, including its hash. Two developers running `uv sync` on the same lockfile get byte-identical environments. It's the same idea as [`package-lock.json`](https://docs.npmjs.com/cli/v10/configuring-npm/package-lock-json) in npm or [`Cargo.lock`](https://doc.rust-lang.org/cargo/guide/cargo-toml-vs-cargo-lock.html) in Rust.

### A tour of `pyproject.toml`

Three sections in the project's [`pyproject.toml`](https://packaging.python.org/en/latest/specifications/pyproject-toml/) are worth pausing on:

```toml
[project]
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "sse-starlette>=2.1",
    "pydantic[email]>=2.9",
    "pydantic-settings>=2.5",
    "sqlalchemy[asyncio]>=2.0.36",
    "asyncpg>=0.30",
    "alembic>=1.14",
    "chromadb>=0.5",
    "pillow>=11",
    "blurhash>=1.1",
    "httpx>=0.27",
    "anthropic>=0.39",
    "sentence-transformers>=3.3",
    "boto3>=1.35",
    "aiofiles>=24.1",
    "structlog>=24.4",
    "python-dotenv>=1.0",
]
```

Each line is a deliberate choice — a one-sentence-each tour:

- **fastapi + uvicorn** — async web framework + an [ASGI](https://asgi.readthedocs.io/en/latest/) server to run it. The async story matters because we stream model output token-by-token in Post 7.
- **sse-starlette** — clean [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) support, which is how the browser receives streamed tokens.
- **pydantic / pydantic-settings** — typed data models and typed environment-variable config (no `os.getenv("…")` strings sprayed across the code).
- **sqlalchemy[asyncio] + asyncpg + alembic** — the database stack: ORM, async Postgres driver, migrations.
- **chromadb** — the vector store that holds embeddings. Embedded in-process; no separate server.
- **pillow + blurhash** — image processing and [blurhash](https://blurha.sh/) generation for placeholder thumbnails.
- **httpx + anthropic + sentence-transformers** — model clients. `httpx` for Ollama's HTTP API, `anthropic` for the cloud swap-in (Post 3), `sentence-transformers` as a fallback embedding provider.
- **boto3 + aiofiles** — S3-compatible storage (Cloudflare R2 in Post 10) and async filesystem.
- **structlog** — structured JSON logging, which makes the deploy logs in Post 10 actually grep-able.

And then the two type/lint tools, which I want to draw extra attention to:

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_ignores = true
disallow_untyped_defs = true

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "RUF"]
```

> *What are these?* [**mypy**](https://mypy-lang.org/) is Python's most widely used static type checker — it reads your type annotations and flags places where the types don't line up. [**ruff**](https://docs.astral.sh/ruff/) is a [Rust](https://www.rust-lang.org/)-based linter and formatter from the same team behind `uv`. Where `mypy` catches type bugs, `ruff` catches style bugs, dead imports, undefined variables, and a few hundred other classes of mistake — all in well under a second on a project this size.

`strict = true` is a strong opinion. It enforces:

- Every function must have type annotations on its arguments and return type.
- No implicit `Any` (the type that silently accepts anything).
- Imports without type stubs are explicit failures.

Setting `mypy --strict` on a brand-new project is *cheap*. Setting it on a 50,000-line legacy codebase is a months-long migration. The right time is now.

### Verify

```bash
# from backend/
uv run mypy app/                # no output = pass
uv run ruff check app/          # "All checks passed!"
uv run python -c "import fastapi, sqlalchemy, chromadb; print('ok')"
```

If all three pass, the Python project is sound.

---

## The First Migration: Real Schema in Real Postgres {#first-migration}

> *What's a migration?* A migration is a versioned script — usually one `.py` file per change — that takes the database from one schema (the set of tables, columns, indexes, constraints) to the next. Run all of them in order on a fresh database and you arrive at the current schema. [**Alembic**](https://alembic.sqlalchemy.org/en/latest/) is the migration tool that ships with SQLAlchemy. It can *autogenerate* a migration by diffing your declared models (Python classes) against the live database — though as we'll see in a minute, you should never trust autogenerate blindly.

### How Alembic knows what to do: `alembic.ini` and `alembic/env.py`

Before `alembic revision` does anything useful, two files on disk wire the tool to this project:

- **`backend/alembic.ini`** is the [INI file](https://en.wikipedia.org/wiki/INI_file) telling Alembic where the migration directory lives (`alembic/`), what to name new revision files, and (nominally) the Postgres URL. The interesting line is `sqlalchemy.url =` — left empty on purpose, because the actual URL is injected at runtime from `.env`. There's also a `[post_write_hooks]` section that pipes every autogenerated migration through `ruff --fix` so the formatting matches the rest of the codebase out of the box.
- **`backend/alembic/env.py`** is the Python script Alembic runs on every command. Three load-bearing lines: it imports `Base` from `app.db.models` (so `Base.metadata` knows every model class), calls `get_settings()` and sets `sqlalchemy.url` to `settings.database_url` (overriding the empty placeholder in `alembic.ini`), and sets `target_metadata = Base.metadata` — that's the "what should the schema look like?" source of truth `--autogenerate` diffs the live database against.

You don't write either from scratch. `alembic init alembic` (a one-time bootstrap in any new SQLAlchemy project) creates both with sensible defaults; the project's `env.py` has a small patch to read the URL from `Settings` and to register the `Base` import, plus the async-engine plumbing that matches the rest of the codebase. If you cloned the repo they exist already; if you're rebuilding from scratch as you read along, run `uv run alembic init alembic` from `backend/` first, then mirror those edits into the generated `env.py`.

### Generate the initial migration

```bash
cd backend
uv run alembic revision --autogenerate -m "initial schema"
```

This reads `backend/app/db/models.py` (the SQLAlchemy 2.0 typed declarative models), compares them against the empty database, and writes a file like:

```
backend/alembic/versions/a1b2c3d4e5f6_initial_schema.py
```

**Open that file and read it.** I mean it. Autogenerate is great but not perfect, and the bugs it introduces are silent and load-bearing. Specifically check:

- All 10 application tables are present: `episodes`, `pages`, `characters`, `page_characters`, `wiki_articles`, `commentary_notes`, `chat_sessions`, `chat_messages`, `world_entities`, `world_relationships`. (An eleventh table, `alembic_version`, gets created automatically.)
- JSONB columns use `postgresql.JSONB`, **not** generic `sa.JSON`. Autogenerate sometimes downgrades these. ChromaDB metadata queries in Post 6 depend on JSONB specifically, so this matters.
- Array columns (`mood_tags`, `aliases`) use `postgresql.ARRAY(sa.String())`.
- The unique constraint on `(episode_id, page_number)` in `pages` exists.
- The `(session_id, created_at)` index on `chat_messages` exists — chat history retrieval scans by it on every request.

If anything is off, **edit the migration file directly** rather than regenerating. Autogenerate occasionally drops things on regeneration if its inference of "current state" drifts.

> *Why we have these specific tables, in one paragraph.* `episodes` and `pages` are the unit-of-content rows the flipbook UI reads from. `characters` and `page_characters` power the (future) character chips on the page indicator. `wiki_articles` holds the curated universe lore that wiki-mode chat retrieves from in Post 8. `chat_sessions` and `chat_messages` are the conversation audit log. (`commentary_notes` is a forward-looking placeholder — see the feature-to-table walk-through below for what it's for and why it's in the schema without a UI yet.) `world_entities` and `world_relationships` are the knowledge-graph tables that power the world-graph overlay in Post 9 — declared in `models.py` from day one, so this same initial migration creates them, even though nothing touches them until later. The full schema with field-by-field rationale will live in `docs/data-model.md` once the repo goes public with Post 10.

### Why not just `Base.metadata.create_all()`?

Now that you have a generated migration in front of you, it's worth asking: was all this necessary? SQLAlchemy ships a one-liner — `Base.metadata.create_all(engine)` — that walks your declared models and issues `CREATE TABLE` for any that don't exist. No Alembic, no `versions/` directory, no autogenerate. Why bother?

Three reasons, each load-bearing:

1. **`create_all` doesn't migrate.** It creates from scratch. If you add a column tomorrow, `create_all` shrugs at your existing tables — it sees they're already there and skips them. Alembic instead generates a fresh `ALTER TABLE ... ADD COLUMN` migration that updates the live schema in place.
2. **It produces an audit trail.** Every change is a file in `alembic/versions/` with a hash, a parent revision, and a human-readable description. `git log` becomes a history of your database too — invaluable when you're three months in and need to know when a column was added.
3. **It's how every production deploy will work.** When we deploy to [Neon](https://neon.tech/) in Post 10, the deploy script runs `alembic upgrade head` against the production database. There's exactly one way to make schema changes safe across environments, and it starts on day one — not the week you first need it on a database with real users in it.

### Apply it

```bash
uv run alembic upgrade head
```

`head` means "the latest revision." On a fresh database, that's our one new migration.

Verify by listing tables:

```bash
psql "postgresql://peppercarrot:peppercarrot_dev@localhost:5432/peppercarrot" -c "\dt"

#                  List of relations
#  Schema |         Name          | Type  |    Owner
# --------+-----------------------+-------+--------------
#  public | alembic_version       | table | peppercarrot
#  public | chat_messages         | table | peppercarrot
#  public | chat_sessions         | table | peppercarrot
#  public | characters            | table | peppercarrot
#  public | commentary_notes      | table | peppercarrot
#  public | episodes              | table | peppercarrot
#  public | page_characters       | table | peppercarrot
#  public | pages                 | table | peppercarrot
#  public | wiki_articles         | table | peppercarrot
#  public | world_entities        | table | peppercarrot
#  public | world_relationships   | table | peppercarrot
```

Eleven rows. Now insert a test row to prove the schema is healthy:

```bash
psql "postgresql://peppercarrot:peppercarrot_dev@localhost:5432/peppercarrot" \
  -c "INSERT INTO characters (id, name, aliases, image_url) \
      VALUES (gen_random_uuid(), 'Pepper', ARRAY['the witch'], NULL); \
      SELECT name, aliases FROM characters;"
#  name  |   aliases
# -------+-------------
#  Pepper | {the witch}
```

If that query lands without an error, your schema is real, queryable, and ready for the next post.

> *A gotcha worth flagging — skip this if you're using the Docker Postgres we set up earlier.* The verification query above uses [`gen_random_uuid()`](https://www.postgresql.org/docs/current/functions-uuid.html), Postgres's built-in function for minting a fresh random ID. In PostgreSQL 13 and later (including the `postgres:16-alpine` image), it works out of the box. On older Postgres — say a system-installed one from 2019 — the same function lives in an optional add-on called an [*extension*](https://www.postgresql.org/docs/current/external-extensions.html), specifically [`pgcrypto`](https://www.postgresql.org/docs/current/pgcrypto.html), that has to be turned on first. Without it you'll see `function gen_random_uuid() does not exist`.
>
> The fix is to enable the extension at the start of the migration. Add this one line to the top of the `upgrade()` function in your generated migration file, then re-run `alembic upgrade head`:
>
> ```python
> op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
> ```
>
> (`op` is the Alembic helper available inside every migration file; `op.execute(...)` runs raw SQL within the same transaction as the migration's other operations, so the extension is in place before any `CREATE TABLE` runs.)

### From features to tables: how the schema mirrors the app

A natural question reading the table list is: *why these eight, and not some other set?* The schema isn't arbitrary — every table comes from one specific feature of the product. Walking through it feature by feature is the fastest way to see what each table is doing.

Start with the **reading experience** — an AI flipbook of a webcomic. The comic is organized into *episodes*, and each episode is an ordered sequence of *pages*. That gives us two tables and one relationship:

- `episodes` — one row per episode, with the metadata the picker needs: `title`, `episode_number`, cover image, plot summary, publish date.
- `pages` — one row per page, with an `episode_id` column pointing back to its episode and a `page_number` marking its slot within. A unique constraint on `(episode_id, page_number)` enforces "no two pages share a slot in the same episode."

Now the **AI part**: the chat needs to talk about what the reader is currently looking at. To do that, every page needs a prose *description* the chat layer can embed and retrieve. That's the `visual_description` text column on `pages` — the same text Post 4 (ingestion) will hand to the embedding model.

Add **character-aware features**: character chips in the page indicator, "next time this character appears" navigation, and anchoring chat answers to canonical names (so the AI says "Pepper" instead of inventing a fresh name for an unfamiliar witch each time). We need a roster *and* we need to know which pages each character appears on. Two more tables:

- `characters` — one row per named character (Pepper, Carrot, the three Chaosah witches, etc.), with an `aliases` array column for the inevitable "the cat" → Carrot mapping.
- `page_characters` — a *join table*: its only purpose is to connect `pages` and `characters` in a many-to-many relationship (a page can show many characters; a character appears on many pages). Each row is two foreign keys (`page_id`, `character_id`) and nothing else — a pure linkage, with no data of its own.

Add **wiki mode**: a second retrieval pipeline for "what is Chaosah?" or "tell me about the Magic Sand" — questions about universe lore that aren't tied to any specific page. Wiki content has different chunking, a different retrieval `k`, and (because universe facts aren't plot spoilers) skips the spoiler filter that page mode applies. It deserves its own table:

- `wiki_articles` — one row per article (`slug`, `title`, full `content` as markdown, `category` like `school` or `creature`).

Finally, the **conversations themselves** need to persist — a user might close the tab and come back tomorrow to keep chatting about ep07, page 3. So:

- `chat_sessions` — one row per conversation, with `episode_id` and a `current_page` column that updates as the reader flips so the AI always knows the spread the conversation is grounded in.
- `chat_messages` — one row per turn (user or assistant), with `mode` (`page` or `wiki`), `content`, and — for assistant messages — the `retrieved_doc_ids` audit column we'll come back to in a minute.

Those are the seven tables driven by current product features. There's also one **forward-looking placeholder** in the schema: `commentary_notes`. The idea behind it was a small UI feature — surfacing excerpts from David Revoy's making-of blog posts (he often writes about how a specific page came together) inline with the comic. The data shape is straightforward (one row per note, tied to an `episode_id` and an optional `page_number_hint`), and the source content is already part of the archive we'll download from peppercarrot.com in the next section. But the UI to surface it isn't built in the current version of the app — the table is in the schema as a hook for a feature that may or may not ship later. I'm leaving it in the migration mostly because removing tables later is cheap; provisioning them retroactively against production data is less so.

Two more tables — `world_entities` and `world_relationships` — round out the schema. They power the world-graph overlay [Post 9](#) covers in depth, but they're already declared in `models.py` and therefore created by *this* initial migration alongside everything else. We won't touch them until the world-graph phase, but they're sitting there in `\dt` from day one because there's nothing to gain from staging them in a later migration.

The point of this exercise isn't to memorize the tables. It's to notice that **the schema is a direct translation of the product**, not a generic data model pulled off a shelf. When you're designing a schema for your own app, the same move applies: list the features the product is going to support — or might plausibly support soon — and ask what data each one needs. Let the answers cluster into tables. The result is a schema where every table can name the specific feature it exists for. You won't find generic `users` / `tags` / `permissions` / `revisions` tables sitting there "just in case" — and that absence is the point. Schemas full of speculative tables are how data models calcify before the product is even shipped.

### Four design decisions worth naming

A migration is just a `CREATE TABLE` script. The interesting parts of a data model are the design decisions that *aren't* visible in the DDL — the choices about what to store where, in what shape, and what to leave out. Four worth flagging now, because each one shows up as a load-bearing choice in a later post:

**1. Postgres holds the text of truth; ChromaDB holds embeddings + IDs only.**

A naive RAG app stores every chunk's text in both places — the embedding in Chroma and the same text duplicated as a column. That gives you two problems: data drift (which copy is canonical?) and storage bloat (a 1024-dim float vector is a few KB; the same text in prose might be twice that, often more). This schema picks the opposite shape. `pages.visual_description` is the canonical text in Postgres. When Post 4 (ingestion) embeds those descriptions, it writes to Chroma only the *embedding* plus a small metadata payload — notably the page's UUID — and *not* the text itself. At retrieval time (Post 6), the orchestrator asks Chroma for matching IDs, then re-fetches the actual text from Postgres. One source of truth, two indexes pointing at it.

**2. `pages.image_url` is a relative key, not a full URL.**

The column stores `episodes/ep01-pollution/pages/001-display.webp`, not `http://localhost:8000/images/episodes/ep01-pollution/pages/001-display.webp`. The full URL is composed at API response time by whichever storage backend is active — a `LocalStorage` implementation when you're on `STORAGE_BACKEND=local`, an `R2Storage` implementation when Post 10 flips that to `r2`. The indirection lets you swap your image host with a one-line config change instead of an `UPDATE` over every row in `pages`. (The latter is easy with three pages and a single shared prefix; it's a Saturday-night incident with three thousand rows and a half-dozen historical prefixes accreted over a year.) Post 3 builds the `StorageClient` interface that consumes these relative keys.

**3. `image_metadata` is JSONB, not separate columns.**

Width, height, blurhash, dominant color — you might expect each as its own typed column. They're a single JSONB blob instead, for two reasons:

- *Co-fetched and rarely queried in isolation.* The frontend reads them all together when rendering a page; nothing in the app queries "every page wider than 1000px." When you never filter on a field, putting it in its own column buys you nothing.
- *The shape will evolve.* Today the blob is `{width, height, blurhash, dominant_color}`. Tomorrow we might add `palette`, `alt_text`, or `dominant_emotion`. With JSONB that's an edit to `models.py`; with separate columns every new field is a migration.

The tradeoff: no [GIN](https://www.postgresql.org/docs/current/gin.html) index by default, no per-field type safety from the DB. Both are worth giving up for metadata that's always read as a bundle.

**4. `chat_messages.retrieved_doc_ids` is the column that makes retrieval debuggable.**

Every assistant message will record exactly which Chroma chunks the orchestrator pulled to compose its prompt — stored as a JSONB array of IDs. This sounds like a boring audit column. It's the single most useful column in the schema.

When you're tuning retrieval in Post 6 and a user reports "the chat answered wrong about page 4," you need to know *what context the model actually saw* at the moment it answered. Rebuilding that from logs is usually impossible — the chunk store keeps moving as you re-ingest, so the same query at a different time returns different results. With `retrieved_doc_ids` recorded inline, every assistant turn is fully replayable: pull the IDs, fetch those chunks from Chroma, look at exactly what the prompt contained, then decide whether the bug is in retrieval (wrong chunks) or generation (right chunks, wrong answer). Mature LLM-app codebases all have a column like this under various names — "trace," "context manifest," "retrieval ledger." Whatever it's called, build it on day one. Adding it retroactively means losing the audit trail for everything that happened before.

---

## Your First Episode in `data/raw/` {#first-episode}

The last preflight step: get one episode of the comic onto disk in a structured way that the ingestion pipeline (Post 4) can consume.

Lots of "chat with X" demos start by web-scraping HTML. *Pepper & Carrot* has a much nicer path: David Revoy's project [publishes a fully structured source archive](https://www.peppercarrot.com/0_sources/) with three JSON manifests that make the whole corpus programmatically accessible:

- **`/0_sources/episodes-v1.json`** — master list of every episode and the canonical filename for each page slot.
- **`/0_sources/{slug}/info.json`** — per-episode metadata: publication date, original language, software credits.
- **`/0_sources/{slug}/hi-res/titles.json`** — episode title in every language.

Page filenames follow a deterministic pattern (`{lang}_Pepper-and-Carrot_by-David-Revoy_E{NN}P{NN}.{ext}`), so an acquisition script can build every URL it needs without scraping any HTML at all.

> *A small portfolio judgement worth saying out loud.* The "right" way to ingest someone else's content is almost always: read their data the way they meant you to read it. Web scraping HTML is a fragile fallback that breaks the moment the publisher restyles a button. Looking for a JSON or RSS feed *first* — even if it takes an extra hour of investigation — is the move. In this case, half a day spent reading `/0_sources/` instead of scraping pays off across the rest of the project.

The project includes `ingestion/acquire.py`, a small CLI that does exactly that. Run it once to grab episode 1 in English:

```bash
cd ingestion
uv run python acquire.py list                                # prints all available episodes
uv run python acquire.py episode \
    --slug ep01_Potion-of-Flight \
    --lang en \
    --out ../data/raw
uv run python acquire.py commentary \
    --slug ep01_Potion-of-Flight \
    --out ../data/raw
```

You should end up with:

```bash
$ ls ../data/raw/ep01-potion-of-flight/
commentary.html  commentary.url  cover.jpg  metadata.yaml  pages/

$ ls ../data/raw/ep01-potion-of-flight/pages/
page_001.jpg  page_002.jpg  page_003.jpg

$ cat ../data/raw/ep01-potion-of-flight/metadata.yaml
slug: ep01-potion-of-flight
title: 'Episode 1: The Potion of Flight'
episode_number: 1
language: en
published_at: 2014-05-31
credits_url: https://www.peppercarrot.com/en/webcomic/ep01_Potion-of-Flight.html
commentary_url: https://www.davidrevoy.com/...
```

**Open `page_001.jpg` in an image viewer** to confirm it's a real comic page and not (e.g.) a 404 HTML body that got saved with a `.jpg` extension. Once that opens to a hand-painted Pepper, you have real input data.

> One episode is enough for the next few posts. When you reach Post 5 (the frontend reading MVP) and want a populated episode picker, come back here and run `uv run python acquire.py all --lang en --out ../data/raw --limit 5` to grab the first five. The full 39-episode pull is roughly 1–2 GB and takes a few minutes — the acquisition script throttles itself to be polite to the upstream server. **Everything is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**: free to use *with attribution*. Don't strip the `credits_url` from `metadata.yaml` — the UI will surface it visibly later.

---

## The `.env` File That Ties It Together {#env}

`.env` is the single source of truth for "what URLs and credentials does this project use?" — it's gitignored, lives at the project root, and is loaded by [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) at startup into a typed `Settings` object.

Copy the template and edit only what you need:

```bash
cp .env.example .env
```

The local-dev defaults already match the workshop you just built. A walk through the keys you'll see:

```bash
# === Database ===
POSTGRES_USER=peppercarrot
POSTGRES_PASSWORD=peppercarrot_dev
POSTGRES_DB=peppercarrot
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# === Vector store ===
CHROMA_PERSIST_DIR=./data/chroma     # ChromaDB writes here as a flat directory

# === Storage (Post 3 swaps this to R2 in cloud builds) ===
STORAGE_BACKEND=local
LOCAL_IMAGE_DIR=./data/images
LOCAL_IMAGE_URL_PREFIX=http://localhost:8000/images

# === Model providers ===
VISION_PROVIDER=json                 # the Claude Code skill (Post 4)
OLLAMA_BASE_URL=http://localhost:11434
CHAT_PROVIDER=ollama
OLLAMA_CHAT_MODEL=qwen2.5:7b         # upgrade to qwen2.5:14b on 24GB+ RAM
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=bge-m3
```

> *Why two keys for the embedding model?* `EMBEDDING_PROVIDER` says *which client library* to talk to (Ollama or [`sentence-transformers`](https://www.sbert.net/)). `EMBEDDING_MODEL` says *which model name* to ask for — and the naming convention is provider-specific. Ollama wants `bge-m3`; sentence-transformers wants `BAAI/bge-m3`. Same underlying weights, same 1024-dimensional vectors, but the strings differ. We pry these apart in Post 3 and you'll see exactly why.

The point of the file isn't to memorize what every key does — most of them get explained in the post that first uses them. The point is that *all* runtime configuration lives in one well-known file, and the code never reads `os.environ` directly. The `Settings` class in `backend/app/config.py` is the only door.

---

## Verification: The Whole Workshop, End to End {#verification}

If you've followed along, run this exact sequence one more time as a clean sanity check. **Every command should succeed.**

```bash
# Tools
docker --version
uv --version
node --version
claude --version

# Postgres reachable, schema in place
docker compose ps                                                       # postgres should be (healthy)
psql "postgresql://peppercarrot:peppercarrot_dev@localhost:5432/peppercarrot" -c "\dt"  # 11 tables

# Ollama serving both models
ollama list                                                              # qwen2.5:7b AND bge-m3
curl -s http://localhost:11434/api/tags | jq '.models[].name'

# Backend Python deps installed and type-clean
cd backend && uv sync
uv run mypy app/                                                         # no errors
uv run ruff check app/                                                   # All checks passed!
uv run python -c "import fastapi, sqlalchemy, chromadb; print('ok')"

# Frontend deps installed
cd ../frontend && npm install && npm run type-check                      # tsc passes

# One episode on disk
ls ../data/raw/ep01-potion-of-flight/pages/                              # at least page_001.jpg
```

If every line in that block prints what's expected, you're done. The workshop is built. Every later post in this series — from "make the page descriptions" in Post 3, through "stream tokens to the browser" in Post 7, through "deploy to Modal and Fly" in Post 10 — assumes a working version of this checklist.

If something fails, **fix it now**. Setup bugs that survive into application development become two-day debugging sessions because they masquerade as application bugs.

---

## Key Takeaways {#key-takeaways}

**1. Treat setup as a real engineering deliverable, not a prelude.** The thirty minutes it takes to verify Postgres, Ollama, mypy, ruff, Alembic, and `data/raw/` independently saves you days of "is this an app bug or a setup bug" later. On a portfolio project, "it runs cleanly on a fresh clone" is the *first* thing a reviewer notices.

**2. `mypy --strict` on day one is cheap; `mypy --strict` on day 600 is a migration.** Bolt the guardrail on while the codebase is still empty. Every later file gets written with full annotations as a matter of course, and the cost is invisible.

**3. Migrations exist from commit #1, not when "we get to production."** Even if today's database is just for one developer, `alembic upgrade head` is the same command on a laptop and on Neon in Post 10. The first migration is a 30-line file. The first migration *retrofitted* onto an established project is a multi-day archaeology project.

**4. Read the data publisher's data, not their HTML.** *Pepper & Carrot* publishes machine-readable JSON manifests at `/0_sources/`. Spending half a day reading those instead of scraping HTML pays off across the rest of the project — and it's a habit that generalizes far beyond webcomics.

**5. Local-first isn't free, but it's worth the cost.** Ollama with `qwen2.5:7b` is genuinely slower than calling [Claude Haiku](https://www.anthropic.com/news/claude-haiku-4-5) or OpenAI's `gpt-4o-mini`. What you get in exchange is unmetered iteration, a forcing function for provider abstraction, and a deploy story (Post 10) where the same `httpx.AsyncClient` talks to a serverless GPU in production. The free iteration loop is what makes prompt engineering (Post 8) feasible at all.

---

## Appendix A: What the SQLAlchemy Models Actually Look Like {#appendix-models}

Throughout this post I've referenced "the SQLAlchemy 2.0 typed declarative models in `backend/app/db/models.py`" without ever showing one. This appendix closes that gap. The code below is exactly what Alembic's autogenerate diffs against to produce the migration you just applied — every column, type, and constraint mentioned in the design discussions earlier is a line of Python below.

If you've never seen SQLAlchemy 2.0's typed style before, here's the one-sentence summary: **each column is a class attribute typed with `Mapped[X]` (the Python type) and assigned to `mapped_column(...)` (the database type and constraints)**. The two halves are kept in agreement by [`mypy --strict`](https://mypy-lang.org/) — if you say a column is `Mapped[str]` and then map it to a column that allows NULL, the type-check fails.

### Three terms used throughout: parent, child, join

The section headings below — and the annotations on each model — lean on three terms that come up whenever a relational database holds connected things. Worth nailing down before the code:

- **Parent table** — a table other tables point at via [foreign keys](https://www.postgresql.org/docs/current/tutorial-fk.html). Nothing references *down into* it; its rows exist on their own. `episodes` is a parent.
- **Child table** — a table whose rows belong to a parent. Each row carries a foreign-key column pointing at the parent's primary key. `pages` is a child of `episodes` (each page row has an `episode_id`).
- **Join table** — a table whose only purpose is to connect *two other tables* in a many-to-many. Its rows are pure linkages: two foreign keys, no data of its own. `page_characters` joins `pages` to `characters`.

The same table can be both: `pages` is a child of `episodes` *and* a parent of `page_characters`. Trace the foreign-key arrows — arrows pointing *into* a table mean it's playing a parent role; arrows pointing *out* mean it's a child or part of a join. Most tables in a real schema have both.

Here's how those four roles play out across this schema's tables. Each arrow points from a child to its parent (the direction the foreign key references):

<div style="margin: 1.5rem 0; overflow-x: auto;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 640" role="img"
     aria-label="Foreign-key diagram of the eight schema tables, showing parent, child, join, and standalone roles."
     style="display: block; width: 100%; height: auto; max-width: 900px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">
  <defs>
    <marker id="fk-arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>
    </marker>
  </defs>

  <!-- Foreign-key arrows: each points from a CHILD up to its PARENT -->
  <!-- pages.episode_id -> episodes.id (CASCADE) -->
  <line x1="105" y1="180" x2="420" y2="104" stroke="#6b7280" stroke-width="1.5" marker-end="url(#fk-arrow)"/>
  <!-- chat_sessions.episode_id -> episodes.id (CASCADE) -->
  <line x1="295" y1="180" x2="440" y2="104" stroke="#6b7280" stroke-width="1.5" marker-end="url(#fk-arrow)"/>
  <!-- commentary_notes.episode_id -> episodes.id (CASCADE) -->
  <line x1="515" y1="180" x2="470" y2="104" stroke="#6b7280" stroke-width="1.5" marker-end="url(#fk-arrow)"/>
  <!-- characters.first_appearance_episode_id -> episodes.id (SET NULL) -->
  <line x1="735" y1="180" x2="490" y2="104" stroke="#6b7280" stroke-width="1.5"
        stroke-dasharray="6,4" marker-end="url(#fk-arrow)"/>
  <!-- chat_messages.session_id -> chat_sessions.id (CASCADE) -->
  <line x1="295" y1="320" x2="295" y2="244" stroke="#6b7280" stroke-width="1.5" marker-end="url(#fk-arrow)"/>
  <!-- page_characters.page_id -> pages.id (CASCADE) -->
  <line x1="365" y1="460" x2="135" y2="244" stroke="#6b7280" stroke-width="1.5" marker-end="url(#fk-arrow)"/>
  <!-- page_characters.character_id -> characters.id (CASCADE) -->
  <line x1="515" y1="460" x2="710" y2="244" stroke="#6b7280" stroke-width="1.5" marker-end="url(#fk-arrow)"/>

  <!-- episodes (PARENT) -->
  <g>
    <rect x="370" y="42" width="160" height="62" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="450" y="70" text-anchor="middle" font-size="15" font-weight="600" fill="#1f2937">episodes</text>
    <text x="450" y="90" text-anchor="middle" font-size="11" fill="#92400e" font-weight="600">PARENT</text>
  </g>

  <!-- pages (CHILD + parent) -->
  <g>
    <rect x="40" y="182" width="130" height="62" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="105" y="210" text-anchor="middle" font-size="15" font-weight="600" fill="#1f2937">pages</text>
    <text x="105" y="230" text-anchor="middle" font-size="11" fill="#1e40af" font-weight="600">CHILD + parent</text>
  </g>

  <!-- chat_sessions (CHILD + parent) -->
  <g>
    <rect x="210" y="182" width="170" height="62" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="295" y="210" text-anchor="middle" font-size="15" font-weight="600" fill="#1f2937">chat_sessions</text>
    <text x="295" y="230" text-anchor="middle" font-size="11" fill="#1e40af" font-weight="600">CHILD + parent</text>
  </g>

  <!-- commentary_notes (CHILD, leaf) -->
  <g>
    <rect x="420" y="182" width="190" height="62" rx="8" fill="#f1f5f9" stroke="#64748b" stroke-width="1.5"/>
    <text x="515" y="210" text-anchor="middle" font-size="15" font-weight="600" fill="#1f2937">commentary_notes</text>
    <text x="515" y="230" text-anchor="middle" font-size="11" fill="#475569" font-weight="600">CHILD, leaf</text>
  </g>

  <!-- characters (optional CHILD + parent) -->
  <g>
    <rect x="650" y="182" width="170" height="62" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="735" y="205" text-anchor="middle" font-size="15" font-weight="600" fill="#1f2937">characters</text>
    <text x="735" y="223" text-anchor="middle" font-size="11" fill="#1e40af" font-weight="600">optional CHILD</text>
    <text x="735" y="237" text-anchor="middle" font-size="11" fill="#1e40af" font-weight="600">+ parent</text>
  </g>

  <!-- chat_messages (CHILD, leaf) -->
  <g>
    <rect x="210" y="322" width="170" height="62" rx="8" fill="#f1f5f9" stroke="#64748b" stroke-width="1.5"/>
    <text x="295" y="350" text-anchor="middle" font-size="15" font-weight="600" fill="#1f2937">chat_messages</text>
    <text x="295" y="370" text-anchor="middle" font-size="11" fill="#475569" font-weight="600">CHILD, leaf</text>
  </g>

  <!-- page_characters (JOIN) -->
  <g>
    <rect x="320" y="460" width="240" height="62" rx="8" fill="#d1fae5" stroke="#059669" stroke-width="1.5"/>
    <text x="440" y="488" text-anchor="middle" font-size="15" font-weight="600" fill="#1f2937">page_characters</text>
    <text x="440" y="508" text-anchor="middle" font-size="11" fill="#065f46" font-weight="600">JOIN</text>
  </g>

  <!-- wiki_articles (STANDALONE) -->
  <g>
    <rect x="680" y="460" width="170" height="62" rx="8" fill="#f3f4f6" stroke="#9ca3af" stroke-width="1.5"/>
    <text x="765" y="488" text-anchor="middle" font-size="15" font-weight="600" fill="#1f2937">wiki_articles</text>
    <text x="765" y="508" text-anchor="middle" font-size="11" fill="#4b5563" font-weight="600">STANDALONE</text>
  </g>

  <!-- Legend -->
  <g>
    <line x1="40" y1="572" x2="86" y2="572" stroke="#6b7280" stroke-width="1.5" marker-end="url(#fk-arrow)"/>
    <text x="96" y="576" font-size="11" fill="#4b5563">solid arrow = FK with ondelete="CASCADE"</text>
    <line x1="40" y1="596" x2="86" y2="596" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="6,4" marker-end="url(#fk-arrow)"/>
    <text x="96" y="600" font-size="11" fill="#4b5563">dashed arrow = FK with ondelete="SET NULL"</text>
    <text x="450" y="625" text-anchor="middle" font-size="11" fill="#6b7280" font-style="italic">Each arrow points from a child to its parent (the direction the foreign key references).</text>
  </g>
</svg>
</div>

Reading the diagram:

- **`(PARENT)`** — nothing FKs out of it. `episodes` is the only pure parent in this schema.
- **`(CHILD + parent)`** — has a foreign key out *and* something else FKs into it. `pages`, `chat_sessions`, and `characters` are all in this dual role.
- **`(CHILD, leaf)`** — has a FK out but nothing points at it. `commentary_notes` and `chat_messages` sit at the ends of their chains.
- **`(JOIN)`** — two FKs out, no other data. `page_characters` is the only one — it links `pages` ↔ `characters` for the many-to-many.
- **`(STANDALONE)`** — no FK in or out. `wiki_articles` is the floor of the schema's complexity, sitting off to the side.

One nuance the diagram glosses over for simplicity: the `characters → episodes` arrow is `first_appearance_episode_id`, which uses `ondelete="SET NULL"` rather than the `CASCADE` used by every other arrow. So deleting an episode doesn't delete the characters who first appeared in it — it just clears their "first appearance" pointer. The `PageCharacter` annotation later in this appendix has the full discussion of `CASCADE` vs `SET NULL`.

### The imports and a couple of helpers

Every model lives in `models.py`. These lines appear once at the top of the file:

```python
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime, Float, ForeignKey, Index, String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Common base for all ORM models."""


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )


def _timestamp_now() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
```

Three things to notice:

- **`from __future__ import annotations`** enables [PEP 563](https://peps.python.org/pep-0563/) lazy evaluation of type hints. Without it, the forward references between models (e.g. `Page` referencing `Episode` before `Episode` is defined) would crash at import time. Add this line to any Python file with model classes that reference each other.
- **`class Base(DeclarativeBase):`** is the parent every model inherits from. It's how SQLAlchemy knows which classes are tables.
- **`_uuid_pk()` and `_timestamp_now()`** are small helpers. Every row in this schema has a UUID primary key and most have a `created_at` / `ingested_at` timestamp. Factoring those out keeps each model class focused on the columns that actually differ.

The next three sections show three representative model classes — `Episode`, `Page`, `ChatMessage` — each connecting directly to a design decision from earlier in the post.

### The full schema, every column

The role diagram above grouped tables by structural role. This one drops a level: every column, type, key, default, and FK target on all ten tables — including `world_entities` and `world_relationships`, which the role diagram skips because they're introduced later in the series.

<iframe src="{{ '/assets/embed/peppercarrot_data_model_erd.html' | relative_url }}" loading="lazy" style="width: 100%; height: 780px; border: 1px solid rgba(115, 114, 108, 0.18); border-radius: 6px; display: block; margin: 1rem 0;" title="Pepper & Carrot data model ERD (interactive — drag to pan, use ± to zoom)"></iframe>

A few patterns worth noticing once everything is in one frame:

- **Every table has a UUID primary key with `default uuid4()`** — the `_uuid_pk()` helper above is the single line of code those ten `id` columns share.
- **Every FK in the schema is one of two flavors:** `ON DELETE CASCADE` (children disappear with the parent) or `ON DELETE SET NULL` (children survive, pointer is cleared). There's no `RESTRICT` and no orphan rows.
- **JSONB and `text[]` appear only on `pages`, `chat_messages` (and `characters.aliases`).** Everywhere else, the schema is plain typed columns — JSONB is reserved for genuinely open-ended payloads (raw image EXIF, per-page mood lists, retrieval doc-id traces, model token counts), not used as a substitute for proper columns.
- **`world_entities` and `world_relationships`** carry their own `created_at` + `updated_at` because, unlike the read-mostly ingestion tables, they're edited by hand as the world graph evolves — the `updated_at` with `onupdate=func.now()` makes that auditable.
- **`world_relationships` is the only self-referential table:** both `source_id` and `target_id` point at `world_entities`, which is how the graph in Post 9 encodes edges like "Pepper is a member of Hippiah Coven."

### `Episode` — a parent table

```python
class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[uuid.UUID] = _uuid_pk()
    slug: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    episode_number: Mapped[int] = mapped_column(nullable=False)
    language: Mapped[str] = mapped_column(String(8), default="en", nullable=False)
    cover_image_url: Mapped[str | None] = mapped_column(Text)
    plot_summary: Mapped[str | None] = mapped_column(Text)
    credits_url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ingested_at: Mapped[datetime] = _timestamp_now()

    pages: Mapped[list[Page]] = relationship(
        back_populates="episode",
        cascade="all, delete-orphan",
        order_by="Page.page_number",
    )
    commentary_notes: Mapped[list[CommentaryNote]] = relationship(
        back_populates="episode", cascade="all, delete-orphan",
    )
```

Most of the body is column declarations — one per attribute on the table. A few things worth pausing on:

- **`Mapped[str | None]` vs `Mapped[str]`** is how nullability is communicated. The type annotation is the source of truth: `Mapped[str | None]` produces a nullable column; `Mapped[str]` (combined with `nullable=False`) produces a NOT NULL column. The two halves must agree or `mypy --strict` yells at you.
- **`Mapped[str] = mapped_column(String(128), unique=True, nullable=False)`** — `String(128)` is `VARCHAR(128)` in Postgres. We bound the length on identifier-style columns (`slug`, `title`, `language`) and leave long-text columns as `Text` (effectively unbounded). The bound is a small belt-and-braces guard against accidentally writing megabytes into a column that's supposed to hold a short identifier.
- **`relationship(back_populates="episode", cascade="all, delete-orphan", order_by="Page.page_number")`** is the ORM-side declaration that "an Episode has many Pages, and the Page side has a matching `episode` back-reference." Three knobs worth knowing:
  - `back_populates="episode"` — names the matching attribute on the *other* class (here, `Page.episode`). What this actually buys you is **automatic in-memory sync**: when you write `episode.pages.append(page)`, SQLAlchemy *also* sets `page.episode = episode` for you — and vice versa, setting `page.episode = ep` automatically appends to `ep.pages`. Without `back_populates`, you'd have to remember to update both sides every time. Note this is purely Python-side ergonomics — the *database* knows about the relationship because of the `ForeignKey("episodes.id")` on `Page.episode_id`; `back_populates` just keeps the two `relationship()` attributes mirrored in your application code so you don't end up with two views of the same connection that disagree.
  - `cascade="all, delete-orphan"` — if you delete an Episode, delete its Pages too. Appropriate here because Pages can't exist without their Episode.
  - `order_by="Page.page_number"` — `episode.pages` is always returned in reading order without you having to remember an `.order_by()` clause on every query.

### `Page` — a child, with most of the design decisions visible

```python
class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (
        UniqueConstraint("episode_id", "page_number", name="uq_pages_episode_page"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("episodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    page_number: Mapped[int] = mapped_column(nullable=False)
    image_url: Mapped[str] = mapped_column(Text, nullable=False)
    thumbnail_url: Mapped[str | None] = mapped_column(Text)
    original_url: Mapped[str | None] = mapped_column(Text)
    ocr_text: Mapped[str | None] = mapped_column(Text)
    visual_description: Mapped[str | None] = mapped_column(Text)
    mood_tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    image_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    episode: Mapped[Episode] = relationship(back_populates="pages")
    characters: Mapped[list[Character]] = relationship(
        secondary="page_characters", back_populates="pages",
    )
```

This class is dense — but every design decision from earlier in the post shows up here in code:

- **`UniqueConstraint("episode_id", "page_number", ...)`** in `__table_args__` enforces the "no two pages share a slot in the same episode" rule from the feature-to-table walkthrough. Defining it once at the class level (instead of trying to assemble it from individual columns) is the readable way to express a multi-column constraint.
- **`ForeignKey("episodes.id", ondelete="CASCADE")`** plus the matching `relationship(back_populates="pages")` on the Episode side is the link that lets you do `page.episode` and `episode.pages` in Python without writing a JOIN by hand. `ondelete="CASCADE"` mirrors the `cascade="all, delete-orphan"` on the Episode side: at both the ORM level *and* the database level, deleting an episode also deletes its pages. (You want both — the ORM cascade handles in-process deletes; the database `ON DELETE CASCADE` handles direct SQL deletes or admin tools.)
- **`image_url: Mapped[str] = mapped_column(Text, nullable=False)`** — this is the "relative key, not full URL" column from **design decision #2**. The Python type is just `str`; storage of the prefix happens in the `StorageClient` interface that Post 3 builds.
- **`visual_description: Mapped[str | None] = mapped_column(Text)`** — the canonical embedded text from **design decision #1**. Stored once here in Postgres; only the embedding and the UUID go to Chroma when Post 4 (ingestion) runs.
- **`mood_tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)`** — the array column you verified existed in the migration. The Python side is a `list[str]`; the database side is `VARCHAR[]` (an array of unbounded `VARCHAR`s — and since [PostgreSQL treats unbounded `VARCHAR` as interchangeable with `TEXT`](https://www.postgresql.org/docs/current/datatype-character.html), the storage and performance are effectively the same as `TEXT[]` would be). `default=list` is a Python-side default (every new row starts with `[]`, not `NULL`).
- **`image_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)`** — **design decision #3** in code form. `dict[str, Any]` on the Python side, `JSONB` on the database side. The blob currently holds `{width, height, blurhash, dominant_color}`; new fields are an edit to this line, not a migration.
- **`relationship(secondary="page_characters", ...)`** is how SQLAlchemy expresses the many-to-many we set up. From Python you write `page.characters` (or `character.pages`); SQLAlchemy handles the join through the `page_characters` association table — which we'll look at in a moment.

Every line is justified by a feature or a design decision we already named.

### `Character` and `PageCharacter` — both sides of the many-to-many

The `secondary="page_characters"` declaration on `Page.characters` only makes sense if there's a matching declaration on the `Character` side and an actual join table called `page_characters`. Here are both, in order:

```python
class Character(Base):
    __tablename__ = "characters"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    bio: Mapped[str | None] = mapped_column(Text)
    first_appearance_episode_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("episodes.id", ondelete="SET NULL"),
    )
    image_url: Mapped[str | None] = mapped_column(Text)

    pages: Mapped[list[Page]] = relationship(
        secondary="page_characters", back_populates="characters",
    )


class PageCharacter(Base):
    __tablename__ = "page_characters"

    page_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("pages.id", ondelete="CASCADE"),
        primary_key=True,
    )
    character_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("characters.id", ondelete="CASCADE"),
        primary_key=True,
    )
```

Four things to notice:

- **Symmetric `relationship(...)` on both sides.** `Page.characters` and `Character.pages` both use `relationship(secondary="page_characters", back_populates=<other side>)`. The `secondary=` string names the join table to walk through; `back_populates=` keeps the two endpoints in sync so SQLAlchemy knows they describe the same M:N. Writing only one half works at runtime but leaves the other side without an attribute — better to write both.
- **The join table is *just* two foreign keys.** `PageCharacter` has no `id` column of its own — its primary key is the composite `(page_id, character_id)`, declared by setting `primary_key=True` on both columns. That's the database-level statement of "a character either appears on a page or doesn't; there's no extra data about the appearance" (no per-appearance timestamp, no panel count, no role). If we ever wanted that extra data, we'd give the join table its own `id` and add columns; for now, two FKs are enough.
- **`ondelete="CASCADE"` on both FKs of the join table.** Delete a page and its character-appearance rows go with it; same if you delete a character. The join table never holds orphan rows pointing at deleted entities.
- **`first_appearance_episode_id` uses `ondelete="SET NULL"`, not `CASCADE`.** This is a small but real design choice worth pausing on. Deleting an episode shouldn't *delete* every character who first appeared in it — that would lose Pepper if you ever dropped episode 1. It should just clear the "first appearance" pointer to `NULL`. The choice of `CASCADE` vs `SET NULL` vs the default `RESTRICT` (which would block the parent delete entirely) is always a "what's the right thing to do if the parent disappears?" question. Spend a few seconds on it for every foreign key — getting it wrong shows up months later as either data loss (`CASCADE` too eager) or deletes that mysteriously fail (`RESTRICT` you didn't realize was there).

### `ChatMessage` — the audit column up close

```python
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_session_created", "session_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)        # 'user' | 'assistant'
    mode: Mapped[str | None] = mapped_column(String(32))                 # 'page' | 'wiki', null on user messages
    content: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_doc_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    latency_ms: Mapped[int | None] = mapped_column()
    token_counts: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = _timestamp_now()

    session: Mapped[ChatSession] = relationship(back_populates="messages")
```

The headline is `retrieved_doc_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)` — that's **design decision #4** in code. Every assistant message stores the list of Chroma chunk IDs the orchestrator used to compose its prompt. JSONB on the database side gives you a real array stored efficiently; `Mapped[list[str]]` on the Python side means your application code reads it as a plain Python list with no parsing. When you tune retrieval in Post 6 and a user reports "the chat answered wrong about page 4," this column is what makes "show me the context for that exact message" a one-line query.

A few smaller points worth flagging:

- **`Index("ix_chat_messages_session_created", "session_id", "created_at")`** — the composite index from the migration checklist. Chat history is always read "messages in this session, in chronological order," so the index covers exactly that access pattern. Without it, listing a session's messages requires a full table scan (cheap at 10 rows, painful at 10,000).

  > *What's a database index, in plain English?* Think of the index at the back of a textbook. To find every page that mentions "Pepper", you don't read the whole book cover to cover — you flip to the index, find the entry, and jump straight to the listed pages. A database index works the same way. Without one, finding "every message in session X" means Postgres has to read every row in `chat_messages` and check each one — a [**full table scan**](https://www.postgresql.org/docs/current/using-explain.html). With the index above, Postgres jumps straight to the right neighborhood and reads only those rows.
  >
  > *Why "composite"?* The index is on **two columns at once**: `(session_id, created_at)`. That's because the query we run on every chat-panel load is "give me all the rows where `session_id = ?`, ordered by `created_at`." A single-column index on `session_id` would solve the *filter* part (jump to that session's rows) but Postgres would still have to sort them by `created_at` afterward. Indexing both columns together means the rows are already grouped by session **and** stored in time order within each session — the query is just a contiguous read.
  >
  > *Is there a cost?* Yes — indexes use disk space, and every `INSERT` / `UPDATE` has to update the index too. So you don't index every column "just in case." The rule of thumb: add an index for query patterns you run **frequently** (every chat-panel load qualifies) on tables that will **grow** (chat history accumulates forever) where the table size will eventually be **large enough that a scan hurts** (every demo user opening a session adds rows). All three are true here, so this index earns its keep.
- **`role: Mapped[str]`** — note this is a plain string, not a Postgres `ENUM` type. The two acceptable values are `"user"` and `"assistant"`; you might reasonably expect an enum to enforce that. We use plain strings because [enums are notoriously painful to migrate](https://www.postgresql.org/docs/current/sql-altertype.html) in Postgres (adding a new variant requires raw SQL inside a transaction with specific ordering rules). A string column plus a Pydantic schema at the API boundary catches the same typos with much less ceremony.
- **`mode: Mapped[str | None]`** — nullable because user messages don't have a mode; only assistant responses are produced by a specific retrieval pipeline (`page` or `wiki`). Storing the mode on every assistant message means you can audit "which mode did this answer go through?" at the row level — useful for the per-mode analytics in Post 8.
- **`latency_ms` and `token_counts`** are observability columns: how long the assistant message took to produce, how many prompt/completion tokens it consumed. Filling them in happens in Post 6/Post 7; including the columns in the schema *now* means the audit trail is complete from the very first chat message you ever store. (Adding observability columns retroactively is the same problem as adding `retrieved_doc_ids` retroactively — you lose the history for everything that happened before.)

### `ChatSession` and `WikiArticle` — completing the picture

Two more classes worth a quick look. `ChatSession` is the other half of the `ChatMessage` story — the parent table its `session_id` FK points at. `WikiArticle` is the simplest model in the schema — a good reference point for "how minimal can a class be?"

```python
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[str | None] = mapped_column(String(256))
    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("episodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    current_page: Mapped[int] = mapped_column(default=1, nullable=False)
    created_at: Mapped[datetime] = _timestamp_now()

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class WikiArticle(Base):
    __tablename__ = "wiki_articles"

    id: Mapped[uuid.UUID] = _uuid_pk()
    slug: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(64))
    source_url: Mapped[str | None] = mapped_column(Text)
```

Three things to notice:

- **`ChatSession.messages` mirrors `Episode.pages` exactly.** Same three knobs: `back_populates` (the matching attribute on the child side), `cascade="all, delete-orphan"` (deleting the parent removes the children), `order_by=` (so reading the relationship returns rows in a useful order without a manual `.order_by()` on every query). Seeing this shape twice is enough to recognize it as the standard "one-to-many where the child belongs entirely to the parent" pattern. Most schemas have several of these.
- **`current_page` is the one mutable column in the chat domain.** Each flip on the frontend triggers a small `PATCH /sessions/<id>` request that bumps this number — which is what makes the AI's grounding context always reflect the spread the reader is currently looking at. Compare to `chat_messages`, which is purely append-only: once a turn is recorded, it never changes.
- **`WikiArticle` has no `relationship(...)` at all.** It's a fully standalone table — wiki articles aren't tied to specific episodes or pages, so there's no FK out and no back-reference in. (The link from a chat question to a wiki article happens at *retrieval* time in Post 6, not via a database relationship.) This is the floor of the schema's complexity: a primary key, five string-ish columns, no joins. When you're modeling your own domain and a table feels like it doesn't need to point at anything, `WikiArticle` is the shape to aim for — clean, minimal, no premature relationships.

---

That's the shape of every model in this schema: a typed Python class, one `mapped_column(...)` per database column, with `Mapped[X]` driving both the Python type and (where it differs) the database type. The remaining models in `models.py` — `CommentaryNote`, `WorldEntity`, `WorldRelationship` — follow exactly the same patterns: `CommentaryNote` is a lean version of `Page` (FK to episodes, an optional page-number hint, content as text); `WorldEntity` and `WorldRelationship` are Post 9 territory but use the same parent/child + cascade choices we just walked through. Read the classes above carefully and the rest are a five-minute skim.

---

Next up: **Post 3 — Provider Abstractions: Why Every External Service Hides Behind an Interface.** With the workshop standing, we start writing code. Specifically: the three [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol) types — `StorageClient`, `EmbeddingClient`, and (foreshadowing) `ChatClient` — that make the rest of the project portable. By the end of that post you'll have a `LocalStorage` working end-to-end against `./data/images/`, a `SentenceTransformersEmbeddingClient` (and an `OllamaEmbeddingClient`) producing real 1024-dimensional vectors, and a factory that picks the right implementation based on the `.env` file you just built.

The full source repository and a public live-demo URL will go up alongside the final post of this series — the deploy guide — once it's published. Until then, every post is self-contained, with the code inline and a verifiable checklist at the bottom.

*Pepper & Carrot* is © [David Revoy](https://www.davidrevoy.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All credit to him for the source material that made this project possible.

**All opinions expressed are my own.**
