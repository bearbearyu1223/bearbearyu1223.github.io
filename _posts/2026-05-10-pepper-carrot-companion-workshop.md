---
title: "Pepper & Carrot AI-powered flipbook · Part 2 — Setting Up a Local AI Dev Stack: Postgres, Ollama, and a Project That Type-Checks"
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
7. [Apply the First Migration](#first-migration)
8. [Your First Episode in `data/raw/`](#first-episode)
9. [The `.env` File That Ties It Together](#env)
10. [Verification: The Whole Workshop, End to End](#verification)
11. [Key Takeaways](#key-takeaways)

---

## Why a "Workshop" Phase Deserves a Whole Post {#why-workshop}

If you've followed AI tutorials before, you've seen this pattern: install five things, run a magic script, look at a chatbot. It works on the author's machine. Then a week later you try to debug a retrieval issue and discover the model server was never actually serving the model you thought, the migrations were never applied, and your code is silently importing the wrong embedding library.

A "workshop" phase is the antidote. Each tool gets installed, configured, and **verified to work in isolation** before any application code touches it. You write zero feature code in this post, but you finish it with a green-lit, reproducible base camp.

There's a portfolio judgement here worth naming: real production systems live or die by their setup story. At the level of professional impression, a demo that needs ten paragraphs of caveats to run for the reviewer is worse than a smaller demo that "just works." Investing one post in setup is the cost of being able to write the next eight posts without "did you remember to also..." sprinkled through each one.

The project encodes this same idea as a hard rule it calls **provider
abstraction**: every external service (database, model server, image
store) sits behind a typed interface, so local Postgres and cloud
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

Before any `cd backend` or `cp .env.example .env`, here's a one-screen orientation to what's on disk. From the next section onward, the post starts `cd`-ing into specific subdirectories and editing specific files, and that's a lot less confusing once you've seen the whole layout once.

```
pepper-carrot-companion-workshop/
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

- **You'll only touch a handful of these in this post.** Top-level `docker-compose.yml` and `.env.example`; `backend/pyproject.toml` and the `backend/alembic/` migration directory; the SQLAlchemy models under `backend/app/db/`; and `ingestion/acquire.py`. Every other path is a structural placeholder for code that arrives in later posts, listed so you can see where things will land as the series builds out.
- **`data/` is gitignored, so it's not part of the source.** It's a working directory each developer creates as they run the workshop. By the end of this post you'll have `./data/postgres/` (live Postgres data files), `./data/raw/ep01-potion-of-flight/` (one downloaded episode of comic source), and a placeholder for `./data/chroma/` that gets populated in Post 4.

> **About the repo URL.** The code that backs this post (and Posts 3–4) is at <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop> — a deliberately scoped workshop starter. Clone it and the tree above is exactly what you'll find on disk. The full project repository (frontend, chat orchestrator, world-graph, cloud deploy) goes up alongside the deploy guide near the end of the series. Until then, the file paths in every later command (`backend/app/clients/`, `backend/app/db/models.py`, etc.) are the same paths in both the starter and the eventual full repo.
>
> **Checking out the code.** The setup, the data model, and the provider abstractions (Posts 2–4) all live at one checkpoint: `git checkout post-02-04-starter` gives you a complete, working tree for all three. Later posts add their own tags (`post-05-06-ingestion`, `post-07-08-fullstack`, `post-09-rag`, …); each names the tag to check out. See the README's [Following along with the blog series](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop#following-along-with-the-blog-series).

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

- **`environment:`** is a map of environment variables passed into the container's process. The Postgres image is designed to read three of them on first boot — `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` — and create the initial superuser and database from those values. Set them once at first boot and they're baked into the data directory thereafter.

  The `${VAR:-default}` syntax is straight from [POSIX shells](https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_06_02): "use the value of `$VAR` if it's set in the environment, otherwise use this fallback." Compose reads `.env` automatically, so a `POSTGRES_PASSWORD=hunter2` line in your `.env` propagates here without any further wiring. (This is why we copied `.env.example` to `.env` before running `docker compose up`.)

- **`ports:`** maps `host_port:container_port`. Postgres listens on 5432 inside the container; we expose it on `localhost:5432` on your laptop, so any tool on your machine can connect with `postgresql://...@localhost:5432/...`. If you already have a system Postgres on 5432, override `POSTGRES_PORT=5433` in `.env` — the `${POSTGRES_PORT:-5432}` substitution then maps host port 5433 to container port 5432.

- **`volumes:`** mounts a directory from your host into the container, a [*bind mount*](https://docs.docker.com/storage/bind-mounts/). `./data/postgres` on your laptop becomes `/var/lib/postgresql/data` inside the container, which is where Postgres stores every byte of every table. Without this, `docker compose down` would also delete your database; with it, your data survives container restarts, image upgrades, and laptop reboots.

  Two follow-on points worth knowing:
  - `./data/postgres` is **gitignored** — binary database files belong in backups, not version control.
  - If you ever want to start fresh: `docker compose down && rm -rf data/postgres && docker compose up -d` gets you a virgin database with all the initial-boot setup re-run.

- **`healthcheck:`** tells Docker how to find out whether the container is *actually ready*, not just running. The image starts the moment the process launches, but Postgres takes a couple of seconds before it accepts connections. The check runs [`pg_isready`](https://www.postgresql.org/docs/current/app-pg-isready.html), Postgres's own "are you up?" command, every 5 seconds, gives up on a single probe after 5 seconds, and tolerates 5 consecutive failures before marking the container *unhealthy*. `docker compose ps` shows `(healthy)` once the first probe succeeds, which is also the signal the FastAPI app uses in Post 3 to know it's safe to open its DB connection pool.

- **`restart: unless-stopped`** auto-restarts the container if it crashes, and brings it back up automatically when you reboot your laptop, but stays out of the way if you explicitly `docker compose stop`. A reasonable default for a dev database you don't want to think about.

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

That single line is doing four things, all of which would be the *first* things to fail if your setup is broken. It's worth taking apart:

| Piece | What it is |
|---|---|
| `psql` | The Postgres command-line client you just installed. |
| `"postgresql://peppercarrot:peppercarrot_dev@localhost:5432/peppercarrot"` | A [**connection URL**](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING) — the standard way to describe a Postgres connection in one string. Anatomy: `postgresql://<user>:<password>@<host>:<port>/<database>`. So this says "connect as user `peppercarrot` with password `peppercarrot_dev` to the Postgres server at `localhost:5432`, and pick the database named `peppercarrot` once connected." Every key here was set in `docker-compose.yml` from `.env` — the same shape of URL is what the FastAPI app will use in later posts. |
| `-c "..."` | "Run this one SQL statement, print the result, then exit." Without `-c`, `psql` drops you into an interactive shell — handy for exploring, but for a one-shot check `-c` is cleaner. |
| `SELECT version();` | A SQL query that asks Postgres for [its own version string](https://www.postgresql.org/docs/current/functions-info.html). It's the database equivalent of `--version` — and it's a better health check than that, because to answer it Postgres has to (1) accept your TCP connection, (2) authenticate your user/password, (3) open the named database, and (4) run a real SQL query against it. If you see a version string come back, **all four** of those work. If any one of them is broken, you get a specific error: `could not connect to server`, `password authentication failed`, `database "peppercarrot" does not exist`, etc. — each of which points at exactly which piece to fix. |

So this isn't just a "did it install?" check. It's an end-to-end probe of the whole local Postgres setup in one line.

**Option B — skip the install and use the `psql` that's already inside the container.** Slightly more typing, but zero new software on your laptop:

```bash
docker exec -it peppercarrot-postgres psql -U peppercarrot -d peppercarrot -c "SELECT version();"
```

Same idea as Option A, but with a different connection mechanism. Here it is taken apart the same way:

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

That's it: you now have a real relational database, running locally, with a known username, password, port, and database name. Every later post talks to it through this URL.

Here's a snapshot of what your laptop looks like at the end of this post — three local services on three known ports, each persisting to a known directory. The third box (FastAPI plus an embedded ChromaDB) doesn't start until Post 3, but it's drawn in so the picture stays complete:

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
- **ChromaDB is not a server.** Unlike Postgres or Ollama, ChromaDB is a *library*: it runs inside the FastAPI process, reads and writes a directory on disk, and has no port of its own. This is why it's drawn *inside* the FastAPI box, not next to it. (Post 3 explains why this is the right call for a portfolio-scale project, and Post 10 covers when you'd outgrow it.)
- **Everything that matters persists under `./data/`** (except Ollama's model weights, which Ollama manages in its own home directory). That whole tree is gitignored — `git status` should never show database rows or model files. To start completely fresh, `rm -rf data/` and re-run the workshop.

---

## Ollama: An LLM Server for Your Laptop {#ollama}

> *What's Ollama, in plain English?* [**Ollama**](https://ollama.com/) is a tiny HTTP server that runs [open-weights language models](https://ollama.com/library) locally on your machine. You `ollama pull` a model (downloads the weights, typically a few GB), and `ollama serve` (or the desktop app) makes that model reachable at `http://localhost:11434` with an [HTTP API](https://github.com/ollama/ollama/blob/main/docs/api.md) similar to the [OpenAI API](https://openai.com/api/). The point: a language model becomes a normal local service, no API key, no per-request billing, no rate limits other than your laptop's CPU/GPU.

### Why "local-first" is a real design decision, not just a budget call

You could absolutely build this project against the [Anthropic API](https://docs.claude.com/en/api/getting-started) or the [OpenAI API](https://openai.com/api/). The project's [`anthropic`](https://github.com/anthropics/anthropic-sdk-python) Python SDK is even in the dependencies, and switching is a config change (Post 3 covers exactly how). But starting local-first has three benefits worth naming explicitly:

1. **You iterate without watching a meter.** During development, you'll hit the model thousands of times — embedding test corpora, debugging prompts, replaying conversations. A free local model removes the small cognitive tax of "is this experiment worth $0.40?".
2. **You'll discover provider abstractions you would have skipped.** Once your code has to talk to a model at `http://localhost:11434` *and* a hosted API behind a Bearer token, you stop hard-coding either one. That [interface](https://docs.python.org/3/library/typing.html#typing.Protocol) is the whole topic of Post 3.
3. **The cloud port becomes a one-day project.** In Post 10, we move the *same* Ollama process onto a serverless GPU on [Modal](https://modal.com/). It's the identical HTTP API, so only the URL changes. No model swap, no prompt rewriting.

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

> **A note on the vision model that *isn't* there.** Ollama also ships a `qwen2.5vl` *vision* variant. We don't use it. Page descriptions in this project come from a Claude Code [skill](https://docs.claude.com/en/docs/claude-code/skills): Claude itself reads each comic page image and writes a structured JSON description next to it. Post 4 explains why this is a deliberate architectural call (better quality, cost-effective at portfolio scale since it rides on an existing Claude subscription rather than a per-request API bill, auditable artifacts on disk) rather than a workaround. Don't pull `qwen2.5vl` — it's 1–2 GB you don't need.

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

Each line is a deliberate choice. Here's a one-sentence tour of each:

- **fastapi + uvicorn** — async web framework plus an [ASGI](https://asgi.readthedocs.io/en/latest/) server to run it. The async story matters because we stream model output token-by-token in Post 7.
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

> *What are these?* [**mypy**](https://mypy-lang.org/) is Python's most widely used static type checker; it reads your type annotations and flags places where the types don't line up. [**ruff**](https://docs.astral.sh/ruff/) is a [Rust](https://www.rust-lang.org/)-based linter and formatter from the same team behind `uv`. Where `mypy` catches type bugs, `ruff` catches style bugs, dead imports, undefined variables, and a few hundred other classes of mistake, all in well under a second on a project this size.

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

## Apply the First Migration {#first-migration}

> *What's a migration?* A versioned script — usually one `.py` file per change — that takes the database from one schema (the set of tables, columns, indexes, constraints) to the next. Run them in order on a fresh database and you arrive at the current schema. [**Alembic**](https://alembic.sqlalchemy.org/en/latest/) is the migration tool that ships with SQLAlchemy.

The workshop starter already ships the initial migration, so applying it is one command from `backend/`:

```bash
cd backend
uv run alembic upgrade head
```

`head` means "the latest revision." On a fresh database, that's the one initial migration — and it creates every table the app will ever use.

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

Eleven rows — ten application tables plus `alembic_version`. Now insert a test row to prove the schema is queryable:

```bash
psql "postgresql://peppercarrot:peppercarrot_dev@localhost:5432/peppercarrot" \
  -c "INSERT INTO characters (id, name, aliases, image_url) \
      VALUES (gen_random_uuid(), 'Pepper', ARRAY['the witch'], NULL); \
      SELECT name, aliases FROM characters;"
#  name  |   aliases
# -------+-------------
#  Pepper | {the witch}
```

If that lands without an error, your schema is real, queryable, and ready for the next post.

> *A gotcha worth flagging — skip this if you're using the Docker Postgres we set up earlier.* The verification query uses [`gen_random_uuid()`](https://www.postgresql.org/docs/current/functions-uuid.html). In PostgreSQL 13 and later (including the `postgres:16-alpine` image) it works out of the box. On an older Postgres it lives in the [`pgcrypto`](https://www.postgresql.org/docs/current/pgcrypto.html) extension — add `op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')` to the top of the migration's `upgrade()` function and re-run `alembic upgrade head`.

That's the migration applied. **What it actually created — how Alembic generates it from your models, why migrations beat `Base.metadata.create_all()`, how the ten tables map onto product features, and a column-by-column tour of the SQLAlchemy models — is the whole subject of the next post, [The Data Model]({% post_url 2026-05-11-pepper-carrot-companion-data-model %}).**

---

## Your First Episode in `data/raw/` {#first-episode}

The last preflight step: get one episode of the comic onto disk in a structured way that the ingestion pipeline (Post 4) can consume.

Lots of "chat with X" demos start by web-scraping HTML. *Pepper & Carrot* has a much nicer path. David Revoy's project [publishes a fully structured source archive](https://www.peppercarrot.com/0_sources/) with three JSON manifests that make the whole corpus programmatically accessible:

- **`/0_sources/episodes-v1.json`** — master list of every episode and the canonical filename for each page slot.
- **`/0_sources/{slug}/info.json`** — per-episode metadata: publication date, original language, software credits.
- **`/0_sources/{slug}/hi-res/titles.json`** — episode title in every language.

Page filenames follow a deterministic pattern (`{lang}_Pepper-and-Carrot_by-David-Revoy_E{NN}P{NN}.{ext}`), so an acquisition script can build every URL it needs without scraping any HTML at all.

> *A small portfolio judgement worth saying out loud.* The "right" way to ingest someone else's content is almost always to read their data the way they meant you to read it. Web scraping HTML is a fragile fallback that breaks the moment the publisher restyles a button. Looking for a JSON or RSS feed *first*, even if it takes an extra hour of investigation, is the move. In this case, half a day spent reading `/0_sources/` instead of scraping pays off across the rest of the project.

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

`.env` is the single source of truth for "what URLs and credentials does this project use?" It's gitignored, lives at the project root, and is loaded by [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) at startup into a typed `Settings` object.

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

> *Why two keys for the embedding model?* `EMBEDDING_PROVIDER` says *which client library* to talk to (Ollama or [`sentence-transformers`](https://www.sbert.net/)). `EMBEDDING_MODEL` says *which model name* to ask for, and the naming convention is provider-specific. Ollama wants `bge-m3`; sentence-transformers wants `BAAI/bge-m3`. Same underlying weights, same 1024-dimensional vectors, but the strings differ. We pry these apart in Post 3 and you'll see exactly why.

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

If something fails, fix it now. Setup bugs that survive into application development become two-day debugging sessions because they masquerade as application bugs.

---

## Key Takeaways {#key-takeaways}

**1. Treat setup as a real engineering deliverable, not a prelude.** The thirty minutes it takes to verify Postgres, Ollama, mypy, ruff, Alembic, and `data/raw/` independently saves you days of "is this an app bug or a setup bug" later. On a portfolio project, "it runs cleanly on a fresh clone" is the *first* thing a reviewer notices.

**2. `mypy --strict` on day one is cheap; `mypy --strict` on day 600 is a migration.** Bolt the guardrail on while the codebase is still empty. Every later file gets written with full annotations as a matter of course, and the cost is invisible.

**3. Migrations exist from commit #1, not when "we get to production."** Even if today's database is just for one developer, `alembic upgrade head` is the same command on a laptop and on Neon in Post 10. The first migration is a 30-line file. The first migration *retrofitted* onto an established project is a multi-day archaeology project.

**4. Read the data publisher's data, not their HTML.** *Pepper & Carrot* publishes machine-readable JSON manifests at `/0_sources/`. Spending half a day reading those instead of scraping HTML pays off across the rest of the project — and it's a habit that generalizes far beyond webcomics.

**5. Local-first isn't free, but it's worth the cost.** Ollama with `qwen2.5:7b` is genuinely slower than calling [Claude Haiku](https://www.anthropic.com/news/claude-haiku-4-5) or OpenAI's `gpt-4o-mini`. What you get in exchange is unmetered iteration, a forcing function for provider abstraction, and a deploy story (Post 10) where the same `httpx.AsyncClient` talks to a serverless GPU in production. That free iteration loop is what makes prompt engineering (Post 8) feasible at all.

---

---

Next up: **Post 3 — The Data Model: Ten Tables, One Migration.** You just applied a migration that created eleven tables; the next post opens them up — how Alembic generates the migration from your models, why that beats `create_all()`, how each table maps to a product feature, and a column-by-column tour of the SQLAlchemy models.

The **workshop starter** that backs this post is at <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop>, tagged `post-02-04-starter`. The **full source repository** and a public live-demo URL go up alongside the deploy guide near the end of the series — once it's published.

*Pepper & Carrot* is © [David Revoy](https://www.davidrevoy.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All credit to him for the source material that made this project possible.

**All opinions expressed are my own.**
