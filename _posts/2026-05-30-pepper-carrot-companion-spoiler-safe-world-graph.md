---
title: "Pepper & Carrot AI-powered flipbook · Part 12 of 16 — A World Graph Built by a Skill: Extraction and a Spoiler-Safe API"
date: 2026-05-30 12:00:00 -0800
categories: [Full-Stack, RAG, Local AI]
tags: [knowledge-graph, claude-skills, postgres, fastapi, sqlalchemy, pydantic, peppercarrot, portfolio]
description: >-
  Post 12 of the Pepper & Carrot AI flipbook series. A second Claude Code
  skill — extract-world-graph — walks the wiki sources and the per-page
  description JSONs and writes a durable YAML pair (entities +
  relationships) that a pydantic loader upserts into Postgres. Then a
  FastAPI route serves the graph through a spoiler filter expressed as a
  Postgres row-value comparison — (episode_debut, page_debut) <=
  (current_episode, current_page) — so an edge whose own debut is past the
  reader's cursor can't leak even when both of its endpoints are visible.
  Ten hermetic tests against in-memory SQLite pin the boundary.
pin: true
---

Post 12 of the [*Pepper & Carrot AI-powered flipbook*]({% post_url 2026-05-09-pepper-carrot-companion-trailer %}) series. [Post 9]({% post_url 2026-05-25-pepper-carrot-companion-spoiler-safe-rag %}) made spoiler safety a property of the *retrieval* query; [Post 10]({% post_url 2026-05-25-pepper-carrot-companion-streaming-chat %}) and [Post 11]({% post_url 2026-05-30-pepper-carrot-companion-prompt-hardening %}) made the chat *answers* stream cleanly and hold their format. This post starts a third reading-time affordance — a **world graph** of Pepper, her godmothers, the rival witches, the covens, and the places — but it stops at the data layer. The graph isn't extracted at runtime; it's authored by a **second Claude Code skill** (the first was [`ingest-from-images`]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}) in Post 5) that walks the wiki sources and the page-description JSONs and writes a YAML pair the workshop loads into Postgres. On top of those rows sits a single FastAPI route that gates what the reader can see behind exactly the same spoiler boundary the chat sits behind — this time expressed as a Postgres row-value comparison. The [next post]({% post_url 2026-05-31-pepper-carrot-companion-world-graph-overlay %}) renders this API as an in-reader React-Flow overlay; here we build the skill, the loader, the scraper, and the spoiler-safe endpoint that feeds it.

> **What you'll build in this post.**
> - **A second Claude Code skill**, `.claude/skills/extract-world-graph/SKILL.md`, that reads the curated wiki under `data/raw/wiki/*.md` (plus the framagit-scraped `data/raw/wiki-upstream/*.md` when present) + every `data/raw/ep*/pages/page_*.json` + the image manifest and writes `data/world-graph/entities.yaml` + `data/world-graph/relationships.yaml`. The shipped graph carries **45 entities** (19 characters, 6 covens, 5 places, 15 creatures) and **57 relationships**.
> - **A pydantic loader + ingestion CLI** in `ingestion/world_graph_loader.py` and `ingestion/ingest_world_graph.py` that validate the YAML and upsert it into the `world_entities` + `world_relationships` Postgres tables. The loader prunes orphan entities on every run so a YAML rename (e.g. `squirrels-end` → `squirrel-s-end`) doesn't leave ghost rows.
> - **A wiki image scraper** in `ingestion/wiki_image_scraper.py` that pulls ~40 character/creature portraits from [framagit](https://framagit.org/peppercarrot/wiki) (CC BY 4.0), processes them into thumb + display WebP variants, and writes `image_manifest.json` for the skill to consume.
> - **A spoiler-filtered API endpoint**, `GET /api/world-graph`, in `backend/app/api/world_graph.py` — same lexicographic boundary as Post 9's RAG layer, expressed as a Postgres row-value comparison: `(episode_debut, page_debut) <= (current_episode, current_page)`. Two response modes layer on top: `mode=full` (the whole spoiler-safe world) and `mode=focus` (canonical characters drawn on the current page(s) + a 1-hop expansion through structural edge kinds like `member_of` / `lives_in` / `familiar_of`). A `right_page` query parameter lets two-page-spread readers seed the focus set from both visible pages. **Ten hermetic tests against in-memory SQLite** pin the boundary — including the gotcha where an edge can debut later than both of its endpoints, the focus-mode fallback when no characters are on the page, and the silent collapse when a flipbook callback sends `right_page < page`.
>
> **Prerequisites.**
> - The workshop starter at the `post-12-13-worldgraph` tag: `git checkout post-12-13-worldgraph` (see [Following along with the blog series](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop#following-along-with-the-blog-series)). Everything [Post 11]({% post_url 2026-05-30-pepper-carrot-companion-prompt-hardening %}) needed — Postgres up, migrations applied, the roster seeded, at least Episode 1 ingested, Ollama running with `qwen2.5:7b` and `bge-m3` pulled.
> - For a richer demo, **ingest a few more episodes** through the `ingest-from-images` skill from [Post 5]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}) — the world-graph extraction reads `data/raw/ep*/pages/page_*.json` files to find when each canonical character first appears. The shipped graph YAML was extracted against ep01–ep12; if you only have ep01, the graph will show ~3 entities and the rest will populate as you ingest more.
> - [Node.js 20+](https://nodejs.org/) and the same Vite frontend setup from [Post 8]({% post_url 2026-05-24-pepper-carrot-companion-flipbook-ui %}) (only needed for the overlay in the next post; nothing in *this* post touches the frontend).

> **Checking out the code.** All the changes in this post — the new skill, the loader + scraper + CLI, the `world_graph.py` API route, the YAML pair, and the database-portability shim that lets the spoiler-filter test run against in-memory SQLite — live in the same workshop starter that backed [Posts 2–11](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop), at the `post-12-13-worldgraph` tag: `git checkout post-12-13-worldgraph`. This is the same checkpoint [Post 13 — Rendering the World Graph]({% post_url 2026-05-31-pepper-carrot-companion-world-graph-overlay %}) uses: the two posts split one checkpoint between the spoiler-safe API (here) and the React-Flow overlay (there).

---

## Table of Contents

1. [The Code in Front of You: Tour + Quick Start](#tour)
2. [What This Adds, and What It Doesn't](#what-this-adds)
3. [Why a Second Skill?](#why-a-second-skill)
4. [The Pipeline, End to End](#diagram)
5. [The YAML: One Schema, Two Authors](#the-yaml)
6. [The Wiki Image Scraper](#scraper)
7. [The `extract-world-graph` Skill](#skill)
8. [The Spoiler Filter, This Time in Postgres](#api)

---

## The Code in Front of You: Tour + Quick Start {#tour}

Get the running overlay in front of you before any of the concepts land. Skim this even if you plan to read carefully — watching a single node debut as you flip the page makes the rest obvious.

### Get the code at this post's tag

```bash
git clone https://github.com/bearbearyu1223/pepper-carrot-companion-workshop
cd pepper-carrot-companion-workshop
git checkout post-12-13-worldgraph
```

Already cloned from an earlier post? `git fetch --tags && git checkout post-12-13-worldgraph`.

### What's new in the workshop starter

Two new modules on the backend, several new modules in `ingestion/`, three skills, a new frontend directory, and the durable artifacts on disk:

```
pepper-carrot-companion-workshop/
├── .claude/skills/
│   ├── ingest-from-images/         (from Post 5 — unchanged)
│   ├── extract-world-graph/        ← NEW (Post 12): the second skill
│   │   └── SKILL.md
│   └── summarize-wiki/             ← NEW (Post 12): the third skill — wiki summaries
│       └── SKILL.md
├── backend/app/
│   ├── api/world_graph.py          ← NEW: GET /api/world-graph (spoiler-filtered)
│   ├── main.py                     ← updated: mount the world-graph router + serve avatars
│   ├── retrieval/service.py        ← updated: _WIKI_K = 3, summary-first aware
│   └── db/models.py                ← updated: .with_variant(JSON, "sqlite") shim
├── backend/tests/
│   └── test_world_graph_api.py     ← NEW: 10 spoiler-boundary tests (in-memory SQLite)
├── ingestion/
│   ├── world_graph_loader.py       ← NEW: pydantic contract for the YAML
│   ├── ingest_world_graph.py       ← NEW: validates + upserts the YAML; prunes orphan rows
│   ├── wiki_image_scraper.py       ← NEW: pulls ~40 portraits from framagit
│   ├── wiki_loader.py              ← NEW: parses .md + YAML frontmatter for ingest_wiki
│   ├── wiki_scraper.py             ← NEW: pulls 7 long-form .md files from framagit
│   ├── ingest_wiki.py              ← UPDATED: reads data/wiki-summaries/ (one chunk per summary)
│   ├── chroma_writer.py            ← updated: paragraph-chunking removed; 1 chunk per article
│   └── repository.py               ← updated: + upsert_world_entity / _relationship
├── frontend/src/components/world-graph/  ← NEW: 8 components, ~900 lines
│   ├── constants.ts                ←   kind + edge-kind → color / label mapping
│   ├── fallback-icons.tsx          ←   per-kind SVG silhouettes
│   ├── AvatarNode.tsx              ←   custom react-flow node (8 invisible handles)
│   ├── InfoCard.tsx                ←   popover + "Ask in wiki mode" button
│   ├── KindFilterBar.tsx           ←   per-kind toggle chips with counts
│   ├── focus-layout.ts             ←   kind-grid layout for focus mode
│   ├── WorldGraph.tsx              ←   canvas + mode toggle + fade-in diff + edge styling
│   └── WorldGraphOverlay.tsx       ←   slide-in side panel (with the title flourish)
├── frontend/src/App.tsx            ← updated: "🌐 World" header button + overlay (passes rightPage on landscape spreads)
├── data/world-graph/               ← the durable graph artifact
│   ├── entities.yaml               ←   45 entities (19 characters, 6 covens, 5 places, 15 creatures)
│   ├── relationships.yaml          ←   57 relationships
│   ├── image_manifest.json         ←   scraper output (committed; small + the audit trail)
│   └── images/                     ←   gitignored (~40 thumb + display webp)
├── data/raw/wiki/                  ← NEW: 9 curated bios (.md + YAML frontmatter)
├── data/raw/wiki-upstream/         ← NEW: 7 framagit articles (scraped by wiki_scraper.py)
└── data/wiki-summaries/            ← NEW: the durable wiki artifact (one .md per entity)
    ├── entities/                   ←   45 entity summaries (~100-300 words each)
    └── topics/                     ←   5 topic summaries (history, time system, magic, …)
```

### Run it: three terminals, then click the world button

```bash
# Terminal 1 — Postgres + Ollama (already up from Post 2)
docker compose up -d

# Terminal 2 — backend on :8000 (picks up the new /api/world-graph route)
cd backend && uv sync && uv run uvicorn app.main:app --reload

# Terminal 3 — Vite dev server on :5173 (picks up @xyflow/react)
cd frontend && npm install && npm run dev
```

Four one-time setup commands the first time you check out `post-12-13-worldgraph`:

```bash
# 1. Pull the character/creature portraits from framagit (~40 small webps).
cd ingestion && uv run python wiki_image_scraper.py

# 2. (Optional) Pull the long-form wiki .md files from framagit so the
#    summarize-wiki skill has framagit detail to draw on in addition to
#    the 9 curated bios already in data/raw/wiki/.
uv run python wiki_scraper.py

# 3. Load the shipped entities.yaml + relationships.yaml into Postgres,
#    and the 50 wiki summaries into Postgres + Chroma's wiki_v1 collection.
uv run python ingest_world_graph.py
uv run python ingest_wiki.py

# 4. (Optional) Regenerate any artifact yourself if you've ingested more
#    episodes than the shipped ep01-12, edited the wiki sources, or want
#    to see the skill pattern in action:
#      • "extract the world graph"   → re-authors entities + relationships YAML
#      • "summarize the wiki"        → re-authors data/wiki-summaries/*.md
```

Open <http://localhost:5173>, pick any episode, and click the new **🌐 World** button in the header. A side panel slides in from the right with avatar nodes for every entity that's spoiler-safe at the page you're on. Flip forward into ep11 — Pepper's three godmothers should fade in. Click any avatar — an info card with the entity's portrait, summary, and an **"Ask in wiki mode"** button opens. Click that button, and the overlay closes while the chat panel posts a wiki-mode question about the entity and streams a focused, grounded answer back — courtesy of the third skill (more on that below).

If you ingested only ep01 through the workshop's starter steps, the graph will show three nodes (Pepper, Carrot, and Chaosah). For the full ep01–ep12 graph used in the screenshots and curl examples below, you'll want to ingest the rest of those episodes via the `ingest-from-images` skill from [Post 5]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}).

### Validate it from the terminal

You don't have to take the overlay's word for any of the spoiler-safety:

```bash
# At episode 1, page 1 — only entities with debut ≤ (1, 1) are visible.
curl -s 'http://localhost:8000/api/world-graph?episode_slug=ep01-potion-of-flight&page=1' \
  | python3 -c "import sys,json; g=json.load(sys.stdin); \
      print(f'{len(g[\"nodes\"])} nodes:', sorted(n['slug'] for n in g['nodes']))"
# 3 nodes: ['carrot', 'chaosah', 'pepper']

# At episode 8, page 6 (the tea party with the Chaosah monsters) —
# Chaosah unlocked, the monsters fade in, but the godmothers (debut ep11)
# are still hidden.
curl -s 'http://localhost:8000/api/world-graph?episode_slug=ep08-pepper-s-birthday-party&page=6' \
  | python3 -c "import sys,json; g=json.load(sys.stdin); \
      print(f'{len(g[\"nodes\"])} nodes,', f'{len(g[\"edges\"])} edges')"
# 23 nodes, 24 edges

# Same page, focus mode — only the on-page seed + 1-hop neighbors.
curl -s 'http://localhost:8000/api/world-graph?episode_slug=ep08-pepper-s-birthday-party&page=6&mode=focus' \
  | python3 -c "import sys,json; g=json.load(sys.stdin); \
      print(f'{len(g[\"nodes\"])} nodes:', sorted(n['slug'] for n in g['nodes']))"
# 4 nodes: ['carrot', 'chaosah', 'pepper', 'squirrel-s-end']
#          (Pepper on-page; Carrot via familiar_of; Chaosah via member_of;
#           Squirrel's End via lives_in. Saffron, Coriander, etc. aren't
#           on this page, so they're filtered out.)

# At episode 12, page 7 — full mode shows the whole spoiler-safe world.
curl -s 'http://localhost:8000/api/world-graph?episode_slug=ep12-autumn-clearout&page=7' \
  | python3 -c "import sys,json; g=json.load(sys.stdin); \
      print(f'{len(g[\"nodes\"])} nodes,', f'{len(g[\"edges\"])} edges')"
# 30 nodes, 42 edges
```

Same shape as the `done` audit trail in Post 10's chat: the URL carries the reader's cursor, the server clamps it to the episode's real page count, and the SQL filter is the security boundary. Nothing about the request body can widen what comes back.

---

## What This Adds, and What It Doesn't {#what-this-adds}

The series has been building one affordance per post — a flipbook (Post 7), a chat answer (Post 9), a streaming chat panel (Post 10), prompt-hardened replies (Post 11). Post 12 adds a third *reading-time* affordance alongside the chat:

| | Affordance | Built by | Bounded by |
|---|---|---|---|
| **Post 7** | Flipbook reader | StPageFlip + REST API | (no spoiler concern at the page layer) |
| **Posts 9–11** | Chat (page + wiki) | Ollama + spoiler-filtered RAG + prompt hardening | `chat_sessions.current_page` |
| **Post 12** | World-graph overlay | YAML authored by a second skill, loaded into Postgres, filtered in SQL | `(episode_number, current_page)` cursor in the URL |

Three things this post **isn't**:

- **It isn't an entity extractor at runtime.** The graph is extracted *once* (per episode-ingestion change) by a Claude Code skill running in your editor session. The runtime never calls a model to figure out who's in the world; it queries Postgres and ships the rows. No per-request inference cost, no model dependency on graph correctness.
- **It isn't a generic knowledge-graph framework.** No SPARQL, no neo4j, no graph-DB. Two SQLAlchemy tables, two SQL `SELECT`s per request, one row-value comparison for the spoiler boundary. A small graph (45 nodes, 57 edges in the workshop) doesn't need graph-native storage to be fast or expressive.
- **It isn't editorial about which schools or which characters exist.** The skill works only from what the wiki sources and the page JSONs say. An entity that's in the wiki but never appears in any ingested episode lands as a `(1, 1)`-defaulted low-confidence row — visible from page 1, no edges to anything else yet. If the wiki sources don't mention an entity at all, it isn't in the graph. **Coverage grows when the source material grows.**

The architectural through-line connecting Post 9 and Post 12 is one sentence: **the spoiler boundary lives in the data layer, not in the prompt.** Post 9 expressed it as a Chroma `where` clause built from `chat_sessions.current_page`. Post 12 expresses it as a SQL row-value comparison built from a query-string parameter that's clamped against `pages.page_number`. Same shape, different store.

---

## Why a Second Skill? {#why-a-second-skill}

The first skill — [`ingest-from-images`]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}) from Post 5 — established a pattern: Claude Code, with its image-reading and file-writing tools, **acts as a one-shot author** for a durable on-disk artifact (per-page `PageDescription` JSON files). The runtime never calls a model to produce those descriptions; the ingestion pipeline reads them off disk. Post 5 spent a section on why this matters:

- **Zero per-call cost.** The skill runs inside the user's existing Claude Code session.
- **Durable artifact.** The JSON is committed; hand-edits stick; the skill is for the first pass, not a runtime dependency.
- **Auditable.** Reviewers see plain JSON, not buried model output.

The same three properties apply to the world graph — only this time the artifact is a YAML pair and the source material is the (now multiple) `PageDescription` JSONs plus the wiki `.md` files:

```
+---------------------------+         +-----------------------------+
| data/raw/wiki/*.md        | ──┐     | data/world-graph/           |
| (curated bios)            |   │     |   entities.yaml             |
+---------------------------+   ├──→  |   relationships.yaml        |
| data/raw/wiki-upstream/   |   │     |   (durable, hand-editable,  |
|   *.md (framagit, opt.)   | ──┤     |    version-controlled)      |
+---------------------------+   │     +─────────────────────────────+
| data/raw/ep*/pages/       | ──┘
|   page_*.json             |
| (one per ingested page)   |
+---------------------------+
        ↑                                          ↓
        |                                  ingest_world_graph.py
+---------------------+                            ↓
| extract-world-graph |                            DB
|   skill (you,       |
|   Claude Code)      |
+---------------------+
```

> *Plain-English aside: what's a "Claude Code skill"?* A **skill** is a file at `.claude/skills/<name>/SKILL.md` that tells Claude Code — the CLI/editor agent reading this very repo — how to perform a specific task, plus what tools (Read / Write / Bash / Glob) it's allowed to use. Skills appear with their description in the interactive skills list; trigger phrases in the description fire them when you type a matching prompt. The full mechanism and the architectural decision to use skills as one-shot artifact authors is in [Post 5]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}#what-is-a-skill).

If a skill is just "instructions for Claude Code," what makes it valuable? The leverage is that *Claude Code is already in this loop*. When you're editing the repo, you can type "extract the world graph" and the skill walks the source files and writes the YAML. No new service to deploy, no model API to authenticate against, no rate limit. The runtime pipeline reads the YAML on next startup. **The model touches the artifact at authoring time; the artifact touches the runtime.**

The contrast worth naming is the alternative: a "smart" runtime that calls a model at every page load to figure out what entities the reader should see. That's a server you have to scale, a cost per visit, and a model whose drift can quietly break the graph. The skill pattern moves all of that to authoring time, where you can `git diff` the output and decide.

---

## The Pipeline, End to End {#diagram}

One picture for what we're building. Notice that the model is in the loop *only* at the top — the runtime is plain SQL + REST + React, with the spoiler boundary as the load-bearing piece.

<div style="margin: 1.5rem 0; overflow-x: auto;">
<a href="/assets/picture/2026-05-30-pepper-carrot-companion-spoiler-safe-world-graph/world-graph-flow.svg" target="_blank" rel="noopener" title="Open the diagram full-size in a new tab" style="display: block; cursor: zoom-in;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 720" role="img"
     aria-label="The world-graph pipeline, three tiers. Authoring time: three input sources (the image manifest, the page-description JSONs, and the wiki .md files) feed into the extract-world-graph Claude Code skill, which writes the entities.yaml and relationships.yaml pair. Ingestion: ingest_world_graph.py reads the YAML pair, validates through the pydantic loader, and upserts into world_entities + world_relationships in Postgres. Runtime: the browser overlay sends GET /api/world-graph with the reader's slug and page; the FastAPI route runs a SQL row-value comparison (tuple_(episode_debut, page_debut) &lt;= (current_episode, current_page)) plus an endpoint-visibility check for edges; one SELECT against Postgres returns the spoiler-filtered nodes and edges back to react-flow."
     style="display: block; width: 100%; height: auto; max-width: 1000px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">
  <defs>
    <marker id="wg-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>
    </marker>
    <marker id="wg-arrow-dim" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af"/>
    </marker>
  </defs>

  <!-- ── TIER 1: AUTHORING ───────────────────────────────────────────── -->
  <text x="20" y="32" font-size="12" font-weight="700" fill="#7c2d12" font-style="italic">AUTHORING TIME · Claude Code as one-shot author</text>

  <!-- Three input boxes stacked vertically on the LEFT -->
  <g>
    <rect x="40" y="55" width="240" height="46" rx="6" fill="#fef3c7" stroke="#b45309" stroke-width="1.3"/>
    <text x="160" y="74" text-anchor="middle" font-size="11" font-weight="600" fill="#1f2937">image_manifest.json</text>
    <text x="160" y="90" text-anchor="middle" font-size="9" fill="#92400e" font-style="italic">from wiki_image_scraper.py</text>
  </g>

  <g>
    <rect x="40" y="115" width="240" height="46" rx="6" fill="#fef3c7" stroke="#b45309" stroke-width="1.3"/>
    <text x="160" y="134" text-anchor="middle" font-size="11" font-weight="600" fill="#1f2937">PageDescription JSONs</text>
    <text x="160" y="150" text-anchor="middle" font-size="9" fill="#92400e" font-style="italic">data/raw/ep*/pages/ (Post 5)</text>
  </g>

  <g>
    <rect x="40" y="175" width="240" height="46" rx="6" fill="#fef3c7" stroke="#b45309" stroke-width="1.3"/>
    <text x="160" y="194" text-anchor="middle" font-size="11" font-weight="600" fill="#1f2937">wiki sources</text>
    <text x="160" y="210" text-anchor="middle" font-size="9" fill="#92400e" font-style="italic">data/raw/wiki + wiki-upstream/*.md</text>
  </g>

  <!-- Skill box CENTER -->
  <g>
    <rect x="400" y="100" width="240" height="80" rx="8" fill="#fde68a" stroke="#7c2d12" stroke-width="1.8"/>
    <text x="520" y="128" text-anchor="middle" font-size="13" font-weight="700" fill="#7c2d12">★ extract-world-graph</text>
    <text x="520" y="146" text-anchor="middle" font-size="10" fill="#7c2d12" font-style="italic">read · synthesize · validate</text>
    <text x="520" y="164" text-anchor="middle" font-size="9" fill="#94a3b8">.claude/skills/extract-world-graph/</text>
  </g>

  <!-- YAML box RIGHT -->
  <g>
    <rect x="760" y="100" width="200" height="80" rx="8" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>
    <text x="860" y="128" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">entities.yaml +</text>
    <text x="860" y="144" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">relationships.yaml</text>
    <text x="860" y="164" text-anchor="middle" font-size="9" fill="#94a3b8" font-style="italic">the durable artifact</text>
  </g>

  <!-- Inputs → skill: three converging dashed lines -->
  <path d="M 280 78 Q 340 90, 400 130" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,3" fill="none"/>
  <path d="M 280 138 L 400 138" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,3" fill="none"/>
  <path d="M 280 198 Q 340 180, 400 150" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,3" fill="none"/>

  <!-- Skill → YAML: solid arrow -->
  <line x1="640" y1="140" x2="755" y2="140" stroke="#6b7280" stroke-width="1.5" marker-end="url(#wg-arrow)"/>

  <!-- ── TIER 2: INGESTION ───────────────────────────────────────────── -->
  <text x="20" y="282" font-size="12" font-weight="700" fill="#1e40af" font-style="italic">INGESTION · one CLI command, run after every YAML edit</text>

  <!-- ingest_world_graph.py (includes pydantic loader as a sub-line) -->
  <g>
    <rect x="320" y="305" width="280" height="80" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="460" y="332" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">ingest_world_graph.py</text>
    <text x="460" y="350" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">uses world_graph_loader (pydantic)</text>
    <text x="460" y="368" text-anchor="middle" font-size="9" fill="#94a3b8">delete relationships → prune orphans → upsert</text>
  </g>

  <!-- Postgres -->
  <g>
    <rect x="720" y="305" width="240" height="80" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="840" y="332" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">Postgres</text>
    <text x="840" y="350" text-anchor="middle" font-size="10" fill="#92400e" font-style="italic">world_entities +</text>
    <text x="840" y="364" text-anchor="middle" font-size="10" fill="#92400e" font-style="italic">world_relationships</text>
  </g>

  <!-- YAML → ingest -->
  <line x1="860" y1="180" x2="600" y2="305" stroke="#6b7280" stroke-width="1.5" marker-end="url(#wg-arrow)"/>
  <text x="715" y="240" font-size="9" fill="#94a3b8" font-style="italic" text-anchor="middle">read + validate</text>

  <!-- ingest → Postgres -->
  <line x1="600" y1="345" x2="715" y2="345" stroke="#6b7280" stroke-width="1.5" marker-end="url(#wg-arrow)"/>
  <text x="657" y="338" text-anchor="middle" font-size="9" fill="#94a3b8" font-style="italic">upsert</text>

  <!-- ── TIER 3: RUNTIME ──────────────────────────────────────────────── -->
  <text x="20" y="442" font-size="12" font-weight="700" fill="#065f46" font-style="italic">RUNTIME · every page-flip the reader makes</text>

  <!-- Browser -->
  <g>
    <rect x="40" y="465" width="240" height="140" rx="8" fill="#d1fae5" stroke="#059669" stroke-width="1.5"/>
    <text x="160" y="492" text-anchor="middle" font-size="13" font-weight="600" fill="#1f2937">Browser · overlay</text>
    <text x="160" y="512" text-anchor="middle" font-size="10" fill="#065f46" font-style="italic">react-flow avatar nodes</text>
    <text x="160" y="528" text-anchor="middle" font-size="10" fill="#065f46" font-style="italic">+ fade-in on new debuts</text>
    <text x="160" y="552" text-anchor="middle" font-size="10" fill="#065f46" font-style="italic">+ "Ask in wiki mode" →</text>
    <text x="160" y="567" text-anchor="middle" font-size="10" fill="#065f46" font-style="italic">  pushes into ChatPanel</text>
  </g>

  <!-- API + spoiler filter -->
  <g>
    <rect x="340" y="465" width="280" height="140" rx="8" fill="#fde68a" stroke="#b45309" stroke-width="1.8"/>
    <text x="480" y="492" text-anchor="middle" font-size="13" font-weight="700" fill="#7c2d12">★ GET /api/world-graph</text>
    <text x="480" y="512" text-anchor="middle" font-size="10" fill="#7c2d12" font-style="italic">spoiler-filter is SQL row-value:</text>
    <text x="480" y="532" text-anchor="middle" font-size="10" fill="#475569"
          font-family="ui-monospace, 'SF Mono', Menlo, monospace">tuple_(ep_debut, pg_debut)</text>
    <text x="480" y="548" text-anchor="middle" font-size="10" fill="#475569"
          font-family="ui-monospace, 'SF Mono', Menlo, monospace">  &lt;= (current_ep, current_pg)</text>
    <text x="480" y="575" text-anchor="middle" font-size="9" fill="#94a3b8" font-style="italic">edges require both endpoints visible</text>
  </g>

  <!-- Postgres (same DB as ingestion tier — keep x aligned) -->
  <g>
    <rect x="680" y="465" width="280" height="140" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="820" y="492" text-anchor="middle" font-size="13" font-weight="600" fill="#1f2937">Postgres</text>
    <text x="820" y="512" text-anchor="middle" font-size="10" fill="#92400e" font-style="italic">world_entities +</text>
    <text x="820" y="528" text-anchor="middle" font-size="10" fill="#92400e" font-style="italic">world_relationships</text>
    <text x="820" y="555" text-anchor="middle" font-size="9" fill="#94a3b8">one SELECT per request</text>
  </g>

  <!-- Browser → API (request) -->
  <line x1="280" y1="525" x2="338" y2="525" stroke="#6b7280" stroke-width="1.5" marker-end="url(#wg-arrow)"/>
  <text x="309" y="515" text-anchor="middle" font-size="9" fill="#94a3b8" font-style="italic">slug + page</text>

  <!-- API → Postgres -->
  <line x1="620" y1="525" x2="678" y2="525" stroke="#6b7280" stroke-width="1.5" marker-end="url(#wg-arrow)"/>

  <!-- Postgres → API → Browser (response, dashed return path) -->
  <path d="M 680 555 L 620 555 M 340 555 L 280 555" stroke="#9ca3af" stroke-width="1.3" stroke-dasharray="5,3" fill="none" marker-end="url(#wg-arrow-dim)"/>
  <line x1="620" y1="555" x2="340" y2="555" stroke="#9ca3af" stroke-width="1.3" stroke-dasharray="5,3"/>
  <text x="480" y="595" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">spoiler-filtered nodes + edges flow back</text>

  <!-- "Same DB" dashed connector between ingestion-Postgres and runtime-Postgres -->
  <line x1="840" y1="385" x2="840" y2="460" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="2,4"/>
  <text x="855" y="425" font-size="9" fill="#94a3b8" font-style="italic">same DB</text>

  <!-- ── Legend ──────────────────────────────────────────────────────── -->
  <g transform="translate(40, 660)">
    <rect x="0" y="0" width="14" height="12" fill="#fde68a" stroke="#7c2d12" stroke-width="1.5"/>
    <text x="20" y="11" font-size="11" fill="#4b5563">★ Post 12 load-bearing</text>
    <rect x="180" y="0" width="14" height="12" fill="#fef3c7" stroke="#b45309" stroke-width="1"/>
    <text x="200" y="11" font-size="11" fill="#4b5563">authoring artifact / on disk</text>
    <rect x="400" y="0" width="14" height="12" fill="#dbeafe" stroke="#2563eb" stroke-width="1"/>
    <text x="420" y="11" font-size="11" fill="#4b5563">ingestion code (Python)</text>
    <rect x="600" y="0" width="14" height="12" fill="#d1fae5" stroke="#059669" stroke-width="1"/>
    <text x="620" y="11" font-size="11" fill="#4b5563">runtime (browser / API)</text>
  </g>
</svg>
</a>
</div>

*The three tiers. Authoring time (amber) runs in your Claude Code session — the skill reads the manifest, the wiki sources, and every page JSON it can find, and writes the YAML pair. Ingestion (blue) is a single `uv run python ingest_world_graph.py` that validates the YAML through pydantic and upserts the rows. Runtime (green) is one FastAPI route + one React overlay; the spoiler-filter SQL is the only thing in the request path that decides what's visible. Click the diagram to open it full-size in a new tab.*

The four sections below take the four boxes I starred — the loader contract, the scraper, the skill, and the spoiler-filter API — one at a time. Then the frontend.

---

## The YAML: One Schema, Two Authors {#the-yaml}

The artifact the skill writes and the loader reads is a pair of YAML files at `data/world-graph/`. Here's the shape, abbreviated to two entities and two edges:

```yaml
# data/world-graph/entities.yaml
entities:
  - slug: pepper
    name: Pepper
    kind: character
    summary: The young witch at the centre of the comic. She practises Chaosah …
    episode_debut: 1
    page_debut: 1
    layout: { x: 0, y: 0 }
    image_url: world-graph/images/pepper-thumb.webp
    character_slug: pepper       # links to the canonical Character row

  - slug: chaosah
    name: Chaosah
    kind: coven
    summary: The school of chaos magic — among the oldest and most feared …
    episode_debut: 8
    page_debut: 5                # first explicit Chaosah mention in any episode
    layout: { x: -300, y: -200 }
    image_url: null              # no scraped art for covens; frontend draws SVG
```

```yaml
# data/world-graph/relationships.yaml
relationships:
  - source: pepper
    target: chaosah
    kind: member_of
    episode_debut: 8
    page_debut: 5                # both endpoints visible by this cursor

  - source: cayenne               # cayenne debuts ep11 → this edge can't appear earlier
    target: pepper
    kind: godmother_of
    episode_debut: 11
    page_debut: 1
```

A handful of design choices worth surfacing:

- **`slug` is the natural key.** Re-running the loader is upsert-by-slug, so hand-editing the YAML (say, fixing a `summary` line) and re-running `ingest_world_graph.py` mutates the existing row instead of inserting a duplicate.
- **`character_slug` is optional and bridges to the canonical roster.** When set on a `kind: character` entity, the loader looks up the matching row in the seeded `characters` table by lower-cased name and stores its UUID on `world_entities.character_id`. This is what lets a future frontend (or a future API) jump from the graph to "what page does this character first appear on?" without a duplicate name lookup.
- **`image_url` is a relative key.** Same convention as `pages.image_url` from Post 7: the DB stores `world-graph/images/pepper-thumb.webp`, and the FastAPI route composes the absolute URL at response time through the `Storage` protocol from [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}). Swapping local disk for Cloudflare R2 in Post 14 won't require a database migration.
- **`(episode_debut, page_debut)` is the spoiler coordinate.** This is the only field the API filter cares about. Edges have their *own* debut tuple — Cayenne exists from ep11, the `godmother_of` relation is also ep11, but a future edge between two ep1-debut entities (say, a rivalry revealed in ep15) would carry an ep15 debut and stay hidden until then. The phase-12 doc calls out that this is non-negotiable: a hand-edit could in principle put a future debut on a pre-existing entity, and an "edges-only when both endpoints are visible" check alone would leak it.

### The pydantic contract

The shape isn't suggested by the skill's docstring — it's *enforced* by [`ingestion/world_graph_loader.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-12-13-worldgraph/ingestion/world_graph_loader.py):

```python
# ingestion/world_graph_loader.py (abridged)
class EntityData(BaseModel):
    slug: str
    name: str
    kind: str
    summary: str | None = None
    episode_debut: int
    page_debut: int
    layout: _Layout              # nested {x: float, y: float}
    image_url: str | None = None
    character_slug: str | None = None

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, value: str) -> str:
        allowed = {"character", "creature", "place", "coven", "object"}
        if value not in allowed:
            raise ValueError(f"kind must be one of {sorted(allowed)}, got {value!r}")
        return value


def load_world_graph(graph_dir: Path) -> tuple[list[EntityData], list[RelationshipData]]:
    # … reads both files, validates, then runs:
    known_slugs = {e.slug for e in entities}
    for rel in relationships:
        if rel.source not in known_slugs:
            raise ValueError(f"Relationship references unknown source slug '{rel.source}' …")
        if rel.target not in known_slugs:
            raise ValueError(f"Relationship references unknown target slug '{rel.target}' …")
    return entities, relationships
```

Two contracts in one file: **pydantic validation** (the kind is one of five strings, slugs are non-empty, layout has x and y) and **a cross-file consistency check** (every relationship's source/target points at a known entity slug). A typo in `relationships.yaml` fails at load time with a clear error, not at runtime as a missing-row 500.

The same module is what the skill imports when validating its own output before exiting — the skill and the loader share one contract by importing the same pydantic models. That's the **"the contract is the schema"** rule from [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}#one-protocol-many-implementations), reapplied at the YAML seam.

> *Plain-English aside: why YAML and not JSON?* The artifact is meant to be *hand-edited* — fixing a summary, nudging a coordinate, correcting a debut episode. YAML wins on readability for that use case (no trailing commas to chase, no `"key":` quoting overhead, comments allowed inline so I can flag a low-confidence row with `# confidence: low — defaulted`). JSON would be marginally faster to parse and trivially friendlier to other tooling, but at ~40 rows of structured data, those don't matter. The discipline of "the artifact is meant to be read, edited, and reviewed by humans" outweighs the parser speed.

---

## The Wiki Image Scraper {#scraper}

Before the skill can run, it needs to know which entities have artwork available. The artwork lives on [framagit.org/peppercarrot/wiki](https://framagit.org/peppercarrot/wiki/-/tree/master/medias/img) — about 43 small CC BY 4.0 JPEGs named `chara_*.jpg` (the named cast) and `creature_*.jpg` (familiars and beasts). The scraper at [`ingestion/wiki_image_scraper.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-12-13-worldgraph/ingestion/wiki_image_scraper.py) does three things, in order:

1. **List the tree** via framagit's REST API (`GET /api/v4/projects/peppercarrot%2Fwiki/repository/tree?path=medias/img`). Filter by the `chara_` / `creature_` prefixes; ignore everything else (logo PNGs, hair-piece sprites).
2. **Download each candidate** from the raw URL and **process to two WebP variants** with Pillow — a 96×96 center-square-crop thumbnail (what the graph node renders) and a 320px-longest-edge display variant (what the info card shows on click). Quality 85 WebP keeps each pair under ~10 KB.
3. **Write `data/world-graph/image_manifest.json`** with the scraped ref's sha, the timestamp, and the lists of character/creature slugs.

The manifest is the seam between the scraper and the skill — the scraper doesn't know what entities the skill will author, and the skill doesn't know what filenames the scraper found. They meet at the manifest:

```json
{
  "scraped_from": "framagit:peppercarrot/wiki",
  "scraped_ref": "master",
  "scraped_sha": "3db92c79",
  "scraped_at": "2026-05-31T01:08:06+00:00",
  "characters": ["acren", "camomile", "carrot", "cayenne", "coriander", "cumin", …],
  "creatures": ["air-dragon", "argiope", "bigfish", "dragon-drake", "dragoncow", …]
}
```

Why bother with a manifest when the skill could just `ls data/world-graph/images/`? Two reasons:

- **The manifest captures the source provenance** (`scraped_from`, `scraped_sha`, `scraped_at`) — the audit trail of where the art came from and when. Committed to the repo even though the image bytes aren't (gitignored). Anyone running the project from a fresh clone runs the scraper once and gets the same manifest.
- **The skill needs to know whether a slug is a `character` or a `creature` to assign the right `kind`** — and that information is in the filename prefix, which gets normalized away by the time the variants are written. Putting the prefix info in the manifest as two parallel lists keeps the skill from having to parse filenames itself.

Run the scraper once per fresh clone (or whenever the framagit wiki updates):

```bash
$ cd ingestion && uv run python wiki_image_scraper.py
Found 40 candidate images at master@3db92c79.
  fetching chara_acren.jpg … → acren-thumb.webp (2188 B), acren-display.webp (4082 B)
  fetching chara_camomile.jpg … → camomile-thumb.webp (2392 B), …
  …
  fetching creature_theorem-the-golem.jpg … → theorem-the-golem-thumb.webp (2390 B), …

Done. Downloaded 40 new image(s); skipped 0 already-present.
  characters: 19
  creatures:  21
  manifest:   ../data/world-graph/image_manifest.json
```

The skill (next section) is the *only* code in the repo that reads this manifest. The runtime never consults it; it only reads `world_entities.image_url`, which the skill set based on the manifest at authoring time.

---

## The `extract-world-graph` Skill {#skill}

Now the part where Claude Code does the actual extraction. The skill lives at [`.claude/skills/extract-world-graph/SKILL.md`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-12-13-worldgraph/.claude/skills/extract-world-graph/SKILL.md). Its frontmatter is the standard skill shape — a description, trigger phrases, and an allowed-tools list:

```yaml
---
description: Extract the Pepper&Carrot world graph from the workshop's local sources — the curated wiki (`data/raw/wiki/*.md`), the framagit-scraped wiki (`data/raw/wiki-upstream/*.md` when present), the per-page description JSONs under `data/raw/ep*/pages/`, and the `data/world-graph/image_manifest.json` — then write `data/world-graph/entities.yaml` and `data/world-graph/relationships.yaml` that the world-graph loader consumes. Trigger phrases include "extract the world graph", "rebuild the world graph", "regenerate world graph YAML", "extract-world-graph".
allowed-tools: [Read, Write, Edit, Bash, Glob]
---
```

The body walks Claude Code through six steps. Here's the shape; the full SKILL.md is the canonical version.

**Step 0 — read the manifest.** Open `data/world-graph/image_manifest.json`. If it doesn't exist, instruct the user to run `wiki_image_scraper.py` and stop.

**Step 1 — read source material.** The required sources are `data/raw/wiki/*.md` (the curated short bios that ship with the workshop — Pepper, Carrot, the four primary covens, etc.) and `data/raw/ep*/pages/page_*.json` (every page Claude Code itself wrote with the `ingest-from-images` skill from Post 5). The optional supplement is `data/raw/wiki-upstream/*.md` (framagit-scraped — populated by running `wiki_scraper.py`), which adds long bios of every minor character, creature, place, and school.

**Step 2 — find debut episodes.** Walk every page JSON, collect `characters_present` lists, find the **earliest** `(episode_number, page_number)` each canonical name appears in. Same for `locations_or_concepts`. Then for any entity from the wiki sources that *never* appears in any page JSON (Hippiah's school, Hereva itself, deep-bestiary creatures), default to `(1, 1)` with a `# confidence: low — defaulted` YAML comment so the human knows what to audit.

**Step 3 — build the entity list.** For each canonical character / coven / place / creature:

- **Align the slug with the image manifest where possible** so `image_url` lights up. *"The Sage"* in the wiki becomes `slug: sage` (matching `chara_sage.jpg`) with `name: The Sage`. *"Mayor of Komona"* becomes `slug: mayor`, `name: Mayor of Komona`. Slug mismatches are the most common reason an entity falls back to the SVG icon when it shouldn't.
- **Summary is 1–2 sentences of plain prose**, paraphrased from the wiki sources and/or the page JSONs. The skill's body explicitly bans markdown headers, bullet lists, and recitation — same anti-recitation discipline from [Post 11]({% post_url 2026-05-30-pepper-carrot-companion-prompt-hardening %}#anti-recitation), applied to the artifact this time.
- **Layout coordinates** follow clustering rules: Pepper at origin, covens at the four compass points, witches in small rings around their coven, geography as a horizontal strip below. These are starting positions — the human edits the YAML to tune them — and the skill's STEP 2a is **non-negotiable**: when re-running, **preserve hand-tweaked layouts** by reading the existing `entities.yaml` and reusing each slug's `layout.{x, y}`.

**Step 4 — build the relationship list.** A short taxonomy of kinds (`member_of`, `godmother_of`, `apprentice_of`, `familiar_of`, `lives_in`, `located_in`, `rival_of`, `friend_of`, `family_of`) keeps the frontend's edge-coloring table small. When the source doesn't pin a reveal episode, default to the later of the two endpoints' debuts and flag the row with `# confidence: low`.

**Step 5 — validate before writing.** Run the loader's pydantic check before exiting:

```bash
cd ingestion && uv run python -c "
from pathlib import Path
from world_graph_loader import load_world_graph
entities, rels = load_world_graph(Path('../data/world-graph'))
print(f'OK: {len(entities)} entities, {len(rels)} relationships')
"
```

If validation fails, **the skill is required to fix the YAML and re-validate before reporting success**. Don't leave a broken YAML on disk — better to abort with a clear error than to falsely report progress.

**Step 6 — report.** Counts by kind, image coverage, low-confidence rows for human review.

### A real run

Trigger the skill in Claude Code (open the repo in the editor, type "extract the world graph"). The skill picked up the workshop's state — ep01-12 ingested, the curated + framagit-scraped wiki sources on disk, the manifest fresh from the scraper — and produced:

- **45 entities**: 19 characters (Pepper, Carrot, the three Chaosah godmothers, Saffron, Coriander, Shichimi, Yuzu, Truffel, Mango, Spirulina, Apiaceae, Camomile, Quassia, Torreya, the Mayor of Komona, Prince Acren, the Sage), 6 covens (Chaosah, Hippiah, Magmah, Aquah, Zombiah, Ah), 5 places (Hereva, Komona, Squirrel's End, Qualicity, the Temples of Ah), and 15 creatures (the bestiary plus the three Chaosah demons Hornuk / Eyeük / Spidük).
- **57 relationships**: familiars (Carrot → Pepper, Yuzu → Shichimi, Truffel → Saffron, Mango → Coriander), the godmother edges, the coven memberships, the rivalries, the friendships across schools, and the structural `lives_in` / `located_in` edges that anchor characters to places.
- **Image coverage**: 18 of 19 characters and 12 of 15 creatures have matching framagit art via the manifest; the remaining 3 (plus all 6 covens and all 5 places) use the kind-based SVG fallbacks.

The numbers grow with what the project ingests. A fresh clone with only ep01 ingested and only the curated `data/raw/wiki/` will produce ~10 entities from this skill; pull `data/raw/wiki-upstream/` with `wiki_scraper.py` and the count climbs as the framagit bestiary becomes visible. Hand-edit the YAML to fix anything. Re-run `ingest_world_graph.py` to push the edits to Postgres. The skill is for first passes and source-material refreshes; **subsequent fixes go straight into the YAML**, and the loader's idempotent upsert handles the rest.

---

## The Spoiler Filter, This Time in Postgres {#api}

The runtime side of Post 12 is a single FastAPI route — `GET /api/world-graph` — with four query parameters: `episode_slug`, `page`, an optional `right_page` (the right page of a two-page spread, so the spoiler cursor uses the rightmost visible page), and `mode` (`full` or `focus`, default `full`). The route runs at most two SQL `SELECT`s — one for entities, one for relationships — and the *shape* of the spoiler filter is the same as Post 9 and worth dwelling on.

The mental model: the reader's position is an integer pair `(episode_number, page_number)`. An entity (or edge) is **visible** when its `(episode_debut, page_debut)` is **lexicographically less than or equal to** the cursor. "Lexicographic" means we compare first by episode number, then by page number — exactly the way you'd order words in a dictionary. The reader on ep5 page 2 sees:

- every entity from ep1–4 (any page)
- entities from ep5 pages 1–2

…and nothing else.

Post 9 expressed this as a Chroma `where` clause (`$or` over `$lt episode` and `$and episode + $lt page`). That worked because Chroma's where DSL has those operators. Postgres has something cleaner: **row-value comparison** directly on tuples.

```python
# backend/app/api/world_graph.py (the load-bearing line)
from sqlalchemy import tuple_

debut_tuple = tuple_(WorldEntity.episode_debut, WorldEntity.page_debut)
node_stmt = (
    select(WorldEntity)
    .where(debut_tuple <= cursor)   # cursor is the Python tuple (current_ep, current_pg)
    .order_by(WorldEntity.kind, WorldEntity.slug)
)
```

This compiles to literal SQL:

```sql
SELECT * FROM world_entities
 WHERE (episode_debut, page_debut) <= ('5', '2')
 ORDER BY kind, slug;
```

And **Postgres compares the row values lexicographically by default** ([SQL standard §8.2, supported by Postgres since at least 9.0](https://www.postgresql.org/docs/current/functions-comparisons.html#ROW-WISE-COMPARISON)). One operator, one expression. The SQL reads like the math.

> *Plain-English aside: row-value comparison.* Most SQL comparisons are scalar — `a <= 5`. **Row-value comparison** lets you compare a *tuple* against another tuple: `(a, b) <= (5, 2)`. It evaluates left-to-right, in lexicographic order: first compare the first elements; if they're equal, compare the second; and so on. So `(1, 9) <= (5, 2)` is true (because `1 < 5`), `(5, 1) <= (5, 2)` is true (because `5 = 5` and `1 < 2`), `(5, 9) <= (5, 2)` is false. This is exactly how a dictionary orders strings letter-by-letter. The advantage over `a < 5 OR (a = 5 AND b <= 2)` is that the row form is one expression instead of three, and harder to typo into a subtly-wrong shape.

### Edges have their own debut

The route's edge SQL has a second requirement that's non-negotiable. The phase-12 design doc spells out the failure mode the workshop must avoid:

> Spoiler leaks via edges. It's tempting to filter edges only by checking "both endpoints are visible." That's wrong — an edge can carry plot meaning that debuts later than both nodes (e.g. Pepper and Coriander exist from ep1; their rivalry might be revealed in ep5). Filter edges by their own `(episode_debut, page_debut)` AND require both endpoints to satisfy the same predicate. Both checks, not either.

In SQL:

```python
edge_debut_tuple = tuple_(
    WorldRelationship.episode_debut, WorldRelationship.page_debut
)
edge_stmt = (
    select(WorldRelationship)
    .where(edge_debut_tuple <= cursor)                          # edge's own debut
    .where(WorldRelationship.source_id.in_(visible_ids))        # source endpoint visible
    .where(WorldRelationship.target_id.in_(visible_ids))        # target endpoint visible
    .order_by(WorldRelationship.kind)
)
```

Three where clauses; all three mandatory. If you drop the first, a hand-edited "rivalry revealed in ep15" leaks at ep10 the moment both rivals are introduced. If you drop the last two, an authored-but-orphan edge leaks at the cursor of its own debut. **The test suite pins this one explicitly**, because it's the part that's plausibly "obvious" and silently wrong.

### Two modes, one boundary

The `mode=focus` branch layers on top of the same spoiler predicate — it just narrows the visible set further before returning it. The seed is the canonical characters drawn on any page in `[page, right_page]` (joined through `page_characters` → `characters` → `world_entities.character_id`), still gated by the lexicographic debut tuple so a hand-edited future-debut entity that's also on the page can't slip past. From the seed, the route runs one more SELECT for edges of three structural kinds (`member_of`, `lives_in`, `familiar_of`) where either endpoint is in the seed and the edge's own debut is at or before the cursor; the *other* endpoint of each surviving edge joins the visible set; then the same edge-filter from full mode runs over the expanded set to pull every spoiler-safe edge among the focused nodes (not just the structural-expansion kinds — so a `rival_of` edge between two members of the focus subset still shows when both rivals are in scope).

If the seed comes back empty — a wordless action panel, a landscape painting, or pages that haven't been ingested yet — the route falls back to the full subset rather than returning an empty graph. That's a UX call: most readers' next action after "empty graph" is "ok then show me the whole world", and the kind-filter bar is right there in the UI to narrow it again.

The `right_page` parameter is a small but load-bearing detail. The chat layer from [Post 9]({% post_url 2026-05-25-pepper-carrot-companion-spoiler-safe-rag %}) reads the page from server-side session state (`chat_sessions.current_page`), but the world-graph route is per-request — the cursor lives in the URL. A landscape reader looking at pages 5–6 has *seen* page 6, so the spoiler cursor is `(episode_number, 6)`; sending only `page=5` would gate them out of an entity that debuts on the very page they're looking at. The flipbook passes both pages on landscape spreads; the route uses `right_page` for the spoiler cursor and `[page, right_page]` for the focus-mode seed.

### Proving it: 10 tests, in-memory SQLite

The test file at [`backend/tests/test_world_graph_api.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-12-13-worldgraph/backend/tests/test_world_graph_api.py) does what Post 9's `test_retrieval.py` did with a hermetic Chroma collection: it spins up a real SQL engine (in-memory `aiosqlite`), creates the tables, seeds a hand-crafted graph, and asserts the API output is what the spoiler-filter mandates.

The seeded graph is small enough to reason about:

```text
Entity A — debut (1, 1) — character
Entity B — debut (1, 5) — character
Entity C — debut (2, 3) — coven
Entity D — debut (2, 7) — character

Edge  A→B  kind=friend_of  debut (1, 5)
Edge  B→C  kind=member_of  debut (2, 3)
Edge  A→C  kind=member_of  debut (2, 5)   ← edge debuts AFTER both endpoints
Edge  C→D  kind=rival_of   debut (2, 7)
```

The test that nails the "edge debuts later than its endpoints" case:

```python
async def test_edge_debut_is_filtered_independently(override_deps: None) -> None:
    """At episode 2 page 4, A + B + C are visible (C debuts (2,3)).

    The A→B edge shows. The B→C edge shows (debut (2,3)). The A→C edge
    must NOT show — it debuts at (2,5), which is past the cursor. This
    is the bug the phase-12 doc warns about: an edge can debut later
    than both of its endpoints.
    """
    r = await client.get("/api/world-graph?episode_slug=ep02&page=4")
    body = r.json()
    assert {n["slug"] for n in body["nodes"]} == {"a", "b", "c"}
    edge_kinds = sorted(e["kind"] for e in body["edges"])
    # Only friend_of (1,5) and member_of B→C (2,3) — NOT A→C member_of (2,5).
    assert edge_kinds == ["friend_of", "member_of"]
```

Plus nine more covering the happy path, the unknown-episode 404, the page-past-end clamp, the lexicographic later-episode-unlocks-earlier rule, the image-URL composition through the storage abstraction, the focus-mode fallback when no characters are on the page, the `right_page` cursor (right-edge of a two-page spread), and the silent collapse when a flipbook ordering glitch sends `right_page < page`. All ten pass; the workshop's total test count is now 43.

> *A small infra trade-off worth flagging:* the workshop's production runtime uses Postgres-only column types — `ARRAY(String)` for tag lists, `JSONB` for blob metadata. SQLite doesn't have those. The fix is a one-line `with_variant(JSON, "sqlite")` on the five affected columns in [`backend/app/db/models.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-12-13-worldgraph/backend/app/db/models.py): Postgres still uses `ARRAY` / `JSONB` at runtime, SQLite gets a portable JSON column at test time. The diff is mechanical, the runtime is unchanged, and tests stay hermetic — they don't need a docker-compose Postgres to run.
>
> ```python
> # backend/app/db/models.py — runtime uses ARRAY/JSONB; tests get a JSON shim.
> _PG_ARRAY_STR = ARRAY(String).with_variant(JSON, "sqlite")
> _PG_JSONB = JSONB().with_variant(JSON, "sqlite")
>
> mood_tags: Mapped[list[str]] = mapped_column(_PG_ARRAY_STR, default=list)
> image_metadata: Mapped[dict[str, Any]] = mapped_column(_PG_JSONB, default=dict)
> ```

---

Next up: **Post 13 — Rendering the World Graph: A React-Flow Overlay and Summary-First Wiki.** The spoiler-safe API is built; now it gets a face. [Post 13]({% post_url 2026-05-31-pepper-carrot-companion-world-graph-overlay %}) renders this route as an in-reader React + xyflow overlay — circular avatar nodes with kind-based SVG fallbacks, a kind-filter bar, kind-colored edges that brighten on the selected node, a soft fade-in for newly-revealed entities, and an "Ask in wiki mode" click that round-trips back through the chat panel — and closes the loop with a third skill, `summarize-wiki`, so that clicking an avatar actually returns a focused, grounded answer instead of drowning the small model in 30 KB of multi-entity articles.

The **workshop starter** that backs this post is at <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop>, tagged `post-12-13-worldgraph` — `git checkout post-12-13-worldgraph` to get exactly the code shown here. The **full source repository** and the public live-demo URL go up alongside the deploy guide near the end of the series.

*Pepper & Carrot* is © [David Revoy](https://www.davidrevoy.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All credit to him for the source material that made this project possible.

**All opinions expressed are my own.**

---
