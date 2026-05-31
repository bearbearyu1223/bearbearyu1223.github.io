---
title: "A World Graph Built by a Second Skill: Spoiler-Aware Knowledge Graph Overlay"
date: 2026-05-30 12:00:00 -0800
categories: [Full-Stack, RAG, Local AI]
tags: [knowledge-graph, claude-skills, react-flow, postgres, fastapi, sqlalchemy, peppercarrot, portfolio]
description: >-
  Post 9 of the Pepper & Carrot AI flipbook series. The chat layer answers
  questions about pages and the wiki; now we add a third affordance — a
  spoiler-aware knowledge graph of the comic's world, rendered as an
  in-reader overlay — plus a third Claude Code skill that closes the
  loop back to the chat. A second skill walks the wiki sources + the
  per-page description JSONs and writes a durable YAML pair. A FastAPI
  route filters the graph with a Postgres row-value comparison so an edge
  whose own debut is past the reader's cursor doesn't leak even when both
  of its endpoints are visible. Two response modes — focus (on-page
  characters + 1-hop structural neighbors) and full (the whole spoiler-
  safe world) — share the same boundary. A React + xyflow overlay renders
  circular avatar nodes with kind-based SVG fallbacks, a kind-filter bar,
  kind-colored edges that brighten on the selected node, soft fade-in
  for newly-revealed entities, and an "Ask in wiki mode" click that
  round-trips back through the chat panel. A third skill — summarize-wiki
  — authors one tight ~150-word .md per entity so that "Ask in wiki mode"
  for a minor character like Truffel or a coven like Magmah actually
  works against qwen2.5:7b, instead of the 30 KB multi-entity articles
  blowing past the prompt-hardening guarantees from Post 8.
pin: true
---

Post 9 of the [*Pepper & Carrot AI-powered flipbook*]({% post_url 2026-05-09-pepper-carrot-companion-trailer %}) series. [Post 6]({% post_url 2026-05-25-pepper-carrot-companion-spoiler-safe-rag %}) made spoiler safety a property of the *retrieval* query; [Post 7]({% post_url 2026-05-25-pepper-carrot-companion-streaming-chat %}) and [Post 8]({% post_url 2026-05-30-pepper-carrot-companion-prompt-hardening %}) made the chat *answers* stream cleanly. This post adds a third reading-time affordance on top of the same session: a **world graph overlay** — Pepper, her godmothers, the rival witches, the covens, the places — rendered as circular avatar nodes that reveal themselves over time, gated by exactly the same spoiler boundary the chat sits behind. The graph isn't extracted at runtime; it's authored by a **second Claude Code skill** (the first was [`ingest-from-images`]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}) in Post 4) that walks the wiki sources and the page-description JSONs and writes a YAML pair the workshop loads into Postgres. Same skill-as-author pattern; new artifact; new affordance. **And then a third skill** — `summarize-wiki` — closes the loop, so that when the reader clicks "Ask in wiki mode" on a graph node, the small local model (qwen2.5:7b) actually has the right context to answer with instead of drowning in 30 KB of multi-entity articles.

> **What you'll build in this post.**
> - **A second Claude Code skill**, `.claude/skills/extract-world-graph/SKILL.md`, that reads the curated wiki under `data/raw/wiki/*.md` (plus the framagit-scraped `data/raw/wiki-upstream/*.md` when present) + every `data/raw/ep*/pages/page_*.json` + the image manifest and writes `data/world-graph/entities.yaml` + `data/world-graph/relationships.yaml`. The shipped graph carries **45 entities** (19 characters, 6 covens, 5 places, 15 creatures) and **57 relationships**.
> - **A pydantic loader + ingestion CLI** in `ingestion/world_graph_loader.py` and `ingestion/ingest_world_graph.py` that validate the YAML and upsert it into the `world_entities` + `world_relationships` Postgres tables. The loader prunes orphan entities on every run so a YAML rename (e.g. `squirrels-end` → `squirrel-s-end`) doesn't leave ghost rows.
> - **A wiki image scraper** in `ingestion/wiki_image_scraper.py` that pulls ~40 character/creature portraits from [framagit](https://framagit.org/peppercarrot/wiki) (CC BY 4.0), processes them into thumb + display WebP variants, and writes `image_manifest.json` for the skill to consume.
> - **A spoiler-filtered API endpoint**, `GET /api/world-graph`, in `backend/app/api/world_graph.py` — same lexicographic boundary as Post 6's RAG layer, expressed as a Postgres row-value comparison: `(episode_debut, page_debut) <= (current_episode, current_page)`. Two response modes layer on top: `mode=full` (the whole spoiler-safe world) and `mode=focus` (canonical characters drawn on the current page(s) + a 1-hop expansion through structural edge kinds like `member_of` / `lives_in` / `familiar_of`). A `right_page` query parameter lets two-page-spread readers seed the focus set from both visible pages. **Ten hermetic tests against in-memory SQLite** pin the boundary — including the gotcha where an edge can debut later than both of its endpoints, the focus-mode fallback when no characters are on the page, and the silent collapse when a flipbook callback sends `right_page < page`.
> - **A React + [`@xyflow/react`](https://reactflow.dev/) overlay panel** in `frontend/src/components/world-graph/` (eight small files): avatar nodes with **eight invisible handles** so edges route through whichever side reads cleanest, a **kind-filter bar** with per-kind counts to thin a busy spread, a **mode toggle pill** ("This page" / "Whole world"), **kind-based edge coloring** that brightens on the selected node's incidents, a **focus-layout** that re-arranges the visible subset into a kind-grid when focus mode is on, an info card with an "Ask in wiki mode" button that round-trips through the chat panel, and a **soft fade-in animation** for entities revealed by the latest page flip. The viewport **auto-fits to the visible nodes** in both modes so the whole-world view doesn't strand entities off the right edge of the panel.
> - **A third Claude Code skill**, `.claude/skills/summarize-wiki/SKILL.md`, that reads the wiki source `.md` files + the entity list and writes **one tight ~100-300 word summary per entity** (plus a handful of topic summaries) to `data/wiki-summaries/`. The wiki ingestion pipeline embeds these summaries — not the raw 30 KB articles — so top-3 wiki retrieval lands ~500 words of focused context per question, small enough that Post 8's `OUTPUT RULES` still hold against qwen2.5:7b when the reader clicks "Ask in wiki mode" on a graph node.
>
> **Prerequisites.**
> - The workshop starter at the [`post-9` tag](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/tree/post-9): `git checkout post-9` (see [Following along with the blog series](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop#following-along-with-the-blog-series)). Everything [Post 8]({% post_url 2026-05-30-pepper-carrot-companion-prompt-hardening %}) needed — Postgres up, migrations applied, the roster seeded, at least Episode 1 ingested, Ollama running with `qwen2.5:7b` and `bge-m3` pulled.
> - For a richer demo, **ingest a few more episodes** through the `ingest-from-images` skill from [Post 4]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}) — the world-graph extraction reads `data/raw/ep*/pages/page_*.json` files to find when each canonical character first appears. The shipped graph YAML was extracted against ep01–ep12; if you only have ep01, the graph will show ~3 entities and the rest will populate as you ingest more.
> - [Node.js 20+](https://nodejs.org/) and the same Vite frontend setup from [Post 5]({% post_url 2026-05-23-pepper-carrot-companion-rest-api-flipbook %}).

> **About the repo URL.** All the changes in this post — the new skill, the loader + scraper + CLI, the `world_graph.py` API route, the `world-graph/` frontend components, the YAML pair, and the database-portability shim that lets the spoiler-filter test run against in-memory SQLite — live in the same workshop starter that backed [Posts 2–8](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop), now tagged `post-9`. The cloud-deploy guide is still Post 10.

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
9. [The Frontend Overlay](#frontend)
10. [The Soft Fade-In: A Diff That Drives a Keyframe](#fade-in)
11. [Closing the Loop: The Third Skill and Summary-First Wiki](#third-skill)
12. [What's Honest, What's Open](#honest)
13. [Key Takeaways](#key-takeaways)

---

## The Code in Front of You: Tour + Quick Start {#tour}

Get the running overlay in front of you before any of the concepts land. Skim this even if you plan to read carefully — watching a single node debut as you flip the page makes the rest obvious.

### Get the code at this post's tag

```bash
git clone https://github.com/bearbearyu1223/pepper-carrot-companion-workshop
cd pepper-carrot-companion-workshop
git checkout post-9
```

Already cloned from an earlier post? `git fetch --tags && git checkout post-9`.

### What's new in the workshop starter

Two new modules on the backend, several new modules in `ingestion/`, three skills, a new frontend directory, and the durable artifacts on disk:

```
pepper-carrot-companion-workshop/
├── .claude/skills/
│   ├── ingest-from-images/         (from Post 4 — unchanged)
│   ├── extract-world-graph/        ← NEW (Post 9): the second skill
│   │   └── SKILL.md
│   └── summarize-wiki/             ← NEW (Post 9): the third skill — wiki summaries
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

Four one-time setup commands the first time you check out `post-9`:

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

If you ingested only ep01 through the workshop's starter steps, the graph will show three nodes (Pepper, Carrot, and Chaosah). For the full ep01–ep12 graph used in the screenshots and curl examples below, you'll want to ingest the rest of those episodes via the `ingest-from-images` skill from [Post 4]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}).

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

Same shape as the `done` audit trail in Post 7's chat: the URL carries the reader's cursor, the server clamps it to the episode's real page count, and the SQL filter is the security boundary. Nothing about the request body can widen what comes back.

---

## What This Adds, and What It Doesn't {#what-this-adds}

The series has been building one affordance per post — a flipbook (Post 5), a chat answer (Post 6), a streaming chat panel (Post 7), prompt-hardened replies (Post 8). Post 9 adds a third *reading-time* affordance alongside the chat:

| | Affordance | Built by | Bounded by |
|---|---|---|---|
| **Post 5** | Flipbook reader | StPageFlip + REST API | (no spoiler concern at the page layer) |
| **Posts 6–8** | Chat (page + wiki) | Ollama + spoiler-filtered RAG + prompt hardening | `chat_sessions.current_page` |
| **Post 9** | World-graph overlay | YAML authored by a second skill, loaded into Postgres, filtered in SQL | `(episode_number, current_page)` cursor in the URL |

Three things this post **isn't**:

- **It isn't an entity extractor at runtime.** The graph is extracted *once* (per episode-ingestion change) by a Claude Code skill running in your editor session. The runtime never calls a model to figure out who's in the world; it queries Postgres and ships the rows. No per-request inference cost, no model dependency on graph correctness.
- **It isn't a generic knowledge-graph framework.** No SPARQL, no neo4j, no graph-DB. Two SQLAlchemy tables, two SQL `SELECT`s per request, one row-value comparison for the spoiler boundary. A small graph (45 nodes, 57 edges in the workshop) doesn't need graph-native storage to be fast or expressive.
- **It isn't editorial about which schools or which characters exist.** The skill works only from what the wiki sources and the page JSONs say. An entity that's in the wiki but never appears in any ingested episode lands as a `(1, 1)`-defaulted low-confidence row — visible from page 1, no edges to anything else yet. If the wiki sources don't mention an entity at all, it isn't in the graph. **Coverage grows when the source material grows.**

The architectural through-line connecting Post 6 and Post 9 is one sentence: **the spoiler boundary lives in the data layer, not in the prompt.** Post 6 expressed it as a Chroma `where` clause built from `chat_sessions.current_page`. Post 9 expresses it as a SQL row-value comparison built from a query-string parameter that's clamped against `pages.page_number`. Same shape, different store.

---

## Why a Second Skill? {#why-a-second-skill}

The first skill — [`ingest-from-images`]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}) from Post 4 — established a pattern: Claude Code, with its image-reading and file-writing tools, **acts as a one-shot author** for a durable on-disk artifact (per-page `PageDescription` JSON files). The runtime never calls a model to produce those descriptions; the ingestion pipeline reads them off disk. Post 4 spent a section on why this matters:

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

> *Plain-English aside: what's a "Claude Code skill"?* A **skill** is a file at `.claude/skills/<name>/SKILL.md` that tells Claude Code — the CLI/editor agent reading this very repo — how to perform a specific task, plus what tools (Read / Write / Bash / Glob) it's allowed to use. Skills appear with their description in the interactive skills list; trigger phrases in the description fire them when you type a matching prompt. The full mechanism and the architectural decision to use skills as one-shot artifact authors is in [Post 4]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}#what-is-a-skill).

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
    <text x="160" y="150" text-anchor="middle" font-size="9" fill="#92400e" font-style="italic">data/raw/ep*/pages/ (Post 4)</text>
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
    <text x="20" y="11" font-size="11" fill="#4b5563">★ Post 9 load-bearing</text>
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
- **`image_url` is a relative key.** Same convention as `pages.image_url` from Post 5: the DB stores `world-graph/images/pepper-thumb.webp`, and the FastAPI route composes the absolute URL at response time through the `Storage` protocol from [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}). Swapping local disk for Cloudflare R2 in Post 10 won't require a database migration.
- **`(episode_debut, page_debut)` is the spoiler coordinate.** This is the only field the API filter cares about. Edges have their *own* debut tuple — Cayenne exists from ep11, the `godmother_of` relation is also ep11, but a future edge between two ep1-debut entities (say, a rivalry revealed in ep15) would carry an ep15 debut and stay hidden until then. The phase-12 doc calls out that this is non-negotiable: a hand-edit could in principle put a future debut on a pre-existing entity, and an "edges-only when both endpoints are visible" check alone would leak it.

### The pydantic contract

The shape isn't suggested by the skill's docstring — it's *enforced* by [`ingestion/world_graph_loader.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-9/ingestion/world_graph_loader.py):

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

The same module is what the skill imports when validating its own output before exiting — the skill and the loader share one contract by importing the same pydantic models. That's the **"the contract is the schema"** rule from [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}#one-protocol-many-implementations), reapplied at the YAML seam.

> *Plain-English aside: why YAML and not JSON?* The artifact is meant to be *hand-edited* — fixing a summary, nudging a coordinate, correcting a debut episode. YAML wins on readability for that use case (no trailing commas to chase, no `"key":` quoting overhead, comments allowed inline so I can flag a low-confidence row with `# confidence: low — defaulted`). JSON would be marginally faster to parse and trivially friendlier to other tooling, but at ~40 rows of structured data, those don't matter. The discipline of "the artifact is meant to be read, edited, and reviewed by humans" outweighs the parser speed.

---

## The Wiki Image Scraper {#scraper}

Before the skill can run, it needs to know which entities have artwork available. The artwork lives on [framagit.org/peppercarrot/wiki](https://framagit.org/peppercarrot/wiki/-/tree/master/medias/img) — about 43 small CC BY 4.0 JPEGs named `chara_*.jpg` (the named cast) and `creature_*.jpg` (familiars and beasts). The scraper at [`ingestion/wiki_image_scraper.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-9/ingestion/wiki_image_scraper.py) does three things, in order:

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

Now the part where Claude Code does the actual extraction. The skill lives at [`.claude/skills/extract-world-graph/SKILL.md`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-9/.claude/skills/extract-world-graph/SKILL.md). Its frontmatter is the standard skill shape — a description, trigger phrases, and an allowed-tools list:

```yaml
---
description: Extract the Pepper&Carrot world graph from the workshop's local sources — the curated wiki (`data/raw/wiki/*.md`), the framagit-scraped wiki (`data/raw/wiki-upstream/*.md` when present), the per-page description JSONs under `data/raw/ep*/pages/`, and the `data/world-graph/image_manifest.json` — then write `data/world-graph/entities.yaml` and `data/world-graph/relationships.yaml` that the world-graph loader consumes. Trigger phrases include "extract the world graph", "rebuild the world graph", "regenerate world graph YAML", "extract-world-graph".
allowed-tools: [Read, Write, Edit, Bash, Glob]
---
```

The body walks Claude Code through six steps. Here's the shape; the full SKILL.md is the canonical version.

**Step 0 — read the manifest.** Open `data/world-graph/image_manifest.json`. If it doesn't exist, instruct the user to run `wiki_image_scraper.py` and stop.

**Step 1 — read source material.** The required sources are `data/raw/wiki/*.md` (the curated short bios that ship with the workshop — Pepper, Carrot, the four primary covens, etc.) and `data/raw/ep*/pages/page_*.json` (every page Claude Code itself wrote with the `ingest-from-images` skill from Post 4). The optional supplement is `data/raw/wiki-upstream/*.md` (framagit-scraped — populated by running `wiki_scraper.py`), which adds long bios of every minor character, creature, place, and school.

**Step 2 — find debut episodes.** Walk every page JSON, collect `characters_present` lists, find the **earliest** `(episode_number, page_number)` each canonical name appears in. Same for `locations_or_concepts`. Then for any entity from the wiki sources that *never* appears in any page JSON (Hippiah's school, Hereva itself, deep-bestiary creatures), default to `(1, 1)` with a `# confidence: low — defaulted` YAML comment so the human knows what to audit.

**Step 3 — build the entity list.** For each canonical character / coven / place / creature:

- **Align the slug with the image manifest where possible** so `image_url` lights up. *"The Sage"* in the wiki becomes `slug: sage` (matching `chara_sage.jpg`) with `name: The Sage`. *"Mayor of Komona"* becomes `slug: mayor`, `name: Mayor of Komona`. Slug mismatches are the most common reason an entity falls back to the SVG icon when it shouldn't.
- **Summary is 1–2 sentences of plain prose**, paraphrased from the wiki sources and/or the page JSONs. The skill's body explicitly bans markdown headers, bullet lists, and recitation — same anti-recitation discipline from [Post 8]({% post_url 2026-05-30-pepper-carrot-companion-prompt-hardening %}#anti-recitation), applied to the artifact this time.
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

The runtime side of Post 9 is a single FastAPI route — `GET /api/world-graph` — with four query parameters: `episode_slug`, `page`, an optional `right_page` (the right page of a two-page spread, so the spoiler cursor uses the rightmost visible page), and `mode` (`full` or `focus`, default `full`). The route runs at most two SQL `SELECT`s — one for entities, one for relationships — and the *shape* of the spoiler filter is the same as Post 6 and worth dwelling on.

The mental model: the reader's position is an integer pair `(episode_number, page_number)`. An entity (or edge) is **visible** when its `(episode_debut, page_debut)` is **lexicographically less than or equal to** the cursor. "Lexicographic" means we compare first by episode number, then by page number — exactly the way you'd order words in a dictionary. The reader on ep5 page 2 sees:

- every entity from ep1–4 (any page)
- entities from ep5 pages 1–2

…and nothing else.

Post 6 expressed this as a Chroma `where` clause (`$or` over `$lt episode` and `$and episode + $lt page`). That worked because Chroma's where DSL has those operators. Postgres has something cleaner: **row-value comparison** directly on tuples.

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

The `right_page` parameter is a small but load-bearing detail. The chat layer from [Post 6]({% post_url 2026-05-25-pepper-carrot-companion-spoiler-safe-rag %}) reads the page from server-side session state (`chat_sessions.current_page`), but the world-graph route is per-request — the cursor lives in the URL. A landscape reader looking at pages 5–6 has *seen* page 6, so the spoiler cursor is `(episode_number, 6)`; sending only `page=5` would gate them out of an entity that debuts on the very page they're looking at. The flipbook passes both pages on landscape spreads; the route uses `right_page` for the spoiler cursor and `[page, right_page]` for the focus-mode seed.

### Proving it: 10 tests, in-memory SQLite

The test file at [`backend/tests/test_world_graph_api.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-9/backend/tests/test_world_graph_api.py) does what Post 6's `test_retrieval.py` did with a hermetic Chroma collection: it spins up a real SQL engine (in-memory `aiosqlite`), creates the tables, seeds a hand-crafted graph, and asserts the API output is what the spoiler-filter mandates.

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

> *A small infra trade-off worth flagging:* the workshop's production runtime uses Postgres-only column types — `ARRAY(String)` for tag lists, `JSONB` for blob metadata. SQLite doesn't have those. The fix is a one-line `with_variant(JSON, "sqlite")` on the five affected columns in [`backend/app/db/models.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-9/backend/app/db/models.py): Postgres still uses `ARRAY` / `JSONB` at runtime, SQLite gets a portable JSON column at test time. The diff is mechanical, the runtime is unchanged, and tests stay hermetic — they don't need a docker-compose Postgres to run.
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

## The Frontend Overlay {#frontend}

Eight components in [`frontend/src/components/world-graph/`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/tree/post-9/frontend/src/components/world-graph). Total ~900 lines. Walked from the leaves up:

- **`constants.ts`** — Kind → border-color mapping (a shared CSS-var dictionary), kind → label, the `displayUrlFor()` helper that swaps `-thumb.webp` for `-display.webp` so the info card can fetch the bigger variant without a separate API call, and **`EDGE_KIND_COLOR`** — a per-edge-kind palette grouped semantically (structural = accent, kinship = plum, opposition = red) so the graph reads as four or five visual categories instead of nine distinct colors.
- **`fallback-icons.tsx`** — Per-kind SVG silhouettes for entities without scraped art: a witch's hat for `coven`, a tower for `place`, a paw print for `creature`, a four-point sparkle for `object`, and the first letter of the name for `character`. Sized by a `size` prop so the same icon scales from the 72-pixel graph node up to the 200-pixel info-card portrait.
- **`AvatarNode.tsx`** — A custom react-flow node. A circular avatar with a kind-colored border, either the scraped image or the fallback icon. **Eight invisible handles** — one source + one target on each of the four sides — let `WorldGraph` pick the side per-edge so a horizontal pair gets a left→right curve and a vertical pair gets a top→bottom curve. Without that, react-flow's default routing makes every edge come out the bottom and re-enter the top, which is fine for one or two edges and a snarl for fifteen.
- **`InfoCard.tsx`** — The popover that opens on click. Larger 320-px portrait, name + kind badge, summary, and the **"Ask in wiki mode"** button that calls `onAskInWiki(entity.name)`. Esc + outside-click dismiss (with `stopPropagation` on the keydown so an open card doesn't leak Esc up to the overlay and close the whole panel).
- **`KindFilterBar.tsx`** — A row of toggleable chips above the canvas, one per kind that has any visible nodes, each showing a per-kind dot in the kind's color and a count badge. Click a chip to hide that kind; the canvas re-renders without those nodes (and without any edge with a hidden endpoint). The bar refuses to turn off the *last* active kind so the reader can't accidentally empty the graph and have nothing to recover from.
- **`focus-layout.ts`** — Computes a fresh kind-grid layout for focus mode: covens in a row at `y = -ROW_GAP`, characters + creatures + objects on the main row at `y = 0` (with each familiar inserted right after its owner so the `familiar_of` edge stays a short horizontal line), places at `y = +ROW_GAP`. The full-world coordinates make a sparse mess when only a handful of nodes are in scope; the kind-grid keeps the focus subset centered and legible. Names sort alphabetically within each row for stability.
- **`WorldGraph.tsx`** — The canvas. Owns the API fetch, the kind filter, the mode toggle, the fade-in diff, the edge-coloring regime (default / focused / dimmed), and the per-edge handle selection. The pivot between focus and full mode is one line — `mode === 'focus' ? computeFocusLayout(...) : null` — and a `FitOnNodesChange` child auto-fits the viewport whenever the focused node set changes so the reader doesn't have to pan-and-zoom to find a four-node spread.
- **`WorldGraphOverlay.tsx`** — The slide-in side panel that wraps the canvas. Right-anchored on desktop (~560 px wide), full-screen sheet on mobile. Backdrop dims the flipbook without hiding it. A small hand-inked tendril SVG sits before the "World" title so the panel reads as a page from a witch's journal rather than as a CMS modal.

The component the rest of the system talks to is `WorldGraphOverlay`. It takes `episodeSlug`, `page`, an optional `rightPage` (for two-page-spread landscape), `onAskInWiki`, `onClose`, and renders. `App.tsx` decides when to mount it (`worldOpen` state) and how to handle the wiki-mode round-trip:

```tsx
// frontend/src/App.tsx (the world-graph wiring, abridged)
const [worldOpen, setWorldOpen] = useState(false);
const [outboundQuestion, setOutboundQuestion] = useState<
  { mode: Mode; text: string } | null
>(null);

// …in the header:
<button type="button" className="header-world" onClick={() => setWorldOpen(true)}>
  🌐 World
</button>

// …after the main content:
{worldOpen && (
  <WorldGraphOverlay
    episodeSlug={selectedEpisode.slug}
    page={currentPage}
    // Landscape spread → pass the right page so focus mode seeds from
    // both visible pages and the spoiler cursor uses the rightmost page.
    rightPage={showSpread ? rightPage : undefined}
    onAskInWiki={(entityName) => {
      setWorldOpen(false);
      setOutboundQuestion({ mode: 'wiki', text: `Tell me about ${entityName}` });
    }}
    onClose={() => setWorldOpen(false)}
  />
)}
```

`ChatPanel` watches `outboundQuestion` as a useEffect dep and submits whenever its object identity changes — so clicking "Ask in wiki mode" on the same entity twice still triggers a fresh turn (a new `{ mode, text }` object each time). The chat panel does the actual SSE call through the same `streamMessage` from [Post 7]({% post_url 2026-05-25-pepper-carrot-companion-streaming-chat %}); the world graph is just one more entry point into the chat pipeline that already exists.

### Focus vs full, in one beat

The mode toggle pill in the panel header flips between two views of the same spoiler-filtered universe:

- **Focus** (default): the canonical characters drawn on the page(s) the reader is currently looking at, plus a 1-hop expansion through three structural edge kinds — `member_of` (which coven the witch belongs to), `lives_in` (where the character lives), and `familiar_of` (which animal companions go with which witch). On ep11 page 4 (where Pepper meets Prince Acren) the focus set is 6 nodes: Pepper + Carrot + Cumin (on-page) + Acren (on-page) + Chaosah + Squirrel's End (1-hop neighbors). The godmothers Cayenne and Thyme are *visible* in full mode at that cursor but aren't expanded into the focus set, because no edge from the seed set walks to them through one of the structural kinds.
- **Full**: every entity whose debut is at or before the reader's cursor, laid out at the curated YAML positions (Pepper at origin, covens at compass points, places along the bottom strip). On ep11 page 4: 30 nodes, 42 edges — every entity from ep01–ep11.

Why two modes and not one? The shape of the question "what's relevant to *this* page?" is structurally different from the shape of "what's the world I'm reading about look like?" Focus mode is for inhabiting the moment; full mode is for the explorer's wide view. The reader picks per session per page — both views are one cheap SQL query, so the toggle is instant.

The two modes also exercise different bits of the data model. Focus mode joins `page_characters` (the per-page character links from [Post 5]({% post_url 2026-05-23-pepper-carrot-companion-rest-api-flipbook %})) and the world-graph's `character_id` foreign key — the same bridge that future chat enrichment would use. Full mode is the row-value comparison straight through. The spoiler boundary is identical in both: the rightmost visible page becomes the cursor, the SQL gates entities and edges, the response only carries what the reader has earned the right to see.

### Edge styling: kind-colored on focus, neutral by default

Edges are drawn in a single warm neutral by default — `rgb(110, 80, 50)` at 55% opacity — so the graph reads as a network rather than a kaleidoscope. When the reader selects a node, every edge incident to it brightens to its kind-specific color (`member_of` accent-orange, `godmother_of` plum, `rival_of` red, …), the rest dim to 18% opacity, and the kind label pops up on a parchment pill so you can see *what* the relationship is without hovering. The transition is 180 ms — fast enough to feel responsive, slow enough that the eye catches what just changed.

This is one of those small details that disproportionately changes how the graph *reads*. Without it, every edge looks the same and you have to click each one to find out what it means. With it, picking a node turns the graph into a sentence: "Pepper is a `member_of` Chaosah, `lives_in` Squirrel's End, `rival_of` Saffron, with familiar `Carrot`…". The data is the same; the affordance is what makes it readable.

> *Why react-flow, and why not write the graph by hand?* The graph is small (~20 nodes for the workshop, ~50 for the full app) and a hand-rolled SVG would render fine — pan/zoom/drag are 200 lines of code. The reason for [`@xyflow/react`](https://reactflow.dev/) is the part that *isn't* rendering: edge routing through configurable handles, focus + selection events, fitView, mini-map (optional), keyboard navigation, and a CSS API stable enough that the workshop's parchment theming layers on top cleanly. The library is ~150 KB minified — visible in the build output — which is real but reasonable for a feature like this.

---

## The Soft Fade-In: A Diff That Drives a Keyframe {#fade-in}

The polish bit. When the reader flips into ep11 for the first time, three new nodes (the godmothers) and three new edges (the `godmother_of` relations) become visible. The MVP would just have them pop into existence. The polish has them **fade in softly** so the reader's eye catches the change.

The implementation is deliberately *not* a state machine — it's a diff against the previous snapshot, mounted as a CSS class for exactly one render cycle, with the animation driven by `@keyframes`:

```tsx
// frontend/src/components/world-graph/WorldGraph.tsx (abridged)
const previousIdsRef = useRef<{ nodes: Set<string>; edges: Set<string> }>({
  nodes: new Set(),
  edges: new Set(),
});
const [newlyRevealed, setNewlyRevealed] = useState<{
  nodes: Set<string>;
  edges: Set<string>;
}>({ nodes: new Set(), edges: new Set() });

useEffect(() => {
  api.fetchWorldGraph(episodeSlug, page).then((res) => {
    // Diff: anything in the new snapshot that wasn't in the previous
    // snapshot is freshly revealed by a page flip — fade it in.
    const prev = previousIdsRef.current;
    const newNodes = new Set(
      res.nodes.filter((n) => !prev.nodes.has(n.id)).map((n) => n.id),
    );
    const newEdges = new Set(
      res.edges.filter((e) => !prev.edges.has(e.id)).map((e) => e.id),
    );
    previousIdsRef.current = {
      nodes: new Set(res.nodes.map((n) => n.id)),
      edges: new Set(res.edges.map((e) => e.id)),
    };
    // First load (prev.nodes empty): treat everything as already-known
    // so we don't animate every node on first paint. Fade-in is
    // reserved for the debut-on-flip moment.
    setNewlyRevealed(
      prev.nodes.size === 0
        ? { nodes: new Set(), edges: new Set() }
        : { nodes: newNodes, edges: newEdges },
    );
    setData(res);
  });
}, [episodeSlug, page]);
```

Each node carries the diff result on its className:

```tsx
const nodes = data.nodes.map((entity) => ({
  id: entity.id,
  type: 'avatar',
  position: { x: entity.x, y: entity.y },
  data: { entity },
  className: newlyRevealed.nodes.has(entity.id) ? 'world-node--new' : '',
}));
```

And the CSS does the rest:

```css
@keyframes world-node-debut {
  0%   { opacity: 0; transform: scale(0.6); }
  60%  { opacity: 1; transform: scale(1.08); }
  100% { opacity: 1; transform: scale(1); }
}
.world-node--new .world-node {
  animation: world-node-debut 520ms ease-out;
}
```

Three small wins from this design:

- **The first render is silent.** When the overlay opens, the diff sees an empty previous-snapshot and falls back to "treat everything as already-known" — no animation. Fade-in is reserved for the debut-on-flip moment, where it actually carries meaning.
- **No animation library.** Pure CSS keyframes. The diff is `O(n)` over a set of ~20 entities; the animation is GPU-cheap.
- **The class is sticky for exactly one render.** Because `setNewlyRevealed` is called once per fetch and the JSX recomputes the class on every render, the next state change (a node click, a selection) clears the `--new` class without re-triggering the animation. The animation plays once per debut.

The same shape applies to edges with a slightly different keyframe (`stroke-dashoffset` walks the dasharray as the edge "draws itself" in). Both fall back gracefully — if your browser doesn't support `@keyframes` (it does), the node just appears.

---

## Closing the Loop: The Third Skill and Summary-First Wiki {#third-skill}

The overlay's most quietly load-bearing UI element is the **"Ask in wiki mode" button** on each entity info card. Click an avatar; the card slides in; click the button; the overlay closes and the chat panel from [Post 7]({% post_url 2026-05-25-pepper-carrot-companion-streaming-chat %}) posts a wiki-mode question about the entity, streaming the answer back. That single click is what makes the graph feel connected to the chat instead of being a fancier `<dl>`.

<div style="margin: 1.5rem 0; text-align: center;">
  <a href="/assets/picture/2026-05-30-pepper-carrot-companion-spoiler-safe-world-graph/world-graph-demo.gif" target="_blank" rel="noopener" title="Open the full-size demo in a new tab">
    <img src="/assets/picture/2026-05-30-pepper-carrot-companion-spoiler-safe-world-graph/world-graph-demo.gif"
         alt="A screen recording of the world-graph overlay in action: the reader opens the World panel from inside an episode, the avatar nodes fade in at their curated positions, the reader clicks a character's avatar to open the info card, then clicks 'Ask in wiki mode' — the overlay closes, the chat panel surfaces a 'Consulting the grimoire…' typing indicator with bouncing dots, and a grounded wiki-mode answer streams in token by token."
         loading="lazy"
         style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--parchment-edge, #d4c8b0); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);"/>
  </a>
  <p style="margin: 0.5rem 0 0; font-size: 0.85rem; color: var(--muted, #6f6e69); font-style: italic;">
    The end-to-end flow: open the World panel, click an avatar, click Ask in wiki mode, watch the typing indicator land in the chat panel while qwen2.5:7b warms up, then read the grounded summary stream in. Click the image to view full-size.
  </p>
</div>

Here's the connection, in three lines of TSX:

```tsx
// frontend/src/components/world-graph/InfoCard.tsx
<button onClick={() => onAskInWiki(entity.name)} className="info-card__ask">
  Ask in wiki mode →
</button>
```

```tsx
// frontend/src/App.tsx — passed down to WorldGraphOverlay
onAskInWiki={(entityName) => {
  setWorldOpen(false);                            // close the overlay
  setOutboundQuestion({                           // new object identity fires the send
    mode: 'wiki',
    text: `Tell me about ${entityName}`,
  });
}}
```

The wire is short. What it lands on, though, is the entire wiki-mode RAG pipeline from Post 7. Which is where this post's *third* skill comes in.

### The problem: small models drown in big wiki context

Post 7 shipped wiki mode against a hand-written `wiki_seed.yaml` — five articles, total under 4 KB. That worked for the five entities those articles covered. Post 9's graph has **45 entities**. The reader can now click any of them — Truffel, Camomile, Quassia, the Mayor, the Mayor's three-line bio — and trigger a wiki-mode question about an entity the seed never covered.

The straightforward fix is to pull more wiki content. The [framagit Pepper&Carrot wiki](https://framagit.org/peppercarrot/wiki) has a handful of long-form `.md` files — `characters.md` (32 KB, one `### Name` section per character), `creatures.md`, `places.md`, `magic-system.md`, `history.md`, `time-system.md`, `timeline.md`. The new `ingestion/wiki_scraper.py` pulls them into `data/raw/wiki-upstream/` in a single CLI run, and `ingest_wiki.py` already accepts both directories.

The first version of this pipeline embedded those articles **whole** — one Chroma chunk per file, top-5 retrieval. "Tell me about Truffel" landed `characters.md` in the retrieved set (Truffel has two paragraphs in there, near the end), but at top-5 it landed `characters.md` alongside `creatures.md` and `places.md`. qwen2.5:7b sees ~80 KB of mostly-unrelated bios and fabricates a backstory about Truffel that isn't in any of them.

I tried paragraph-chunking next: split each `.md` on blank lines, embed each paragraph as its own Chroma row, lower the retrieval `k` to 3. That helped (`characters.md` became 130 small chunks; "Tell me about Truffel" pulled the right paragraph), but the model still leaked. The retrieved paragraphs are short, but the chat orchestrator fetches the *whole article* by `source_id` to feed the prompt, since that's the contract the runtime resolver inherited from Post 7. Top-3 paragraphs from `characters.md` = the whole 32 KB file in the prompt. qwen2.5:7b doesn't ignore the noise; it lets the unrelated bios bleed into Truffel's answer.

You can stare at this for a while and reach for retrieval engineering — re-chunking, dynamic rerankers, a second-pass summarization at query time. All of which add complexity at the runtime layer, which is exactly where the project has been keeping things plain. **The real bug is asking the small model to do the filtering work upstream.** The Post 8 prompt hardening — `OUTPUT RULES`, `GROUNDING CONTRACT`, the markdown-stripping safety net — only holds for context windows the model can actually attend to. A 32 KB wiki article isn't one of those.

### The fix: pre-filter the documents, not the retrieval

The runtime can only land what the index gives it. So change what the index has. Instead of embedding `characters.md` whole (or in 130 paragraph-shaped pieces), **embed one ~150-word summary per entity**, written ahead of time. Top-3 retrieval lands three focused summaries totaling ~500 words. qwen2.5:7b sees a context window it can actually keep at the top of its attention. Post 8's `OUTPUT RULES` apply cleanly.

Authoring 45 entity summaries + 5 topic summaries by hand is a lot of typing. So is keeping them in sync with the framagit wiki when David Revoy publishes another bio. So the workshop introduces a **third Claude Code skill** that does the typing:

```yaml
# .claude/skills/summarize-wiki/SKILL.md (frontmatter abridged)
description: Author per-entity and per-topic wiki summaries from the
  upstream wiki sources. Reads data/raw/wiki/ (curated), data/raw/wiki-
  upstream/ (framagit), and data/world-graph/entities.yaml (the canonical
  entity list). Writes one tight focused summary per entity to
  data/wiki-summaries/entities/<slug>.md and a handful of non-entity
  topic summaries to data/wiki-summaries/topics/<slug>.md. These
  summaries — not the raw wiki articles — become the documents the wiki
  ingestion pipeline embeds.
```

The skill body walks Claude Code through six steps. Read the entity list. Read the source `.md` files. For each entity, find its source paragraphs in the appropriate file. Pick a length tier — ~300 words for major entities (Pepper, the godmothers, Saffron, the primary covens, Komona, Squirrel's End, Hereva), ~100 words for everyone else. Write the summary in flowing prose — **no markdown headers, no bullets, no panel-by-panel structure** (same anti-recitation discipline from the `ingest-from-images` and `extract-world-graph` skills; same reason: any markdown in the embedded document leaks into the chat answer). Frontmatter goes at the top with `slug`, `title`, `category`, and `source_url`. Validate by running the loader at the end.

One run of the skill produces a directory like this:

```
data/wiki-summaries/
├── entities/
│   ├── pepper.md            (~300 words, major)
│   ├── carrot.md            (~150 words)
│   ├── thyme.md             (~280 words, major)
│   ├── cayenne.md           (~280 words, major)
│   ├── cumin.md             (~280 words, major)
│   ├── saffron.md           (~290 words, major)
│   ├── shichimi.md          (~280 words, major)
│   ├── truffel.md           (~90 words, minor)
│   ├── mango.md             (~80 words, minor)
│   ├── apiaceae.md          (~100 words, minor)
│   └── …40 more
└── topics/
    ├── magic-system-overview.md
    ├── history-of-hereva.md
    ├── great-war.md
    ├── time-system.md
    └── chaosah-tradition.md
```

Each file looks like:

```markdown
---
slug: truffel
title: Truffel
category: character
source_url: https://www.peppercarrot.com/en/wiki/Characters.html
---

Truffel is Saffron's familiar, a female white angora cat. She lives
with Saffron in the heart of Komona and travels with her wherever
Saffron goes. Her hobbies are sleeping and grooming. Truffel plays
with Pepper's cat Carrot whenever the two witches visit, which is one
of the small constants that keeps the rivalry between Pepper and
Saffron from drifting into outright hostility. Like Carrot she does
not speak and has no known magic of her own; she is a familiar in the
everyday sense, a constant companion rather than a magical instrument.
```

### One chunk per summary

The wiki ingestion code shrinks rather than grows:

```python
# ingestion/chroma_writer.py — paragraph chunking is gone
async def upsert_wiki_articles(self, articles: list[WikiArticle]) -> None:
    """Embed each wiki summary as a single document in `wiki_v1`."""
    if not articles:
        return
    ids = [str(a.id) for a in articles]
    texts = [format_wiki_for_embedding(a.title, a.content) for a in articles]
    metadatas = [{"source_table": "wiki", "source_id": str(a.id)} for a in articles]
    embeddings = await self._embedding.embed_batch(texts)

    collection = self.get_or_create_collection(WIKI_COLLECTION)
    for a in articles:
        collection.delete(where={"source_id": str(a.id)})
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
```

One chunk per summary keeps the embedding signal concentrated. The Chroma `wiki_v1` collection now holds **50 rows** — one per `.md` file — instead of the 469 paragraph-chunks the previous design produced from 16 sprawling articles.

### Before / after, on the same question

The change shows up plainly in qwen2.5:7b's wiki answers. Same prompt, same chat orchestrator, same Post 8 prompt hardening — only the wiki index differs.

| Question | Before (raw articles, 469 paragraph chunks) | After (50 per-entity summaries, k=3) |
|---|---|---|
| *Tell me about Truffel* | Three meandering paragraphs that confuse Truffel with the broader catalog of Hereva creatures. | "Truffel is Saffron's familiar, a female white angora cat. She lives with Saffron in the heart of Komona and travels with her wherever Saffron goes…" — grounded in the 90-word summary. |
| *Tell me about Cayenne* | "Cayenne is a type of red pepper used in cooking…" — the small model leaked culinary domain knowledge through a tenuous embedding match. | "Cayenne is the tall, thin, and rigid Chaosah witch who serves as Pepper's spell-casting tutor and one of her three godmothers…" — grounded in the 280-word summary. |
| *What is Magmah?* | Vague summary that confuses Magmah with magma (the substance). | "Magmah is the school of cooking, baking, grilling, boiling, frying, steaming, and toasting, and also the house of alchemy and rare metals…" — grounded in the major-tier summary. |

The reading experience is what changed. Click Truffel's avatar on the world-graph overlay, click **Ask in wiki mode**, and the answer that streams back is small, accurate, and warm in the project's voice. Click the Cayenne avatar and the godmother's bio streams in instead of an essay on cuisine. Click Magmah and the model knows it's a school, not a substance.

### Three skills, one pattern

With `summarize-wiki` in place, the workshop now has three Claude Code skills, each authoring a different durable artifact from the project's source material:

| Skill | Reads | Writes | Why |
|---|---|---|---|
| `ingest-from-images` (Post 4) | page images | `data/raw/ep*/pages/page_*.json` | The vision step is too tactile for a runtime call; better as durable JSON. |
| `extract-world-graph` (Post 9) | wiki sources + page JSONs + image manifest | `data/world-graph/entities.yaml` + `relationships.yaml` | The entity list shouldn't be re-derived per request; better as a hand-editable YAML. |
| `summarize-wiki` (Post 9) | wiki sources + entity list | `data/wiki-summaries/{entities,topics}/*.md` | The retrieval index shouldn't have to do disambiguation the small model can't follow; better as pre-filtered documents. |

All three are **one-shot authors of durable artifacts**. Same architectural rationale every time: move the model touch to authoring time, where the artifact can be `git diff`-ed, hand-edited, and version-controlled, so the runtime never depends on a model call to be correct. The runtime is plain Postgres, plain Chroma, plain FastAPI, plain React.

The pattern is composable in a useful way too. The world graph defines what entities the reader can ask about; the summarize-wiki output is what the chat sees when they ask. Same source material (the wiki) flows into both, in two different shapes — one a graph of names, one a set of focused paragraphs. If David Revoy publishes a new bio tomorrow, you run the wiki scraper and re-trigger both skills. The shipped runtime never noticed.

---

## What's Honest, What's Open {#honest}

Three things to name plainly, because portfolio posts that don't are the ones that age badly.

**The workshop graph at the `post-9` tag carries 45 entities and 57 relationships** — the curated wiki plus the framagit scrape, cross-referenced against the page JSONs from ep01–ep12. The number isn't sacred: a fresh clone with only ep01 ingested and only the curated wiki present will produce a much smaller graph (~10 entities) when you re-run the skill, and that's the point. **Coverage grows when the source material grows.** If you scrape additional framagit pages, ingest more episodes, or hand-write a new bio into `data/raw/wiki/`, re-running the `extract-world-graph` skill will surface the additions in the YAML — with hand-tweaked layouts preserved (STEP 2a of the SKILL.md is non-negotiable about this).

**The skill is "Layout idempotent, summary not idempotent."** STEP 2a of the SKILL.md is explicit: the skill must reuse hand-tweaked layout coordinates across re-runs. But summaries get re-authored every time the skill runs. If you've polished a summary line and want to keep it, commit the YAML and `git diff` after a re-run. The reason for the asymmetry is that summaries are the part where the source material's evolution should propagate (a fresh page-description might reveal a new fact), and layouts are the part where the human's curation should dominate. Both are defensible; both are documented; neither is the "right" answer if your priorities differ.

**Several entities are wiki-seeded with low confidence.** Hereva is the *world* and is implicit everywhere but named explicitly in no `locations_or_concepts` list. The same is true for several covens (Aquah, Zombiah, Ah) and many of the bestiary creatures — they're in the wiki sources but no `ingest-from-images` page JSON has yet pinned their first appearance. The skill defaults their debut to a sensible fallback (`(1, 1)` for foundational entities like Hereva and the four primary covens, or the wiki-pinned episode for those it knows about) and writes `# confidence: low — defaulted` next to each one. If you find a default aesthetically wrong, edit the YAML directly — the layout STEP 2a discipline means hand-edits survive re-runs.

**Wiki coverage scales with two parallel knobs.** The world graph (entity list, debut tuples, layout) and the wiki summaries (the per-entity `.md` documents the chat retrieves) are independent artifacts produced by two different skills from the same source material. Add a new bio to `data/raw/wiki/`, re-run `extract-world-graph`, and the new entity surfaces on the overlay. Re-run `summarize-wiki` and the new entity is also askable in wiki mode. Forget the second step and the avatar shows up but "Ask in wiki mode" returns a vague answer for the new node — the symptom is informative, and the fix is one CLI command.

**The world-graph data model bridges, but doesn't currently power, the chat.** The `world_entities.character_id` foreign key links a world entity to the canonical character row, which means future chat features could use the graph to enrich answers ("who's on this page?" → look up the page's characters → look up each character's `member_of` edges → mention what coven they belong to). The workshop doesn't do this yet — the chat pipeline from Posts 6–8 retrieves from `pages_v1` and `wiki_v1` only. The current bridge from graph → chat is one-way ("Ask in wiki mode" sends a question; the chat doesn't yet enrich answers with graph context). Wiring the graph into the chat is a natural follow-up; the workshop ships without it to keep the post's scope on the overlay.

---

## Key Takeaways {#key-takeaways}

**1. Skills are one-shot authors of durable artifacts.** Same pattern as Post 4's `ingest-from-images`: Claude Code reads source material, synthesizes structured data, validates against the loader's contract, writes to disk. The runtime never calls a model to figure out who's in the world or what to say about them; it queries Postgres and Chroma. The artifact is hand-editable, version-controlled, and stable across model versions. Skills are for first passes and source-material refreshes — once the artifact is good, subsequent fixes go directly into the file.

**2. Spoiler safety is a row-value comparison.** Post 6 expressed it as a Chroma `where` clause; Post 9 expresses it as `tuple_(episode_debut, page_debut) <= cursor` in SQL. Same lexicographic shape, different store. Postgres compares row values lexicographically by default; SQLAlchemy renders the expression cleanly. One operator, one expression, no `OR`-with-`AND`-shaped subtlety to typo.

**3. Edges need their own debut filter.** "Both endpoints visible" is not enough. An edge can carry plot meaning that debuts later than both of its endpoints — a rivalry revealed in ep15 between two ep1-debut witches must stay hidden until ep15. Filter the edge's *own* debut AND require both endpoints to satisfy the same predicate. Three where clauses; all three mandatory; one test pins it explicitly because it's the most plausible bug.

**4. The pydantic loader is the contract.** The skill imports the same `EntityData` / `RelationshipData` models the ingestion CLI uses. A misspelled slug fails fast at load time with a clear error, not at runtime as a missing-row 500. The contract isn't a markdown spec; it's a Python file that both authors and consumers import. Same shape applies to the wiki summaries — `WikiArticleData` is the pydantic contract `summarize-wiki` writes to and `ingest_wiki.py` reads.

**5. Image scraping captures provenance, not just pixels.** `image_manifest.json` records the framagit ref, the short SHA, the timestamp, and the lists of character/creature slugs that were on disk at that ref. The manifest is the seam between the scraper and the skill, and the audit trail for "where did these portraits come from?" It's committed (small); the bytes aren't (gitignored). Anyone running the project re-runs the scraper to populate them.

**6. The frontend is small because the back end is structural.** Eight components, ~900 lines of TSX, react-flow handling the canvas mechanics. The kind-grid focus-layout is 100 lines; the rest is mapping API rows to nodes and edges, picking handle sides per edge for cleaner bezier routing, and the fade-in diff. No node-positioning algorithm beyond focus mode's kind-grid; no edge-routing math beyond "horizontal pair → right→left, vertical pair → top→bottom"; no graph-state library. The only state the component owns is `selectedNodeId`, the active-kind set, the mode, and the fade-in diff — and the viewport auto-fits on every node-set change so neither focus nor full mode strands entities off-screen at a fixed default zoom.

**7. Two views, one boundary.** Focus mode ("This page") and full mode ("Whole world") layer on top of the same spoiler-filtered SELECT — the focus seed joins `page_characters` to find who's drawn on the visible spread, then expands one hop via the structural edge kinds (`member_of`, `lives_in`, `familiar_of`). Same boundary; same row-value comparison; different scoping.

**8. Animate when it carries information.** First render is silent; new debuts after a page flip fade in. The animation isn't decoration — it draws the reader's eye to *the change*, which is exactly the affordance a spoiler-revealing graph wants. CSS `@keyframes` driven by a one-render class is enough; no animation library required. The diff is `O(n)` over a set; the animation is GPU-cheap. Edge selection is the same idea: neutral by default, kind-colored on focus, with a 180 ms transition so the reader's eye catches *what just lit up*.

**9. Skill-as-author scales, runtime-extraction doesn't.** A runtime model call per overlay open would be: a service to scale, a cost per visit, a model whose drift can quietly break the graph. The skill pattern moves all of that to authoring time. You can `git diff` the YAML, hand-edit it, and re-run a 1-second ingest. The model touches the artifact at authoring time; the artifact touches the runtime.

**10. When the model is too small, shrink the documents, not the prompt.** Post 8's `OUTPUT RULES` work against a 600-word context window. They don't work against 30 KB of multi-entity wiki articles, no matter how the prompt is reworded. The `summarize-wiki` skill responds by pre-filtering the documents themselves: one tight ~150-word summary per entity, one Chroma chunk per summary, top-3 retrieval landing ~500 words. The cheap fixes — bigger model, more retrieval engineering, better prompts — all assume the runtime should keep doing the disambiguation work. The actual fix is to give the small model a small problem.

---

Next up: **Post 10 — Shipping It: Cloudflare Pages + Fly + Modal + R2 + Neon for ~\$10/mo.** The flipbook, the chat, and the world graph all run beautifully on your laptop. Post 10 takes the same architecture and puts it on the internet — Cloudflare Pages for the static frontend, Fly for the FastAPI backend, Modal for the GPU-served Ollama, Neon for managed Postgres, Cloudflare R2 for the image bytes. The seam each provider abstraction from [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) was designed for finally pays off.

The **workshop starter** that backs this post is at <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop>, tagged `post-9` — `git checkout post-9` to get exactly the code shown here. The **full source repository** and the public live-demo URL go up alongside Post 10.

*Pepper & Carrot* is © [David Revoy](https://www.davidrevoy.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All credit to him for the source material that made this project possible.

**All opinions expressed are my own.**

---
