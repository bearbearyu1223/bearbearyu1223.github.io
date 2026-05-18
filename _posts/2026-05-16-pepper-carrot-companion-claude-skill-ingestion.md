---
title: "Claude Skills as an Ingestion Tool: When the Best Vision Model Is the One Driving Your Editor"
date: 2026-05-16 00:00:00 -0800
categories: [Full-Stack, RAG, Local AI]
tags: [claude-skills, claude-code, vision, peppercarrot, ingestion, postgres, chromadb, python, portfolio]
description: >-
  Post 4 of the Pepper & Carrot AI flipbook series. The comic is images,
  not text — so before any RAG can happen, every page needs a description.
  This post walks through using a Claude Code skill as the vision provider
  for the ingestion pipeline of this portfolio-project specifically: no
  per-call API cost beyond the Claude Code subscription, auditable JSON
  artifacts on disk, same Claude model as Anthropic's hosted vision API. By the end, one full episode is ingested
  into Postgres + ChromaDB + local storage. The right vision provider is
  context-specific — local VLM, hosted API, and Claude Code each win under
  different constraints (budget, whether the pipeline runs unattended,
  throughput) — and the post includes a decision matrix mapping each
  constraint to the right choice.
pin: true
---

Post 4 of the [*Pepper & Carrot AI-powered flipbook*]({% post_url 2026-05-09-pepper-carrot-companion-trailer %}) series. With the workshop standing from [Post 2]({% post_url 2026-05-10-pepper-carrot-companion-workshop %}) and the four `Protocol`-typed seams from [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}), it's time to put real data behind them. The comic is images — not text — so before RAG can answer questions about a page, every page needs a description. This post is about who writes those descriptions and how they land in the database.

> **What you'll build in this post.**
> - One full episode of *Pepper & Carrot* ingested end-to-end: image variants written to `LocalStorage`, descriptions in `pages`, character links in `page_characters`, embedded chunks in ChromaDB's `pages_v1` collection, and a 2–3 sentence episode plot summary on `episodes.plot_summary`.
> - The `ingest-from-images` [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills) installed at `.claude/skills/ingest-from-images/`, working as the vision provider for the offline pipeline.
> - A working mental model for what a Claude Code skill is, why this beats running a vision model in production, and the honest trade-offs that come with it.
>
> **Prerequisites.**
> - The workshop starter from [Posts 2–3](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop), with `docker compose up -d`, `alembic upgrade head`, and `seed.py` already run.
> - Episode 1 already downloaded into `data/raw/ep01-potion-of-flight/` via `acquire.py` (Post 2 step 7).
> - A [Claude Code](https://docs.claude.com/en/docs/claude-code) install on a subscription that includes Claude usage (Pro / Max / Team work).
> - Ollama running locally with `qwen2.5:7b` and `bge-m3` pulled (Post 2 step 2) — used by Stage 2 for the embeddings and the plot-summary call.

> **About the repo URL.** The code in this post — the `ingest-from-images` skill at `.claude/skills/ingest-from-images/`, the Stage 2 pipeline at `ingestion/ingest.py`, and the `JsonFileVisionClient` at `backend/app/clients/vision.py` — lives in the same workshop starter that backed [Posts 2]({% post_url 2026-05-10-pepper-carrot-companion-workshop %}) and [3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}): <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop>. Pull the latest to pick up Post 4's additions. The full project repository — frontend, chat orchestrator, world-graph overlay, cloud deploy — lands alongside the deploy guide in Post 10.

---

## Table of Contents

1. [Ingestion Is a Vision Problem](#vision-problem)
2. [Three Vision-Provider Options](#three-options)
3. [What's a Claude Code Skill?](#what-is-a-skill)
4. [The `ingest-from-images` Skill, Walked Through](#the-skill)
5. [The Python Pipeline (Stage 2)](#stage-two)
6. [Two-Stage Flow in One Picture](#two-stage-flow)
7. [Why This Beats a Runtime Vision Model](#why-this-wins)
8. [The Honest Trade-offs](#trade-offs)
9. [Running It on Episode 1](#running-it)
10. [Key Takeaways](#key-takeaways)
11. [Appendix: How a VLM Actually Sees an Image](#appendix-vlm)

---

## Ingestion Is a Vision Problem {#vision-problem}

Most "chat with X" tutorials skip a layer that's the load-bearing part of *this* project. They start by parsing a PDF, scraping HTML, or pulling text out of a document store, and then go straight into chunking, embedding, retrieval. Their X is already text. The interesting work is downstream of *"some text is sitting on disk."*

A webcomic isn't. The pages of *Pepper & Carrot* are hand-painted watercolor images. There is no native text to chunk. There is no caption layer. There is no markup. There is a `page_001.jpg` on disk and there is a reader who wants to ask the AI *"why is Carrot angry?"*

Today's multimodal LLMs can, in fact, look at that page and answer the question directly — hand the JPEG to Claude or GPT-4o every turn and you could skip ingestion entirely. But the chat layer almost never has the luxury of reasoning about *just* the current page. *"Why is Carrot angry?"* usually needs the prior pages of the episode for narrative context (*"Pepper took the last carrot on page 4"*). *"What's a Komona witch?"* needs a wiki article retrieved from across the whole corpus. So every turn drags in multi-page context, and pulling that context in *fresh, as images* is the part that doesn't scale: expensive on hosted vision APIs, and slow on the local Ollama path even with multimodal models like `llava` or `qwen2-vl`. For the wiki side, text embeddings (bge-m3) are also cheap and semantically sharper on narrative — *"the moment Pepper realizes the potion was a trick"* — than multimodal embeddings are yet, which makes corpus-wide retrieval feasible at all. And a prose description written once is inspectable, diffable, and seedable for tests in a way *"whatever the VLM saw this turn"* never is.

So before any of the RAG plumbing built in later posts can fire, every page gets collapsed to prose once, up front — and the rest of the stack stays text-native. That description is the document the chat layer will embed, search, and ground answers in.

This step is called **ingestion**, and the central question of this post is: **who writes the description?** The choice of who shapes everything downstream — the quality of chat answers, the cost of iterating on prompts, the operational reliability of the pipeline, even how easy it is to fix a description after the fact.

The whole post is about answering that question in a slightly unusual way: have **Claude Code itself** do the looking. Same model as Anthropic's hosted vision API, no metered per-call cost — the ingestion work is subsumed under the Claude Code subscription, JSON artifacts on disk you can hand-edit. The architecture from [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) makes this possible without any special-casing — the rest of the pipeline still talks to a `VisionClient` Protocol, same shape as if a real model client sat behind it.

---

## Three Vision-Provider Options {#three-options}

Before settling on the Claude-Code-as-vision-provider path, the project actually built and evaluated the two more conventional options. The choice between them is **context-specific** — there's no universal "best" vision provider; the right one for any given project depends on a small set of constraints (per-call budget, whether the pipeline needs to run unattended on a schedule with no human present, iteration speed, throughput requirements, privacy posture). Pepper & Carrot is a portfolio / demo project, so its constraints are very particular: no per-call API budget, no unattended-automation requirement, and free-at-the-margin prompt iteration as the dominant priority. **Under those constraints**, the third option wins. A different project — say, a paid product ingesting hundreds of comics nightly under a hosted-vision-API contract — would land somewhere else.

The point of walking through all three isn't to anoint Option 3 as universally best. It's to give you the decision frame so you can pick the right one for **your** project.

### Option 1 — Local VLM (Ollama, `qwen2.5vl:7b`)

> *Plain-English aside: what's a VLM?* A **vision-language model** (VLM) is a language model with a vision encoder bolted on the front: you give it an image, it produces text. [Qwen2.5-VL](https://qwenlm.github.io/blog/qwen2.5-vl/) is Alibaba's open-weights vision model in the same family as the `qwen2.5:7b` text model we already pulled in [Post 2]({% post_url 2026-05-10-pepper-carrot-companion-workshop %}). Running it locally via [Ollama](https://ollama.com/library/qwen2.5vl) means images go from your filesystem to Ollama's HTTP API, get tokenized + encoded + decoded into output text, and come back as a description — no network round-trip to a cloud API. (For the actual mechanics — what *tokenized*, *encoded*, and *decoded* really mean inside the model, and why a decoder-only transformer is sufficient even though "understanding an image" sounds like it needs an encoder — see the [Appendix](#appendix-vlm).)

This is the "obvious" local-first choice. It fits cleanly behind the `VisionClient` Protocol; you build an `OllamaVisionClient` that base64-encodes the image, POSTs to `/api/chat` with the image attached, parses the response. The project actually did this. It worked, sort of, for six episodes. Three problems killed it:

**Markdown leakage.** Prompted for *"3–5 sentences of flowing prose, no markdown,"* `qwen2.5vl:7b` produces this:

```
**Panel 1:** **Setting:** A potion shop in Komona.
**Action:** Pepper is reading a book.
**Mood:** Quiet, studious.

**Panel 2:** **Setting:** Same shop.
**Action:** Carrot knocks over a vial. ...
```

Not every page comes out this way — some pages come back as clean prose — but enough do that the output is unreliable. The training-data distribution for "describe this image" pulls the model toward the structured panel-by-panel format, and prompt instructions only partially override it, with no way to predict from the input which way a given page will go. Worse: when this text gets retrieved by the chat layer (built in a later post) and fed to *another* small local model for synthesis, that model mirrors the same `**bold-label:**` shape in its replies — so the user sees wiki-style answers instead of conversational prose. The leak isn't cosmetic; it propagates.

**GGML crashes mid-run.** Specific page images would trigger a hard assertion failure inside [llama.cpp](https://github.com/ggml-org/llama.cpp)'s vision pipeline:

```
GGML_ASSERT(a->ne[2] * 4 == b->ne[0]) failed
```

(Tracked in [`ollama/ollama#15828`](https://github.com/ollama/ollama/issues/15828). The same assertion hits other vision models too — e.g. [`glm-ocr`](https://github.com/ollama/ollama/issues/14171) — so it's a llama.cpp vision-pipeline issue surfaced via Ollama, not a qwen-specific bug.) The entire run dies, mid-episode. Rerunning might succeed, might fail differently, might fail at a different page. **There is no application-layer fix.** Either the underlying tensor shapes align or they don't, and the error message is opaque from a Python caller's perspective.

**Latency.** A nine-page episode through `qwen2.5vl:7b` on Apple Silicon took **~10 minutes** per run, sometimes longer under memory pressure or on cold start — roughly a minute per page after the model warmed. Iterating on the description prompt meant 10+ minutes between each change, which adds up across a session of prompt-tuning.

### Option 2 — Hosted vision API (Anthropic)

Same wire shape, different endpoint. `AnthropicVisionClient` base64-encodes images, calls the [Anthropic Messages API](https://docs.claude.com/en/api/messages) with image content blocks, parses the response. The model behind it (Claude Opus) is, as of writing, the best vision model on the market for this kind of task.

> *Plain-English aside: what's "base64-encoding" an image?* The Anthropic Messages API speaks JSON, and JSON strings can't safely carry the raw bytes of a JPEG — null bytes and control characters would either break the parser or get mangled in transit. **Base64** is a way of representing arbitrary binary data using only 64 printable ASCII characters (`A–Z`, `a–z`, `0–9`, `+`, `/`); every 3 bytes become 4 base64 characters, about 33% bigger than the original. So `AnthropicVisionClient` reads the image as bytes, runs `base64.b64encode(...)` on them, and drops the resulting ASCII string (something like `/9j/4AAQSkZJRg...`) into the JSON body inside an `{"type": "image", "source": {"type": "base64", ...}}` content block. Anthropic's server decodes it back to bytes on its end and feeds those into the vision encoder. A 200 KB comic page becomes about a 270 KB string in the request body.

This fixed everything from Option 1. No GGML bugs. Clean prose output. ~5 seconds per page. The descriptions were genuinely good.

The one downside: **roughly $30–$45 per full re-ingest** of a ~40-episode corpus at Opus pricing — about $0.08–$0.12 per page, given comic-page-sized images, a system prompt with the cast list, and a few hundred output tokens per description. That sounds cheap, and per run it is — until you realise you're going to want to re-ingest dozens of times while iterating on the description prompt. After ten iterations you've spent a few hundred dollars testing a prompt that mostly didn't change. (Dropping to Sonnet roughly fifths the bill with some quality trade-off; Haiku is cheaper still.) Not catastrophic, but a real friction tax on the loop.

### Option 3 — Claude Code itself as the vision provider

Then a third option emerged. Claude Code's `Read` tool can read image files directly, and the model behind it is **the same Claude Opus model that powers the Anthropic vision API**. The marginal cost is zero — it's folded into the Claude Code subscription you're already paying for. The quality is identical.

The shape of the solution: instead of writing an `OllamaVisionClient` or `AnthropicVisionClient` that the ingestion script calls at runtime, you write a **[Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills)** — a project-local file that auto-loads into a Claude Code session — that instructs Claude to:

1. Read each page image with the `Read` tool.
2. Compose a `PageDescription` matching a strict schema.
3. Write the JSON to disk next to the image with the `Write` tool.
4. Run the Python pipeline that consumes those JSONs.

The Python pipeline then talks to a `JsonFileVisionClient` — the fourth and last implementation of the four `Protocol`-typed seams introduced in [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}). (Post 3 landed `Storage`, `EmbeddingClient`, and `ChatClient`; `VisionClient` was named but left for this post.) The class does nothing model-y itself — it just reads the sibling JSON file the skill wrote and validates it against the `PageDescription` schema. **The rest of the architecture is unchanged.** The pipeline still goes through the `VisionClient` Protocol; the model is just outside the Python process this time, doing its work *before* the script runs. The actual implementation is short enough to read end-to-end — we walk through it below in [§ The Python Pipeline (Stage 2)](#stage-two).

Three options, side-by-side:

| Provider | Quality | Cost per re-ingest | Failure modes | Time for 9 pages |
|---|---|---|---|---|
| Local VLM (`qwen2.5vl:7b`) | Markdown leakage; structure propagates downstream | $0 | GGML assertion crashes | ~10 min |
| Hosted API (Anthropic vision) | Excellent | ~$30–$45 (Opus, full corpus) | Network / rate limits | ~45 sec |
| **Claude Code as vision provider** | **Excellent** (same model) | **$0** *(subscription)* | None observed | ~5–15 min interactive |

For **this project's** constraints — portfolio scope, no unattended-automation requirement, free iteration as the dominant priority — the Claude Code path is the right call. No metered per-call cost (subsumed under the Claude Code subscription you're already paying for), high-quality descriptions, and JSONs on disk that you can hand-edit. But it's not a universal winner. The moment your project needs ingestion that runs automatically on a schedule (a nightly cron, a webhook fire, anything with no human in the loop), or you're ingesting at a volume where one Claude Code session per episode doesn't fit your workflow, **the hosted-API option is the right call** — same quality, you just pay for it. And if budget is tight *and* privacy posture requires keeping pixels on your own infrastructure, modern open-weights VLMs (e.g. [Qwen3-VL](https://qwenlm.github.io/blog/qwen3-vl/)) have closed enough of the markdown-leakage gap that **the local-VLM option becomes viable again** — you trade latency and GPU cost for the privacy.

The architecture supports any of the three. Flipping `VISION_PROVIDER` in `.env` (plus rebuilding the corresponding client) is the only code-level difference. The decision is about constraints, not about code.

The rest of this post walks through how the skill is built and what the pipeline does on the back of it — but read it as a *case study* of one solution that fit one project's constraints, not as a prescription.

---

## What's a Claude Code Skill? {#what-is-a-skill}

> *Plain-English aside.* A **Claude Code skill** is a small piece of text — a Markdown file with a YAML header — that **auto-loads itself into a Claude Code session when the user's request matches a trigger description.** Think of it as a project-pinned prompt that activates on demand. The skill's body is read by Claude, in context, the moment any of its trigger phrases appear in your message; from then on, Claude follows the skill's instructions to handle your request.

Concretely:

- A skill is a directory at `.claude/skills/<name>/` containing at minimum a `SKILL.md` file.
- The `SKILL.md` starts with [YAML frontmatter](https://docs.claude.com/en/docs/claude-code/skills) declaring two things: a `description` (whose text Claude scans against incoming user messages to decide whether to load this skill) and an `allowed-tools` list (which Claude tools the skill is permitted to use).
- The body of `SKILL.md` is the skill's prompt — written for Claude, in instructional voice. Step-by-step instructions, schemas, examples, gotchas.
- Skills can ship **additional files** alongside `SKILL.md`: shell scripts, examples, JSON Schemas — anything the skill's body references.

Two flavors:

- **Project-local skills** live in `.claude/skills/` inside a repository. They travel with the codebase. Clone the repo, open Claude Code in it, and the skills are available. This is what we're using here.
- **Personal skills** live in `~/.claude/skills/` on a single machine. They're available across all your projects but don't travel.

> *Plain-English aside: how does Claude actually pick a skill?* When you type a message in Claude Code, the system reads the `description` field of every available skill. If your message looks like it might match — *"ingest episode 7 from images"* clearly matches the description we'll write in a moment — Claude loads that skill's full body into the conversation and starts following its instructions. You can also invoke a skill explicitly with [`/<skill-name>`](https://docs.claude.com/en/docs/claude-code/skills#invoking-a-skill) if you want to skip the matching step. There's no separate "install" or registration; dropping a `SKILL.md` into `.claude/skills/<name>/` is enough.

Why this is useful for ingestion specifically: the skill is the **piece of context Claude needs to do the task correctly.** Without it, asking Claude *"describe the pages of episode 1"* would get you Claude's best guess at what you want — probably a single paragraph per page, no structured fields, no canonical character names, no validation. With the skill loaded, Claude knows exactly what JSON shape to produce, which fields are required, which character names to use, where to write the files, and what script to run when it's done. **The skill replaces the prompt-engineering work you'd otherwise re-do every time.**

---

## The `ingest-from-images` Skill, Walked Through {#the-skill}

The skill lives at [`.claude/skills/ingest-from-images/SKILL.md`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/.claude/skills/ingest-from-images/SKILL.md) in the workshop starter. Let's go through it section by section.

### The frontmatter

```yaml
---
description: >-
  Ingest a Pepper&Carrot episode by describing each page image yourself
  (via the Read tool) and writing PageDescription JSON files that the
  JsonFileVisionClient picks up. This is the standard ingestion path for
  the project — there is no other vision provider. Trigger phrases
  include "ingest episode N", "ingest episode N from images", "describe
  pages for episode N", "re-describe pages for episode N", "describe the
  missing pages of episode N", "ingest-from-images", "set up episode N
  for the chat".
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
---
```

Two things to notice:

- The **`description` is long on purpose.** It's not just documentation for humans — it's the prompt-matching text Claude uses to decide whether to load this skill. The more trigger phrases you list, the more reliably the skill activates on natural-language requests. *"ingest episode 1"*, *"re-describe pages for ep07"*, *"set up episode 14 for the chat"* — all should match. Skill descriptions that are too short tend to under-activate.
- **`allowed-tools` is the security boundary.** This skill needs `Read` (to view images), `Write` and `Edit` (to produce JSON files), `Bash` (to run the validation and ingestion scripts), and `Glob` (to list page files). It does **not** need `WebFetch`, `WebSearch`, or any of the MCP tools — and so they aren't granted. If the skill tries to invoke one of those, Claude refuses. This is the "principle of least privilege" applied to AI-driven workflows.

### The body — six steps

The body of `SKILL.md` opens with a short `Inputs` preamble and then walks Claude through six numbered steps. I'll show the structure and pull out the load-bearing details.

**Inputs — Identify the target episode.**

Before the numbered steps begin, the skill resolves the user's request to a specific episode directory under `data/raw/`:

- A slug like `ep07-the-wish` → use directly.
- A number like *"episode 7"* / *"ep7"* → resolve via `ls data/raw/ | grep '^ep07-'`; ask the user if multiple match.
- No episode named → list `data/raw/ep*` and ask which one.

This is the same `data/raw/` directory the acquisition script wrote into in [Post 2]({% post_url 2026-05-10-pepper-carrot-companion-workshop %}).

**Step 1 — Read the cast list from Postgres.**

```bash
docker exec peppercarrot-postgres psql -U peppercarrot -d peppercarrot -tA \
  -c "SELECT name FROM characters ORDER BY name;"
```

This is critical. The schema's `characters_present` field has to contain **canonical names** from the seeded `characters` table — not Claude's guess at a character name. *"the young witch"* should resolve to `"Pepper"`. *"the cat"* to `"Carrot"`. *"the bird"* to `"Mango"`. The skill pulls the cast list at the top of each run so Claude has it in working memory while describing pages.

> *Plain-English aside: why does this matter so much?* The chat layer (covered in a later post) joins page descriptions back to canonical characters via `page_characters` — that's what powers the "next time Pepper appears" navigation feature and lets the chat anchor "Pepper" to a specific known entity rather than re-inferring her existence on every turn. If Claude writes `"the witch"` instead of `"Pepper"` in `characters_present`, the join fails silently and that page never shows up in any character-anchored query. Pulling the cast list at the start fixes this *structurally* — Claude knows the only acceptable strings up front.

**Step 2 — List the pages.**

```bash
ls data/raw/<slug>/pages/page_*.jpg
```

Note the count, then process the pages in order — narrative continuity from earlier pages carries forward into the descriptions of later ones because Claude is describing them all in the same conversation. (The `PageDescription` schema doesn't pass previous-page context between calls — `JsonFileVisionClient` ignores the `previous_page_description` argument from the Protocol contract entirely — but Claude itself has continuity because every page is described in the same Claude Code session.)

**Step 3 — Describe each page.**

For each `page_NNN.jpg`, in order:

1. Use the `Read` tool to view the image. (Claude's `Read` tool accepts image paths and the model sees the image natively.)
2. Compose a `PageDescription` matching the schema:

```json
{
  "visual_description": "3-5 sentences of flowing prose, present tense. NO markdown, NO panel-by-panel breakdown (no 'Panel 1:', 'Setting:', 'Mood:' headers), NO bullet points. Describe what the page shows as a coherent narrative paragraph that a friend reading over your shoulder would understand.",
  "dialogue": [
    {"speaker": "Pepper", "text": "verbatim from the speech bubble"},
    {"speaker": null, "text": "SFX IN CAPS"}
  ],
  "characters_present": ["Pepper", "Carrot"],
  "locations_or_concepts": ["Komona", "the Potion Contest"],
  "mood_tags": ["surprised", "comedic"]
}
```

3. Write it to `data/raw/<slug>/pages/page_NNN.json` — sibling of the image.

Each field has rules attached, and **every constraint is load-bearing.** A short tour:

- **`visual_description`: prose only.** No markdown headers, no bullet lists, no "Panel 1: ... Panel 2: ..." structure. The whole point of running this through Claude rather than `qwen2.5vl:7b` is to escape the structured-output gravity that the smaller model couldn't. If Claude slips into a structured format anyway (it sometimes does for very complex pages), re-prompt for prose.
- **`dialogue`: verbatim, one entry per bubble, in reading order.** Don't paraphrase — these strings end up retrievable in the embedding index, and paraphrase distorts what the model can recall. `speaker = null` for sound effects, narration, and unidentified speakers.
- **`characters_present`: canonical names only.** Pulled from Step 1's list. Don't invent names for unnamed creatures — leave them out of this field and describe them in `visual_description` instead.
- **`locations_or_concepts`: named places, magic schools, named potions, currencies.** Keeps the universe-specific vocabulary in a queryable field for later.
- **`mood_tags`: 1–4 short adjectives.** Powers a future filter feature in the UI ("show me funny pages"); harmless to include even if you never build that filter.

**Step 4 — Validate the JSONs.**

Before invoking the ingestion script, the skill makes Claude verify every JSON parses against the Pydantic model:

```bash
cd backend && uv run python -c "
from pathlib import Path
from app.clients.vision import PageDescription
for p in sorted(Path('../data/raw/<slug>/pages').glob('*.json')):
    PageDescription.model_validate_json(p.read_text())
    print(f'{p.name}: OK')
"
```

This catches missing fields, wrong types, and JSON syntax errors *before* they break a longer pipeline run. The error you get from `model_validate_json` points at the exact field that failed — easier to diagnose than a `KeyError` two layers deeper.

**Step 5 — Invoke the wrapper script.**

The skill invokes the wrapper via an **absolute path** rather than a relative one, because Claude Code's Bash tool can run from any working directory and `./...` would resolve against the wrong place. The source pattern:

```bash
# Resolve the project root from anywhere the Bash tool happens to be in:
ROOT=$(git -C "$(pwd)" rev-parse --show-toplevel)
"$ROOT/.claude/skills/ingest-from-images/scripts/reingest_with_json.sh" <slug>
```

That script is the bridge between Stage 1 (Claude writes JSONs) and Stage 2 (Python pipeline does everything else). Two interesting things in it worth pulling out:

```bash
# Capture the original VISION_PROVIDER line so we can put it back exactly
# as it was (preserves comments/spacing on adjacent lines).
orig_line=$(grep -E '^VISION_PROVIDER=' "$ENV_FILE" || true)

# Always restore on exit, even on Ctrl-C or ingest failure.
restore() {
  sed -i '' "s|^VISION_PROVIDER=.*|$orig_line|" "$ENV_FILE"
  echo "[reverted] $orig_line"
}
trap restore EXIT

sed -i '' "s|^VISION_PROVIDER=.*|VISION_PROVIDER=json|" "$ENV_FILE"
```

> *Plain-English aside: what's `trap … EXIT`?* In bash, [`trap`](https://www.gnu.org/software/bash/manual/html_node/Signals.html) registers a function to run when a specific event happens. `trap restore EXIT` says *"no matter how this script ends — clean finish, exception, Ctrl-C, the OS sending a kill signal — run the `restore` function before exiting."* It's the bash equivalent of Python's `try / finally`: guarantees cleanup. Here it guarantees that `VISION_PROVIDER` in `.env` gets put back to whatever it was before the script started, even if the ingestion script crashes halfway through. Without this, a failed run could leave `.env` in a weird state and future runs would behave unexpectedly.

**Step 6 — Verify the ingestion landed.**

After the script finishes, the skill makes Claude run two psql queries: one confirming the episode row has a `plot_summary` and the right page count, and one listing per-page `(page_number, characters)` so any "unknown character" warnings get caught right then.

```bash
docker exec peppercarrot-postgres psql -U peppercarrot -d peppercarrot -c \
  "SELECT episode_number, slug, plot_summary IS NOT NULL AS has_summary,
   (SELECT COUNT(*) FROM pages WHERE episode_id = e.id) AS page_count
   FROM episodes e WHERE slug = 'ep01-potion-of-flight';"
```

Expect `has_summary = t` and `page_count` matching the number of JSONs.

That's the whole skill — ~170 lines of Markdown plus a ~50-line wrapper script. Modest surface area, hidden in plain sight, doing the highest-leverage step in the entire ingestion stack.

---

## The Python Pipeline (Stage 2) {#stage-two}

Once Stage 1's JSONs are sitting on disk, Stage 2 is just code. [`ingestion/ingest.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/ingestion/ingest.py) orchestrates everything. The script is ~540 lines but the per-episode flow is straightforward — most of the length is wiki-ingestion and world-graph paths we'll cover in later posts.

Here's the per-episode flow at the right level of detail to follow. We'll show the load-bearing parts inline and link the rest.

### Load + validate, then build clients

```python
async def _run_episode(episode_dir: Path, *, reembed: bool) -> None:
    settings = get_settings()

    # Step 1: validate input — fail fast on malformed metadata before
    # opening any external connections.
    metadata = load_episode_metadata(episode_dir)
    page_files = list_page_files(episode_dir)
    if not page_files:
        raise click.ClickException(f"No page files found in {episode_dir}/pages/")

    # Build clients (the one place we wire up SDKs — CLAUDE.md rule #1).
    vision = get_vision_client(settings)           # → JsonFileVisionClient
    embedding = get_embedding_client(settings)     # → Ollama or sentence-transformers
    storage = get_storage(settings)                # → LocalStorage or R2Storage
    chat = get_chat_client(settings)               # → for the plot-summary call
```

The four `get_*_client` calls return Protocol-typed instances — exactly the seams from [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}). The factory in `clients/__init__.py` reads `.env` and picks an implementation. For Stage 2 in the workshop starter, that resolves to: `JsonFileVisionClient` for vision (the only option in `VISION_PROVIDER`), `OllamaEmbeddingClient` for embeddings, `LocalStorage` for image storage, and `OllamaChatClient` for the plot-summary call.

### A look at `JsonFileVisionClient` {#json-file-vision-client}

The vision client deserves a closer look because it's where the [§ Three Vision-Provider Options](#three-options) architectural choice lands as actual code — and because it shows the payoff of the [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) `Protocol`-and-factory pattern.

The `Protocol` itself, from `backend/app/clients/vision.py`:

```python
class VisionClient(Protocol):
    async def describe_page(
        self,
        image_path: Path,
        previous_page_description: str | None,
        cast_list: list[str],
    ) -> PageDescription: ...

    async def answer_about_page(
        self, image_path: Path, prompt: str
    ) -> str: ...
```

Two methods. `describe_page` is the one Stage 2's per-page loop calls; `answer_about_page` is reserved for future runtime page-Q&A and is optional for any given implementation.

And the concrete `JsonFileVisionClient` (same file, abbreviated for readability):

```python
class JsonFileVisionClient:
    """Reads pre-written `PageDescription` JSON files from disk."""

    async def describe_page(
        self,
        image_path: Path,
        previous_page_description: str | None,
        cast_list: list[str],
    ) -> PageDescription:
        del previous_page_description, cast_list  # JSON is the source of truth
        json_path = image_path.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(
                f"No description JSON at {json_path}. Run the "
                f"`ingest-from-images` skill against this episode first."
            )
        return PageDescription.model_validate_json(
            json_path.read_text(encoding="utf-8")
        )

    async def answer_about_page(self, image_path: Path, prompt: str) -> str:
        raise NotImplementedError(
            "Ingestion-only — no model behind this client."
        )
```

Two things matter here. First, **the class implements the Protocol structurally, not nominally** — `JsonFileVisionClient` does not inherit from `VisionClient`. It just has methods of the right shape, which is the [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) rule playing out: any class with `async describe_page(...) -> PageDescription` and `async answer_about_page(...) -> str` satisfies the contract, regardless of inheritance. `mypy --strict` checks the structural shape, not the class hierarchy.

Second, **the implementation contains no model calls at all** — no `httpx`, no `anthropic` SDK, no `ollama` SDK. The vision work has already happened out of process, in a Claude Code session that ran before this Python script started, and the result lives on disk as a JSON file the client just reads and validates. From the pipeline's perspective it *is* calling a vision model. The "vision model" just happens to be a developer plus Claude sitting in another window.

The factory entry in `backend/app/clients/__init__.py` is correspondingly tiny:

```python
def get_vision_client(settings: Settings) -> VisionClient:
    # Single implementation: each page image must have a sibling .json file
    # containing a serialised PageDescription. The `ingest-from-images`
    # Claude Code skill produces those JSON files.
    del settings  # no settings consumed today; kept for future provider opts
    return JsonFileVisionClient()
```

No `if settings.vision_provider == ...` branch the way `get_chat_client` has — because in this project we only need the one implementation, and CLAUDE.md's *"never import an SDK outside `clients/`"* rule means nothing else in the codebase has any visibility into what the vision client actually does internally. If the constraints ever flipped — say, the chat layer one day needed live page-Q&A and you wanted a real runtime VLM behind it — you'd add an `AnthropicVisionClient` class in `vision.py` and a config branch here, and **nothing else in the codebase would change**. The route handlers, the ingestion script, the chat orchestrator, the retrieval layer — none of them would notice. That's the whole point of the [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) architecture: the abstraction absorbs even very unconventional choices — like *"the vision model is actually a Claude Code session"* — without leaking through the rest of the system.

### The per-page loop

For each page, in order:

```python
async def _process_one_page(
    *, session, storage, vision,
    episode_id, episode_slug, page_number, page_file,
    previous_visual_description, cast_list,
):
    # (a) Generate image variants with Pillow.
    processed: ProcessedPageImages = process_page_image(page_file)

    # (b) Upload variants via the storage abstraction.
    base_key = f"episodes/{episode_slug}/pages/{page_number:03d}"
    await storage.put(f"{base_key}-thumbnail.webp", processed.thumbnail_bytes, "image/webp")
    await storage.put(f"{base_key}-original{page_file.suffix}", processed.original_bytes, ...)
    if processed.is_animated:
        display_key = f"{base_key}-original{page_file.suffix}"   # GIFs keep motion
    else:
        display_key = f"{base_key}-display.webp"
        await storage.put(display_key, processed.display_bytes, "image/webp")

    # (c) Read the page description. JsonFileVisionClient looks for a
    #     sibling page_NNN.json next to the image — written by the skill.
    description = await vision.describe_page(
        image_path=page_file,
        previous_page_description=previous_visual_description,
        cast_list=cast_list,
    )

    # (d) Upsert the pages row.
    page = await upsert_page(session, episode_id, page_number,
                              {"display": display_key, "thumbnail": thumbnail_key,
                               "original": original_key}, description)

    # (e) Set image_metadata (width / height / blurhash / dominant_color).
    page.image_metadata = {
        "width": processed.metadata.width,
        "height": processed.metadata.height,
        "blurhash": processed.metadata.blurhash,
        "dominant_color": processed.metadata.dominant_color,
    }

    # (f) Link characters via the page_characters join table.
    await link_page_characters(session, page.id, list(description.characters_present))

    return page, description
```

Five things in this loop earn their lines:

**Image variants are pre-computed.** [`process_page_image`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/ingestion/images.py) (in `ingestion/images.py`) returns three variants for every page: a 1600-px-on-the-long-edge WebP for display, a 300-px WebP for thumbnails, and the original bytes pass-through. All three get uploaded under the same `episodes/<slug>/pages/<NNN>` key prefix. The display variant is the one the flipbook UI loads (covered in the next post); the thumbnail is for the episode picker; the original is kept for archival and high-res views.

**Animated pages (GIFs) bypass the display-variant resize.** *Pepper & Carrot* episode 4 has at least one animated page. Pillow's WebP encoder loses the motion if you write the resized version as a static WebP, so the code detects animated source files (`processed.is_animated`) and serves the original directly as the display URL. Same key shape; different bytes behind it.

**`vision.describe_page` is the Protocol seam.** It accepts an image path, an optional previous-page description, and a cast list. `JsonFileVisionClient` ignores the second and third arguments — they're part of the contract because *other* implementations of `VisionClient` would need them, but the JSON version pulls everything from the sibling JSON file. (Reminder: the `previous_page_description` argument is for hypothetical future clients that need narrative continuity context. Claude's reading inside the skill *already has* continuity because pages are described in order in the same Claude Code session.)

**Per-page commits.** Look at the call site of `_process_one_page` in the parent function:

```python
for page_num, page_file in enumerate(progress, start=1):
    page, description = await _process_one_page(...)
    await session.commit()                   # ← per-page commit
    page_results.append((page, description))
    previous_desc = description.visual_description
```

The DB transaction is committed *after every page*. If page 7 fails for any reason — bad JSON, storage hiccup, network blip — pages 1 through 6 stay committed in the database. The next ingestion run resumes from a known good state. The alternative (one giant transaction per episode) means a single bad page rolls back hours of work.

**The image metadata is JSONB.** From Post 2's design decisions: `pages.image_metadata` is a JSONB column holding `{width, height, blurhash, dominant_color}` as a single blob, not separate columns. The frontend reads them all together when rendering a page; nothing in the app queries "every page wider than 1000 px." JSONB keeps the column count flat and lets us add `palette` or `alt_text` later as one-line model edits.

### After all pages: plot summary + embeddings

Two more steps run after the per-page loop finishes:

```python
# Step 4: episode plot summary — one chat call across all descriptions.
summary = await _summarize_episode(chat, [d for _, d in page_results], metadata)
await upsert_episode_summary(session, episode_id, summary)
await session.commit()

# Step 6: Chroma writes (after DB commit so PKs are durable on disk).
chroma_writer = ChromaWriter(settings.chroma_persist_dir, embedding)
await chroma_writer.upsert_page_chunks(page_results, episode_number=metadata.episode_number)
```

The plot summary is **one chat call** (`chat.stream(...)`) that takes all the page descriptions as input and asks for a 2–3 sentence summary. That's stored on `episodes.plot_summary` and is what the episode picker UI shows underneath each episode card. The prompt is deliberately tight:

```python
system = (
    "You are a literary editor writing concise, spoiler-aware plot summaries "
    "for a children's webcomic."
)
```

Spoiler-aware because the model only sees the descriptions *of pages in this episode*, not future ones — so it can't accidentally reveal something downstream.

The Chroma writes happen **after** the DB commit. Reasoning: if Chroma fails after Postgres has committed, you can re-run safely — re-ingestion is idempotent. If we wrote to Chroma first and Postgres failed, the vector store would have orphan chunks pointing at IDs that don't exist anywhere.

> *Plain-English aside: what's "idempotent" in this context?* An **idempotent** operation is one you can run multiple times and the end state is the same as running it once. Re-ingesting episode 1 ten times produces the same set of rows in `pages` and the same set of chunks in `pages_v1`. Achieved here via three primary keys: `pages.(episode_id, page_number)` is unique (so re-runs upsert the same row, not a new one), Chroma chunks are keyed on a stable composite ID, and `LocalStorage.put` (from [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %})) is a no-op when the bytes match. Idempotency is what makes a partial-failure recovery just "run it again" instead of "manually clean up state first."

The full embedding format for `pages_v1` chunks — `visual_description + "\n\nDialogue:\n" + ...` — lives in [`ingestion/chroma_writer.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/ingestion/chroma_writer.py) so the chat layer can import the same helper and stay in sync. If retrieval-time text and ingestion-time text drift, retrieval quality silently degrades.

---

## Two-Stage Flow in One Picture {#two-stage-flow}

Stage 1 (interactive, Claude Code session) writes JSONs. Stage 2 (the Python pipeline, ~30–60 sec) reads them and produces everything else.

<div style="margin: 1.5rem 0; overflow-x: auto;">
<a href="/assets/picture/2026-05-16-pepper-carrot-companion-claude-skill-ingestion/two-stage-flow.svg" target="_blank" rel="noopener" title="Open the diagram full-size in a new tab" style="display: block; cursor: zoom-in;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 580" role="img"
     aria-label="Two-stage ingestion flow. Stage 1: image goes into a Claude Code session with the ingest-from-images skill loaded; Claude reads the image and writes a sibling JSON file. Stage 2: the Python pipeline (ingest.py) reads both the image and the JSON, then writes to LocalStorage, Postgres, and ChromaDB."
     style="display: block; width: 100%; height: auto; max-width: 960px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">
  <defs>
    <marker id="ing-arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>
    </marker>
  </defs>

  <!-- Stage labels -->
  <text x="220" y="30" text-anchor="middle" font-size="13" font-weight="700" fill="#475569" letter-spacing="0.05em">STAGE 1</text>
  <text x="220" y="48" text-anchor="middle" font-size="11" fill="#94a3b8" font-style="italic">interactive Claude Code session · ~5–15 min · $0</text>
  <text x="730" y="30" text-anchor="middle" font-size="13" font-weight="700" fill="#475569" letter-spacing="0.05em">STAGE 2</text>
  <text x="730" y="48" text-anchor="middle" font-size="11" fill="#94a3b8" font-style="italic">Python script · ~30–60 sec · $0</text>

  <line x1="475" y1="20" x2="475" y2="540" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4,4"/>

  <!-- Stage 1 box -->
  <g>
    <rect x="80" y="70" width="280" height="68" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="220" y="95" text-anchor="middle" font-size="13" font-weight="600" fill="#1f2937">page_001.jpg on disk</text>
    <text x="220" y="115" text-anchor="middle" font-size="10" fill="#475569" font-style="italic"
          font-family="ui-monospace, 'SF Mono', Menlo, Monaco, monospace">data/raw/ep01-potion-of-flight/pages/</text>
    <text x="220" y="129" text-anchor="middle" font-size="10" fill="#94a3b8" font-style="italic">(downloaded by acquire.py, Post 2)</text>
  </g>

  <!-- Arrow down to Claude session -->
  <line x1="220" y1="138" x2="220" y2="178" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ing-arrow)"/>
  <text x="230" y="162" font-size="10" fill="#475569"
        font-family="ui-monospace, 'SF Mono', Menlo, Monaco, monospace">Read tool</text>

  <!-- Claude Code session box -->
  <g>
    <rect x="60" y="180" width="320" height="120" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="220" y="204" text-anchor="middle" font-size="14" font-weight="600" fill="#1f2937">Claude Code session</text>
    <text x="220" y="222" text-anchor="middle" font-size="11" fill="#1e40af">+ ingest-from-images skill loaded</text>
    <text x="220" y="244" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">Claude reads the image,</text>
    <text x="220" y="258" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">composes a `PageDescription` matching</text>
    <text x="220" y="272" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">the strict schema, repeats per page.</text>
    <text x="220" y="290" text-anchor="middle" font-size="10" fill="#94a3b8" font-style="italic">Same model as Anthropic vision API.</text>
  </g>

  <!-- Arrow down to JSON -->
  <line x1="220" y1="300" x2="220" y2="340" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ing-arrow)"/>
  <text x="230" y="324" font-size="10" fill="#475569"
        font-family="ui-monospace, 'SF Mono', Menlo, Monaco, monospace">Write tool</text>

  <!-- JSON file box -->
  <g>
    <rect x="80" y="342" width="280" height="68" rx="8" fill="#d1fae5" stroke="#059669" stroke-width="1.5"/>
    <text x="220" y="367" text-anchor="middle" font-size="13" font-weight="600" fill="#1f2937">page_001.json on disk</text>
    <text x="220" y="387" text-anchor="middle" font-size="10" fill="#065f46" font-style="italic">structured PageDescription</text>
    <text x="220" y="401" text-anchor="middle" font-size="10" fill="#475569"
          font-style="italic" font-family="ui-monospace, 'SF Mono', Menlo, Monaco, monospace">(sibling of page_001.jpg)</text>
  </g>

  <!-- Stage 1 → Stage 2 hand-off arrow (curves right) -->
  <path d="M 360 376 Q 475 376 475 305 Q 475 235 590 235" stroke="#6b7280" stroke-width="1.5" fill="none" stroke-dasharray="6,4" marker-end="url(#ing-arrow)"/>
  <text x="475" y="265" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">JSON + JPG</text>
  <text x="475" y="278" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">both on disk</text>

  <!-- Stage 2 box: ingest.py -->
  <g>
    <rect x="590" y="180" width="320" height="160" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="750" y="204" text-anchor="middle" font-size="14" font-weight="600" fill="#1f2937">ingestion/ingest.py</text>
    <text x="750" y="222" text-anchor="middle" font-size="11" fill="#1e40af">Python pipeline · per-page loop</text>
    <text x="750" y="246" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">1. Pillow image variants (1600 / 300 / orig)</text>
    <text x="750" y="260" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">2. storage.put → display / thumb / original keys</text>
    <text x="750" y="274" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">3. JsonFileVisionClient reads sibling .json</text>
    <text x="750" y="288" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">4. upsert_page() + link_page_characters()</text>
    <text x="750" y="302" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">5. commit (per-page)</text>
    <text x="750" y="320" text-anchor="middle" font-size="10" fill="#94a3b8" font-style="italic">After loop: plot_summary + Chroma embeds</text>
  </g>

  <!-- Three output arrows fanning out from Stage 2 -->
  <line x1="640" y1="340" x2="610" y2="408" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ing-arrow)"/>
  <line x1="750" y1="340" x2="750" y2="408" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ing-arrow)"/>
  <line x1="860" y1="340" x2="890" y2="408" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ing-arrow)"/>

  <!-- Output 1: LocalStorage -->
  <g>
    <rect x="510" y="410" width="180" height="72" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="600" y="434" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">LocalStorage</text>
    <text x="600" y="452" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">data/images/episodes/…</text>
    <text x="600" y="468" text-anchor="middle" font-size="10" fill="#92400e">display / thumb / original</text>
  </g>

  <!-- Output 2: Postgres -->
  <g>
    <rect x="660" y="410" width="180" height="72" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="750" y="434" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">PostgreSQL</text>
    <text x="750" y="452" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">pages · page_characters</text>
    <text x="750" y="468" text-anchor="middle" font-size="10" fill="#92400e">+ episodes.plot_summary</text>
  </g>

  <!-- Output 3: ChromaDB -->
  <g>
    <rect x="810" y="410" width="140" height="72" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="880" y="434" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">ChromaDB</text>
    <text x="880" y="452" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">pages_v1 collection</text>
    <text x="880" y="468" text-anchor="middle" font-size="10" fill="#92400e">N embedded chunks</text>
  </g>

  <!-- Legend -->
  <g>
    <rect x="80" y="510" width="16" height="14" fill="#fef3c7" stroke="#f59e0b" stroke-width="1"/>
    <text x="104" y="522" font-size="11" fill="#4b5563">on disk / external store</text>
    <rect x="270" y="510" width="16" height="14" fill="#dbeafe" stroke="#2563eb" stroke-width="1"/>
    <text x="294" y="522" font-size="11" fill="#4b5563">code · process</text>
    <rect x="430" y="510" width="16" height="14" fill="#d1fae5" stroke="#059669" stroke-width="1"/>
    <text x="454" y="522" font-size="11" fill="#4b5563">artifact produced by Stage 1</text>
    <line x1="660" y1="517" x2="710" y2="517" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="6,4"/>
    <text x="718" y="522" font-size="11" fill="#4b5563">handoff between stages</text>
  </g>
</svg>
</a>
</div>

*Stage 1 reads the image and writes a sibling JSON; Stage 2 reads the image **and** the JSON and writes to all three persistent stores. The handoff between stages is the JSON file on disk — durable, auditable, hand-editable. Click the diagram to open it full-size in a new tab.*

The cleanest property of this design is that **the two stages are independent.** You can re-run Stage 1 without re-running Stage 2 (you've changed the description prompt and want fresh JSONs but haven't loaded them yet). You can re-run Stage 2 without re-running Stage 1 (you've changed the chunking strategy in `chroma_writer.py` and want to re-embed against the existing JSONs). Either flow is one command.

---

## Why This Was the Right Choice For *This* Project {#why-this-wins}

Five reasons this approach fits the *portfolio / demo* shape of Pepper & Carrot specifically. Each one lines up with a constraint that's load-bearing here and may *not* be load-bearing for your project — read these as "here's how this approach maps to one specific project's priorities," not as "here's why this beats everything else."

**1. Cost matches a portfolio budget.** No metered per-call dollars — the work runs inside an existing Claude Code subscription ($20/mo Pro, $100–200/mo Max), which the author already pays for. The marginal cost of one more page, or one more prompt-iteration pass, is zero. Iterating on the description prompt — *"actually, lean more whimsical; capture the mood the panel border evokes; don't say 'panel'"* — costs nothing but the Claude Code session time. Compare to ~$13 per full corpus re-ingest with a hosted vision API; harmless at production volume, but a friction tax on a project where the *whole point* is to iterate prompt phrasing freely. **For a paid product where the cost per call is amortized over high request volume, this calculus flips** — the hosted API's per-call price becomes a rounding error and the ability to run ingestion automatically on a schedule matters more than free iteration.

**2. Quality matches what you'd get from a paid call to the same model.** Claude Code uses the same Claude Opus model that powers Anthropic's hosted vision API, prompted via the skill to produce flowing prose with the canonical cast list and a strict schema as the contract. The descriptions are essentially as good as you can get for this task today. This isn't a "free thing that's almost as good as the paid thing" — it's literally the same model. The trade-off is *how* you call it, not *what calls back*.

**3. Reliability matches a single-developer workflow.** No GGML assertion failures, no model-server crashes mid-run, no HTTP rate limits to budget for. The one failure mode that does exist — Claude misreads a page, invents a character — is *visible in the JSON on disk* and can be hand-edited in 10 seconds. **For a single developer iterating in interactive bursts**, that's a much friendlier loop than chasing transient errors in a batch pipeline. For an unattended nightly job, you'd want different failure-handling — retry policies, dead-letter queues, alerting — that a hosted-API path naturally fits into.

**4. Auditability is the killer feature, period.** Every page description sits on disk as a structured JSON file you can read, diff, and edit. If a description has a factual error (Claude said *"Pepper is reading a book"* when she's clearly reading a scroll), you fix the JSON, re-run Stage 2 in 30 seconds, and the new text is in the DB and Chroma. With a runtime VLM, fixing one bad description means either re-running the whole vision pass or maintaining a separate "manual overrides" table the runtime layer checks. **This benefit isn't portfolio-specific** — any project that values "fix one bad description in 30 seconds" gets it, regardless of the model behind it. Worth knowing this is the part that generalizes.

**5. Architecture leaves the door open.** `JsonFileVisionClient` implements the existing `VisionClient` Protocol exactly the same way an `AnthropicVisionClient` would. Nothing else in the pipeline — image processing, storage, DB upsert, character linking, Chroma write, plot summary — knows or cares that the "vision model" is actually a Claude Code session writing files. **The seam from [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) absorbs the unconventional choice cleanly.** If the project's constraints ever flip — a runtime vision provider for live page-Q&A, an automated scheduled pipeline — reintroducing the right vision client is a config change, not a refactor. **The right takeaway from this section isn't "always use Claude Code as your vision provider"; it's "build the seam that lets you swap providers when your constraints change."**

---

## The Honest Trade-offs {#trade-offs}

Three real costs of the Claude-Code-as-vision-provider path, all named up front:

**Stage 1 requires an interactive Claude Code session.** It can't run unattended on a schedule, or be triggered by a webhook, or wired into any "no human present" automation. You have to open a Claude Code window, type *"ingest episode 5"*, and let it work for 10–15 minutes. For a portfolio project where ingestion is a once-per-episode-added concern, this is fine. For a real product that ingests new content on a schedule or under SLAs, this rules the approach out — you'd reintroduce a hosted vision client and pay the per-call cost.

**Doesn't scale to a full corpus in one session.** Each Claude Code session has a finite context window. Reading 100+ images in one conversation fills it up — the recommended pattern is **one Claude Code session per episode** for batch work. For *Pepper & Carrot*'s ~40 episodes, that's 40 sessions over a few weekends. Workable for a portfolio project; unworkable as a per-publishing-cycle workflow in a CMS.

**No live VLM for runtime page-Q&A.** `JsonFileVisionClient.answer_about_page` raises `NotImplementedError`. The chat orchestrator (covered in a later post) operates on retrieved *text* — not images — so this isn't a feature gap today. If a future feature needs live pixel inspection at chat time (*"describe panel 3 in detail"*), you'd reintroduce a different vision provider for that path. The Protocol is still in place; the abstraction holds.

### When you'd choose differently

To make the framing concrete, here's the decision tree for *another* project picking a vision provider:

| Your project's situation | What we'd pick |
|---|---|
| Paid product · ingestion runs unattended on a schedule | **Hosted vision API** (Anthropic, OpenAI, Google). Same quality as Claude Code; you pay for the unattended automation and per-call SLAs. |
| Privacy posture requires pixels stay on your own infra | **Local VLM** ([Qwen3-VL](https://qwenlm.github.io/blog/qwen3-vl/), [Llama 4 Vision](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), etc., on owned GPUs). Pay the latency and GPU-amortization cost for the privacy. The markdown-leakage problem we hit on `qwen2.5vl:7b` is much smaller on the current generation of open-weights VLMs. |
| One-off internal tool · you already have a Claude subscription · no scheduled/unattended runs needed | **Claude Code as vision provider** (this post's path). Free, high-quality, JSONs on disk. |
| Tiny corpus (<50 images), one-time job | Any of the above — at this size the choice barely matters. Pick whichever you already have credentials for. |
| Very large corpus (millions of images), unit-economics dominate | Probably a self-hosted optimized VLM behind a batch API. Per-call hosted-API pricing adds up; Claude Code path doesn't scale. |

The portfolio-project version of this project is in row three. Most real products live in rows one or two. **Pick the row your project is actually in.** The architecture in this codebase — the `VisionClient` Protocol from [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) — supports any of the rows. **That's the durable lesson; the specific row we picked is just a case study.**

---

## Running It on Episode 1 {#running-it}

Here's the full reproduction recipe. You should have everything from Posts 2–3 already set up.

### 1. Confirm prerequisites

If you've worked through Posts 2 and 3 recently, most of this is already in place — but each command below is idempotent, so you can run the whole block top-to-bottom whether starting fresh or resuming. Everything assumes you start at the repo root.

```bash
# 1a. Repo at the right state
cd path/to/pepper-carrot-companion-workshop
git pull                                          # pick up Post 4's ingestion code + the skill

# 1b. .env in place (docker-compose reads it for Postgres credentials,
#     and the wrapper script reads/writes VISION_PROVIDER in it)
[[ -f .env ]] || cp .env.example .env             # no-clobber: keeps any existing .env

# 1c. Postgres up + healthy + schema applied + characters seeded
docker compose up -d
docker compose ps                                 # postgres should show (healthy)
(cd backend && uv sync \
            && uv run alembic upgrade head \
            && uv run python -m app.db.seed)     # 31 canonical characters upserted

# 1d. Ollama serving + both models installed
#     (qwen2.5:7b ≈ 4.7 GB for the plot-summary call; bge-m3 ≈ 1.2 GB for embeddings.
#      qwen2.5vl:7b is NOT needed — Claude Code itself is the vision provider.)
ollama serve &                                    # skip if you already run the menu-bar app
ollama pull qwen2.5:7b                            # no-op if already installed
ollama pull bge-m3                                # no-op if already installed
ollama list                                       # both should appear

# 1e. Episode 1 on disk + ingestion deps installed
(cd ingestion && uv sync \
              && uv run python acquire.py episode \
                   --slug ep01_Potion-of-Flight \
                   --lang en \
                   --out ../data/raw)
ls data/raw/ep01-potion-of-flight/pages/          # page_001.jpg, page_002.jpg, page_003.jpg
```

If any step fails, fix it before continuing — the skill will assume all five are in place. The `acquire.py` script is idempotent too: re-running only re-downloads what's missing.

### 2. Open Claude Code in the repo

```bash
cd path/to/pepper-carrot-companion-workshop
claude
```

The `ingest-from-images` skill auto-loads because its description matches your repo's local `.claude/skills/`.

### 3. Ask Claude to ingest the episode

Type into the Claude Code prompt:

```
ingest episode 1 from images
```

What happens next (transcribed from a real run):

1. **Claude resolves the slug.** Runs `ls data/raw/ | grep '^ep01-'`, finds `ep01-potion-of-flight`.
2. **Claude pulls the cast list.** Runs `docker exec peppercarrot-postgres psql ...` and surfaces the 31 canonical names.
3. **Claude reads each page.** For `page_001.jpg`, `page_002.jpg`, `page_003.jpg`, it uses the `Read` tool to view each image, then composes a `PageDescription` and writes the JSON next to the image.
4. **Claude validates.** Runs the `model_validate_json` loop. Output should be `page_001.json: OK / page_002.json: OK / page_003.json: OK`.
5. **Claude invokes the wrapper script.** `.claude/skills/ingest-from-images/scripts/reingest_with_json.sh ep01-potion-of-flight`. You'll see Stage 2's per-page progress bar tick through three pages, the plot summary call, and the Chroma writes.
6. **Claude runs verification queries.** Two `psql` queries confirming the row landed and characters linked. Expect `has_summary = t`, `page_count = 3`.

If you want to watch it without retyping at each step, hit `Tab` (autocomplete) or just stay out of the way — the skill is set up to drive the whole flow autonomously.

![Live recording of the `ingest-from-images` skill ingesting episode 1 — Claude reads each page, writes the JSONs, runs the script, and verifies the database. (Click to enlarge.)](/assets/picture/2026-05-16-pepper-carrot-companion-claude-skill-ingestion/ingest-episode-1.gif){: width="720" .shadow }
*A live walkthrough of `ingest-from-images` on episode 1 — skill loads, Claude reads each page image, writes the JSONs, runs the wrapper script, Stage 2 produces variants + DB rows + Chroma chunks + plot summary in ~30 seconds. (Click to enlarge.)*

### 4. Verify the result yourself

A few one-line queries you can run from a second terminal:

```bash
# All three pages in the DB, with the first 80 chars of each description.
docker exec peppercarrot-postgres psql -U peppercarrot -d peppercarrot -c "
  SELECT p.page_number, LEFT(p.visual_description, 80) AS preview
  FROM pages p JOIN episodes e ON e.id = p.episode_id
  WHERE e.slug = 'ep01-potion-of-flight'
  ORDER BY p.page_number;
"

# Episode plot summary on the episodes row.
docker exec peppercarrot-postgres psql -U peppercarrot -d peppercarrot -c "
  SELECT plot_summary FROM episodes WHERE slug = 'ep01-potion-of-flight';
"

# Character links per page.
docker exec peppercarrot-postgres psql -U peppercarrot -d peppercarrot -c "
  SELECT p.page_number, STRING_AGG(c.name, ', ' ORDER BY c.name) AS chars
  FROM pages p
  LEFT JOIN page_characters pc ON pc.page_id = p.id
  LEFT JOIN characters c ON c.id = pc.character_id
  WHERE p.episode_id = (SELECT id FROM episodes WHERE slug = 'ep01-potion-of-flight')
  GROUP BY p.page_number ORDER BY p.page_number;
"

# ChromaDB chunks.
cd backend && uv run python -c "
import chromadb
client = chromadb.PersistentClient(path='../data/chroma')
col = client.get_collection('pages_v1')
print(f'pages_v1 has {col.count()} chunks')
print(col.peek(1))
"

# Image variants on disk.
ls data/images/episodes/ep01-potion-of-flight/pages/
# 001-display.webp  001-thumbnail.webp  001-original.jpg
# 002-display.webp  002-thumbnail.webp  002-original.jpg
# 003-display.webp  003-thumbnail.webp  003-original.jpg
```

If all five queries return the expected shape, **you have one episode fully ingested.** Every later post in the series builds on top of this state.

The SQL queries confirm the *shape* of the data landed. The actual *quality* of the descriptions is something only your eyes can verify — so here's each page of episode 1 with the JSON the skill wrote next to it, side by side. Click any image to open it full-size in a new tab.

**Page 1.**

<table style="width: 100%; table-layout: fixed; border-collapse: separate; border-spacing: 1em 0; margin: 1.5em 0; border: none;"><tbody><tr>
<td style="width: 50%; vertical-align: top; padding: 0; border: none;">
<a href="/assets/picture/2026-05-16-pepper-carrot-companion-claude-skill-ingestion/ep01-page-001.jpg" target="_blank" rel="noopener"><img src="/assets/picture/2026-05-16-pepper-carrot-companion-claude-skill-ingestion/ep01-page-001.jpg" alt="Pepper &amp; Carrot, episode 1, page 1 — Pepper standing beside a bubbling cauldron in a cozy witch's workshop, consulting a small spellbook while her orange cat Carrot dozes on a shelf behind her." style="width: 100%; max-height: 480px; object-fit: contain; display: block; border-radius: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.12);"></a>
</td>
<td style="width: 50%; vertical-align: top; padding: 0; border: none;">
<pre style="margin: 0; padding: 0.9em 1em; max-height: 480px; overflow-y: auto; font-size: 0.78em; line-height: 1.55; white-space: pre-wrap; word-break: break-word; background: #f6f8fa; border-radius: 6px; border: 1px solid rgba(0,0,0,0.08); box-sizing: border-box;"><code>{
  "visual_description": "In a cozy witch's workshop lit by a glowing round window, young Pepper stands beside a bubbling cauldron, consulting a small spellbook while her orange cat Carrot dozes on a shelf behind her. The room is cluttered with potion jars, kettles, and a broomstick leaning against the wall, all bathed in warm green-amber light from the cauldron's steam. She tastes the brew with a tentative sip, then frowns and reaches for more ingredients, plopping them into the pot to strengthen the mixture.",
  "dialogue": [
    {"speaker": "Pepper", "text": "...and the last touch."},
    {"speaker": null, "text": "SHI SHH"},
    {"speaker": "Pepper", "text": "...mmm probably not strong enough."},
    {"speaker": null, "text": "PLOP"},
    {"speaker": null, "text": "PLOP"}
  ],
  "characters_present": ["Pepper", "Carrot"],
  "locations_or_concepts": ["Pepper's house"],
  "mood_tags": ["quiet", "focused", "cozy"]
}</code></pre>
</td>
</tr></tbody></table>

*`page_001.json`. `visual_description` is prose, no markdown leakage. `dialogue` is verbatim (`SHI SHH`, `PLOP` are sound effects with `speaker = null`). Both characters are canonical names from the seeded `characters` table.*

**Page 2.**

<table style="width: 100%; table-layout: fixed; border-collapse: separate; border-spacing: 1em 0; margin: 1.5em 0; border: none;"><tbody><tr>
<td style="width: 50%; vertical-align: top; padding: 0; border: none;">
<a href="/assets/picture/2026-05-16-pepper-carrot-companion-claude-skill-ingestion/ep01-page-002.jpg" target="_blank" rel="noopener"><img src="/assets/picture/2026-05-16-pepper-carrot-companion-claude-skill-ingestion/ep01-page-002.jpg" alt="Pepper &amp; Carrot, episode 1, page 2 — A golden potion erupts from the cauldron; Carrot leaps into it despite Pepper's warning, sending shimmering droplets across the workshop." style="width: 100%; max-height: 480px; object-fit: contain; display: block; border-radius: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.12);"></a>
</td>
<td style="width: 50%; vertical-align: top; padding: 0; border: none;">
<pre style="margin: 0; padding: 0.9em 1em; max-height: 480px; overflow-y: auto; font-size: 0.78em; line-height: 1.55; white-space: pre-wrap; word-break: break-word; background: #f6f8fa; border-radius: 6px; border: 1px solid rgba(0,0,0,0.08); box-sizing: border-box;"><code>{
  "visual_description": "The cauldron erupts in a brilliant golden burst of sparkling magic, a glowing wand-shape arcing out of the brew as Pepper claps her hands in delight at the successful potion. Moments later her glee turns to alarm as Carrot, eyes wide with mischief, leaps up to investigate — she shouts him away while clutching her spellbook. Despite the warning, the orange tabby splashes straight into the bubbling golden potion, sending shimmering droplets everywhere across the workshop.",
  "dialogue": [
    {"speaker": "Pepper", "text": "ha... perfect."},
    {"speaker": "Pepper", "text": "NO! Don't even think about it."},
    {"speaker": null, "text": "SPLASH"}
  ],
  "characters_present": ["Pepper", "Carrot"],
  "locations_or_concepts": ["Pepper's house", "Potion of Flight"],
  "mood_tags": ["triumphant", "alarmed", "comedic"]
}</code></pre>
</td>
</tr></tbody></table>

*`page_002.json`. The mood progression `triumphant → alarmed → comedic` traces the page's three-beat narrative arc. `SPLASH` is another sound effect (`speaker = null`). `Potion of Flight` now appears in `locations_or_concepts` — the named potion this episode introduces, a hook for later retrieval.*

**Page 3.**

<table style="width: 100%; table-layout: fixed; border-collapse: separate; border-spacing: 1em 0; margin: 1.5em 0; border: none;"><tbody><tr>
<td style="width: 50%; vertical-align: top; padding: 0; border: none;">
<a href="/assets/picture/2026-05-16-pepper-carrot-companion-claude-skill-ingestion/ep01-page-003.jpg" target="_blank" rel="noopener"><img src="/assets/picture/2026-05-16-pepper-carrot-companion-claude-skill-ingestion/ep01-page-003.jpg" alt="Pepper &amp; Carrot, episode 1, page 3 — Pepper on her broom mid-race at night while Carrot, glittering with potion-gold, streaks ahead as a luminous flying cat." style="width: 100%; max-height: 480px; object-fit: contain; display: block; border-radius: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.12);"></a>
</td>
<td style="width: 50%; vertical-align: top; padding: 0; border: none;">
<pre style="margin: 0; padding: 0.9em 1em; max-height: 480px; overflow-y: auto; font-size: 0.78em; line-height: 1.55; white-space: pre-wrap; word-break: break-word; background: #f6f8fa; border-radius: 6px; border: 1px solid rgba(0,0,0,0.08); box-sizing: border-box;"><code>{
  "visual_description": "Under a starry night sky and a checkered finish-line banner, Pepper sits perched on her broom mid-race, soaring through wisps of cloud beside other young witches and their familiars. Carrot streaks ahead of her, his orange body wrapped in glittering golden sparkles from the potion, transformed into a luminous flying cat that easily outpaces the broom. Pepper looks up at him with a wry, exasperated smile, the racing crescent moon and rival competitors framing the scene as her unintended advantage carries the day.",
  "dialogue": [
    {"speaker": "Carrot", "text": "Happy?!"}
  ],
  "characters_present": ["Pepper", "Carrot"],
  "locations_or_concepts": ["Potion of Flight", "broom race"],
  "mood_tags": ["wondrous", "comedic", "triumphant"]
}</code></pre>
</td>
</tr></tbody></table>

*`page_003.json`. A mostly-action page: only one dialogue line — Carrot's — captured under his canonical name (the schema doesn't enforce real-world plausibility). `Potion of Flight` carries forward as the cross-page narrative anchor.*

### 5. Inspect a JSON (and edit if you want)

The JSONs are sitting next to the images. Open one:

```bash
cat data/raw/ep01-potion-of-flight/pages/page_001.json | jq .
```

```json
{
  "visual_description": "In a sunlit potion shop in Komona, Pepper concentrates …",
  "dialogue": [
    {"speaker": "Pepper", "text": "Now where did I put that vial of moonsilver?"},
    {"speaker": null, "text": "MEOW"}
  ],
  "characters_present": ["Pepper", "Carrot"],
  "locations_or_concepts": ["Komona"],
  "mood_tags": ["focused", "cozy"]
}
```

Edit it in your text editor — fix a phrase, add a character, adjust a mood tag — then re-run the wrapper script:

```bash
./.claude/skills/ingest-from-images/scripts/reingest_with_json.sh ep01-potion-of-flight
```

Thirty seconds later, the new description is in `pages.visual_description`, re-embedded in `pages_v1`, and the new plot summary reflects the change. **The JSON-on-disk hand-edit loop is the killer feature of this design.** You couldn't easily do this with a runtime vision model — to retouch one description you'd have to either re-run the whole VLM pass or maintain a separate "manual overrides" table the runtime checks.

---

## Key Takeaways {#key-takeaways}

**1. Pick the vision provider for your project's *actual* constraints.** Local VLM, hosted API, or Claude Code itself — none is universally best. The right choice depends on whether you need unattended automation, what your per-call budget is, whether iteration speed dominates over throughput, and what privacy posture you're operating under. **For this portfolio project**, free iteration and zero-ops-cost made Claude Code the right call. **For a paid product** ingesting comics nightly under SLAs, a hosted API would be. **For a privacy-sensitive deployment** keeping pixels in-house, a self-hosted open-weights VLM would. Read this post as one case study, not a prescription — and run your own version of the decision tree under "When you'd choose differently" before committing.

**2. Skills are pinned, executable prompts.** A Claude Code skill is the difference between *"explaining the task to Claude every time"* and *"explaining it once and having Claude remember."* When a task has rigid structural requirements — strict schemas, canonical vocabulary, validation steps, follow-up scripts — wrapping it in a skill makes the work reliable and the friction near zero.

**3. Make the high-leverage step auditable.** Page descriptions are the highest-leverage input to chat quality in the entire system. Putting them on disk as inspectable JSON — rather than treating them as ephemeral model output — means **fixing a bad description is a 30-second hand-edit** instead of a multi-hour pipeline re-run or a brittle "manual overrides" layer. Push correctness-critical artifacts onto disk; treat them as source of truth.

**4. Two-stage pipelines are recoverable pipelines.** Stage 1 is the slow expensive step; Stage 2 is the fast deterministic step. **Separating them with a durable artifact (the JSON) means partial failures are cheap.** Network blip during Chroma writes? Re-run Stage 2; Stage 1's work is on disk. Bug in your chunking strategy? Re-run Stage 2 against existing JSONs in 30 seconds; no need to re-describe anything. Architectures that hand-off through durable, inspectable files almost always end up easier to operate than monolithic ones.

**5. Protocols absorb unconventional choices.** Substituting "a Claude Code session" for "a model client" sounds like it should require a special path through the codebase. It doesn't. `JsonFileVisionClient` satisfies the same `VisionClient` Protocol any model-backed implementation would — the rest of the pipeline doesn't know or care. **Good abstractions let you make weird choices without paying weirdness tax across the codebase.**

**6. Cost shapes iteration; iteration shapes quality.** Whichever vision provider you pick, the per-call price you pay during *prompt iteration* (before a stable prompt ships) has compounding effects on output quality. If every prompt change costs $30+ at Opus pricing, you'll make fewer changes; if every change is free, you'll iterate to better outputs. The portfolio-project decision matrix here weighted "free iteration" very heavily — but a real product launching against a fixed deadline might rationally weight "ship a known-acceptable prompt now" over "iterate to a slightly better one." Both are defensible; just be deliberate about which one you're optimizing.

---

## Appendix: How a VLM Actually Sees an Image {#appendix-vlm}

The *tokenized + encoded + decoded* gloss in [Option 1](#three-options) compresses about six distinct steps into three words. For readers who want the actual mechanics, here is the full path an image takes through a model like Qwen2.5-VL between the moment it arrives at Ollama's HTTP API and the moment a description comes back out.

### The image side — from pixels to embeddings

**Patchification.** The raw image (say 1024×1024 RGB) is resized and normalized, then sliced into a regular grid of non-overlapping **patches** — typically 14×14 pixels each. A 448×448 image at 14px patches → 32×32 = 1024 patches. Each patch is flattened (14×14×3 = 588 numbers) and run through a single linear projection that maps it to a vector in some `d_vision` dimension (e.g. 1024). These vectors are *patch embeddings* — the image-side counterpart to what subword tokenization produces for text, except they're continuous embeddings produced by a linear layer rather than discrete IDs from a fixed vocabulary. Position information is added via 2D rotary embeddings (Qwen2.5-VL's choice) or learned 2D positional embeddings (older ViTs).

**Vision encoder (ViT).** The patch embeddings pass through a Vision Transformer: a stack of self-attention + MLP layers. Attention here is **bidirectional / full** — every patch can attend to every other patch. This is what lets the ViT do the heavy semantic work of *"this patch is the corner of Pepper's hat, this patch is sky, this patch is Carrot's ear"* by aggregating information across the whole image. Qwen2.5-VL also applies 2×2 patch merging at this stage, which roughly quarters the sequence length without losing much fidelity. Output: a sequence of contextualized visual embeddings, ~256 of them for a 448×448 input.

**Projector (the bridge).** The ViT outputs live in `d_vision`. The LLM's text embeddings live in `d_llm` (e.g. 3584 for Qwen2.5-7B). A small MLP — typically two linear layers with a GELU between them — maps each visual embedding from `d_vision` → `d_llm`. After this projection, **visual embeddings are dimensionally and semantically interchangeable with the LLM's text token embeddings.** This MLP is the "bolt on the front" in literal terms; it's also the cheapest part of the stack and usually the only part trained from scratch when assembling a new VLM.

### The text side

The text prompt (*"Describe this page"*) goes through the LLM's existing BPE tokenizer into integer token IDs (*"De"*, *"scribe"*, *" this"*, *" page"*), and each ID is looked up in the LLM's token embedding table to produce a `d_llm`-dimensional vector. Standard LLM input pipeline, unchanged. Because the visual embeddings (from the projector) and the text embeddings (from this step) live in the same vector space, they can be concatenated into a single input sequence.

### Sequence construction and generation

The visual and text embeddings are concatenated into one sequence, with special marker tokens around the image region. In Qwen2.5-VL the layout looks roughly like:

```
[<|im_start|>][user][<|vision_start|>][v_1][v_2]...[v_256][<|vision_end|>][De][scribe][ this][ page][<|im_end|>][<|im_start|>][assistant]
```

From the LLM's perspective, this is just one input sequence of `d_llm`-dimensional vectors. The LLM does not know or care which positions came from pixels and which came from words — both arrive as embeddings in the same space.

This unified sequence is fed into the LLM — **one stack of decoder-only transformer layers with causal self-attention**. The model computes attention over the whole input, then samples the next token from the output distribution at the final position. That token is appended to the sequence, the model is run again, the next token is sampled, and so on, until an end-of-sequence token is emitted. The generated token IDs are then run back through the tokenizer (*detokenization*) to produce the human-readable description string that comes back over Ollama's HTTP API.

### Why decoder-only is enough

This is the part worth understanding properly, because it isn't obvious why *"the model that generates text one token at a time"* can also *"understand an image."*

There are three transformer flavors:

1. **Encoder-only** (BERT). Bidirectional self-attention. Good at *understanding* a fixed input. Bad at *generating* sequences of arbitrary length.
2. **Decoder-only** (GPT, Llama, Qwen). Causal self-attention — each position can only attend to earlier positions. Inherently autoregressive: generation is just "predict the next position." All modern frontier LLMs are this.
3. **Encoder-decoder** (the original 2017 Transformer, T5). Encoder bidirectionally processes the input; decoder generates the output and cross-attends back to the encoder's representation. Classical seq2seq machine translation used this.

The naive intuition for VLMs would be: *"I need to understand the image, so I need an encoder. Then I need to generate a description, so I need a decoder. So a VLM should be encoder-decoder."* And that **is** a valid design — early VLMs like BLIP-2 (Q-Former + Flan-T5) worked exactly this way.

But Qwen2.5-VL, LLaVA, and most modern VLMs collapse this into just a ViT plus a decoder-only LLM. The reason it works:

**The ViT *is* the bidirectional encoder, just for the image modality.** By the time visual embeddings reach the LLM, the ViT has already done the bidirectional understanding pass over the image. Each visual embedding has already aggregated information from every other patch via the ViT's full attention. The LLM doesn't need to re-encode them with bidirectional attention; that work is done before the LLM ever sees the sequence.

From the LLM's perspective, the image embeddings are just a chunk of **prefix context** — exactly like a system prompt or a long document the user pasted in. Causal attention is sufficient for using prefix context, because:

- When the LLM processes the prompt tokens (*"Describe this page"*), each prompt position attends back to all the earlier image positions. Causal mask doesn't block this — earlier positions are always visible to later ones.
- When the LLM generates output tokens one at a time, each newly-generated token attends back to all of: image embeddings, prompt embeddings, and previously-generated output tokens.

The only thing causal attention forbids is *future* positions being visible to *past* positions — and the image is always in the past relative to whatever's being generated. The constraint never bites.

**The deeper reason this is elegant:** VLM construction becomes mostly composition rather than re-architecting. Take a pre-trained ViT (the vision community has lots — CLIP-ViT, SigLIP, etc.). Take a pre-trained decoder-only LLM (a huge investment in itself). Wedge a small MLP projector between them. Train mostly the projector, plus a lighter fine-tune of the LLM on vision-instruction data, to align the two. You get a VLM without retraining either of the two giant components from scratch. This is roughly how Qwen2.5-VL, LLaVA, and most current open-weights VLMs are built — and it's why the field has been able to ship competitive VLMs so quickly on top of existing text-only LLMs.

---

Next up: **Post 5 — From Database to Browser: A REST API and a Real Flipbook.** With one episode sitting in Postgres + ChromaDB + LocalStorage, it's time to surface it. We'll wire up the first FastAPI route handlers (episodes list, page detail), build the React + [StPageFlip](https://github.com/Nodlik/StPageFlip) flipbook component, and connect the two with a typed API client. By the end of that post you'll have an episode picker plus a real page-flipping flipbook rendering real data from your local backend.

The **workshop starter** that backs this post is at <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop> — pull the latest to pick up `ingestion/ingest.py` and the `ingest-from-images` skill. The **full source repository** and a public live-demo URL go up alongside the final post of this series — the deploy guide — once it's published.

*Pepper & Carrot* is © [David Revoy](https://www.davidrevoy.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All credit to him for the source material that made this project possible.

**All opinions expressed are my own.**
