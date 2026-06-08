---
title: "Pepper & Carrot AI-powered flipbook · Part 5 of 16 — Claude Skills as a Vision Provider: Ingesting a Comic by Reading It"
date: 2026-05-16 00:00:00 -0800
categories: [Full-Stack, RAG, Local AI]
tags: [claude-skills, claude-code, vision, peppercarrot, ingestion, postgres, chromadb, python, portfolio]
description: >-
  Post 5 of the Pepper & Carrot AI flipbook series. The comic is images,
  not text — so before any RAG can happen, every page needs a description.
  This post is about who writes those descriptions: a tour of the three
  vision-provider options (local VLM, hosted API, Claude Code itself), why
  the Claude Code path wins for a portfolio project, and a section-by-section
  walk through the `ingest-from-images` skill that produces a JSON
  description per page. The right vision provider is context-specific, and
  the post includes a decision matrix mapping each constraint to the right
  choice — plus an appendix on how a VLM actually sees an image.
pin: true
---

Post 5 of the [*Pepper & Carrot AI-powered flipbook*]({% post_url 2026-05-09-pepper-carrot-companion-trailer %}) series. With the workshop standing from [Post 2]({% post_url 2026-05-10-pepper-carrot-companion-workshop %}) and the four `Protocol`-typed seams from [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}), it's time to put real data behind them. The comic is images, not text, so before RAG can answer questions about a page, every page needs a description. This post is about *who writes those descriptions*: the three vision-provider options, why a Claude Code skill is the right one here, and how that skill is built.

> **What you'll build in this post.**
> - A working decision frame for the three vision-provider options — local VLM, hosted vision API, and Claude Code itself — and the constraints that pick between them.
> - The `ingest-from-images` [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills) installed at `.claude/skills/ingest-from-images/`, working as the vision provider for the offline pipeline.
> - One episode's worth of `PageDescription` JSON files written next to the page images — the durable artifact the next post's pipeline consumes.
> - A working mental model for what a Claude Code skill is, why this beats running a vision model in production, and a feel for how a VLM actually turns pixels into text.
>
> **Prerequisites.**
> - The workshop starter from [Posts 2]({% post_url 2026-05-10-pepper-carrot-companion-workshop %})–[4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}), with `docker compose up -d`, `alembic upgrade head`, and `seed.py` already run.
> - Episode 1 already downloaded into `data/raw/ep01-potion-of-flight/` via `acquire.py` (the Post 2 acquisition step).
> - A [Claude Code](https://docs.claude.com/en/docs/claude-code) install on a subscription that includes Claude usage (Pro / Max / Team work).

> **About the repo URL.** The code in this post — the `ingest-from-images` skill at `.claude/skills/ingest-from-images/` and the `JsonFileVisionClient` at `backend/app/clients/vision.py` — lives in the same workshop starter that backed [Posts 2]({% post_url 2026-05-10-pepper-carrot-companion-workshop %})–[4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}): <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop>. Pull the latest to pick up the ingestion additions. The full project repository — frontend, chat orchestrator, world-graph overlay, cloud deploy — lands alongside the deploy guide near the end of the series.
>
> **Checking out the code.** The ingestion work spans two posts that share one checkpoint: `git checkout post-05-06-ingestion` gives you a complete, working tree for both the `ingest-from-images` skill (this post) and the Stage-2 pipeline ([Post 6 — The Ingestion Pipeline]({% post_url 2026-05-17-pepper-carrot-companion-ingestion-pipeline %})). Each later post then adds its own tag (`post-07-08-fullstack`, `post-09-rag`, …) — see the README's [Following along with the blog series](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop#following-along-with-the-blog-series).

---

## Table of Contents

1. [Ingestion Is a Vision Problem](#vision-problem)
2. [Three Vision-Provider Options](#three-options)
3. [What's a Claude Code Skill?](#what-is-a-skill)
4. [The `ingest-from-images` Skill, Walked Through](#the-skill)
5. [Appendix: How a VLM Actually Sees an Image](#appendix-vlm)

---

## Ingestion Is a Vision Problem {#vision-problem}

Most "chat with X" tutorials skip a layer that's the load-bearing part of *this* project. They start by parsing a PDF, scraping HTML, or pulling text out of a document store, and then go straight into chunking, embedding, retrieval. Their X is already text. The interesting work is downstream of *"some text is sitting on disk."*

A webcomic isn't. The pages of *Pepper & Carrot* are hand-painted watercolor images. There is no native text to chunk. There is no caption layer. There is no markup. There is a `page_001.jpg` on disk and there is a reader who wants to ask the AI *"why is Carrot angry?"*

Today's multimodal LLMs can, in fact, look at that page and answer the question directly: hand the JPEG to Claude or GPT-4o every turn and you could skip ingestion entirely. But the chat layer almost never has the luxury of reasoning about *just* the current page. *"Why is Carrot angry?"* usually needs the prior pages of the episode for narrative context (*"Pepper took the last carrot on page 4"*). *"What's a Komona witch?"* needs a wiki article retrieved from across the whole corpus. So every turn drags in multi-page context, and pulling that context in *fresh, as images* is the part that doesn't scale: expensive on hosted vision APIs, and slow on the local Ollama path even with multimodal models like `llava` or `qwen2-vl`. For the wiki side, text embeddings (bge-m3) are also cheap and semantically sharper on narrative (*"the moment Pepper realizes the potion was a trick"*) than multimodal embeddings are yet, which makes corpus-wide retrieval feasible at all. And a prose description written once is inspectable, diffable, and seedable for tests in a way *"whatever the VLM saw this turn"* never is.

So before any of the RAG plumbing built in later posts can fire, every page gets collapsed to prose once, up front, and the rest of the stack stays text-native. That description is the document the chat layer will embed, search, and ground answers in.

This step is called **ingestion**, and the central question of this post is: who writes the description? The choice shapes everything downstream: the quality of chat answers, the cost of iterating on prompts, the operational reliability of the pipeline, even how easy it is to fix a description after the fact.

The whole post is about answering that question in a slightly unusual way: have **Claude Code itself** do the looking. It's the same model as Anthropic's hosted vision API, with no metered per-call cost (the ingestion work is subsumed under the Claude Code subscription), and JSON artifacts on disk you can hand-edit. The architecture from [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) makes this possible without any special-casing: the rest of the pipeline still talks to a `VisionClient` Protocol, same shape as if a real model client sat behind it.

---

## Three Vision-Provider Options {#three-options}

Before settling on the Claude-Code-as-vision-provider path, the project actually built and evaluated the two more conventional options. The choice between them is context-specific. There's no universal "best" vision provider; the right one for any given project depends on a small set of constraints (per-call budget, whether the pipeline needs to run unattended on a schedule with no human present, iteration speed, throughput requirements, privacy posture). Pepper & Carrot is a portfolio / demo project, so its constraints are very particular: no per-call API budget, no unattended-automation requirement, and free-at-the-margin prompt iteration as the dominant priority. Under those constraints, the third option wins. A different project — say, a paid product ingesting hundreds of comics nightly under a hosted-vision-API contract — would land somewhere else.

The point of walking through all three isn't to anoint Option 3 as universally best. It's to give you the decision frame so you can pick the right one for *your* project.

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

Not every page comes out this way (some come back as clean prose) but enough do that the output is unreliable. The training-data distribution for "describe this image" pulls the model toward the structured panel-by-panel format, and prompt instructions only partially override it, with no way to predict from the input which way a given page will go. Worse: when this text gets retrieved by the chat layer (built in a later post) and fed to *another* small local model for synthesis, that model mirrors the same `**bold-label:**` shape in its replies, so the user sees wiki-style answers instead of conversational prose. The leak isn't cosmetic; it propagates.

**GGML crashes mid-run.** Specific page images would trigger a hard assertion failure inside [llama.cpp](https://github.com/ggml-org/llama.cpp)'s vision pipeline:

```
GGML_ASSERT(a->ne[2] * 4 == b->ne[0]) failed
```

(Tracked in [`ollama/ollama#15828`](https://github.com/ollama/ollama/issues/15828). The same assertion hits other vision models too — e.g. [`glm-ocr`](https://github.com/ollama/ollama/issues/14171) — so it's a llama.cpp vision-pipeline issue surfaced via Ollama, not a qwen-specific bug.) The entire run dies, mid-episode. Rerunning might succeed, might fail differently, might fail at a different page. There is no application-layer fix. Either the underlying tensor shapes align or they don't, and the error message is opaque from a Python caller's perspective.

**Latency.** A nine-page episode through `qwen2.5vl:7b` on Apple Silicon took ~10 minutes per run, sometimes longer under memory pressure or on cold start, roughly a minute per page after the model warmed. Iterating on the description prompt meant 10+ minutes between each change, which adds up across a session of prompt-tuning.

### Option 2 — Hosted vision API (Anthropic)

Same wire shape, different endpoint. `AnthropicVisionClient` base64-encodes images, calls the [Anthropic Messages API](https://docs.claude.com/en/api/messages) with image content blocks, parses the response. The model behind it (Claude Opus) is, as of writing, the best vision model on the market for this kind of task.

> *Plain-English aside: what's "base64-encoding" an image?* The Anthropic Messages API speaks JSON, and JSON strings can't safely carry the raw bytes of a JPEG — null bytes and control characters would either break the parser or get mangled in transit. **Base64** is a way of representing arbitrary binary data using only 64 printable ASCII characters (`A–Z`, `a–z`, `0–9`, `+`, `/`); every 3 bytes become 4 base64 characters, about 33% bigger than the original. So `AnthropicVisionClient` reads the image as bytes, runs `base64.b64encode(...)` on them, and drops the resulting ASCII string (something like `/9j/4AAQSkZJRg...`) into the JSON body inside an `{"type": "image", "source": {"type": "base64", ...}}` content block. Anthropic's server decodes it back to bytes on its end and feeds those into the vision encoder. A 200 KB comic page becomes about a 270 KB string in the request body.

This fixed everything from Option 1. No GGML bugs. Clean prose output. ~5 seconds per page. The descriptions were genuinely good.

The one downside: roughly $30–$45 per full re-ingest of a ~40-episode corpus at Opus pricing, about $0.08–$0.12 per page, given comic-page-sized images, a system prompt with the cast list, and a few hundred output tokens per description. That sounds cheap, and per run it is, until you realise you're going to want to re-ingest dozens of times while iterating on the description prompt. After ten iterations you've spent a few hundred dollars testing a prompt that mostly didn't change. (Dropping to Sonnet roughly fifths the bill with some quality trade-off; Haiku is cheaper still.) Not catastrophic, but a real friction tax on the loop.

### Option 3 — Claude Code itself as the vision provider

Then a third option emerged. Claude Code's `Read` tool can read image files directly, and the model behind it is the same Claude Opus model that powers the Anthropic vision API. The marginal cost is zero: it's folded into the Claude Code subscription you're already paying for. The quality is identical.

The shape of the solution: instead of writing an `OllamaVisionClient` or `AnthropicVisionClient` that the ingestion script calls at runtime, you write a **[Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills)** — a project-local file that auto-loads into a Claude Code session — that instructs Claude to:

1. Read each page image with the `Read` tool.
2. Compose a `PageDescription` matching a strict schema.
3. Write the JSON to disk next to the image with the `Write` tool.
4. Run the Python pipeline that consumes those JSONs.

The Python pipeline then talks to a `JsonFileVisionClient`, the fourth and last implementation of the four `Protocol`-typed seams introduced in [Post 4]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}). (Post 4 landed `Storage`, `EmbeddingClient`, and `ChatClient`; `VisionClient` was named but left for this post.) The class does nothing model-y itself: it just reads the sibling JSON file the skill wrote and validates it against the `PageDescription` schema. The rest of the architecture is unchanged. The pipeline still goes through the `VisionClient` Protocol; the model is just outside the Python process this time, doing its work *before* the script runs. The actual implementation is short enough to read end-to-end, and we walk through it in the next post, [Post 6 — The Ingestion Pipeline]({% post_url 2026-05-17-pepper-carrot-companion-ingestion-pipeline %}).

Three options, side-by-side:

| Provider | Quality | Cost per re-ingest | Failure modes | Time for 9 pages |
|---|---|---|---|---|
| Local VLM (`qwen2.5vl:7b`) | Markdown leakage; structure propagates downstream | $0 | GGML assertion crashes | ~10 min |
| Hosted API (Anthropic vision) | Excellent | ~$30–$45 (Opus, full corpus) | Network / rate limits | ~45 sec |
| **Claude Code as vision provider** | **Excellent** (same model) | **$0** *(subscription)* | None observed | ~5–15 min interactive |

For this project's constraints — portfolio scope, no unattended-automation requirement, free iteration as the dominant priority — the Claude Code path is the right call. No metered per-call cost (subsumed under the Claude Code subscription you're already paying for), high-quality descriptions, and JSONs on disk that you can hand-edit. But it's not a universal winner. The moment your project needs ingestion that runs automatically on a schedule (a nightly cron, a webhook fire, anything with no human in the loop), or you're ingesting at a volume where one Claude Code session per episode doesn't fit your workflow, the hosted-API option is the right call: same quality, you just pay for it. And if budget is tight *and* privacy posture requires keeping pixels on your own infrastructure, modern open-weights VLMs (e.g. [Qwen3-VL](https://qwenlm.github.io/blog/qwen3-vl/)) have closed enough of the markdown-leakage gap that the local-VLM option becomes viable again, trading latency and GPU cost for the privacy.

The architecture supports any of the three. Flipping `VISION_PROVIDER` in `.env` (plus rebuilding the corresponding client) is the only code-level difference. The decision is about constraints, not about code.

The rest of this post walks through how the skill is built and what the pipeline does on the back of it. But read it as a *case study* of one solution that fit one project's constraints, not as a prescription.

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

Why this is useful for ingestion specifically: the skill is the piece of context Claude needs to do the task correctly. Without it, asking Claude *"describe the pages of episode 1"* would get you Claude's best guess at what you want — probably a single paragraph per page, no structured fields, no canonical character names, no validation. With the skill loaded, Claude knows exactly what JSON shape to produce, which fields are required, which character names to use, where to write the files, and what script to run when it's done. The skill replaces the prompt-engineering work you'd otherwise re-do every time.

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

- The **`description` is long on purpose.** It's not just documentation for humans; it's the prompt-matching text Claude uses to decide whether to load this skill. The more trigger phrases you list, the more reliably the skill activates on natural-language requests. *"ingest episode 1"*, *"re-describe pages for ep07"*, *"set up episode 14 for the chat"* — all should match. Skill descriptions that are too short tend to under-activate.
- **`allowed-tools` is the security boundary.** This skill needs `Read` (to view images), `Write` and `Edit` (to produce JSON files), `Bash` (to run the validation and ingestion scripts), and `Glob` (to list page files). It does not need `WebFetch`, `WebSearch`, or any of the MCP tools, so they aren't granted. If the skill tries to invoke one of those, Claude refuses. This is the "principle of least privilege" applied to AI-driven workflows.

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

This is critical. The schema's `characters_present` field has to contain **canonical names** from the seeded `characters` table, not Claude's guess at a character name. *"the young witch"* should resolve to `"Pepper"`. *"the cat"* to `"Carrot"`. *"the bird"* to `"Mango"`. The skill pulls the cast list at the top of each run so Claude has it in working memory while describing pages.

> *Plain-English aside: why does this matter so much?* The chat layer (covered in a later post) joins page descriptions back to canonical characters via `page_characters`. That's what powers the "next time Pepper appears" navigation feature and lets the chat anchor "Pepper" to a specific known entity rather than re-inferring her existence on every turn. If Claude writes `"the witch"` instead of `"Pepper"` in `characters_present`, the join fails silently and that page never shows up in any character-anchored query. Pulling the cast list at the start fixes this *structurally*: Claude knows the only acceptable strings up front.

**Step 2 — List the pages.**

```bash
ls data/raw/<slug>/pages/page_*.jpg
```

Note the count, then process the pages in order: narrative continuity from earlier pages carries forward into the descriptions of later ones because Claude is describing them all in the same conversation. (The `PageDescription` schema doesn't pass previous-page context between calls — `JsonFileVisionClient` ignores the `previous_page_description` argument from the Protocol contract entirely — but Claude itself has continuity because every page is described in the same Claude Code session.)

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

Each field has rules attached, and every constraint is load-bearing. A short tour:

- **`visual_description`: prose only.** No markdown headers, no bullet lists, no "Panel 1: ... Panel 2: ..." structure. The whole point of running this through Claude rather than `qwen2.5vl:7b` is to escape the structured-output gravity that the smaller model couldn't. If Claude slips into a structured format anyway (it sometimes does for very complex pages), re-prompt for prose.
- **`dialogue`: verbatim, one entry per bubble, in reading order.** Don't paraphrase: these strings end up retrievable in the embedding index, and paraphrase distorts what the model can recall. `speaker = null` for sound effects, narration, and unidentified speakers.
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

> *Plain-English aside: what's `trap … EXIT`?* In bash, [`trap`](https://www.gnu.org/software/bash/manual/html_node/Signals.html) registers a function to run when a specific event happens. `trap restore EXIT` says *"no matter how this script ends — clean finish, exception, Ctrl-C, the OS sending a kill signal — run the `restore` function before exiting."* It's the bash equivalent of Python's `try / finally`: it guarantees cleanup. Here it guarantees that `VISION_PROVIDER` in `.env` gets put back to whatever it was before the script started, even if the ingestion script crashes halfway through. Without this, a failed run could leave `.env` in a weird state and future runs would behave unexpectedly.

**Step 6 — Verify the ingestion landed.**

After the script finishes, the skill makes Claude run two psql queries: one confirming the episode row has a `plot_summary` and the right page count, and one listing per-page `(page_number, characters)` so any "unknown character" warnings get caught right then.

```bash
docker exec peppercarrot-postgres psql -U peppercarrot -d peppercarrot -c \
  "SELECT episode_number, slug, plot_summary IS NOT NULL AS has_summary,
   (SELECT COUNT(*) FROM pages WHERE episode_id = e.id) AS page_count
   FROM episodes e WHERE slug = 'ep01-potion-of-flight';"
```

Expect `has_summary = t` and `page_count` matching the number of JSONs.

That's the whole skill: ~170 lines of Markdown plus a ~50-line wrapper script. Modest surface area, hidden in plain sight, doing the highest-leverage step in the entire ingestion stack.

---

## Appendix: How a VLM Actually Sees an Image {#appendix-vlm}

The *tokenized + encoded + decoded* gloss in [Option 1](#three-options) compresses about six distinct steps into three words. For readers who want the actual mechanics, here is the full path an image takes through a model like Qwen2.5-VL between the moment it arrives at Ollama's HTTP API and the moment a description comes back out.

### The image side — from pixels to embeddings

**Patchification.** The raw image (say 1024×1024 RGB) is resized and normalized, then sliced into a regular grid of non-overlapping **patches**, typically 14×14 pixels each. A 448×448 image at 14px patches → 32×32 = 1024 patches. Each patch is flattened (14×14×3 = 588 numbers) and run through a single linear projection that maps it to a vector in some `d_vision` dimension (e.g. 1024). These vectors are *patch embeddings*: the image-side counterpart to what subword tokenization produces for text, except they're continuous embeddings produced by a linear layer rather than discrete IDs from a fixed vocabulary. Position information is added via 2D rotary embeddings (Qwen2.5-VL's choice) or learned 2D positional embeddings (older ViTs).

**Vision encoder (ViT).** The patch embeddings pass through a Vision Transformer: a stack of self-attention + MLP layers. Attention here is **bidirectional / full** — every patch can attend to every other patch. This is what lets the ViT do the heavy semantic work of *"this patch is the corner of Pepper's hat, this patch is sky, this patch is Carrot's ear"* by aggregating information across the whole image. Qwen2.5-VL also applies 2×2 patch merging at this stage, which roughly quarters the sequence length without losing much fidelity. Output: a sequence of contextualized visual embeddings, ~256 of them for a 448×448 input.

**Projector (the bridge).** The ViT outputs live in `d_vision`. The LLM's text embeddings live in `d_llm` (e.g. 3584 for Qwen2.5-7B). A small MLP — typically two linear layers with a GELU between them — maps each visual embedding from `d_vision` → `d_llm`. After this projection, visual embeddings are dimensionally and semantically interchangeable with the LLM's text token embeddings. This MLP is the "bolt on the front" in literal terms; it's also the cheapest part of the stack and usually the only part trained from scratch when assembling a new VLM.

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

The naive intuition for VLMs would be: *"I need to understand the image, so I need an encoder. Then I need to generate a description, so I need a decoder. So a VLM should be encoder-decoder."* And that *is* a valid design — early VLMs like BLIP-2 (Q-Former + Flan-T5) worked exactly this way.

But Qwen2.5-VL, LLaVA, and most modern VLMs collapse this into just a ViT plus a decoder-only LLM. The reason it works:

**The ViT *is* the bidirectional encoder, just for the image modality.** By the time visual embeddings reach the LLM, the ViT has already done the bidirectional understanding pass over the image. Each visual embedding has already aggregated information from every other patch via the ViT's full attention. The LLM doesn't need to re-encode them with bidirectional attention; that work is done before the LLM ever sees the sequence.

From the LLM's perspective, the image embeddings are just a chunk of **prefix context** — exactly like a system prompt or a long document the user pasted in. Causal attention is sufficient for using prefix context, because:

- When the LLM processes the prompt tokens (*"Describe this page"*), each prompt position attends back to all the earlier image positions. Causal mask doesn't block this — earlier positions are always visible to later ones.
- When the LLM generates output tokens one at a time, each newly-generated token attends back to all of: image embeddings, prompt embeddings, and previously-generated output tokens.

The only thing causal attention forbids is *future* positions being visible to *past* positions — and the image is always in the past relative to whatever's being generated. The constraint never bites.

**The deeper reason this is elegant:** VLM construction becomes mostly composition rather than re-architecting. Take a pre-trained ViT (the vision community has lots — CLIP-ViT, SigLIP, etc.). Take a pre-trained decoder-only LLM (a huge investment in itself). Wedge a small MLP projector between them. Train mostly the projector, plus a lighter fine-tune of the LLM on vision-instruction data, to align the two. You get a VLM without retraining either of the two giant components from scratch. This is roughly how Qwen2.5-VL, LLaVA, and most current open-weights VLMs are built, and it's why the field has been able to ship competitive VLMs so quickly on top of existing text-only LLMs.

---

Next up: **Post 6 — The Ingestion Pipeline: From Page JSONs to Postgres + Chroma.** The skill above writes one JSON description per page; the next post is the Stage-2 Python pipeline that consumes those JSONs, generating image variants with Pillow, uploading through the storage abstraction, upserting `pages` rows and `page_characters` links, writing a per-episode plot summary, and embedding every chunk into ChromaDB's `pages_v1` collection. By the end of it you'll have one full episode ingested end-to-end.

The **workshop starter** that backs this post is at <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop>, tagged `post-05-06-ingestion` — the same checkpoint the Post 6 pipeline post uses. The **full source repository** and a public live-demo URL go up alongside the deploy guide near the end of the series — once it's published.

*Pepper & Carrot* is © [David Revoy](https://www.davidrevoy.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All credit to him for the source material that made this project possible.

**All opinions expressed are my own.**
