---
title: "Making Small Models Behave: Wiki Mode and the Long Road to Concise Answers"
date: 2026-05-30 00:00:00 -0800
categories: [Full-Stack, RAG, Local AI]
tags: [prompt-engineering, qwen, ollama, react-markdown, suggestion-chips, rag, fastapi, peppercarrot, portfolio]
description: >-
  Post 8 of the Pepper & Carrot AI flipbook series. Post 7 left a streaming
  chat panel and an honest admission: the engineering guarantees structure
  and safety, but it doesn't guarantee taste. This post is the
  prompt-engineering pass that closes that gap on a 7B local model — a
  markdown stripper on every piece of text entering the prompt, a
  closed-world grounding contract, a page-mode anti-recitation block, a
  strict response-format cap, a much sharper suggestion-chip prompt with
  bad/good examples, and react-markdown in the chat panel as the
  belt-and-suspenders safety net.
pin: true
---

Post 8 of the [*Pepper & Carrot AI-powered flipbook*]({% post_url 2026-05-09-pepper-carrot-companion-trailer %}) series. [Post 7]({% post_url 2026-05-25-pepper-carrot-companion-streaming-chat %}) ended with an honest admission: streaming worked, retrieval was spoiler-safe, the suggestion chips were structurally sound — and the actual *replies* were still occasionally rough. A 7B local model under pressure invents details, mirrors `### headers` from its notes, treats prepared context as user-provided input and politely asks the user for "more information," and emits a wiki chip that's really a page question wearing a wiki label. None of that is a bug in the plumbing; it's a gap between *the engineering* and *the model's taste*. This post is the prompt-engineering pass that closes most of it — a strict response-format contract, a closed-world grounding rule, a page-mode anti-recitation block, a markdown stripper on every piece of text on its way into the prompt, a much sharper suggestion-chip prompt with concrete bad/good examples, and `react-markdown` in the chat panel as the safety net for the rest.

> **What you'll build in this post.**
> - A `_strip_markdown` helper in `backend/app/orchestration/chat.py` and apply it to **every** piece of text on its way into the prompt — `episode.plot_summary`, `page.visual_description`, `page.ocr_text`, each retrieved page's description, each retrieved wiki article body. Small chat models mirror whatever formatting they see; remove the markers at the source.
> - Four composable prompt blocks in `backend/app/core/prompts.py` — `_SPOILER_DISCIPLINE`, `_GROUNDING_CONTRACT`, `_PAGE_ANTI_RECITATION`, `_RESPONSE_FORMAT` — re-mixed into a sharper `PAGE_MODE_SYSTEM` and `WIKI_MODE_SYSTEM`.
> - A much longer `SUGGESTIONS_SYSTEM` with explicit BAD-CHIP and GOOD-CHIP examples — the abstract rules from Post 7 get ignored by 7B models; the worked examples don't.
> - `react-markdown` + `remark-gfm` in `frontend/src/components/ChatPanel.tsx` as a last-line safety net for any markdown that *does* leak past the system prompt, so a stray `### header` reads as bubble content rather than ugly raw text.
> - Three new unit tests in `backend/tests/test_chat.py` that pin `_strip_markdown` against the most common Markdown shapes we feed it.
>
> **Prerequisites.**
> - The workshop starter at the [`post-8` tag](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/tree/post-8): `git checkout post-8` (see [Following along with the blog series](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop#following-along-with-the-blog-series)). Everything [Post 7]({% post_url 2026-05-25-pepper-carrot-companion-streaming-chat %}) needed — Postgres up, migrations applied, the roster seeded, Episode 1 ingested, the wiki seed ingested, Ollama running with `qwen2.5:7b` and `bge-m3` pulled.
> - [Node.js 20+](https://nodejs.org/) for the Vite frontend, exactly as in [Post 5]({% post_url 2026-05-23-pepper-carrot-companion-rest-api-flipbook %}).

> **About the repo URL.** All the changes in this post — `core/prompts.py`, the `_strip_markdown` helper and its call sites in `orchestration/chat.py`, the new tests, the `react-markdown` dependency, and the chat panel's markdown rendering — live in the same workshop starter that backed [Posts 2–7](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop), now tagged `post-8`. File links below point at that tag. The full project repository (world-graph overlay, cloud deploy) goes up alongside the deploy guide in Post 10.

---

## Table of Contents

1. [The Code in Front of You: Tour + Quick Start](#tour)
2. [What "Small Model" Actually Means Here](#small-model)
3. [The Thesis: Structure Carries Safety, Prompts Carry Taste](#thesis)
4. [Tightener 1 — Strip Markdown From Everything That Enters the Prompt](#strip-markdown)
5. [Tightener 2 — The Grounding Contract](#grounding)
6. [Tightener 3 — Page-Mode Anti-Recitation](#anti-recitation)
7. [Tightener 4 — The Strict Response-Format Block](#response-format)
8. [Wiki Mode: The Long Road to Concise Answers](#wiki-concise)
9. [Suggestion Chips: A Schema Can't Carry Taste Either](#chips)
10. [The Markdown Safety Net in the Chat Panel](#safety-net)
11. [The Prompt, End to End](#diagram)
12. [The Honest Part: What Prompts Still Can't Fix](#honest)
13. [Key Takeaways](#key-takeaways)

---

## The Code in Front of You: Tour + Quick Start {#tour}

Before any concepts, let's get the hardened pipeline in front of you and orient around what changed. Skim this even if you read the rest carefully — watching the *same question* go from rough to clean once is worth more than the rest of the post combined.

### Get the code at this post's tag

Every file referenced below lives at the **`post-8`** tag:

```bash
git clone https://github.com/bearbearyu1223/pepper-carrot-companion-workshop
cd pepper-carrot-companion-workshop
git checkout post-8
```

Already cloned from an earlier post? `git fetch --tags && git checkout post-8`. Each post from Post 5 onward has its own tag (`post-5`, `post-6`, `post-7`, `post-8`), and `git checkout main` returns you to the latest.

### What's new in the workshop starter

Three backend files changed, two frontend files changed, one test file grew. Nothing structural moved — the changes are concentrated where the model meets prose:

```
pepper-carrot-companion-workshop/
├── backend/app/
│   ├── core/prompts.py              ← rewritten: 4 composable blocks; sharper chips
│   └── orchestration/chat.py        ← + _strip_markdown applied at every prompt-bound site
├── backend/tests/test_chat.py       ← + 3 tests for _strip_markdown
└── frontend/
    ├── package.json                 ← + react-markdown + remark-gfm
    ├── src/components/ChatPanel.tsx ← assistant bubble now renders through ReactMarkdown
    └── src/styles/global.css        ← + .chat-markdown styles
```

That's it. **No new endpoints, no new schemas, no new routes.** Every change is a string change in `prompts.py`, a function call in `chat.py`, or six lines of JSX. The story this post tells — and the portfolio signal — is that the hardest part of a small-model RAG project is the part that looks the smallest in the diff.

### Run it: two terminals, then ask the same question twice

```bash
# Terminal 1 — backend on :8000 (Postgres + Ollama already up from earlier posts)
cd backend && uv sync && uv run uvicorn app.main:app --reload

# Terminal 2 — Vite dev server on :5173 (npm install picks up react-markdown)
cd frontend && npm install && npm run dev
```

Open <http://localhost:5173>, pick Episode 1, flip to page 3, and ask **"who is on this page and what are they doing?"** in page mode, then ask **"what is Chaosah?"** in wiki mode. The replies should arrive as **plain conversational prose**, **at most four sentences**, with **no `###` headers, no bullet lists, no "Certainly!" preamble, and no "In essence," wrap-up**. The suggestion chips that appear underneath should each be a **complete, specific question** — the wiki chip should reference something universe-flavored (a school, a place, a creature), not a page detail dressed up in wiki tags.

You don't have to take the demo's word for any of that. Here's the same exchange via `curl`, so you can see what's actually on the wire:

```bash
SID=$(curl -s -X POST localhost:8000/api/sessions -H 'content-type: application/json' \
  -d '{"episode_slug":"ep01-potion-of-flight"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
curl -s -X PATCH localhost:8000/api/sessions/$SID -H 'content-type: application/json' \
  -d '{"current_page":3}'

# Capture the `done` frame so you can inspect chips + audit trail in one shot.
curl -sN -X POST localhost:8000/api/sessions/$SID/messages -H 'content-type: application/json' \
  -d '{"mode":"page","message":"who is on this page and what are they doing?"}' \
  | grep '^data: {"message_id"' | sed 's/^data: //' | python3 -m json.tool
# {
#   "message_id": "…",
#   "retrieved_doc_ids": ["b97f…", "7a58…"],
#   "suggestions": [
#     {"mode": "page", "text": "Why is Carrot glowing in the third panel"},
#     {"mode": "wiki", "text": "What kind of magic is Chaosah known for"}
#   ]
# }
```

> *If you skipped Posts 6–7:* the `done` frame is the SSE event that closes a chat turn. `retrieved_doc_ids` is the audit trail of which Chroma chunks grounded the answer; `suggestions` is the two follow-up chips. The token-by-token text streams in `event: token` frames before it (`curl -N` lets you watch them live).

### What's *not* in the diff is the point

If you've followed Posts 6–7, the most striking thing about Post 8 is what *doesn't* change. The Chroma `where` clause is byte-for-byte the same. The `EventSourceResponse` framing is the same. The named-slot JSON schema for chips is the same. The retrieval boundary, the spoiler filter, the SSE wire format — all untouched.

**Every change in this post lives where the model meets prose.** Retrieval doesn't get safer; the model's *answers* get cleaner. That separation — *the safety lives in the structure, the polish lives in the prompts* — is the spine of the post.

---

## What "Small Model" Actually Means Here {#small-model}

A short detour before we get to the tighteners, because the right prompt for a 7B local model is *different in kind* from the right prompt for a frontier API. If you've only ever used GPT-4 or Claude Sonnet, the rules below will feel paranoid.

> *Plain-English aside: what's a "7B" model?* The "7B" means **7 billion parameters** — the numbers the model adjusts during training to encode what it knows about language. For comparison, [Llama 3.1 405B](https://ai.meta.com/blog/meta-llama-3-1/) has fifty-seven times as many, and frontier models like Claude Sonnet are larger still (the exact count isn't published). More parameters means more capacity to store nuance, hold long context coherently, and follow soft instructions — *"be concise"* — without further prodding. A 7B model is at the bottom of the useful range: small enough to fit on a laptop's GPU or even run on a fast CPU, big enough to write coherent prose and follow *strict* instructions if you spell them out. Soft instructions, on the other hand, tend to bounce off.

We're on [`qwen2.5:7b`](https://qwenlm.github.io/blog/qwen2.5-llm/), served by [Ollama](https://ollama.com/), running on the same laptop as the backend. The choice was deliberate (cost: $0, latency: predictable, privacy: total), and it's the dominant constraint on every prompt we write. Three patterns matter:

- **Mirroring.** A 7B model will reproduce in the answer whatever shape it sees in the context. Notes written in markdown produce markdown answers; notes with bullet lists produce bullet-listed answers; notes phrased as "Background:", "Motivations:", "Personality:" produce essays with those exact subsections. The fix is structural, at the data layer: scrub the markers before the model sees them. That's [Tightener 1](#strip-markdown).
- **Parametric bleed-through.** *Pepper&Carrot* has been on the public internet since 2014, the comic and its wiki are CC-BY, and a 7B model trained on web text has absorbed *some* of it — usually fragments, often wrong. The model will confidently fill gaps in our notes with plausible-but-fabricated lore unless we explicitly tell it: *the notes are the only source of truth, ignore what you "know" from elsewhere*. That's [Tightener 2](#grounding).
- **Soft instructions get ignored.** *"Be concise"* doesn't work. *"Maximum 4 sentences. Hard cap. Stop after the answer."* works most of the time. *"Answer using the wiki context"* doesn't reliably stop the model from reciting a tour of the article; *"WIKI ANSWERS MUST BE TIGHT. Pick the ONE OR TWO facts that directly answer the question, and answer in 2-3 sentences"* with a concrete contrast example does. That's [Tightener 4](#response-format) and [§ Wiki Concise](#wiki-concise).

On a frontier API the soft versions of those instructions usually carry the day. On a 7B local model, you write the strict version, and you write it in shouting caps if that's what it takes. The point of using qwen2.5:7b for the demo isn't masochism — it's that **the *Pepper & Carrot* companion runs locally on the reader's laptop** ([ADR 0001](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-8/docs/decisions/0001-local-first-architecture.md)). When the model is your reader's hardware, you write prompts for *that* model, not the model you wish you had.

---

## The Thesis: Structure Carries Safety, Prompts Carry Taste {#thesis}

Two ideas split the work between Posts 6–7 and Post 8 cleanly. Both have appeared earlier in the series; this post leans on them together.

**Structural guarantees go in the data layer.** [Post 6]({% post_url 2026-05-25-pepper-carrot-companion-spoiler-safe-rag %}) made spoiler safety a property of the Chroma `where` clause, not a line in the prompt — because *"please don't spoil"* is a request a model can ignore, and a hardcoded filter is one it cannot. [Post 7]({% post_url 2026-05-25-pepper-carrot-companion-streaming-chat %}) extended the same instinct to the suggestion chips: a JSON Schema with two **named slots** (`page_chip` + `wiki_chip`) made "two page chips in a row" structurally impossible. Both decisions remove an entire class of failure from the model's hands by removing the affordance to fail that way.

**Taste — concision, register, the right level of specificity, the right facts to surface, the right *length* — can't be made structural.** It lives in the system prompt, in the few-shot examples, and (last-line) in the rendering layer. No `where` clause can make the model lead with the answer instead of "Certainly!"; no JSON schema can make a wiki chip about a witch school instead of a panel detail. Those land through prompt discipline: *what you write*, *how strictly you write it*, and *how concretely you ground it in worked examples* the model can pattern-match against.

That's the split, and it's the whole post. The structural guarantees from Posts 6–7 don't change — every chunk the orchestrator hands the model is still gated by reading position, every chip still occupies a typed slot, every assistant message is still validated server-side before reaching the DOM. Post 8 adds the *taste* layer on top, in four named blocks of prompt plus one helper function that scrubs markdown before any of it can leak into context.

The next four sections take the tighteners one at a time, in roughly the order they pay off if you're starting from a working-but-rough pipeline.

---

## Tightener 1 — Strip Markdown From Everything That Enters the Prompt {#strip-markdown}

This is the smallest tightener and probably the single biggest visible win. Local chat models *mirror their context*: if the notes have `**bold**`, the answer comes back with `**bold**`; if the notes are organized under `### Scene` / `### Characters` headers, the answer comes back with `### Scene` / `### Characters` headers. The right place to fix this isn't the answer; it's the input.

Two of our text sources are markdown-heavy at origin:

- **Page descriptions** are written by the [`ingest-from-images` Claude Code skill]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %}), which produces prose that often carries `**character names**` for emphasis and the occasional list. Auditable on disk, but markdown-flavored.
- **Wiki seed articles** ([`ingestion/wiki_seed.yaml`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-8/ingestion/wiki_seed.yaml)) are hand-written in flowing prose, but the upstream framagit wiki (which informs them) is dense with `## Section` headers and bullet lists, and that style bleeds into anything paraphrased from it.

A handful of regular expressions strips the markers and leaves the text:

```python
# backend/app/orchestration/chat.py
_MARKDOWN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\*\*([^*\n]+?)\*\*"), r"\1"),                # **bold**
    (re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)"), r"\1"),       # *italic*
    (re.compile(r"__([^_\n]+?)__"), r"\1"),                    # __bold__
    (re.compile(r"(?<!_)_([^_\n]+?)_(?!_)"), r"\1"),           # _italic_
    (re.compile(r"`([^`\n]+?)`"), r"\1"),                      # `code`
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),             # # headers
    (re.compile(r"^\s*[-*•]\s+", re.MULTILINE), ""),           # - bullets
    (re.compile(r"^\s*\d+\.\s+", re.MULTILINE), ""),           # 1. numbered
    (re.compile(r"^\s*>\s?", re.MULTILINE), ""),               # > blockquotes
    (re.compile(r"^\s*-{3,}\s*$", re.MULTILINE), ""),          # --- rules
    (re.compile(r"\n{3,}"), "\n\n"),                           # collapse blank runs
)


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting so the chat model sees plain prose."""
    for pattern, replacement in _MARKDOWN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()
```

> *Why not use a proper markdown library?* A real parser (`markdown-it-py`, `mistune`) handles edge cases these regexes don't — nested emphasis, tables, fenced code blocks. We don't need it. The inputs are short, the structures are simple, and the failure mode of an over-zealous regex (an asterisk that survives, a dash that doesn't) is invisible at the answer layer — the model still produces clean prose. The cost-benefit tipped toward "10 regexes in 20 lines" rather than "a parsing dependency for two callers."

The interesting part is the **discipline of where to apply it**: *everywhere text enters the prompt*. Forget one site and that one site reintroduces the mirroring. Five call sites cover it:

```python
# backend/app/orchestration/chat.py — every site that hands text to the prompt
description = _strip_markdown(page.visual_description or "")         # retrieved page
text_lookup[("wiki", str(article.id))] = (                          # retrieved wiki
    f"{article.title}\n\n{_strip_markdown(article.content)}"
)
# … and inside _render_page_block:
parts.append(_strip_markdown(episode.plot_summary))                  # episode summary
parts.append(_strip_markdown(page.visual_description))               # current page
parts.append(_strip_markdown(page.ocr_text))                         # dialogue OCR
```

Notice that the **article title is *not* stripped** — titles are short, never carry markdown, and going through the regex pass would be pure ceremony. The discipline is "strip what's likely to have markdown," not "strip everything." A type-checker won't tell you which is which; that's a judgement call you make per site.

And because this is the kind of helper that's easy to forget about and easy to break, [`test_chat.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-8/backend/tests/test_chat.py) pins the contract — inline markers like `**bold**` and `*italic*` go away, block markers like `## Scene` and `- bullets` go away, and plain prose passes through unchanged. Twelve regexes have a lot of room for a typo to slip in and silently swallow the wrong thing; the three tests below catch the obvious shapes:

```python
def test_strip_markdown_removes_inline_markers() -> None:
    assert _strip_markdown("**Pepper** and *Carrot*") == "Pepper and Carrot"
    assert _strip_markdown("__bold__ and _italic_") == "bold and italic"
    assert _strip_markdown("call `embed_batch()` first") == "call embed_batch() first"

def test_strip_markdown_removes_block_markers() -> None:
    src = "## Scene\nPepper brews a potion.\n\n- the cauldron bubbles\n…"
    out = _strip_markdown(src)
    assert "##" not in out
    assert "- " not in out
    assert "Pepper brews a potion." in out          # text content survives
```

Run them:

```bash
cd backend && uv run pytest tests/test_chat.py -v
# test_strip_markdown_removes_inline_markers PASSED
# test_strip_markdown_removes_block_markers PASSED
# test_strip_markdown_leaves_plain_prose_untouched PASSED
# 33 passed
```

> *Aside: this is exactly the kind of helper [Tightener 4](#response-format)'s "no markdown in the answer" rule depends on.* The system prompt asks for plain prose; the model usually obliges; when it doesn't, the markdown safety net in [§ The Markdown Safety Net](#safety-net) renders it readably. But the *first* line of defense is "never put markdown in the context to begin with" — because the model can ignore a rule, but it can't mirror what isn't there. Belt-*and*-suspenders, with this regex as the belt.

---

## Tightener 2 — The Grounding Contract {#grounding}

The second tightener fixes a failure mode that's easy to miss because the *answer still reads well*: the model invents lore. Pepper has been on the web for over a decade, and qwen2.5:7b has seen some of it during training. Ask it "who founded Chaosah?" with a wiki article in context that *doesn't* mention the founder, and a 7B model will sometimes pull a plausible-sounding name from its training residue rather than admit the wiki doesn't say. The answer sounds confident; it might even be partially right; but the moment the user fact-checks, the demo loses credibility.

The fix is to make the contract explicit in the system prompt — a *closed-world rule* the model is told to apply when it picks where to look:

```python
# backend/app/core/prompts.py
_GROUNDING_CONTRACT = """\

GROUNDING CONTRACT — read carefully:
The notes provided in the user message (the "Current spread", "Reference \
context", and "Wiki context" sections) are your ONLY source of truth about \
*Pepper&Carrot*. You may have encountered this comic during training; ignore \
what you "know" from elsewhere. If a character, place, event, or detail is \
not in the notes below, it does not exist for this answer — even if you are \
confident you've seen it before. Prior turns in this same conversation also \
count as grounded context: when the user refers back to something ("that \
witch you mentioned earlier"), look at the conversation history rather than \
guessing.

When the notes don't cover what the user asked, say so directly. Pick \
whichever of these shapes fits the situation, paraphrasing to keep the \
conversational tone:
  - "the page doesn't show that"
  - "the wiki doesn't cover this"
  - "I don't see that in the notes for this spread"
  - "the comic hasn't said yet"

Admitting the notes don't cover something is always better than filling the \
gap with a plausible-sounding invention. Inventing a detail to sound helpful \
is the worst possible failure mode here.
"""
```

> *Plain-English aside: closed-world vs open-world reasoning.* In **open-world** reasoning, anything not in your notes might still be true — you just don't know about it. In **closed-world** reasoning, anything not in your notes is *defined* not to be true *for this answer*. Most language models default to open-world (they're trained to be helpful, and "I don't know" feels unhelpful). For a RAG companion, you want closed-world: the retrieval layer is doing the choosing, and the model's job is to faithfully reflect *what was retrieved*, not to supplement it from memory. The `_GROUNDING_CONTRACT` block is what flips the default.

Two non-obvious details:

- **"Prior turns in this same conversation also count as grounded context."** This sentence was added after a real failure: the user asked a follow-up like "what's her favorite potion?" referring to a witch named in the previous turn, and the model — taking *"the notes are your only source of truth"* too literally — refused to recognize the back-reference. The fix is to say so explicitly. **Prompt rules are sticky; they don't gracefully handle "but also" cases unless you write the "but also" in.**
- **The list of acceptable "I don't know" shapes is a teaching example, not an enumeration.** Without it, 7B models default to either *"I don't have enough information"* (corporate-cold) or *"based on the descriptions provided, I cannot determine…"* (essay-cold). The four conversational shapes show the *register* you want, and the model interpolates from there. *Showing* the right register beats *telling* it.

This block is mode-neutral — both `PAGE_MODE_SYSTEM` and `WIKI_MODE_SYSTEM` compose it. It's the closest thing the prompts have to a constitution: *don't make stuff up; if the notes don't cover it, say so.* Every later block leans on it.

---

## Tightener 3 — Page-Mode Anti-Recitation {#anti-recitation}

This one is specific to page mode, and it fixes two related failure modes the grounding contract alone doesn't cover.

**Failure A: the model treats the prepared notes as user-provided input.** The prompt assembly hands the model a labelled user turn:

```
=== Current spread (pages 1 and 2, both visible to the reader) ===
-- Page 1 --
Characters on this page: Pepper
<the page's visual description>

=== Reference context (earlier pages you've already read) ===
(empty on page 1)

=== User question ===
who is on this page?
```

A naive 7B model reads the `Current spread` heading, sees that it was *provided* alongside the question, and concludes that the user typed the whole thing. It then evades the actual question with *"Could you specify which character you're asking about?"* or *"There are several details — could you clarify your question?"* — as if the notes were context the user wrote that *also* needed clarifying. The user, who can clearly see *one* witch in the panel, is now confused.

**Failure B: on a two-page spread, the model conflates "Current spread" with "Reference context."** The reader sees pages 1 and 2 side-by-side and asks "who's on this page?" — but `Reference context` is empty (it's page 1), and the model has been *trained* to use everything in context. So a model that doesn't know better will sometimes pad the answer with characters from the *Reference context* heading on a later spread, listing characters from earlier pages as candidates for "who's here."

Both failures get addressed by spelling out, in the system prompt, *what each labelled block actually is and isn't*:

```python
# backend/app/core/prompts.py
_PAGE_ANTI_RECITATION = """\

The user message contains "Current spread" (or "Current page") and possibly \
"Reference context" sections. These are facts prepared for you behind the \
scenes — your private notes about the page(s) the user is reading. The user \
did not write them. Treat them as authoritative, and answer the user's \
question directly using them.

CRUCIAL: the "Current spread" section and the "Reference context" section are \
SEPARATE and describe DIFFERENT things.
  - "Current spread" describes what is on the page(s) the user is reading \
RIGHT NOW. …
  - "Reference context" describes other pages from earlier in the comic — it \
is supporting background, NOT what is on the current spread.

When the user asks about "this page", "this spread", "here", "who's on this \
page", or names something shown ("the girl", "the witch", "the cat") — look \
ONLY at the "Current spread" section. …

Never evade by asking for clarification when the answer is in the notes. The \
following responses are FORBIDDEN:
  - "Could you specify which character/page/girl?"
  - "There are many characters mentioned"
  - "I don't have enough information"
  - "Please provide more context"

Don't refer to the notes as something the user gave you (no "your cast \
list", no "the info you shared", no "the information provided").
"""
```

Two non-obvious moves:

- **A forbidden-phrase list beats an abstract instruction.** *"Don't ask for clarification"* is a soft rule that bends under pressure. *"The following responses are FORBIDDEN: 'Could you specify…', 'I don't have enough information', 'Please provide more context'"* is a concrete list of strings the model can pattern-match against and avoid. Same effect on a frontier model; very different effect on a 7B.
- **The "don't refer to the notes as user-provided" rule** is what stops the model from saying things like *"From your cast list above, the character on this page is Pepper."* That phrasing is technically correct, but it tears the fourth wall of the demo — the user never *wrote* a cast list. Telling the model not to acknowledge its scaffolding keeps the answer feeling like a friend reading along, not like an LLM reading a brief.

This block is page-mode-only because wiki mode has a different shape (one or two articles, no "current spread") and a different failure mode (recitation, not evasion). Wiki mode gets its own discipline in [§ Wiki Concise](#wiki-concise).

---

## Tightener 4 — The Strict Response-Format Block {#response-format}

This is the loudest block in the prompt — and the one that does the most work in shifting *how* the model answers, irrespective of *what* it answers. The whole thing is shouted in caps and concrete bans because soft rules — *"keep it short"*, *"plain prose"*, *"avoid markdown"* — get ignored by 7B models. Strict rules with concrete forbidden phrases don't:

```python
# backend/app/core/prompts.py
_RESPONSE_FORMAT = """\

OUTPUT RULES — these are strict:
  - Maximum 4 sentences. Hard cap. Stop after the answer.
  - Plain conversational prose. No headers (no "###", "##", "**Heading:**"). \
No bullet points. No numbered lists. No bold or italic markdown.
  - No preamble. Do not start with "Certainly!", "Of course!", "Based on the \
descriptions provided", "Great question", or similar acknowledgements. Begin \
with the answer itself.
  - No wrap-ups. Do not end with "In essence,", "Overall,", "In summary,", \
"This shows that…" or similar essayistic closings.
  - No sub-sections breaking the answer into parts ("Background:", \
"Motivations:", "Personality:"). Answer in one continuous short paragraph.
  - Do not speculate or invent backstory. If the notes don't say something, \
either don't mention it or admit "the comic doesn't say". Phrases like "it's \
likely that…", "she may have…", "her background might involve…" are FORBIDDEN.

If the user explicitly asks for a deeper dive ("tell me more", "go into \
detail", "expand on that"), you may go up to ~8 sentences — still no headers, \
still no bullet lists, still grounded in the notes.
"""
```

Five non-obvious moves are buried in there:

- **A hard cap with a number.** *"Be concise"* doesn't work; *"Maximum 4 sentences. Hard cap."* does, because the model can count. 4 was picked empirically — 3 is often too short for an explanation, 5 starts to feel essay-shaped. The 8-sentence escape hatch for explicit *"tell me more"* keeps the cap from feeling brittle: the user can always pull more out by asking.
- **Bans on preamble and wrap-ups, by example.** *"Don't start with 'Certainly!'"* works much better than *"Get to the point."* The model has been trained on millions of essays that open with "Certainly!" — it has to be *told*, in the same phrasing, not to.
- **Forbidden sub-section names by literal example.** *"Background:", "Motivations:", "Personality:"* are the specific labels qwen2.5:7b reaches for when given a character description. Banning the abstraction *"don't break the answer into sections"* moves nothing; banning the strings does.
- **The speculation phrases — "it's likely that…", "she may have…"** — are the verbal tics of a model trying to be helpful past the edge of its notes. Banning them in the same shape they appear is what makes the model stop reaching beyond the grounding contract.
- **The escape hatch is part of the constraint.** Without it, a user who genuinely wants more detail gets a curt four-sentence reply and feels brushed off. With it, the constraint is *default behavior with a known opt-out* — which is how concision actually works in conversation.

Both `PAGE_MODE_SYSTEM` and `WIKI_MODE_SYSTEM` end with this block. Combined with the [grounding contract](#grounding), the result is a model that lands the answer in the first sentence, in plain prose, in roughly the length a friend would actually use.

> *Sanity check the assembled prompt yourself.* If you want to see what the model actually receives, drop a one-liner at the bottom of `_assemble_messages` and watch the full prompt log:
>
> ```python
> import logging
> logging.getLogger(__name__).warning("PROMPT:\n%s", parts)
> ```
>
> Run a chat turn against the running backend; you'll see the four labelled sections plus the user question, with every piece of text scrubbed by `_strip_markdown`. The diff between this and what you'd see at `post-7` is *the entire content* of this post.

---

## Wiki Mode: The Long Road to Concise Answers {#wiki-concise}

The post's title points at wiki mode specifically, and for a reason. Page mode's failures are easy to spot (the model invents a panel detail; the spoiler filter or the user catches it immediately). Wiki mode's failure is subtler: **the model answers correctly, but at length**. *"What is Chaosah?"* prompts a full essay on the school's history, philosophy, headquarters, rival schools, colour palette, and notable students — when what the user wanted was a sentence or two of orientation.

This is *the* hardest behavior to shift on a 7B model, because the model was trained on encyclopedias and the wiki articles in context *look* like encyclopedia entries. Mirroring kicks in: the model writes what it sees. The grounding contract doesn't help — the recited material is from the article. The response-format cap helps, but the model often spends the 4-sentence budget on a tour of subtopics rather than a focused answer.

The mode-specific block that does the work is structured as a *contrast example*:

```python
# backend/app/core/prompts.py (excerpt from WIKI_MODE_SYSTEM)
"""
WIKI ANSWERS MUST BE TIGHT. Wiki articles contain many facts; your job is to \
pick the ONE OR TWO that directly answer what the user asked, and answer in \
2-3 sentences. Do not list everything you know about the topic. Do not give \
an encyclopedia entry. Do not enumerate sub-topics.

Concrete contrast:
  - User asks "what is Chaosah?" → "Chaosah is the witch school of chaos \
magic, one of the major schools in the *Pepper&Carrot* universe. Pepper is \
its youngest student." That's it. Do not also explain the founder, the \
philosophy, the rivals, the headquarters, the colour palette.
  - User asks "who founded Chaosah?" → answer with the founder, in one \
sentence, plus at most one sentence of context. Do not pivot to a tour of \
the school.

If the user wants more, they will ask. The shorter the answer, the better — \
brevity is not a tradeoff against quality here, it IS the quality.
"""
```

Three moves are worth naming here, because each one comes from a real demo failure:

- **A concrete answer that the model can pattern-match against.** The literal sentence *"Chaosah is the witch school of chaos magic, one of the major schools…"* gives the model a *shape* to follow. Abstract instructions like *"answer the question and stop"* shift behavior much less. The model copies what it sees.
- **An explicit list of "do not also explain…" topics.** *"The founder, the philosophy, the rivals, the headquarters, the colour palette"* are exactly the subsections qwen2.5:7b reaches for when expanding "What is X?" into an essay. Naming them is what stops them.
- **"Brevity is not a tradeoff against quality here, it IS the quality."** A flagged-up reframe. A 7B model under pressure thinks longer = more helpful; saying the inverse explicitly, in italics, in the prompt, moves it. Treat the prompt like a one-shot lesson: explain the *why*, not just the *what*.

The page-mode equivalent is the per-question focus rule (*"Focus on what's happening on the current spread and the immediate story"*) — same idea, different angle. Both are mode-specific because the kinds of *length-failure* are mode-specific: page mode tends to invent panel detail to pad the answer; wiki mode tends to tour the article.

---

## Suggestion Chips: A Schema Can't Carry Taste Either {#chips}

[Post 7]({% post_url 2026-05-25-pepper-carrot-companion-streaming-chat %}) introduced the two follow-up chips — one `page_chip`, one `wiki_chip` — generated by a second, schema-constrained model call and validated server-side before they render. The schema (named slots, both required, string-typed with min/max lengths) guaranteed that *something* parseable comes back in each slot. It did *not* guarantee that the content in the `wiki_chip` slot was actually about the universe rather than the page, and Post 7 closed with that honest admission.

Post 8 is where we make the chip prompt do more work. The Post 7 `SUGGESTIONS_SYSTEM` was ~100 words and mostly abstract: *"page_chip must be about the page; wiki_chip must be about the universe; both must be complete questions."* The Post 8 version is ~400 words and structured around **concrete bad/good examples** — the form that actually moves 7B behavior:

```python
# backend/app/core/prompts.py (excerpt from SUGGESTIONS_SYSTEM)
"""
BAD CHIPS (never generate these):
  - "Can you tell me more about Pepper and how she interacts with Carrot in \
her daily life"  ← "Can you tell me" stem, generic, two ideas stacked
  - "What kind of mischief does Carrot usually get into, and how does \
Pepper deal with it"  ← two questions stacked with "and"
  - "Tell me about the characters"  ← generic, no specificity
  - "Carrot, Pepper's curious cat, is a frequent flyer"  ← NOT A QUESTION
  - "In the story, the mischievous cat named Carrot is"  ← NOT A QUESTION, \
truncated mid-sentence
  - "What kind of mischief does Carrot,"  ← truncated, ends with comma

GOOD CHIPS — note that some are short, some are longer; what matters is \
that each is a complete, specific question:
  - page: "What's in the rainbow potion bottles"
  - page: "Why is Carrot reaching for the top shelf"
  - page: "What does the sign on the cottage door say"
  - wiki: "Tell me about Chaosah magic"
  - wiki: "Who founded Pepper's witch school and why"
  - wiki: "What is the difference between Chaosah and Hippiah"
"""
```

A few things to call out, because the choices weren't arbitrary:

- **Each BAD chip is annotated with the *reason*** ("← truncated, ends with comma", "← two questions stacked with 'and'", "← 'Can you tell me' stem"). The annotation is what teaches the model the *category* of mistake, not just the specific example. Without it, the model dutifully avoids the literal strings and finds other ways to make the same class of mistake.
- **The BAD examples are real model output**, copied from logs during Post 7's testing. Made-up bad examples don't carry the same *pattern weight* — the model has actually produced these shapes, and naming them is how you teach it not to.
- **The GOOD examples are mixed-length.** *"What's in the rainbow potion bottles"* is 6 words; *"What is the difference between Chaosah and Hippiah"* is 9. The note "*some are short, some are longer; what matters is that each is a complete, specific question*" reframes length as a *consequence* of specificity, not a target on its own.

The structural validation in `_parse_suggestions` ([Post 7]({% post_url 2026-05-25-pepper-carrot-companion-streaming-chat %}#validation)) still runs after this — chips that *do* slip through as a statement or a truncated fragment get dropped on the server. **The schema guarantees parseability; the prompt examples guarantee *taste*; the validator catches what slips past both.** Three layers of defense, none of them sufficient alone.

> *What about a better model?* This is the right next question. Switching the suggestion call to a frontier API (Claude Haiku, GPT-4.1-mini) would clean the chips up dramatically with a much shorter prompt — and at portfolio scale the cost would be negligible. The provider abstraction from [Post 3]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) lets us do exactly that with a one-line config flip: the chip call goes through `ChatClient.complete()`, which has both an Ollama and an Anthropic implementation. The reason we don't here is the *post's* premise — "if your whole pipeline runs locally, can you still make the demo good?" — but the seam to escape that premise is one config line away. Knowing *when to upgrade the model* and *when to upgrade the prompt* is part of the design story; pretending only one of them is the answer would be dishonest.

---

## The Markdown Safety Net in the Chat Panel {#safety-net}

Three lines of defense aren't quite enough. Even with the system prompt forbidding markdown, even with `_strip_markdown` scrubbing the context, a 7B model under a tricky question will occasionally emit *one* `### header` or *one* `**bold**` phrase. With Post 7's chat panel, that markdown rendered as literal characters in the bubble — `### Background` showed up exactly that way, hashes and all, and the bubble suddenly looked like a code dump.

The last-line fix is to render the assistant's output through a markdown renderer. Install [`react-markdown`](https://github.com/remarkjs/react-markdown) and [`remark-gfm`](https://github.com/remarkjs/remark-gfm) (GitHub-flavored markdown — tables, strikethrough, task lists, the things `react-markdown` doesn't enable by default):

```bash
cd frontend && npm install react-markdown remark-gfm
```

Then six lines of JSX in [`ChatPanel.tsx`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/post-8/frontend/src/components/ChatPanel.tsx):

```tsx
// frontend/src/components/ChatPanel.tsx (the assistant-bubble branch)
{m.role === 'assistant' && m.content ? (
  // Safety net for Post 8: the system prompt asks for plain prose, but a
  // 7B model under pressure will occasionally emit `### headers`,
  // `**bold**`, or `- bullets` anyway. Rendering markdown turns that ugly
  // raw output into something readable; remark-gfm covers tables and
  // strikethrough if the model reaches for them.
  <div className="chat-markdown">
    <ReactMarkdown remarkPlugins={[remarkGfm]}>
      {m.content}
    </ReactMarkdown>
  </div>
) : (
  m.content || (streaming ? '…' : '')
)}
```

User bubbles stay plain text — the user typed it; if they put markdown in their question, they meant it. Assistant bubbles go through `ReactMarkdown`. The CSS in `global.css` tones down the browser defaults — a stray `### header` shouldn't render at h3 *document* sizing in a chat bubble:

```css
/* frontend/src/styles/global.css */
.chat-markdown { white-space: normal; }
.chat-markdown > :first-child { margin-top: 0; }
.chat-markdown > :last-child { margin-bottom: 0; }
.chat-markdown p { margin: 0 0 0.5rem; }
.chat-markdown h1, .chat-markdown h2, .chat-markdown h3,
.chat-markdown h4, .chat-markdown h5, .chat-markdown h6 {
  font-size: 1rem;
  font-weight: 600;
  margin: 0.4rem 0 0.2rem;
}
.chat-markdown ul, .chat-markdown ol { margin: 0.3rem 0 0.5rem; padding-left: 1.2rem; }
.chat-markdown code {
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
  font-size: 0.85em;
  background: var(--parchment-edge);
  padding: 0 0.2em;
  border-radius: 3px;
}
```

> *Why is this the *last* line of defense and not the *only* one?* Two reasons. **First, mirroring is contagious in long conversations.** Once the model has emitted one `###` header and the user follow-ups, every subsequent reply tends to also be in that shape — the conversation history is now in context, and the model mirrors *that*. Stripping markdown out of the *retrieved* text in Tightener 1 stops it from getting a foothold in the first place. **Second, prompt rules teach the model the right *behavior*, not just the right *rendering*.** A model that doesn't emit `### Background` is also less likely to *structure its thinking* as a four-section essay; that's harder to measure but the demo benefits visibly. The renderer fixes the symptom; the system prompt fixes the cause. We want both.

This is also the section to flag a security note: `react-markdown` sanitizes its output by default — no embedded HTML, no `<script>` tags, no inline-`onclick` handlers. You don't have to bolt on a separate sanitizer. If you ever swap the renderer (for `markdown-it`, say), check the same; rendering arbitrary model output in a browser is one of the easier ways to introduce an XSS bug.

---

## The Prompt, End to End {#diagram}

One picture of where each tightener lives in the pipeline. Notice that *only* the prompt-bound paths gain anything — retrieval, the spoiler filter, the chip schema, the SSE framing are all from Posts 6–7 and unchanged.

<div style="margin: 1.5rem 0; overflow-x: auto;">
<a href="/assets/picture/2026-05-30-pepper-carrot-companion-prompt-hardening/prompt-stack.svg" target="_blank" rel="noopener" title="Open the diagram full-size in a new tab" style="display: block; cursor: zoom-in;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 540" role="img"
     aria-label="The prompt-hardening stack. The user message and the saved current_page enter retrieval (unchanged from Post 6 — spoiler-filtered for page mode, unfiltered for wiki mode). Retrieved chunks plus the current page text pass through _strip_markdown before they reach the prompt. The system prompt is composed of named blocks: _SHARED_VOICE, _SPOILER_DISCIPLINE, _GROUNDING_CONTRACT, then per-mode _PAGE_ANTI_RECITATION or _WIKI_OUTPUT_DISCIPLINE, then _RESPONSE_FORMAT. The assembled prompt goes to ChatClient.stream(); tokens stream back through ReactMarkdown in the chat panel as the last-line safety net. After the answer, a second non-streaming call uses SUGGESTIONS_SYSTEM (with bad/good examples) and the named-slot schema; chips pass through the question-shape parser before reaching the DOM."
     style="display: block; width: 100%; height: auto; max-width: 1180px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">
  <defs>
    <marker id="ph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>
    </marker>
  </defs>

  <text x="590" y="26" text-anchor="middle" font-size="13" font-weight="600" fill="#1f2937">Retrieval and framing are unchanged from Posts 6–7.</text>
  <text x="590" y="46" text-anchor="middle" font-size="13" font-weight="600" fill="#b45309">Every Post 8 change lives where the model meets prose.</text>

  <!-- Input: user message + current_page -->
  <g>
    <rect x="30" y="80" width="200" height="84" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="130" y="106" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">User message</text>
    <text x="130" y="126" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">+ saved current_page</text>
    <text x="130" y="146" text-anchor="middle" font-size="9" fill="#94a3b8" font-style="italic">spoiler boundary lives here</text>
  </g>

  <line x1="230" y1="120" x2="298" y2="120" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ph-arrow)"/>

  <!-- Retrieval -->
  <g>
    <rect x="300" y="80" width="220" height="84" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="410" y="106" text-anchor="middle" font-size="13" font-weight="600" fill="#1f2937">RetrievalService</text>
    <text x="410" y="126" text-anchor="middle" font-size="10" fill="#92400e" font-style="italic">page: spoiler-filtered (Post 6)</text>
    <text x="410" y="146" text-anchor="middle" font-size="10" fill="#92400e" font-style="italic">wiki: unfiltered (Post 7)</text>
  </g>

  <line x1="520" y1="120" x2="588" y2="120" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ph-arrow)"/>

  <!-- _strip_markdown -->
  <g>
    <rect x="590" y="80" width="260" height="84" rx="8" fill="#fde68a" stroke="#b45309" stroke-width="1.8"/>
    <text x="720" y="105" text-anchor="middle" font-size="12" font-weight="700" fill="#7c2d12">★ Post 8 · _strip_markdown</text>
    <text x="720" y="125" text-anchor="middle" font-size="10" fill="#7c2d12" font-style="italic">applied to every prompt-bound text</text>
    <text x="720" y="147" text-anchor="middle" font-size="9" fill="#94a3b8" font-style="italic">**bold**, ### header, - bullet → gone</text>
  </g>

  <!-- System-prompt stack -->
  <g>
    <rect x="40" y="220" width="640" height="230" rx="8" fill="#fde68a" stroke="#b45309" stroke-width="1.8"/>
    <text x="360" y="246" text-anchor="middle" font-size="13" font-weight="700" fill="#7c2d12">★ Post 8 · render_system_prompt(mode)</text>
    <text x="360" y="264" text-anchor="middle" font-size="10" fill="#7c2d12" font-style="italic">composed top to bottom from these named blocks</text>

    <rect x="60" y="284" width="600" height="22" rx="3" fill="#fef3c7" stroke="#b45309" stroke-width="1"/>
    <text x="360" y="299" text-anchor="middle" font-size="11" fill="#1f2937">_SHARED_VOICE (warm, playful, not corporate)</text>

    <rect x="60" y="310" width="600" height="22" rx="3" fill="#fef3c7" stroke="#b45309" stroke-width="1"/>
    <text x="360" y="325" text-anchor="middle" font-size="11" fill="#1f2937">_SPOILER_DISCIPLINE (mode-neutral, position interpolated from session)</text>

    <rect x="60" y="336" width="600" height="22" rx="3" fill="#fed7aa" stroke="#b45309" stroke-width="1"/>
    <text x="360" y="351" text-anchor="middle" font-size="11" font-weight="600" fill="#1f2937">_GROUNDING_CONTRACT — closed-world rule (Tightener 2)</text>

    <rect x="60" y="362" width="290" height="22" rx="3" fill="#fed7aa" stroke="#b45309" stroke-width="1"/>
    <text x="205" y="377" text-anchor="middle" font-size="10" font-weight="600" fill="#1f2937">page: _PAGE_ANTI_RECITATION (3)</text>

    <rect x="370" y="362" width="290" height="22" rx="3" fill="#fed7aa" stroke="#b45309" stroke-width="1"/>
    <text x="515" y="377" text-anchor="middle" font-size="10" font-weight="600" fill="#1f2937">wiki: _WIKI_OUTPUT_DISCIPLINE (concise contrast)</text>

    <rect x="60" y="388" width="600" height="22" rx="3" fill="#fed7aa" stroke="#b45309" stroke-width="1"/>
    <text x="360" y="403" text-anchor="middle" font-size="11" font-weight="600" fill="#1f2937">_RESPONSE_FORMAT — 4-sentence cap, no preamble/wrap-ups (Tightener 4)</text>

    <text x="360" y="430" text-anchor="middle" font-size="9" fill="#94a3b8" font-style="italic">same composition as Post 7; the blocks themselves are sharper</text>
  </g>

  <line x1="720" y1="164" x2="500" y2="220" stroke="#9ca3af" stroke-width="1.3" stroke-dasharray="4,4" marker-end="url(#ph-arrow)"/>
  <text x="610" y="200" font-size="10" fill="#94a3b8" font-style="italic">stripped notes</text>

  <!-- Chat call -->
  <g>
    <rect x="720" y="220" width="200" height="84" rx="8" fill="#d1fae5" stroke="#059669" stroke-width="1.5"/>
    <text x="820" y="246" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">ChatClient.stream()</text>
    <text x="820" y="266" text-anchor="middle" font-size="10" fill="#065f46" font-style="italic">qwen2.5:7b via Ollama</text>
    <text x="820" y="286" text-anchor="middle" font-size="10" fill="#065f46" font-style="italic">tokens out (Post 7 SSE)</text>
  </g>
  <line x1="680" y1="285" x2="718" y2="262" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ph-arrow)"/>

  <!-- ReactMarkdown safety net -->
  <g>
    <rect x="950" y="220" width="200" height="84" rx="8" fill="#fde68a" stroke="#b45309" stroke-width="1.8"/>
    <text x="1050" y="244" text-anchor="middle" font-size="12" font-weight="700" fill="#7c2d12">★ Post 8 · &lt;ReactMarkdown&gt;</text>
    <text x="1050" y="264" text-anchor="middle" font-size="10" fill="#7c2d12" font-style="italic">renders the bubble</text>
    <text x="1050" y="286" text-anchor="middle" font-size="9" fill="#94a3b8" font-style="italic">safety net for leaked markdown</text>
  </g>
  <line x1="920" y1="262" x2="948" y2="262" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ph-arrow)"/>

  <!-- Chip side-channel -->
  <g>
    <rect x="720" y="338" width="430" height="106" rx="8" fill="#fde68a" stroke="#b45309" stroke-width="1.8"/>
    <text x="935" y="362" text-anchor="middle" font-size="12" font-weight="700" fill="#7c2d12">★ Post 8 · second call · SUGGESTIONS_SYSTEM</text>
    <text x="935" y="382" text-anchor="middle" font-size="10" fill="#7c2d12" font-style="italic">bad/good chip examples replace abstract rules</text>
    <text x="935" y="402" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">named-slot schema (Post 7) — unchanged</text>
    <text x="935" y="422" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">_parse_suggestions question-shape filter — unchanged</text>
  </g>
  <line x1="820" y1="304" x2="850" y2="338" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ph-arrow)"/>

  <!-- Legend -->
  <g>
    <rect x="30" y="478" width="14" height="12" fill="#fde68a" stroke="#b45309" stroke-width="1.5"/>
    <text x="50" y="489" font-size="11" fill="#4b5563">★ Post 8 (this post)</text>
    <rect x="210" y="478" width="14" height="12" fill="#dbeafe" stroke="#2563eb" stroke-width="1"/>
    <text x="230" y="489" font-size="11" fill="#4b5563">unchanged from Post 7</text>
    <rect x="380" y="478" width="14" height="12" fill="#fef3c7" stroke="#f59e0b" stroke-width="1"/>
    <text x="400" y="489" font-size="11" fill="#4b5563">retrieval / framing (Post 6)</text>
    <rect x="580" y="478" width="14" height="12" fill="#d1fae5" stroke="#059669" stroke-width="1"/>
    <text x="600" y="489" font-size="11" fill="#4b5563">provider abstraction (Post 3)</text>
  </g>
</svg>
</a>
</div>

*Every box outlined in dark amber is something Post 8 either added or substantially sharpened. Notice that the *path* of the request is exactly Posts 6–7's path — retrieval still happens before the prompt, the prompt is still composed top-to-bottom from named blocks, the chip call is still a second schema-constrained pass, the chat panel still consumes SSE. The hardening is concentrated at three points: scrubbing input text, sharpening system prompts (per mode), and rendering output safely. Click the diagram to open it full-size in a new tab.*

---

## The Honest Part: What Prompts Still Can't Fix {#honest}

Two honest things to name, because the post is a portfolio piece and the line between *engineering judgment* and *salesmanship* is "tell me what you tried that didn't work."

**The wiki chip is still sometimes a page question wearing a wiki label.** The Post 8 prompt cuts the failure rate substantially — running the chip call ten times on a typical page-3 page-mode answer in Episode 1, you'll see something like 7–8 chips that clearly land in wiki territory ("What is Chaosah?", "How do potions work in Hereva?") and 2–3 that are really about the panel ("How does Pepper feel about Carrot winning?"). Compared to Post 7's roughly 50/50, that's a real shift; compared to *zero* page-flavored wiki chips, it's incomplete. The schema *can't* tell the difference (both are valid strings, both are questions), and the few-shot examples can teach the model what *good* wiki chips look like but can't reliably make qwen2.5:7b *infer the topic* of a question it just answered. A reliable fix means a stronger model on the chip call — and the provider abstraction makes that a config flip, not a refactor.

**Sentence counting isn't quite reliable.** The 4-sentence cap holds most of the time. Occasionally — particularly on questions that genuinely warrant nuance, like "what are Pepper and Carrot's personalities like?" — qwen2.5:7b will produce six or seven sentences and mostly stop. The cap is *guidance with strong examples*, not a hard limit at the sampler level; the model can choose to overshoot. Two real options to close this further: a length-penalty in `complete()` (provider-specific and finicky), or a post-generation check that truncates at sentence 4 if the response runs long (cheap, ugly). The workshop ships neither — at portfolio scale the occasional 6-sentence reply isn't worth the new failure mode of mid-sentence truncation.

The third honest thing: **`_strip_markdown` is regex-based, not a real parser.** A pathological input (deeply nested emphasis, tables, fenced code blocks with markdown *inside* them) would survive partially intact. Our actual inputs don't have those shapes, and the failure mode is benign (a stray asterisk in context), so the choice is deliberate. If the input shape ever changes (e.g., re-ingesting a richer wiki source), the right move is to swap in a proper parser, not to keep patching regexes.

The bigger meta-point: **prompts are a software discipline, not a magic incantation.** You write them with specific failures in mind, you measure against those failures, you sharpen the prompt where the failures concentrate, and you don't pretend the prompt is doing more than it is. The thing that makes the demo *safe* is structural — the retrieval boundary, the schema, the server-side validation. The thing that makes the demo *feel good* is the prompt. Confusing those two is how a portfolio project gets a CVE; pretending the prompt-only version is enough is how it gets a one-star review.

---

## Key Takeaways {#key-takeaways}

**1. Strip the markdown out of the context, not just out of the answer.** Small chat models mirror what they see — `**bold**` in context produces `**bold**` in the reply, `### Section` headers in notes produce `### Section` answers. A 30-line regex pass applied to every prompt-bound site (page descriptions, wiki articles, `plot_summary`, `ocr_text`) prevents the mirroring at the source. The frontend's markdown renderer is the *safety net*; the regex is the *belt*.

**2. Make the contract about parametric memory explicit.** A 7B model trained on web text has absorbed fragments of *Pepper&Carrot* and will confidently fill gaps with plausible-sounding inventions unless you tell it not to. The `_GROUNDING_CONTRACT` block says it directly: *the notes are your only source of truth; if a detail isn't in them, it doesn't exist for this answer; admitting "the notes don't say" is always better than filling the gap.* Closed-world reasoning isn't the default — flip the default in the prompt.

**3. Page-mode evasion is a labeling problem, not a model problem.** When the model sees a `=== Current spread ===` heading next to the user question, it sometimes concludes the user typed the whole thing and politely asks for clarification — about a question it could have answered from the notes. The `_PAGE_ANTI_RECITATION` block tells the model what the labelled sections *are* (private notes prepared for it, not user input) and enumerates the forbidden evasion phrases by literal example. Forbidden-string lists beat abstract rules at 7B.

**4. Soft format rules don't work; strict ones with numbers and forbidden phrases do.** *"Be concise"* gets ignored. *"Maximum 4 sentences. Hard cap. No preamble. Do not start with 'Certainly!'"* does not. The strict rules sound paranoid on first read; they're paranoid because the soft versions empirically failed. The `_RESPONSE_FORMAT` block is the loudest block in the prompt for a reason.

**5. Wiki-mode concision needs a contrast example, not an instruction.** *"Pick one or two facts and stop"* moves the model less than *"User asks 'what is Chaosah?' → 'Chaosah is the witch school of chaos magic, one of the major schools in the universe. Pepper is its youngest student.' That's it. Do not also explain the founder, the philosophy, the rivals, the headquarters, the colour palette."* The concrete answer-shape gives the model something to pattern-match against; the explicit *"do not also explain…"* list names the exact subtopics it was going to reach for anyway.

**6. The chip schema guarantees parseability; bad/good examples guarantee taste.** Post 7's named-slot schema made "two page chips in a row" structurally impossible, which was the right structural guarantee. It can't tell whether a `wiki_chip` is actually about the universe or about a page detail — that's a *taste* problem and lives in the prompt. The Post 8 `SUGGESTIONS_SYSTEM` quadrupled in length, almost entirely with concrete BAD-CHIP and GOOD-CHIP examples annotated with the *category* of mistake each one makes. Real-failure examples teach more than rules.

**7. The markdown renderer is the *last* line of defense, not the only one.** Rendering assistant output through `react-markdown` + `remark-gfm` means a stray `### header` reads as a small heading rather than as hashes. It does *not* mean the prompt rules can relax — mirroring is contagious in long conversations, and one markdown reply tends to beget more. Belt + suspenders + safety net is the right number of layers for a small model on local hardware.

**8. The portfolio signal is "knowing when to stop." `_strip_markdown` is a regex pass, not a parser; the 4-sentence cap is guidance, not a sampler constraint; the wiki chip is sometimes off-topic. Each of those has a more elaborate solution available, and the workshop ships none of them — because the marginal improvement is small and the failure mode of the *more* elaborate solution is worse. Calibrating cost-of-engineering against value-of-fix, on a small model, in a small portfolio project, is itself the design decision.

---

Next up: **Post 9 — A World Graph Built by a Second Skill: Spoiler-Aware Knowledge Graph Overlay.** Post 8 closed the gap on the chat *replies*; Post 9 adds a new affordance to the reader. A second Claude Code skill (mirroring `ingest-from-images` from [Post 4]({% post_url 2026-05-16-pepper-carrot-companion-claude-skill-ingestion %})) walks the upstream *Pepper&Carrot* wiki + the existing page-description JSONs and writes a durable YAML of entities + relationships. That YAML loads into Postgres, gets exposed through a spoiler-filtered API (same lexicographic boundary as the page retrieval — `(episode_debut, page_debut) ≤ (current_episode, current_page)`), and renders in the browser as an avatar-node overlay built on [react-flow](https://reactflow.dev/) with kind-based SVG fallbacks for the entities without art. Click a node, read the blurb, click "Ask in wiki mode" — the chip routes back through the same `WIKI_MODE_SYSTEM` you sharpened here.

The **workshop starter** that backs this post is at <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop>, tagged `post-8` — `git checkout post-8` to get exactly the code shown here (see [Following along with the blog series](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop#following-along-with-the-blog-series)). The **full source repository** and the public live-demo URL go up alongside the final post — the deploy guide — once it's published.

*Pepper & Carrot* is © [David Revoy](https://www.davidrevoy.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All credit to him for the source material that made this project possible.

**All opinions expressed are my own.**

---
