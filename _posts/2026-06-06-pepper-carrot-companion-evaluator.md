---
title: "Pepper & Carrot AI-powered flipbook · Part 18 — Grading the Companion: An Agentic Evaluator on the Other Side of MCP"
date: 2026-06-06 13:00:00 -0800
categories: [Full-Stack, RAG, MCP]
tags: [mcp, model-context-protocol, evaluation, llm-as-judge, ragas, retrieval, failure-attribution, claude, anthropic, voyage-ai, peppercarrot, portfolio]
description: >-
  Part 18 of the Pepper & Carrot AI flipbook series — the other half of the MCP
  story. Post 17 built an MCP *server* that exposed the deployed reading
  companion as two tools (search, ask). This post builds an MCP *client* that
  consumes them to actually grade the app: a deterministic retrieval harness
  (recall@k, nDCG, MRR, plus an end-to-end spoiler-boundary check) and an
  LLM-as-judge answer layer (correctness, faithfulness, relevance, completeness)
  with explicit variance guards — joined by the one thing a single-number eval
  can't give you: failure attribution, telling a retrieval miss apart from a
  generation miss. It's written for someone new to RAG evaluation. The throughline
  is a hard line between what stays deterministic (the metrics) and what's allowed
  to be agentic (inventing test cases, judging open prose) — including a
  self-verifying gold generator that drafts candidates and auto-discards the ones
  the live index can't actually surface. Everything is reproducible from the repo.
pin: true
---

Part 18 of the [*Pepper & Carrot AI-powered flipbook*]({% post_url 2026-05-09-pepper-carrot-companion-trailer %}) series, and the second half of a two-part thread on the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). [Post 17]({% post_url 2026-06-06-pepper-carrot-companion-mcp-server %}) built an MCP **server** — a thin adapter that exposed the deployed reading companion as two tools, `search` (retrieval) and `ask` (the real answer pipeline). That post ended on a promise: *"we built the grip; next, we squeeze."* This is the squeeze.

Because here's the thing a demo can't tell you: **is the companion actually any good?** When it answers "who's on this page?", did it retrieve the right context, or get lucky? When it sounds confident, is it grounded in the comic or quietly hallucinating? And the question that matters most when something goes wrong — *whose fault was it, the retriever or the model?* This post builds the program that answers those: an **agentic evaluator** that is itself an MCP **client**, consuming the exact same `search` and `ask` tools to grade the app from the outside.

> **▶ The repo: [`pepper-carrot-eval`](https://github.com/bearbearyu1223/pepper-carrot-eval).** A separate repo from the app and the server — because the evaluator is a *consumer* of the system under test, not a part of it. Clone it, point it at the live MCP server, and `uv run pepper-carrot-eval --retrieval-only` produces a real scored retrieval report with **no API key at all** (the deterministic layer runs entirely through the server). *Pepper & Carrot* is © [David Revoy](https://www.peppercarrot.com), CC BY 4.0.

> **What you'll build in this post.**
> - **A deterministic retrieval harness** driven by the `search` tool — recall@k, precision@k, nDCG@k, MRR over a frozen gold set, plus an **end-to-end spoiler-boundary check** that proves the [Post 9 security property]({% post_url 2026-05-25-pepper-carrot-companion-spoiler-safe-rag %}) holds *through the tool*, not just in a unit test.
> - **An LLM-as-judge answer layer** driven by the `ask` tool — correctness, faithfulness, relevance, completeness against golden Q&A, with **concrete judge-variance guards** (a cross-model judge, an anchored rubric, forced structured output, median-of-N, a cache) plus a cheap deterministic similarity signal.
> - **Failure attribution** — the centerpiece. Because the same question runs through *both* tools, a bad answer can be blamed on **retrieval** (the right chunk never surfaced) vs. **generation** (the model had the chunk and blew it anyway).
> - **A self-verifying gold generator** — the agentic part: an LLM drafts candidate test cases, each is *verified against the live index*, and the ones the index can't actually surface are **auto-discarded** before a human ever sees them.
> - **Reasoning traces** — every `search`/`ask`/judge call recorded to a JSONL file you can `jq`, so every score is auditable back to the model's own reasoning.
>
> **Prerequisites.**
> - [Post 17]({% post_url 2026-06-06-pepper-carrot-companion-mcp-server %}) (the MCP server) — this post is its other half and assumes the two tools exist.
> - The MCP server reachable (the public one at `https://pepper-carrot-mcp.fly.dev/mcp` is fine).
> - Python 3.11+ and [`uv`](https://docs.astral.sh/uv/). For the answer layer only: an `ANTHROPIC_API_KEY` (the judge) and a `VOYAGE_API_KEY` (similarity). The retrieval layer needs neither.
> - **No eval experience assumed** — the concepts are built from zero.

> **About the repos.** Three repos, one system. The app ([`pepper-carrot-companion-workshop`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop)) is the thing under test; the server ([`pepper-carrot-mcp`](https://github.com/bearbearyu1223/pepper-carrot-mcp)) exposes it; this evaluator ([`pepper-carrot-eval`](https://github.com/bearbearyu1223/pepper-carrot-eval)) grades it. Keeping them separate is the point — the evaluator only ever touches the app through the public MCP tools, the same way any outside grader would.

---

## Table of Contents

1. [Why a Demo Isn't Evidence](#why)
2. [The One Rule: What Stays Deterministic](#rule)
3. [Instrument 1 — Grading Retrieval (deterministically)](#retrieval)
4. [Instrument 2 — Grading Answers (with a judge)](#answers)
5. [Failure Attribution: Whose Fault Was It?](#attribution)
6. [Where the Gold Comes From — and a Self-Verifying Generator](#gold)
7. [Reasoning Traces: Every Score Is Auditable](#traces)
8. [What's Honest, What's Open](#honest)
9. [Key Takeaways](#key-takeaways)

---

## Why a Demo Isn't Evidence {#why}

The companion *looks* great in a demo. You ask "who's brewing the potion?", a fluent, on-topic answer streams back, and it feels done. But "feels right on the three questions I tried" is not evidence, and a portfolio reviewer who has shipped real systems knows it. Two things can be wrong under a confident answer:

- **Retrieval** can miss. The vector search might surface the wrong chunks — an unrelated wiki article that happens to share vocabulary — and you'd never know, because the model writes a smooth paragraph anyway.
- **Generation** can drift. Even handed the *right* chunks, the model can ignore them, embellish, or hallucinate a fact that isn't in the comic.

> *Plain-English aside: what "evaluating RAG" actually means.* A retrieval-augmented system has two stages — *find the relevant text*, then *write an answer from it* — so it has two ways to fail, and evaluating it means grading both **separately**. Retrieval is a *search* problem: did the right chunks come back, and were they ranked near the top? That's measurable with classic information-retrieval numbers, no LLM required. Answer quality is a *writing* problem: is the response correct, grounded, on-topic, complete? That's fuzzier — you need judgment. The mistake beginners make is to grade only the final answer; then a retrieval bug and a generation bug look identical, and you can't fix either with confidence. The whole design of this evaluator is to keep the two measurements apart, and then — the payoff — *join them* to assign blame.

An MCP server makes this clean, because it hands you the two stages as two separate tools. `search` is the retrieval stage with the scores and metadata exposed; `ask` is the full pipeline producing the real user-facing answer. Grade `search`, grade `ask`, and you've graded the two failure modes independently — using the exact same tools any other MCP client (Claude included) would call.

---

## The One Rule: What Stays Deterministic {#rule}

Before any code, the single most important design decision — and the one a thoughtful reviewer will look for. An evaluator is only useful if its numbers are **trustworthy over time**: you change the prompt, re-run, and a moved number means a real regression, not judge noise. So the rule is:

> **Keep the scored metrics deterministic. Allow agency only where open-ended reasoning genuinely pays — and never let it compute the score.**

That draws a hard line through the system:

| Component | Deterministic? | Why it sits there |
|---|---|---|
| recall@k, nDCG, MRR | **Deterministic** | Pure set/rank math over fixed `search` output — bit-identical run to run |
| Spoiler-boundary check | **Deterministic** | A structural assertion on returned positions |
| Answer↔reference similarity | **Deterministic** | A fixed embedding model's cosine |
| Failure attribution | **Deterministic** | A join over the two layers' outputs |
| LLM-as-judge (answer quality) | **Agentic, guarded** | Open prose needs judgment — but variance-controlled and cached |
| Generating / inventing test cases | **Agentic, frozen after review** | Needs creativity — done offline, never in the scored loop |

The agentic parts *feed* the deterministic metrics (by inventing test cases) and *explain* them (by judging prose), but they don't *compute* them. That distinction is the spine of everything below. (It's the same boundary the [provider-abstraction post]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) drew between "what changes and what doesn't" — here it's "what must be reproducible and what's allowed to reason.")

The diagram below is the whole architecture: one MCP client, two instruments, a deterministic core, and a thin agentic edge that generates the gold and judges the prose.

<div style="margin: 1.5rem 0; overflow-x: auto;">
<a href="/assets/picture/2026-06-06-pepper-carrot-companion-evaluator/evaluator.svg" target="_blank" rel="noopener" title="Click to enlarge — opens the diagram full-size in a new tab" style="display: block; cursor: zoom-in;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 470" role="img"
     aria-label="The evaluator as an MCP client. On the right, the deployed pepper-carrot MCP server exposes the search and ask tools. In the middle, the evaluator consumes them: the search tool drives a deterministic retrieval harness (recall@k, nDCG, MRR, spoiler-boundary check); the ask tool drives an LLM-as-judge answer layer (correctness, faithfulness, relevance, completeness) plus a deterministic Voyage similarity. Both feed a deterministic failure-attribution join that classifies each item as pass, generation fault, retrieval fault, or masked gap, producing a scored report and a JSONL trace. On the left, an agentic, offline self-verifying gold generator drafts candidate test cases, verifies each against the live index, auto-discards failures, and — after human review — freezes them into the gold set the harness reads. A colored band marks the deterministic core versus the agentic edge."
     style="display: block; width: 100%; height: auto; max-width: 1080px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">
  <defs>
    <marker id="ev-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#b45309"/>
    </marker>
  </defs>

  <!-- LEFT: agentic gold generation -->
  <rect x="24" y="60" width="210" height="150" rx="9" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>
  <text x="129" y="82" text-anchor="middle" font-size="11.5" font-weight="700" fill="#7c2d12">Agentic (offline)</text>
  <text x="129" y="100" text-anchor="middle" font-size="9.5" font-style="italic" fill="#7c2d12">self-verifying gold gen</text>
  <text x="129" y="124" text-anchor="middle" font-size="9.5" fill="#7c2d12">LLM drafts a candidate</text>
  <text x="129" y="140" text-anchor="middle" font-size="9.5" fill="#7c2d12">→ verify vs. live index</text>
  <text x="129" y="156" text-anchor="middle" font-size="9.5" fill="#7c2d12">→ auto-discard misses</text>
  <text x="129" y="180" text-anchor="middle" font-size="9.5" font-weight="700" fill="#7c2d12">→ human review → freeze</text>
  <line x1="234" y1="170" x2="300" y2="210" stroke="#b45309" stroke-width="1.4" marker-end="url(#ev-arrow)"/>
  <rect x="60" y="220" width="140" height="26" rx="5" fill="#fff7ed" stroke="#9a3412" stroke-width="1.2"/>
  <text x="130" y="237" text-anchor="middle" font-size="10" font-weight="700" fill="#7c2d12">gold_*.yaml (frozen)</text>
  <line x1="200" y1="233" x2="300" y2="250" stroke="#94a3b8" stroke-width="1.3" marker-end="url(#ev-arrow)"/>

  <!-- MIDDLE: evaluator -->
  <rect x="300" y="40" width="430" height="300" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="1.6"/>
  <text x="515" y="62" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a8a">pepper-carrot-eval · MCP client</text>
  <text x="515" y="79" text-anchor="middle" font-size="9" font-style="italic" fill="#1e3a8a">deterministic core · agentic edge</text>

  <rect x="318" y="92" width="394" height="58" rx="7" fill="#ecfdf5" stroke="#059669" stroke-width="1.3"/>
  <text x="515" y="110" text-anchor="middle" font-size="10.5" font-weight="700" fill="#065f46">Retrieval harness (deterministic) ← search</text>
  <text x="515" y="127" text-anchor="middle" font-size="9.5" fill="#065f46">recall@k · nDCG · MRR · spoiler-boundary check</text>
  <text x="515" y="141" text-anchor="middle" font-size="9" font-style="italic" fill="#065f46">bit-identical run to run</text>

  <rect x="318" y="158" width="394" height="64" rx="7" fill="#fefce8" stroke="#ca8a04" stroke-width="1.3"/>
  <text x="515" y="176" text-anchor="middle" font-size="10.5" font-weight="700" fill="#713f12">Answer layer ← ask</text>
  <text x="515" y="192" text-anchor="middle" font-size="9.5" fill="#713f12">LLM-as-judge (guarded) + Voyage similarity</text>
  <text x="515" y="206" text-anchor="middle" font-size="9" font-style="italic" fill="#713f12">correctness · faithfulness · relevance · completeness</text>

  <rect x="318" y="230" width="394" height="42" rx="7" fill="#f1f5f9" stroke="#475569" stroke-width="1.3"/>
  <text x="515" y="248" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1f2937">Failure attribution (deterministic join)</text>
  <text x="515" y="263" text-anchor="middle" font-size="9" fill="#475569">pass · generation-fault · retrieval-fault · masked-gap</text>

  <rect x="360" y="288" width="140" height="34" rx="6" fill="#fff" stroke="#475569"/>
  <text x="430" y="309" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">report.md / .json</text>
  <rect x="528" y="288" width="150" height="34" rx="6" fill="#fff" stroke="#475569"/>
  <text x="603" y="309" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">traces/*.jsonl</text>

  <!-- RIGHT: MCP server -->
  <rect x="800" y="120" width="256" height="150" rx="10" fill="#fde68a" stroke="#b45309" stroke-width="1.6"/>
  <text x="928" y="146" text-anchor="middle" font-size="12" font-weight="700" fill="#7c2d12">pepper-carrot-mcp</text>
  <text x="928" y="163" text-anchor="middle" font-size="9" font-style="italic" fill="#7c2d12">the server from Post 17</text>
  <rect x="820" y="178" width="216" height="30" rx="6" fill="#fff7ed" stroke="#9a3412" stroke-width="1.2"/>
  <text x="928" y="198" text-anchor="middle" font-size="11" font-weight="700" fill="#7c2d12">tool: search</text>
  <rect x="820" y="214" width="216" height="30" rx="6" fill="#fff7ed" stroke="#9a3412" stroke-width="1.2"/>
  <text x="928" y="234" text-anchor="middle" font-size="11" font-weight="700" fill="#7c2d12">tool: ask</text>

  <line x1="730" y1="120" x2="800" y2="180" stroke="#b45309" stroke-width="1.6" marker-end="url(#ev-arrow)"/>
  <line x1="800" y1="229" x2="732" y2="190" stroke="#b45309" stroke-width="1.6" marker-end="url(#ev-arrow)"/>
  <text x="772" y="150" text-anchor="middle" font-size="8.5" fill="#7c2d12" font-style="italic">MCP</text>

  <!-- bottom band -->
  <text x="515" y="372" text-anchor="middle" font-size="10.5" font-style="italic" font-weight="600" fill="#334155">Green = deterministic (reproducible metrics) · Amber = agentic (invents cases, judges prose) · the agentic edge never computes a score</text>
</svg>
</a>
</div>

*The evaluator consumes the same two MCP tools the app exposes. Green is the reproducible core; amber is the agentic edge that feeds and explains it. Click to enlarge.*

---

## Instrument 1 — Grading Retrieval (deterministically) {#retrieval}

The retrieval harness is the easy, satisfying part: pure math, no model, perfectly reproducible. For each test query you have a **gold set** — the chunk(s) that *should* come back — and you call `search`, then score the ranked results against the gold.

> *Plain-English aside: recall@k, nDCG, MRR in one breath.* All three ask "did the right chunks come back, and were they near the top?", differently. **recall@k** — of the chunks that should appear, what fraction landed in the top *k*? (Did we find them at all?) **MRR** (mean reciprocal rank) — how high was the *first* correct chunk? Rank 1 scores 1.0, rank 2 scores 0.5, rank 3 scores 0.33. (How good is the top of the list?) **nDCG@k** — a rank-weighted blend that rewards putting relevant chunks higher. You report them at several *k* values: recall@1 is "is the best answer literally first?", recall@3 is "is it in the top 3 the model actually gets fed?" These are decades-old information-retrieval metrics, and that's the point — they're boring, standard, and trustworthy.

One design detail that matters: **gold keys are stable, not Chroma ids.** A vector store assigns a fresh UUID to every chunk each time you re-ingest, so gold that referenced UUIDs would rot on the next ingestion. Instead a page chunk is keyed `("page", episode, page)` and a wiki chunk by its title — identifiers that survive a rebuild. The harness resolves those keys against whatever `search` returns today, so the gold outlives the index.

Run it against the live server with no key at all, and the headline on the real (39-item) gold set is:

```
overall  recall@1 = 0.846   recall@3 = 1.000   nDCG@3 = 0.937   MRR = 0.915
page     recall@1 = 0.700   recall@3 = 1.000   MRR  = 0.817
wiki     recall@1 = 0.897   recall@3 = 1.000   MRR  = 0.948
```

Read those two columns together, because they tell a story. **recall@3 = 1.000 everywhere** means every gold chunk is reachable — the gold isn't broken, and within the top 3 the model always gets the right context. But **recall@1 = 0.846** means the *best* chunk is first only ~85% of the time. That gap is the eval doing its job: it's discriminating. (If every number were 1.000 you'd have a *useless* eval — a ruler that says everything is perfect measures nothing. More on engineering that honestly in §6.)

### The spoiler boundary, proven end-to-end

There's one retrieval metric that isn't about quality — it's about safety. The whole series leans on a [spoiler-safe boundary]({% post_url 2026-05-25-pepper-carrot-companion-spoiler-safe-rag %}): page-mode retrieval must never surface a page the reader hasn't reached. That was unit-tested deep in the app. But a unit test proves the *function* works; it doesn't prove the *deployed system, reached through a tool* still honors it. So the evaluator re-checks it **structurally, end-to-end**: drive `search` at a reader position and assert that no returned chunk sits at or past the cursor. It's deterministic, it needs no key, and it turns "we wrote a spoiler filter" into "we continuously verify, from the outside, that the live system doesn't leak." That's a much stronger claim — and exactly the kind a reviewer trusts.

---

## Instrument 2 — Grading Answers (with a judge) {#answers}

Retrieval math can't tell you whether *"Carrot is the only one here, pouring a bright orange potion with guilty glug-glug sounds"* is a good answer. Prose needs judgment, and the standard tool for judgment at scale is **LLM-as-judge**: you ask a strong model to score the answer against a rubric.

> *Plain-English aside: LLM-as-judge, and why it needs guardrails.* Using a model to grade another model's output sounds circular, and done naively it is. Models are inconsistent graders — ask twice, get two scores — and they have biases: they reward fluent, long answers, and they tend to *prefer outputs from their own family* (self-enhancement bias). So an LLM judge without guards is a wobbly ruler. The fix isn't to abandon it — for open-ended prose there's no better option at scale — it's to *engineer the variance down* until the number is stable enough to track.

The judge here scores four dimensions — **correctness** (vs. a reference answer), **faithfulness** (is every claim supported by the retrieved context?), **relevance** (does it address the question?), and **completeness** (are the must-have facts present?) — and it carries five concrete guards:

- **A cross-model judge.** The app answers with `claude-haiku-4-5`; the judge is `claude-sonnet-4-6` — a stronger, *different* model, to sidestep self-preference bias.
- **An anchored rubric.** Each dimension has described 0 / 0.5 / 1 anchors ("1.0 = every fact matches; 0.5 = mostly right with one wrong fact; 0.0 = contradicts the reference"). Anchoring is the single biggest variance reducer.
- **Forced structured output.** The judge must return its scores through a tool call with a fixed schema — no free-form drift, and a `reasoning` field *per dimension* so every score comes with its justification.
- **Temperature 0 + median-of-N.** It samples the verdict three times and takes the median, reporting the spread as a variance signal — a high spread flags the item for human review.
- **A disk cache.** Verdicts are keyed by `(rubric version, item, answer hash)`, so re-running an unchanged answer reuses the verdict. That makes the report *effectively* reproducible and keeps re-runs cheap.

Alongside the judge runs a cheap, fully deterministic second opinion: a **Voyage cosine similarity** between the answer and the reference. When the judge and the cosine disagree, that's a flag — two signals are better than one, and one of them never wobbles.

> *Why not just use a framework?* [RAGAS](https://docs.ragas.io/), [DeepEval](https://docs.confident-ai.com/), and [TruLens](https://www.trulens.org/) all ship these metrics, and they're good. But the *point* of a portfolio piece about RAG quality is to make the rubric and the variance guards **visible**, not bury them inside a wrapper. So the retrieval metrics are ~50 lines of from-scratch numpy, and the judge harness is explicit — with the frameworks as an optional second opinion rather than the source of truth. Knowing *when* to reach for the framework and when to show your work is itself the signal.

On a live single-item run, the judge scored a Chaosah answer all 1.0s with a per-dimension reason for each — *"faithfulness: all claims … are directly supported by the retrieved context"* — and the deterministic cosine agreed at 0.89. Two independent signals, one auditable.

---

## Failure Attribution: Whose Fault Was It? {#attribution}

Here's the move that makes this more than two metrics in a trench coat. Because the **same question** runs through *both* tools, the evaluator can tell retrieval failures apart from generation failures — the single most useful thing an eval can hand you, because it tells you *which half to fix*.

The logic is a deterministic 2×2. "Did the gold chunk reach generation?" comes from what `ask` reports it retrieved (cross-checked against `search` at the production `k` — they should match, since both call the identical retrieval path). "Was the answer good?" comes from the judge (or, without a key, the cosine). Cross them:

| Gold reached generation? | Answer good? | Verdict |
|---|---|---|
| ✅ | ✅ | **Pass** |
| ✅ | ❌ | 🔴 **Generation fault** — the model had the context and still blew it (hallucination / incompleteness / reasoning error) |
| ❌ | ❌ | 🟠 **Retrieval fault** — the evidence never surfaced; the answer never had a chance |
| ❌ | ✅ | ⚪ **Masked gap** — answered well *without* the gold → either the gold is mis-specified, or the model leaned on parametric knowledge (a faithfulness risk worth reviewing) |

A single quality score would collapse all four of those into "good" or "bad." The attribution table keeps them apart, so the report's headline isn't a number — it's a *diagnosis*: "3 generation faults, 1 retrieval fault, 1 masked gap to review." That tells you to go fix the prompt, not the retriever (or vice versa). **Building the consumer is how you learn what the producer actually does** — and attribution is where that pays off.

---

## Where the Gold Comes From — and a Self-Verifying Generator {#gold}

Every metric above leans on **gold** — the queries paired with their right answers. And here's the iron rule, the one a careful reviewer checks for: **the tools run the eval; they never supply the ground truth.** If the system under test also defined "correct," you'd be grading it against itself. So gold lives in the evaluator's repo, version-controlled and frozen, authored by a human-reviewed process — never pulled live from the app.

But hand-authoring gold doesn't scale, and that's where a little agency earns its place. `pepper-carrot-bootstrap` is a **self-verifying generator**: for each entity or page in a corpus snapshot, an LLM *drafts* a candidate — a query, a reference answer, the chunk that should be gold, even a harder **name-free relational** variant — and then **each candidate is verified against the live index before it's written**:

- Does `search` actually surface the proposed gold chunk, and at what rank? If it never appears in the top-k, the candidate is **auto-discarded** — the query is bad or the gold unreachable.
- For spoilers: is the reader position genuinely before the ending, and does the boundary hold? If not, discarded.

So a human reviewer only ever sees candidates that *already passed verification*. The agency is the **generate → verify → auto-discard** loop, and it's a clean example of the §2 boundary: the *drafting* is agentic, the *verification* reuses the same deterministic `search` the scored harness uses, and the output is a *candidate* — never the frozen gold — pending review.

The relational queries it writes are genuinely good. From the *Prince Acren* wiki summary, the generator proposed: *"What is the name of the young, uncrowned king whose land contains Squirrel's End, and who is unable to subjugate the flying city of Komona because his armies lack airborne combat training?"* — and the verifier confirmed Acren still ranks first. That's exactly the kind of hard, name-free query that makes recall@1 dip below 1.0 and the eval *discriminate*. Which is the honest engineering point: a too-easy gold set (every query literally names its answer) scores 1.000 and measures nothing; the generator helps manufacture *appropriately* hard cases, then proves they're still answerable. Tools invent and vet; a human reviews and freezes.

---

## Reasoning Traces: Every Score Is Auditable {#traces}

A score you can't explain is a score you can't trust. So every LLM and tool interaction — each `search`, each `ask`, each judge verdict, each generated candidate — is appended to a run-stamped JSONL file with the gold item it belongs to, the model, the Anthropic request id, the latency, and the full input/output *including the model's reasoning*. No dependency, just one JSON object per line, greppable with `jq`:

```bash
jq 'select(.item_id=="qa-wiki-chaosah")' traces/run-*.jsonl          # one item's whole chain
jq 'select(.component=="judge") | .output.reasoning' traces/run-*.jsonl   # per-dimension reasons
```

Here's a real item's trace, rendered — the nine records that produced **one** score in the report. The `ask` and `search` calls, the three judge samples (median-of-three) with their actual request ids and latencies, and the deterministic attribution join that turns them into a verdict:

<div style="margin: 1.5rem 0; overflow-x: auto;">
<a href="/assets/picture/2026-06-06-pepper-carrot-companion-evaluator/trace.svg" target="_blank" rel="noopener" title="Click to enlarge — opens the trace full-size in a new tab" style="display: block; cursor: zoom-in;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 716" role="img"
     aria-label="A detailed, field-level reasoning trace for one gold item (qa-wiki-chaosah, answer phase), verbatim from a real run of nine records. The gold item input question is 'What is Chaosah?' in wiki mode. Step 1, the ask tool (3.6 seconds) returns a 443-character answer beginning 'Chaosah is the school of chaos magic and the foundation from which all other magic on Hereva flows. It deals with time, gravity, particle physics', plus three retrieved document ids and two suggestion chips. Step 2, a search at k=3 (0.4 seconds) returns three wiki chunks with cosine scores 0.588, 0.535, and 0.465 — exactly what generation saw; their ids equal the ask call's retrieved_doc_ids, the consistency guarantee. Step 3, the judge (claude-sonnet-4-6) is sampled three times, median-of-3, with request ids msg_01Dj, msg_01CP, msg_01D3 and latencies 6.1, 11.6, 6.1 seconds; its input is the question plus reference plus the must-include fact 'chaos' plus the three chunks, and it returns per-dimension reasoning: correctness 1.0 (every listed fact matches the reference and context), faithfulness 1.0 (all claims supported by the retrieved context), relevance 1.0 (directly addresses the question), completeness 1.0 (the must-include fact is present). Median is all 1.0, variance 0.0, cached. A deterministic Voyage cosine second opinion is 0.89, agreeing. Step 4, a deterministic attribution join: retrieval_ok because the gold wiki chunk chaosah is in the retrieved chunks, and answer_ok because correctness and faithfulness clear 0.6, so the verdict is PASS."
     style="display: block; width: 100%; height: auto; max-width: 1000px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">
  <rect x="0" y="0" width="760" height="716" fill="#ffffff"/>

  <text x="14" y="28" font-size="16" font-weight="700" fill="#1f2937">Detailed trace — <tspan font-family="ui-monospace, Menlo, monospace" fill="#7c2d12">qa-wiki-chaosah</tspan> · answer phase</text>
  <text x="14" y="46" font-size="11" font-style="italic" fill="#64748b">verbatim from traces/run-20260606-223046.jsonl — text, scores, request ids and latencies are real</text>

  <!-- INPUT -->
  <rect x="14" y="58" width="732" height="34" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.2"/>
  <text x="28" y="80" font-size="13" fill="#1e3a8a"><tspan font-weight="700">gold item input</tspan> — question: “What is Chaosah?”   (mode: wiki)</text>

  <!-- ① ask -->
  <rect x="14" y="104" width="732" height="130" rx="7" fill="#ecfdf5" stroke="#059669" stroke-width="1.4"/>
  <text x="28" y="130" font-size="13.5" font-weight="700" fill="#065f46">① ask · 3.6s</text>
  <text x="150" y="130" font-size="12" fill="#065f46">· input {question: "What is Chaosah?", mode: wiki}</text>
  <text x="28" y="155" font-size="12.5" fill="#065f46">→ the real 443-character answer:</text>
  <text x="28" y="178" font-size="12.5" font-style="italic" fill="#047857">“Chaosah is the school of chaos magic and the foundation from which all</text>
  <text x="28" y="197" font-size="12.5" font-style="italic" fill="#047857">other magic on Hereva flows. It deals with time, gravity, particle physics…”</text>
  <text x="28" y="222" font-size="11.5" fill="#065f46">retrieved_doc_ids: [4f9dbb64…, 5c041e65…, d738cb2d…]  ·  +2 suggestion chips</text>

  <!-- ② search -->
  <rect x="14" y="246" width="732" height="86" rx="7" fill="#ecfdf5" stroke="#059669" stroke-width="1.4"/>
  <text x="28" y="272" font-size="13.5" font-weight="700" fill="#065f46">② search · k=3 · 0.4s</text>
  <text x="222" y="272" font-size="12" fill="#065f46">— recovers the faithfulness context</text>
  <text x="28" y="296" font-size="12.5" fill="#065f46">wiki #1  score 0.588      ·      #2  score 0.535      ·      #3  score 0.465</text>
  <text x="28" y="318" font-size="11" font-style="italic" fill="#475569">↳ these ids == ask.retrieved_doc_ids — the consistency guarantee (one retrieval path, both tools)</text>

  <!-- ③ judge -->
  <rect x="14" y="346" width="732" height="208" rx="7" fill="#fefce8" stroke="#ca8a04" stroke-width="1.4"/>
  <text x="28" y="372" font-size="13.5" font-weight="700" fill="#713f12">③ judge · claude-sonnet-4-6 · ×3 (median-of-3)</text>
  <text x="28" y="393" font-size="11" font-family="ui-monospace, Menlo, monospace" fill="#92400e">req msg_01Dj… / msg_01CP… / msg_01D3…   ·   latency 6.1 / 11.6 / 6.1 s</text>
  <text x="28" y="415" font-size="12" fill="#713f12">input: question + reference + must-include[“chaos”] + the 3 chunks  →  verdict + reasoning:</text>
  <text x="28" y="440" font-size="12.5" fill="#854d0e">●  <tspan font-weight="700">correctness 1.0</tspan> — every listed fact matches the reference and the retrieved context</text>
  <text x="28" y="464" font-size="12.5" fill="#854d0e">●  <tspan font-weight="700">faithfulness 1.0</tspan> — all claims are directly supported by the retrieved context</text>
  <text x="28" y="488" font-size="12.5" fill="#854d0e">●  <tspan font-weight="700">relevance 1.0</tspan> — directly and fully addresses “What is Chaosah?”</text>
  <text x="28" y="512" font-size="12.5" fill="#854d0e">●  <tspan font-weight="700">completeness 1.0</tspan> — must-include “chaos” present; core reference details covered</text>
  <text x="28" y="540" font-size="11" font-style="italic" fill="#713f12">median 1.0 / 1.0 / 1.0 / 1.0 · variance 0.00 (samples agree) · cached by (rubric v2, item, answer-hash)</text>

  <!-- similarity -->
  <text x="28" y="576" font-size="12.5" fill="#334155">＋  deterministic <tspan font-weight="700">Voyage cosine(answer, reference) = 0.89</tspan> — an independent second opinion (agrees)</text>

  <!-- ④ attribution -->
  <rect x="14" y="588" width="732" height="92" rx="7" fill="#dcfce7" stroke="#16a34a" stroke-width="1.8"/>
  <text x="28" y="614" font-size="13.5" font-weight="700" fill="#166534">④ failure attribution (deterministic join)</text>
  <text x="28" y="638" font-size="12.5" fill="#166534"><tspan font-weight="700">retrieval_ok ✓</tspan>  —  gold {wiki: chaosah} ∈ retrieved chunks</text>
  <text x="28" y="664" font-size="12.5" fill="#166534"><tspan font-weight="700">answer_ok ✓</tspan>  —  correctness ≥ 0.6  AND  faithfulness ≥ 0.6    →    <tspan font-weight="700" font-size="14">VERDICT: PASS</tspan></text>

  <text x="14" y="704" font-size="11" fill="#64748b">Every number in report.md expands into records like these — model, request id, latency, inputs, and the model's own reasoning.</text>
</svg>
</a>
</div>

*A real `qa-wiki-chaosah` trace: `ask` → `search` (the faithfulness context) → three judge samples → a deterministic PASS. Click to enlarge.*

And drilling into one of those judge records, the reasoning is stored **per dimension** — here, verbatim from the trace:

```
faithfulness: All claims — chaos magic school, base of Hereva's magic, time/gravity/particle
physics/underground forces, practical magic ethos, secrecy, black holes, demon summoning —
are directly supported by the retrieved context.
```

The tricky bit is correlation: the judge runs in a worker thread (the Anthropic SDK is synchronous), so how does its verdict know which gold item triggered it? A `ContextVar` set by the harness per item is *copied into* the worker thread by `asyncio.to_thread`, so a judge record traces cleanly back to its question. The result is that any score in the report can be expanded into the exact reasoning that produced it — and that JSONL ships straight into a trace UI like [Arize Phoenix](https://phoenix.arize.com/) or [Langfuse](https://langfuse.com/) the day you want a dashboard instead of `jq`.

---

## What's Honest, What's Open {#honest}

In the spirit of the series:

**The gold set is a reviewed seed, not a benchmark.** 39 retrieval / 14 Q&A / 14 refusal items, authored from the corpus and verified against the index. That's enough to exercise every metric and produce an honest, discriminating report — not enough to make confident claims about absolute quality. The self-verifying generator exists precisely so the set can grow without the authoring becoming a second job, but growth still gates on human review. The number that matters isn't "the app scored X"; it's "here is a reproducible harness that *would* catch a regression."

**The judge is not bit-reproducible, and the report says so.** LLM-as-judge wobbles; the guards (cross-model, anchored rubric, temperature 0, median-of-three, the cache) shrink the wobble but don't erase it. The deterministic layers — retrieval, similarity, attribution — *are* bit-reproducible, which is why the design pushes everything load-bearing into them and treats the judge as a guarded, cached signal rather than the source of truth. An honest eval is loud about which of its numbers are which.

**"All 1.000" is a failure mode, not a victory.** The first version of the gold scored a perfect 1.0 on everything, because every query named its own answer. That's a ruler with no markings. Making the eval *useful* meant deliberately authoring harder cases — name-free relational queries, page reach-backs where the right page ranks third because other episodes have similar scenes — until recall@1 fell to a believable 0.85. A perfect score is usually a sign your test is too easy, and saying so is part of the honesty.

**It costs real money to run the full thing.** The retrieval layer and the bootstrap *verification* are essentially free (embeddings only). The answer layer calls `ask` (real generations) and the judge (median-of-three on a Sonnet) — a few cents for the full set. The cache makes re-runs cheap, and `--limit 1` dry-runs the whole pipeline for a fraction of a cent. None of that is a problem at portfolio scale; it's just worth naming, because an eval you can't afford to run is an eval you won't.

---

## Key Takeaways {#key-takeaways}

**1. Grade the two failure modes separately, then join them.** Retrieval and generation fail differently, and a single answer-quality score can't tell them apart. Measuring each — then attributing a bad answer to one or the other — is the difference between "the eval said 6/10" and "the eval said *fix the retriever*."

**2. Draw a hard line between deterministic and agentic.** The metrics that track regressions must be reproducible; only case-invention and prose-judging get to be agentic, and even those never compute a score. An eval whose numbers move because the judge had a bad day is worse than no eval.

**3. An LLM judge is usable only with its guards on.** Cross-model, anchored rubric, forced structure, median-of-N, a cache. Without them you have a fluent random-number generator; with them you have a stable, auditable signal. Show the guards — they're the engineering.

**4. The tools run the eval; they never supply the ground truth.** Gold is owned by the evaluator, human-reviewed and frozen. A self-verifying generator can *scale* the authoring — drafting candidates and auto-discarding the ones the index can't surface — but a human still reviews before anything is trusted.

**5. A perfect score usually means a broken test.** If everything reads 1.000, your gold is too easy. Engineering *appropriately hard* cases (and being honest that you did) is what turns an eval from a rubber stamp into a measurement.

**6. The two halves of MCP are a server and a client — and the client is where the rigor lives.** Post 17 exposed the app as tools; this post consumed them to grade it. Same protocol, same two tools, opposite ends — and "I can measure my system, not just demo it" is the stronger portfolio sentence.

---

*The evaluator is its own repo: [`pepper-carrot-eval`](https://github.com/bearbearyu1223/pepper-carrot-eval) — clone it, point `MCP_SERVER_URL` at the [live server](https://pepper-carrot-mcp.fly.dev/mcp), and `uv run pepper-carrot-eval --retrieval-only` for a real scored retrieval report with no API key. It consumes the [`pepper-carrot-mcp`](https://github.com/bearbearyu1223/pepper-carrot-mcp) server from [Post 17]({% post_url 2026-06-06-pepper-carrot-companion-mcp-server %}), which wraps the [deployed companion](https://pepper-carrot-ai-flipbook.devcloudweb.com/). Three repos — app, server, evaluator — one system, graded from the outside through the same two tools any MCP client would use. Pepper & Carrot is © [David Revoy](https://www.peppercarrot.com), CC BY 4.0.*
