---
title: "Pepper & Carrot AI-powered flipbook · Part 18 — Grading the Companion: An Agentic Evaluator on the Other Side of MCP"
date: 2026-06-06 13:00:00 -0800
categories: [Full-Stack, RAG, MCP]
tags: [mcp, model-context-protocol, evaluation, llm-as-judge, ragas, retrieval, failure-attribution, claude, anthropic, voyage-ai, peppercarrot, portfolio]
description: >-
  Part 18 of the Pepper & Carrot AI flipbook series — the other half of the MCP
  story. Post 17 built an MCP *server* that exposed the deployed reading
  companion as two tools (search, ask). This post builds an *MCP client* that
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

But a demo can't answer the question that actually matters: is the companion any good? When it answers "who's on this page?", did it retrieve the right context, or just get lucky? When it sounds confident, is it grounded in the comic, or quietly making things up? And when something goes wrong, which half is to blame, the retriever or the model? This post builds the program that answers those questions: an **agentic evaluator** that is itself an MCP **client**, using the same `search` and `ask` tools to grade the app from the outside. ("Agentic" is narrow here — the evaluator *judges* answers and *drafts* test cases with an LLM, but it never lets a model decide which tool to call; that stays plain code. The three labels worth keeping straight — *evaluator*, *MCP client*, and *agentic* — are untangled in [§How an MCP Client Talks to the Server](#client).)

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

- [Table of Contents](#table-of-contents)
- [Why a Demo Isn't Evidence](#why)
- [How an MCP Client Talks to the Server](#client)
  - [Why a client, and not a Claude skill (or a direct API call)?](#why-a-client-and-not-a-claude-skill-or-a-direct-api-call)
- [The One Rule: What Stays Deterministic](#rule)
- [Instrument 1 — Grading Retrieval (deterministically)](#retrieval)
  - [The spoiler boundary, proven end-to-end](#the-spoiler-boundary-proven-end-to-end)
- [Instrument 2 — Grading Answers (with a judge)](#answers)
- [Failure Attribution: Whose Fault Was It?](#attribution)
- [Where the Gold Comes From — and a Self-Verifying Generator](#gold)
- [Reasoning Traces: Every Score Is Auditable](#traces)
- [What's Honest, What's Open](#honest)
- [Key Takeaways](#key-takeaways)
- [What's Next: Red-Teaming the Companion](#next)

---

## Why a Demo Isn't Evidence {#why}

The companion *looks* great in a demo. You ask "who's brewing the potion?", a fluent, on-topic answer streams back, and it feels done. But "it felt right on the three questions I tried" isn't evidence, and any reviewer who has shipped a real system knows it. Two things can be wrong underneath a confident answer:

- **Retrieval** can miss. The vector search might surface the wrong chunks — say, an unrelated wiki article that happens to share some vocabulary — and you'd never notice, because the model writes a smooth paragraph regardless.
- **Generation** can drift. Even when it's handed the *right* chunks, the model can ignore them, embellish, or invent a fact that isn't anywhere in the comic.

> *Plain-English aside: what "evaluating RAG" actually means.* A retrieval-augmented system has two stages — *find the relevant text*, then *write an answer from it* — so it has two ways to fail, and evaluating it means grading both **separately**. Retrieval is a *search* problem: did the right chunks come back, and were they ranked near the top? That's measurable with classic information-retrieval numbers, no LLM required. Answer quality is a *writing* problem: is the response correct, grounded, on-topic, complete? That's fuzzier — you need judgment. The mistake beginners make is to grade only the final answer; then a retrieval bug and a generation bug look identical, and you can't fix either with confidence. The whole design of this evaluator is to keep the two measurements apart, and then — the payoff — *join them* to assign blame.

An MCP server makes this clean, because it hands you the two stages as two separate tools. `search` is the retrieval stage with its scores and metadata exposed; `ask` is the full pipeline producing the real user-facing answer. Grade `search`, grade `ask`, and you've graded the two failure modes independently, using the same tools any other MCP client (Claude included) would call.

---

## How an MCP Client Talks to the Server {#client}

[Post 17]({% post_url 2026-06-06-pepper-carrot-companion-mcp-server %}) built the **server** — the thing that *offers* `search` and `ask`. This evaluator is the **client** — the thing that *calls* them. The two roles are worth pulling apart for a second, because the whole architecture rests on the split.

> *Plain-English aside: MCP client vs. server.* Picture the server as a wall socket that exposes a fixed set of tools, and a client as anything that plugs in and uses them. A client connects to the server's URL, asks "what do you offer?" (discovery), then calls a tool by name with JSON arguments and reads the result back. That's the *exact* handshake Claude performs when you add a custom connector — Claude is just an MCP client with a chat UI bolted on. The evaluator performs the same handshake, except it's a plain Python program, so every call is scripted and every response is captured to disk.

In practice it's a thin wrapper over a client library:

```python
# src/pepper_carrot_eval/client.py  (abbreviated)
from fastmcp import Client

class EvalMCPClient:
    def __init__(self, url):
        self._client = Client(url)            # the deployed server's /mcp URL

    async def search(self, *, query, mode, k, current_episode=None, current_page=None):
        args = {"query": query, "mode": mode, "k": k}
        if current_episode is not None: args["current_episode"] = current_episode
        if current_page is not None:    args["current_page"] = current_page
        result = await self._client.call_tool("search", args)   # the call Claude would make
        return result.structured_content
```

`call_tool("search", args)` is the same request a chat client fires when it decides to search — we've just put it under programmatic control, with the inputs fixed and the outputs recorded.

So when the intro calls the evaluator "an MCP client," that's shorthand. Strictly, the eval repo is **two** programs, and each one *contains* its own small MCP client that it uses as its line to the app. Laying them out untangles the labels, which are easy to conflate:

```text
pepper-carrot-eval — two programs, each its own MCP client of one server

Scored evaluator  (run.py)                  → writes the scored report
├─ gold sets ............. frozen test data
├─ deterministic metrics . recall@k · nDCG · MRR · spoiler check · attribution
├─ LLM judge ............. scores prose       (agentic, guarded)
├─ trace writer .......... the JSONL audit log
└─ EvalMCPClient ......... calls search / ask
         │
Gold generator  (bootstrap.py)              → writes candidate gold, offline
├─ LLM drafter ........... invents candidate cases   (agentic)
└─ EvalMCPClient ......... calls search to verify each candidate
         │                 (verified + human-reviewed → frozen into the gold sets above)
         │
         ▼   both clients call_tool(...) on the same endpoint
    MCP server (Post 17)  →  deployed app (the system under test)
```

Two things to read off that. First, **an MCP client is a small box, not the whole program** — each of the two programs has its own, and it's only their connection to the server (the scored evaluator also calls `ask`; the generator calls `search` alone). Second, **"agentic" is a property of particular boxes** — the judge and the drafter, where an LLM reasons — not of the tool calls, which plain code still dispatches. So the labels sit on different axes:

| Term | What it is | Decides which tool to call? |
|---|---|---|
| **MCP client** | a role — the thing that calls the tools | No — plain code dispatches it |
| **evaluator** | the whole program that grades the app | No |
| **"agentic"** | a property — it has LLM-reasoning parts | No |

In one line: an MCP client is *how* each program reaches the app; "agentic" is about the *reasoning* parts (judging answers, drafting cases); and neither means a program autonomously picks tools. An MCP client *can* be driven by an autonomous agent — that's exactly what Claude is — but these two are driven by scripts.

### Why a client, and not a Claude skill (or a direct API call)?

Two reasonable-sounding alternatives, and why neither fits *measurement*:

- **A Claude skill / agent.** You could hand Claude the tools and a prompt and let it grade the app. But an agent *improvises* — it decides what to ask, in what order, and when it's satisfied. That's the right behavior for an assistant and the wrong behavior for a ruler. An eval has to drive identical inputs every run and capture every output, or a moved number might just be the agent having a different day. The MCP client gives you the agent's exact interface with a script's determinism.
- **Calling the app's HTTP API directly.** You could skip MCP and hit the app's own endpoints. But then you'd be grading a *private* path, not the one real clients use. Driving the same `search`/`ask` tools that Claude calls means the eval exercises the actual integration surface — if a tool's schema, description, or behavior regresses, the eval catches it, because it's calling identically. There's no app-specific test backdoor to quietly drift out of sync with production.

The payoff is the symmetry the series keeps coming back to: because client and server speak one protocol, the evaluator is *just another MCP client* — the same role Claude plays — so nothing about the app or the server has to change to be graded. Post 17 built the grip; the evaluator is simply a second hand reaching for it.

---

## The One Rule: What Stays Deterministic {#rule}

Before any code, the single most important design decision — and the one a thoughtful reviewer will look for. An evaluator is only useful if its numbers are trustworthy over time: you change the prompt, re-run, and a number that moved means a real regression, not judge noise. So the rule is:

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

The agentic parts *feed* the deterministic metrics (by inventing test cases) and *explain* them (by judging prose), but they never *compute* them. That distinction is the spine of everything below. It's the same boundary the [provider-abstraction post]({% post_url 2026-05-13-pepper-carrot-companion-provider-abstractions %}) drew between "what changes and what doesn't" — here it's "what must be reproducible and what's allowed to reason."

You can see the line in the dispatch itself. The evaluator is an MCP client, but it is *not* an agent that decides which tool to reach for — the harness calls `search` and `ask` directly, in fixed order, gated only by which API keys are present:

```python
# src/pepper_carrot_eval/run.py  (abbreviated)
async with EvalMCPClient(cfg.mcp_server_url) as client:
    # Retrieval layer: deterministic, calls `search` only. Always runs (needs no key).
    retrieval_results = await retrieval_eval.evaluate_retrieval(client, gold_retrieval, index)

    if not retrieval_only and cfg.anthropic_api_key:      # answer + refusal layers
        # Answer layer: calls `ask` then `search`, then the judge + attribution join.
        answer_results = await answer_eval.evaluate_answers(
            client, gold_qa, index, judge=judge, similarity=similarity,
            production_k=cfg.production_k,
        )
        attributions = [attribution.attribute(item, by_id[item["id"]]) for item in gold_qa]
        refusal_results = await refusal_eval.evaluate_refusals(
            client, gold_refusal, index, judge=judge, production_k=cfg.production_k,
        )
```

No model chooses the path; `--retrieval-only` (or a missing key) just skips a branch. So the *conditions under which a tool is called* are visible, fixed, and greppable — which is exactly what "deterministic" has to mean if a moved number is going to mean a real regression.

> *Plain-English aside: but what if we let the evaluator drive the tools itself?* Tempting — give an LLM the tools and let it decide what to search and ask. For the *scored* loop, it's the wrong move, and the reasons are the same ones that make the eval worth running. **You'd lose reproducibility:** if an agent picks the query and the `k`, recall@k stops measuring the retriever and starts measuring how the agent phrased things that day, so a moved number no longer means a regression. **Attribution breaks:** the 2×2 needs the *same* question through both tools at the production `k`; an improvised paraphrase makes the retrieval-vs-generation comparison apples-to-oranges. **You'd grade two systems at once:** a bad score could be the app's fault or the driver's, and the clean "system under test" boundary is gone. There *is* a right place for an agent driving tools — but it's **discovery, not measurement**: an adversarial explorer that pokes the app to *find* failures you didn't think to write gold for (red-teaming, fuzzing). That's the same family as the [self-verifying gold generator](#gold) below — it lives on the agentic *edge*, proposing cases that get verified and frozen, and the deterministic harness still does the scoring. Chain the two (agent finds, harness measures); don't merge them (agent inside the scored loop).

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

The retrieval harness is the easy, satisfying part: pure math, no model, perfectly reproducible. For each test query you have a **gold set** — the chunk or chunks that *should* come back. You call `search`, then score the ranked results against that gold.

> *Plain-English aside: recall@k, nDCG, MRR in one breath.* All three ask "did the right chunks come back, and were they near the top?", differently. **recall@k** — of the chunks that should appear, what fraction landed in the top *k*? (Did we find them at all?) **MRR** (mean reciprocal rank) — how high was the *first* correct chunk? Rank 1 scores 1.0, rank 2 scores 0.5, rank 3 scores 0.33. (How good is the top of the list?) **nDCG@k** — a rank-weighted blend that rewards putting relevant chunks higher. You report them at several *k* values: recall@1 is "is the best answer literally first?", recall@3 is "is it in the top 3 the model actually gets fed?" These are decades-old information-retrieval metrics, and that's the point — they're boring, standard, and trustworthy.

All three are a few lines of plain Python over the keys `search` returns — no model, no framework, bit-identical every run:

```python
# src/pepper_carrot_eval/metrics.py
def recall_at_k(retrieved, gold, k):
    """Fraction of gold keys that appear in the top-k retrieved."""
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)

def mrr(retrieved, gold):
    """Reciprocal rank of the first gold hit (1-based); 0 if none."""
    for rank, key in enumerate(retrieved, start=1):
        if key in gold:
            return 1.0 / rank
    return 0.0

def _dcg(relevances):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

def ndcg_at_k(retrieved, gold, k):
    """Binary-relevance nDCG@k. 0 if no gold (IDCG would be 0)."""
    if not gold:
        return 0.0
    gains = [1.0 if key in gold else 0.0 for key in retrieved[:k]]
    idcg = _dcg([1.0] * min(len(gold), k))
    return _dcg(gains) / idcg if idcg else 0.0
```

One design detail that matters: **gold keys are stable, not Chroma ids.** A vector store assigns a fresh UUID to every chunk each time you re-ingest, so gold that referenced those UUIDs would rot the next time you ingested. Instead, a page chunk is keyed `("page", episode, page)` and a wiki chunk by its title — identifiers that survive a rebuild. The harness resolves those keys against whatever `search` returns today, so the gold outlives the index:

```python
# src/pepper_carrot_eval/corpus.py
def chunk_key(chunk):
    """Stable key for a chunk the `search` tool returned."""
    meta = chunk.get("metadata", {}) or {}
    table = chunk.get("source_table") or meta.get("source_table", "")
    if table == "pages":
        return ("page", int(meta["episode_number"]), int(meta["page_number"]))
    if table == "wiki":
        return ("wiki", str(chunk.get("text", "")).split("\n\n", 1)[0].strip().lower())
    return ("unknown", chunk.get("chroma_id", ""))

def gold_key(entry):
    """Same key shape, computed from a frozen gold entry."""
    if entry["type"] == "page":
        return ("page", int(entry["episode"]), int(entry["page"]))
    if entry["type"] == "wiki":
        return ("wiki", str(entry["title"]).strip().lower())
    raise ValueError(f"unknown gold chunk type: {entry['type']!r}")
```

Scoring is then just set math over `gold_key(...)` and `chunk_key(...)` — the gold and the live results meet on a key that no re-ingestion can churn.

Run it against the live server with no key at all, and the headline on the real (39-item) gold set is:

```
overall  recall@1 = 0.846   recall@3 = 1.000   nDCG@3 = 0.937   MRR = 0.915
page     recall@1 = 0.700   recall@3 = 1.000   MRR  = 0.817
wiki     recall@1 = 0.897   recall@3 = 1.000   MRR  = 0.948
```

Read those two columns together, because they tell a story. **recall@3 = 1.000 everywhere** means every gold chunk is reachable: the gold isn't broken, and within the top 3 the model always gets the right context. But **recall@1 = 0.846** means the *best* chunk lands first only about 85% of the time. That gap is the eval doing its job — it's discriminating. If every number came back 1.000, you'd have a useless eval: a ruler that says everything is perfect measures nothing. (More on engineering that honestly in §6.)

### The spoiler boundary, proven end-to-end

There's one retrieval metric that isn't about quality at all — it's about safety. The whole series leans on a [spoiler-safe boundary]({% post_url 2026-05-25-pepper-carrot-companion-spoiler-safe-rag %}): page-mode retrieval must never surface a page the reader hasn't reached yet. That was unit-tested deep in the app. But a unit test only proves the *function* works; it says nothing about whether the *deployed system, reached through a tool,* still honors it. So the evaluator re-checks it structurally, end-to-end: drive `search` at a reader position and assert that no returned chunk sits at or past the cursor. The whole check is six lines:

```python
# src/pepper_carrot_eval/refusal_eval.py
def _past_boundary(seen_keys, episode, page):
    """True if any retrieved page key is at/after the (episode, page) cursor."""
    for key in seen_keys:
        if len(key) == 3 and key[0] == "page" and (int(key[1]), int(key[2])) >= (episode, page):
            return True
    return False
```

It's deterministic, it needs no key, and it turns "we wrote a spoiler filter" into "we continuously verify, from the outside, that the live system doesn't leak." That's a much stronger claim, and exactly the kind a reviewer trusts.

---

## Instrument 2 — Grading Answers (with a judge) {#answers}

Retrieval math can't tell you whether *"Carrot is the only one here, pouring a bright orange potion with guilty glug-glug sounds"* is a good answer. Prose needs judgment, and the standard way to get judgment at scale is **LLM-as-judge**: you ask a strong model to score the answer against a rubric.

> *Plain-English aside: LLM-as-judge, and why it needs guardrails.* Using a model to grade another model's output sounds circular, and done naively it is. Models are inconsistent graders — ask twice, get two scores — and they have biases: they reward fluent, long answers, and they tend to *prefer outputs from their own family* (self-enhancement bias). So an LLM judge without guards is a wobbly ruler. The fix isn't to abandon it — for open-ended prose there's no better option at scale — it's to *engineer the variance down* until the number is stable enough to track.

The judge here scores four dimensions — **correctness** (against a reference answer), **faithfulness** (is every claim supported by the retrieved context?), **relevance** (does it address the question?), and **completeness** (are the must-have facts present?) — and it carries five concrete guards:

- **A cross-model judge.** The app answers with `claude-haiku-4-5`; the judge is `claude-sonnet-4-6`, a stronger and deliberately *different* model, to sidestep self-preference bias.
- **An anchored rubric.** Each dimension has described 0 / 0.5 / 1 anchors ("1.0 = every fact matches; 0.5 = mostly right with one wrong fact; 0.0 = contradicts the reference"). Anchoring is the single biggest variance reducer.
- **Forced structured output.** The judge must return its scores through a tool call with a fixed schema, so there's no free-form drift, plus a `reasoning` field *per dimension* so every score comes with its justification.
- **Temperature 0 and median-of-N.** It samples the verdict three times and takes the median, reporting the spread as a variance signal; a wide spread flags the item for human review.
- **A disk cache.** Verdicts are keyed by `(rubric version, item, answer hash)`, so re-running an unchanged answer reuses the verdict. That makes the report *effectively* reproducible and keeps re-runs cheap.

The anchored rubric isn't a separate config file — it's the judge's system prompt, with the 0 / 0.5 / 1 anchors written straight into each dimension so the model has a fixed yardstick instead of its own mood:

```python
# src/pepper_carrot_eval/judge.py
_ANSWER_SYSTEM = """You are a rigorous evaluator of a reading-companion's answers about the \
webcomic Pepper & Carrot. Score the ANSWER on four dimensions, each from 0.0 to 1.0:

- correctness: does the answer agree with the REFERENCE answer's facts?
  1.0 = every fact matches; 0.5 = mostly right with one wrong/missing fact; 0.0 = contradicts it.
- faithfulness: is every claim in the answer supported by the RETRIEVED CONTEXT?
  1.0 = fully grounded; 0.5 = one unsupported aside; 0.0 = invents facts not in the context.
- relevance: does the answer actually address the QUESTION?
  1.0 = directly answers; 0.5 = partially; 0.0 = off-topic.
- completeness: are the MUST-INCLUDE facts present?
  1.0 = all present; 0.5 = some; 0.0 = none.

Judge only what is shown. Ignore length and writing style. Do not reward fluency that isn't \
grounded. For EACH dimension give a one-sentence reason, and an overall rationale. Record \
everything with the record_judgment tool."""
```

Two phrases there are doing real anti-bias work: *"Ignore length and writing style"* and *"Do not reward fluency that isn't grounded"* directly counter the length/fluency bias the aside above warned about.

Three more guards are only a handful of lines — sample N times, take the median, report the spread, and key the cache so an unchanged answer is never re-judged:

```python
# src/pepper_carrot_eval/judge.py  (abbreviated)
for _ in range(self._samples):                       # temperature 0, N samples (default 3)
    out = self._call(_ANSWER_SYSTEM, user, _ANSWER_TOOL, "record_judgment")
    samples.append({d: float(out.get(d, 0.0)) for d in dims})

median = {d: statistics.median(s[d] for s in samples) for d in dims}
variance = max(                                      # wide spread → flag for human review
    (max(s[d] for s in samples) - min(s[d] for s in samples)) for d in dims
)

def _cache_path(self, kind, item_id, answer):        # an unchanged answer reuses its verdict
    digest = hashlib.sha256(
        f"{RUBRIC_VERSION}|{self._model}|{kind}|{item_id}|{answer}".encode()
    ).hexdigest()[:16]
    return self._cache_dir / f"{kind}-{item_id}-{digest}.json"
```

The `_call` there hands the model a fixed `record_judgment` tool whose schema carries the four numeric scores plus a `reasoning` string *per dimension* — that's the "forced structured output" guard, with the per-dimension reason coming back attached to every score. And "forced" is literal: the call pins the temperature *and* the tool, so the model can't free-form a paragraph or skip the schema:

```python
# src/pepper_carrot_eval/judge.py  (the judge call)
resp = self._client.messages.create(
    model=self._model,                                   # claude-sonnet-4-6, not the haiku under test
    temperature=0,
    system=system,                                       # the anchored rubric above
    tools=[tool],
    tool_choice={"type": "tool", "name": tool_name},     # MUST call record_judgment
    messages=[{"role": "user", "content": user}],
)
```

This is the one place a *model* emits a tool call anywhere in the evaluator — and even here it isn't the model deciding to; `tool_choice` compels it. Every `search` and `ask` is dispatched by the plain harness control flow from §2, never by a model. That's what keeps "agentic" confined to the judging itself.

Alongside the judge runs a cheap, fully deterministic second opinion: a **Voyage cosine similarity** between the answer and the reference. When the judge and the cosine disagree, that's a flag. Two signals are better than one, and one of them never wobbles.

> *Why not just use a framework?* [RAGAS](https://docs.ragas.io/), [DeepEval](https://docs.confident-ai.com/), and [TruLens](https://www.trulens.org/) all ship these metrics, and they're good. But the *point* of a portfolio piece about RAG quality is to make the rubric and the variance guards **visible**, not bury them inside a wrapper. So the retrieval metrics are ~50 lines of from-scratch Python (you saw them above), and the judge harness is explicit — with the frameworks as an optional second opinion rather than the source of truth. Knowing *when* to reach for the framework and when to show your work is itself the signal.

On a live single-item run, the judge scored a Chaosah answer all 1.0s, with a reason attached to each dimension — *"faithfulness: all claims … are directly supported by the retrieved context"* — and the deterministic cosine agreed at 0.89. Two independent signals, both auditable.

---

## Failure Attribution: Whose Fault Was It? {#attribution}

Here's the move that makes this more than two metrics in a trench coat. Because the **same question** runs through *both* tools, the evaluator can tell retrieval failures apart from generation failures — the single most useful thing an eval can hand you, because it tells you which half to go fix.

The logic is a deterministic 2×2. "Did the gold chunk reach generation?" comes from what `ask` reports it retrieved, cross-checked against `search` at the production `k` — they should match, since both call the identical retrieval path. "Was the answer good?" comes from the judge, or, without a key, from the cosine. Cross them:

| Gold reached generation? | Answer good? | Verdict |
|---|---|---|
| ✅ | ✅ | **Pass** |
| ✅ | ❌ | 🔴 **Generation fault** — the model had the context and still blew it (hallucination / incompleteness / reasoning error) |
| ❌ | ❌ | 🟠 **Retrieval fault** — the evidence never surfaced; the answer never had a chance |
| ❌ | ✅ | ⚪ **Masked gap** — answered well *without* the gold → either the gold is mis-specified, or the model leaned on parametric knowledge (a faithfulness risk worth reviewing) |

In code, the join is literally that 2×2 — two booleans and four comparisons, no model anywhere in it:

```python
# src/pepper_carrot_eval/attribution.py  (abbreviated)
retrieval_ok = bool(must_cite & set(result.seen_keys)) if must_cite else None
answer_ok = _answer_ok(result, correctness_min=0.6, faithfulness_min=0.6, ...)

if retrieval_ok and answer_ok:
    verdict = PASS              # gold reached generation and the answer is good
elif retrieval_ok and not answer_ok:
    verdict = GENERATION_FAULT  # had the context, still blew it
elif not retrieval_ok and not answer_ok:
    verdict = RETRIEVAL_FAULT   # the evidence never surfaced
else:                           # not retrieval_ok and answer_ok
    verdict = MASKED_GAP        # answered well without the gold — review it
```

A single quality score would collapse all four of those into "good" or "bad." The attribution table keeps them apart, so the report's headline isn't a number — it's a diagnosis: "3 generation faults, 1 retrieval fault, 1 masked gap to review." That tells you to go fix the prompt rather than the retriever, or the other way around. Building the consumer is how you learn what the producer actually does, and attribution is where that pays off.

---

## Where the Gold Comes From — and a Self-Verifying Generator {#gold}

Every metric above leans on **gold** — the queries paired with their right answers. And here's the iron rule, the one a careful reviewer checks for: the tools run the eval, but they never supply the ground truth. If the system under test also got to define "correct," you'd be grading it against itself. So gold lives in the evaluator's repo, version-controlled and frozen, authored by a human-reviewed process, never pulled live from the app.

But hand-authoring gold doesn't scale, and that's where a little agency earns its place. `pepper-carrot-bootstrap` is a **self-verifying generator**: for each entity or page in a corpus snapshot, an LLM *drafts* a candidate — a query, a reference answer, the chunk that should be gold, even a harder name-free relational variant — and then each candidate is verified against the live index before it's written:

- Does `search` actually surface the proposed gold chunk, and at what rank? If it never appears in the top-k, the candidate is **auto-discarded** — the query is bad or the gold unreachable.
- For spoilers: is the reader position genuinely before the ending, and does the boundary hold? If not, discarded.

So a human reviewer only ever sees candidates that already passed verification. And the verifier isn't anything new — it's the same deterministic `search`, asked "does the proposed gold actually rank, and does the boundary still hold?":

```python
# src/pepper_carrot_eval/bootstrap.py  (abbreviated)
async def _wiki_rank(client, query, gold_title):
    result = await client.search(query=query, mode="wiki", k=10)
    titles = [k[1] for k in retrieved_keys(result) if k[0] == "wiki"]
    gold = gold_title.strip().lower()
    return titles.index(gold) + 1 if gold in titles else 0   # 0 = never surfaced → discard

async def _spoiler_holds(client, query, ep, reader_page):
    result = await client.search(query=query, mode="page", k=10,
                                 current_episode=ep, current_page=reader_page)
    pages = [(k[1], k[2]) for k in retrieved_keys(result) if k[0] == "page"]
    return not any((e, p) >= (ep, reader_page) for e, p in pages)
```

A `rank` of 0, or a boundary that doesn't hold, drops the candidate before a human ever sees it. The agency is the **generate → verify → auto-discard** loop, and it's a clean example of the §2 boundary: the *drafting* is agentic, the *verification* reuses the same deterministic `search` the scored harness uses, and the output is a candidate — never the frozen gold — pending review.

The relational queries it writes are genuinely good. From the *Prince Acren* wiki summary, the generator proposed: *"What is the name of the young, uncrowned king whose land contains Squirrel's End, and who is unable to subjugate the flying city of Komona because his armies lack airborne combat training?"* — and the verifier confirmed Acren still ranks first. That's exactly the kind of hard, name-free query that makes recall@1 dip below 1.0 and forces the eval to discriminate. And that's the honest engineering point: a too-easy gold set, where every query literally names its answer, scores 1.000 and measures nothing. The generator helps manufacture *appropriately* hard cases, then proves they're still answerable. Tools invent and vet; a human reviews and freezes.

---

## Reasoning Traces: Every Score Is Auditable {#traces}

A score you can't explain is a score you can't trust. So every LLM and tool interaction — each `search`, each `ask`, each judge verdict, each generated candidate — gets appended to a run-stamped JSONL file, along with the gold item it belongs to, the model, the Anthropic request id, the latency, and the full input and output *including the model's reasoning*. No dependency, just one JSON object per line, greppable with `jq`:

```bash
jq 'select(.item_id=="qa-wiki-chaosah")' traces/run-*.jsonl          # one item's whole chain
jq 'select(.component=="judge") | .output.reasoning' traces/run-*.jsonl   # per-dimension reasons
```

Here's a real item's trace, pretty-printed straight from the JSONL — the five records behind one score in the report. The `ask` call, the `search` that recovers the same chunks generation saw, and the judge (sampled three times for median-of-three; one sample shown). The attribution verdict isn't a stored record — it's computed deterministically from these:

```jsonc
// $ jq 'select(.item_id=="qa-wiki-chaosah")' traces/run-20260606-223046.jsonl

{ "component": "ask", "phase": "answer", "latency_s": 3.565,
  "input":  { "question": "What is Chaosah?", "mode": "wiki" },
  "output": {
    "answer_chars": 443,
    "answer_head": "Chaosah is the school of chaos magic and the foundation from which all other magic on Hereva flows. It deals with time, gravity, particle physics…",
    "retrieved_doc_ids": ["4f9dbb64…", "5c041e65…", "d738cb2d…"],
    "suggestions": [
      { "mode": "page", "text": "What Chaosah magic is being used on this page" },
      { "mode": "wiki", "text": "How do Chaosah witches create black holes and portals" }
    ] } }

{ "component": "search", "phase": "answer", "latency_s": 0.376,
  "input":  { "query": "What is Chaosah?", "mode": "wiki", "k": 3 },
  "output": { "n_chunks": 3, "chunks": [          // the same chunks ask saw — one retrieval path
    { "table": "wiki", "score": 0.588 },
    { "table": "wiki", "score": 0.535 },
    { "table": "wiki", "score": 0.465 } ] } }

{ "component": "judge", "model": "claude-sonnet-4-6",          // sample 1 of 3
  "request_id": "msg_01DjKDpo1soVrBX7xGsLPsPP", "latency_s": 6.147,
  "input":  { "tool": "record_judgment",
              "user_head": "QUESTION:\nWhat is Chaosah?\n\nREFERENCE ANSWER:\n…\n\nMUST-INCLUDE FACTS:\n- chaos" },
  "output": {
    "correctness": 1.0, "faithfulness": 1.0, "relevance": 1.0, "completeness": 1.0,
    "reasoning": {
      "faithfulness": "All claims — chaos magic school, base of Hereva's magic, time/gravity/particle physics/underground forces, practical magic ethos, secrecy, black holes, demon summoning — are directly supported by the retrieved context.",
      "completeness": "The must-include fact 'chaos' is explicitly present ('school of chaos magic'), and the core details from the reference answer are all covered."
      // correctness + relevance reasoning omitted for length
    },
    "rationale": "Fully correct, grounded in the retrieved context, directly addresses the question, includes all must-have facts." } }
```

Three of the five records are above; the other two are the second and third judge samples — near-identical, and the report uses their **median** (all `1.0`, variance `0.00`). Two things to notice. The `search` record's chunks are exactly the ones `ask` retrieved, which is the consistency guarantee both tools share. And there is *no* attribution record: the verdict is a deterministic join over what you see here — `retrieval_ok` (the gold `chaosah` chunk is in the set) AND `answer_ok` (correctness and faithfulness both clear `0.6`) — which lands this item on **PASS**.

The `reasoning` object carries one sentence **per dimension**; the `correctness` reason on that same record, verbatim from the trace, reads:

```
correctness: Every fact in the answer (chaos magic, foundation of all magic on Hereva, time,
gravity, particle physics, underground divine forces, black holes, demon summoning) matches
the reference and context perfectly.
```

The tricky bit is correlation: the judge runs in a worker thread (the Anthropic SDK is synchronous), so how does its verdict know which gold item triggered it? A `ContextVar` the harness sets per item is *copied into* the worker thread by `asyncio.to_thread`, so a judge record traces cleanly back to its question:

```python
# src/pepper_carrot_eval/tracing.py
_ctx: ContextVar[dict | None] = ContextVar("trace_ctx", default=None)

def set_item(item_id, phase):
    """Tag subsequent records with the gold item + phase (call at each loop top)."""
    _ctx.set({"item_id": item_id, "phase": phase})

# …later, in the answer harness — the ContextVar rides into the worker thread:
judgment = await asyncio.to_thread(judge.score_answer, item_id=item["id"], ...)
```

Every record the tracer writes reads `_ctx.get()` for the item id, so the judge's verdict — emitted from another thread — still lands under the right question. The upshot is that any score in the report can be expanded into the exact reasoning that produced it — and that JSONL ships straight into a trace UI like [Arize Phoenix](https://phoenix.arize.com/) or [Langfuse](https://langfuse.com/) the day you want a dashboard instead of `jq`.

---

## What's Honest, What's Open {#honest}

In the spirit of the series:

**The gold set is a reviewed seed, not a benchmark.** 39 retrieval / 14 Q&A / 14 refusal items, authored from the corpus and verified against the index. That's enough to exercise every metric and produce an honest, discriminating report, but not enough to make confident claims about absolute quality. The self-verifying generator exists precisely so the set can grow without the authoring becoming a second job, though growth still gates on human review. The number that matters isn't "the app scored X"; it's "here is a reproducible harness that *would* catch a regression."

**The judge is not bit-reproducible, and the report says so.** LLM-as-judge wobbles; the guards — cross-model, anchored rubric, temperature 0, median-of-three, the cache — shrink the wobble but don't erase it. The deterministic layers (retrieval, similarity, attribution) *are* bit-reproducible, which is why the design pushes everything load-bearing into them and treats the judge as a guarded, cached signal rather than the source of truth. An honest eval is loud about which of its numbers are which.

**"All 1.000" is a failure mode, not a victory.** The first version of the gold scored a perfect 1.0 on everything, because every query named its own answer. That's a ruler with no markings. Making the eval useful meant deliberately authoring harder cases — name-free relational queries, page reach-backs where the right page ranks third because other episodes have similar scenes — until recall@1 fell to a believable 0.85. A perfect score is usually a sign your test is too easy, and saying so out loud is part of the honesty.

**It costs real money to run the full thing.** The retrieval layer and the bootstrap *verification* are essentially free (embeddings only). The answer layer calls `ask` (real generations) and the judge (median-of-three on a Sonnet), which runs a few cents for the full set. The cache makes re-runs cheap, and `--limit 1` dry-runs the whole pipeline for a fraction of a cent. None of that is a problem at portfolio scale; it's just worth naming, because an eval you can't afford to run is an eval you won't.

---

## Key Takeaways {#key-takeaways}

**1. Grade the two failure modes separately, then join them.** Retrieval and generation fail differently, and a single answer-quality score can't tell them apart. Measuring each, then attributing a bad answer to one or the other, is the difference between "the eval said 6/10" and "the eval said *fix the retriever*."

**2. Draw a hard line between deterministic and agentic.** The metrics that track regressions have to be reproducible; only case-invention and prose-judging get to be agentic, and even those never compute a score. An eval whose numbers move because the judge had a bad day is worse than no eval.

**3. An LLM judge is only usable with its guards on.** Cross-model, anchored rubric, forced structure, median-of-N, a cache. Without them you have a fluent random-number generator; with them you have a stable, auditable signal. Show the guards — they're the engineering.

**4. The tools run the eval; they never supply the ground truth.** Gold is owned by the evaluator, human-reviewed and frozen. A self-verifying generator can *scale* the authoring — drafting candidates and auto-discarding the ones the index can't surface — but a human still reviews before anything is trusted.

**5. A perfect score usually means a broken test.** If everything reads 1.000, your gold is too easy. Engineering *appropriately hard* cases, and being honest that you did, is what turns an eval from a rubber stamp into a measurement.

**6. The two halves of MCP are a server and a client, and the client is where the rigor lives.** Post 17 exposed the app as tools; this post consumed them to grade it. Same protocol, same two tools, opposite ends — and "I can measure my system, not just demo it" is the stronger portfolio sentence.

---

## What's Next: Red-Teaming the Companion {#next}

This post's evaluator is the *measurement* half, and it has one structural blind spot — the one [§The One Rule](#rule) keeps circling: it only catches the failures someone *already wrote a gold case for*. The complement is an agent that goes looking for the rest.

> A deterministic eval can only catch failures you already wrote a test for. An agentic red-teamer finds the failures you didn't think of — then hands them to the deterministic harness to guard forever.

That's the next build: **`pepper-carrot-redteam`** — a *third* MCP client of the same server, but this one is genuinely agentic. It's handed the same `search`/`ask` tools and a mission ("try to make it spoil," "get it to confidently invent lore"), and *it* decides the probes, adapts across turns, and reports what broke. It keeps the eval's discipline, though: it explores agentically but judges *structurally* wherever it can — the spoiler oracle is the very same boundary check from [Instrument 1](#retrieval) — and every confirmed failure is written back as **candidate gold** for the deterministic harness. Find once, guard forever. That's the post where the agentic edge finally gets to drive.

---

*The evaluator is its own repo: [`pepper-carrot-eval`](https://github.com/bearbearyu1223/pepper-carrot-eval) — clone it, point `MCP_SERVER_URL` at the [live server](https://pepper-carrot-mcp.fly.dev/mcp), and `uv run pepper-carrot-eval --retrieval-only` for a real scored retrieval report with no API key. It consumes the [`pepper-carrot-mcp`](https://github.com/bearbearyu1223/pepper-carrot-mcp) server from [Post 17]({% post_url 2026-06-06-pepper-carrot-companion-mcp-server %}), which wraps the [deployed companion](https://pepper-carrot-ai-flipbook.devcloudweb.com/). Three repos — app, server, evaluator — one system, graded from the outside through the same two tools any MCP client would use. Pepper & Carrot is © [David Revoy](https://www.peppercarrot.com), CC BY 4.0.*
