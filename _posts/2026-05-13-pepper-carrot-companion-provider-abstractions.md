---
title: "Provider Abstractions: Why Every External Service Hides Behind an Interface"
date: 2026-05-13 00:00:00 -0800
categories: [Full-Stack, RAG, Local AI]
tags: [python, protocol, fastapi, ollama, sentence-transformers, httpx, aiofiles, cloudflare-r2, peppercarrot, portfolio]
description: >-
  Post 3 of the Pepper & Carrot AI flipbook series. Build three typed
  Protocol interfaces — Storage, EmbeddingClient, ChatClient — and the
  factory that picks the right implementation from a .env file. By the
  end you have LocalStorage serving images end-to-end and a working
  embedding client producing real 1024-dim vectors against the local
  Ollama you set up in Post 2.
pin: true
---

Post 3 of the [*Pepper & Carrot AI-powered flipbook*]({% post_url 2026-05-09-pepper-carrot-companion-trailer %})
series. With the workshop standing from
[Post 2]({% post_url 2026-05-10-pepper-carrot-companion-workshop %}), we
write the first code that the rest of the project will sit on top of:
three small Python [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)
interfaces and a factory that picks the right implementation from a
`.env` file. Nothing fancy — ~250 lines of code total — but everything
the next seven posts touches is going to flow through these three
seams.

> **What you'll build in this post.**
> - A `Storage` Protocol with a working `LocalStorage` implementation that writes to `./data/images/` and serves images via FastAPI's [`StaticFiles`](https://fastapi.tiangolo.com/tutorial/static-files/).
> - An `EmbeddingClient` Protocol with two implementations — `OllamaEmbeddingClient` (the project default, talking to the local Ollama you set up in Post 2) and `SentenceTransformersEmbeddingClient` (a zero-network fallback).
> - A `ChatClient` Protocol with `OllamaChatClient` for the local path and `AnthropicChatClient` for the cloud swap-in. (We don't *use* the chat clients in this post — the streaming pipeline lands in Post 6 — but defining them now lets us see all three abstractions in one place.)
> - A factory in `backend/app/clients/__init__.py` that reads `Settings` and hands the rest of the app back a Protocol-typed instance.
> - A smoke test that proves you can swap `EMBEDDING_PROVIDER=ollama` for `EMBEDDING_PROVIDER=sentence-transformers` without changing a single line of caller code.
>
> **Prerequisites.**
> - The workshop from [Post 2]({% post_url 2026-05-10-pepper-carrot-companion-workshop %}) up and green: Postgres healthy in Docker, `ollama serve` running with `qwen2.5:7b` and `bge-m3` pulled, `uv run mypy app/` and `uv run ruff check app/` both clean, and a copy of `.env.example` sitting at `.env`.
> - No new tools to install. We're spending this post entirely inside `backend/app/clients/`.

---

## Table of Contents

1. [The Rule, in One Sentence](#the-rule)
2. [What "Provider Abstraction" Actually Means](#what-it-means)
3. [Three Things This Buys You](#three-things)
4. [The Three Seams in One Picture](#three-seams)
5. [Seam 1 — Storage: LocalStorage End to End](#seam-storage)
6. [Seam 2 — EmbeddingClient: Two Implementations of the Same Protocol](#seam-embedding)
7. [Seam 3 — ChatClient: A Preview of What Post 6 Will Use](#seam-chat)
8. [The Factory in `clients/__init__.py`](#factory)
9. [Verification: Prove the Swap Works](#verification)
10. [The Discipline That Makes This Work](#discipline)
11. [Key Takeaways](#key-takeaways)

---

## The Rule, in One Sentence {#the-rule}

The project's [`CLAUDE.md`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/CLAUDE.md) — the file every contributor (human or AI) reads first — states the rule plainly:

> **Never import `anthropic`, `openai`, `chromadb`, `boto3`, or `ollama` SDKs directly outside of `backend/app/clients/`. Every external service goes through an interface. This is what makes local→cloud migration trivial.**

That's the whole rule. Everything in this post is what it takes to make that rule cheap to follow.

If you've worked on a codebase that *didn't* follow this rule, you know the texture of the pain: someone added an `import openai` to a route handler nine months ago because it was easy; a year later, switching to a different provider means grepping for `openai` across 40 files, untangling differences in parameter names, and discovering you've subtly relied on OpenAI-specific behavior in three places. The fix takes a week. The damage isn't the week — it's the *next* time you want to swap something, when you already know the cost up front and don't bother.

A typed [`Protocol`](https://peps.python.org/pep-0544/) interface is the cheapest possible insurance policy against that. Cheaper than picking a "right" provider; cheaper than building a [DI framework](https://en.wikipedia.org/wiki/Dependency_injection); cheaper than a config-loading library. Roughly 30 lines of Python, *most of which is the docstring*. We'll see exactly how cheap as we walk through the file.

---

## What "Provider Abstraction" Actually Means {#what-it-means}

> *Plain-English aside.* The phrase "provider abstraction" sounds bigger than it is. It just means: **the calling code doesn't know which concrete service is on the other end of the call**. A route handler says "give me an embedding for this string"; whatever object answers that call could be talking to a local Ollama, a hosted Anthropic API, or a stub returning zeros for tests. The route handler can't tell the difference, and — this is the whole point — *doesn't have to*.
>
> In statically typed languages this pattern usually shows up as an "interface" (Java, C#, Go). Python doesn't have a keyword for it, but `Protocol` from the [`typing`](https://docs.python.org/3/library/typing.html) module is the equivalent.

The mechanism in this codebase is a Python [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol) — Python's name for *structural* typing.

```python
# from backend/app/clients/storage.py
from typing import Protocol

class Storage(Protocol):
    async def put(self, key: str, content: bytes, content_type: str) -> None: ...
    async def url_for(self, key: str) -> str: ...
    async def exists(self, key: str) -> bool: ...
```

That's the entire interface. Three async methods. The `...` is Python syntax for "no body" — a Protocol declares the *shape*, not the *behavior*. Any class that defines those three methods with those signatures *is* a `Storage` as far as the type-checker is concerned, even without inheriting from anything.

> *Why "structural"?* In [nominal](https://en.wikipedia.org/wiki/Nominal_type_system) typing — Java, C#, etc. — a class only implements an interface if it explicitly says `class LocalStorage implements Storage`. In [structural](https://en.wikipedia.org/wiki/Structural_type_system) typing, the relationship is inferred from the shape: if `LocalStorage` has those three methods, it *is* a `Storage`. Python's `Protocol` is structural by default. [`mypy --strict`](https://mypy-lang.org/) catches the mismatch if a method signature drifts; the runtime doesn't need to know anything about the relationship.
>
> The practical payoff: implementations don't have to import the Protocol. `LocalStorage` doesn't say `class LocalStorage(Storage):` — it just defines the right methods, and `mypy` keeps everyone honest.

Everything downstream of the rule follows from this one line: `Storage` is a `Protocol`, not a base class. Three other Protocols live alongside it in the same package — `EmbeddingClient`, `ChatClient`, and `VisionClient` (covered in Post 4) — and they all look the same shape.

---

## Three Things This Buys You {#three-things}

Before we look at code, three concrete payoffs from the next ~250 lines. Each one shows up in a later post in this series.

**1. Local↔cloud swap is a config change.**

The same backend binary boots two different deployment shapes depending on the value of three environment variables. Run it with `STORAGE_BACKEND=local CHAT_PROVIDER=ollama EMBEDDING_PROVIDER=ollama` and you get the all-local workshop you built in Post 2. Run it with `STORAGE_BACKEND=r2 OLLAMA_BASE_URL=<modal-url>` and it talks to [Cloudflare R2](https://www.cloudflare.com/developer-platform/products/r2/) for images and a Modal-hosted Ollama for chat — same code, same imports, same call sites. Post 10 is just a long checklist of "set this env var to that production value"; the codebase changes by zero lines.

**2. Hybrid setups become free.**

You might want local embeddings (cheap, no rate limits, runs while you sleep ingesting 39 episodes) but cloud chat (better quality on small models, no GPU on your laptop). With three independent provider selectors — `STORAGE_BACKEND`, `CHAT_PROVIDER`, `EMBEDDING_PROVIDER` — this is literally just `CHAT_PROVIDER=anthropic EMBEDDING_PROVIDER=ollama` in your `.env`. There's no "hybrid mode" feature to design — every combination already works, because none of the three seams knows the others exist.

**3. Tests stop touching the network.**

When the only place SDKs are imported is `clients/`, every other module can be tested with a stub that implements the Protocol in five lines. The integration tests in Post 6 (retrieval) and Post 8 (prompt assembly) will use this — neither one hits a real model. The pattern: pass the test a `FakeChatClient` whose `stream()` yields pre-recorded tokens, and assert on the prompt assembly. Zero network, deterministic, runs in milliseconds.

A worked example, since this comes up often enough to be worth pinning:

```python
# A fake that implements ChatClient in 8 lines. mypy --strict accepts this
# as a ChatClient because the structural shape matches.
class FakeChatClient:
    def __init__(self, scripted_tokens: list[str]) -> None:
        self._tokens = scripted_tokens

    async def stream(self, system, messages, max_tokens=1024):
        for tok in self._tokens:
            yield tok

    async def complete(self, system, messages, *, max_tokens=256, json_format=False):
        return "".join(self._tokens)
```

That's the *whole* test double. The retrieval logic in Post 6 can be tested by handing this to the orchestrator and asserting on what got streamed back. None of the real provider code is exercised; none of it needs to be — that code is tested separately, against real model servers.

---

## The Three Seams in One Picture {#three-seams}

By the end of this post, three Protocols hide three different concerns from the rest of the codebase. Each has at least two implementations selected by config:

<div style="margin: 1.5rem 0; overflow-x: auto;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 580" role="img"
     aria-label="Three-tier diagram: callers (routes, services, ingestion) at the top import Protocol-typed clients from a factory. Three Protocols — Storage, ChatClient, and EmbeddingClient — sit in the middle. Six implementations sit at the bottom, with R2Storage marked as a stub to be filled in later."
     style="display: block; width: 100%; height: auto; max-width: 940px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">
  <defs>
    <marker id="ts-arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>
    </marker>
  </defs>

  <!-- Top: callers box -->
  <g>
    <rect x="70" y="30" width="800" height="80" rx="8" fill="#f1f5f9" stroke="#64748b" stroke-width="1.5"/>
    <text x="470" y="58" text-anchor="middle" font-size="15" font-weight="600" fill="#1f2937">
      Routes · Services · Ingestion
    </text>
    <text x="470" y="80" text-anchor="middle" font-size="12" fill="#475569"
          font-family="ui-monospace, 'SF Mono', Menlo, Monaco, monospace">
      from app.clients import get_storage, get_chat_client, get_embedding_client
    </text>
    <text x="470" y="100" text-anchor="middle" font-size="11" fill="#64748b" font-style="italic">
      the rest of the codebase only sees Protocol-typed instances
    </text>
  </g>

  <!-- Arrows from callers down to three Protocols -->
  <line x1="200" y1="110" x2="200" y2="178" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ts-arrow)"/>
  <line x1="470" y1="110" x2="470" y2="178" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ts-arrow)"/>
  <line x1="740" y1="110" x2="740" y2="178" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ts-arrow)"/>

  <!-- Three Protocol boxes -->
  <g>
    <rect x="120" y="180" width="160" height="65" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="200" y="208" text-anchor="middle" font-size="15" font-weight="600" fill="#1f2937">Storage</text>
    <text x="200" y="228" text-anchor="middle" font-size="11" fill="#92400e" font-weight="600">PROTOCOL</text>
  </g>
  <g>
    <rect x="390" y="180" width="160" height="65" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="470" y="208" text-anchor="middle" font-size="15" font-weight="600" fill="#1f2937">ChatClient</text>
    <text x="470" y="228" text-anchor="middle" font-size="11" fill="#92400e" font-weight="600">PROTOCOL</text>
  </g>
  <g>
    <rect x="660" y="180" width="160" height="65" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="740" y="208" text-anchor="middle" font-size="15" font-weight="600" fill="#1f2937">EmbeddingClient</text>
    <text x="740" y="228" text-anchor="middle" font-size="11" fill="#92400e" font-weight="600">PROTOCOL</text>
  </g>

  <!-- Fan-out arrows from each Protocol to its two implementations -->
  <!-- Storage -> LocalStorage, R2Storage -->
  <line x1="170" y1="245" x2="135" y2="338" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ts-arrow)"/>
  <line x1="230" y1="245" x2="265" y2="338" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ts-arrow)"/>
  <!-- ChatClient -> OllamaChat, AnthropicChat -->
  <line x1="440" y1="245" x2="405" y2="338" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ts-arrow)"/>
  <line x1="500" y1="245" x2="535" y2="338" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ts-arrow)"/>
  <!-- EmbeddingClient -> OllamaEmbed, SentenceTrans -->
  <line x1="710" y1="245" x2="675" y2="338" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ts-arrow)"/>
  <line x1="770" y1="245" x2="820" y2="338" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ts-arrow)"/>

  <!-- Implementation row -->
  <!-- Storage column -->
  <g>
    <rect x="70" y="340" width="130" height="82" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="135" y="368" text-anchor="middle" font-size="14" font-weight="600" fill="#1f2937">LocalStorage</text>
    <text x="135" y="390" text-anchor="middle" font-size="11" fill="#1e40af" font-weight="600">this post</text>
    <text x="135" y="408" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">filesystem + StaticFiles</text>
  </g>
  <g>
    <rect x="200" y="340" width="130" height="82" rx="8" fill="#f3f4f6" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="5,3"/>
    <text x="265" y="368" text-anchor="middle" font-size="14" font-weight="600" fill="#1f2937">R2Storage</text>
    <text x="265" y="390" text-anchor="middle" font-size="11" fill="#4b5563" font-weight="600">stub · Post 10</text>
    <text x="265" y="408" text-anchor="middle" font-size="10" fill="#6b7280" font-style="italic">Cloudflare R2 (S3-compatible)</text>
  </g>

  <!-- ChatClient column -->
  <g>
    <rect x="340" y="340" width="130" height="82" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="405" y="362" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">OllamaChat</text>
    <text x="405" y="378" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">Client</text>
    <text x="405" y="397" text-anchor="middle" font-size="11" fill="#1e40af" font-weight="600">this post</text>
    <text x="405" y="412" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">local Ollama or Modal</text>
  </g>
  <g>
    <rect x="470" y="340" width="130" height="82" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="535" y="362" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">AnthropicChat</text>
    <text x="535" y="378" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">Client</text>
    <text x="535" y="397" text-anchor="middle" font-size="11" fill="#1e40af" font-weight="600">this post</text>
    <text x="535" y="412" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">Claude · prompt cache</text>
  </g>

  <!-- EmbeddingClient column -->
  <g>
    <rect x="610" y="340" width="130" height="82" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="675" y="362" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">OllamaEmbedding</text>
    <text x="675" y="378" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2937">Client</text>
    <text x="675" y="397" text-anchor="middle" font-size="11" fill="#1e40af" font-weight="600">this post · default</text>
    <text x="675" y="412" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">HTTP · bge-m3</text>
  </g>
  <g>
    <rect x="740" y="340" width="160" height="82" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="820" y="362" text-anchor="middle" font-size="11" font-weight="600" fill="#1f2937">SentenceTransformers</text>
    <text x="820" y="378" text-anchor="middle" font-size="11" font-weight="600" fill="#1f2937">EmbeddingClient</text>
    <text x="820" y="397" text-anchor="middle" font-size="11" fill="#1e40af" font-weight="600">this post · fallback</text>
    <text x="820" y="412" text-anchor="middle" font-size="10" fill="#475569" font-style="italic">in-process · PyTorch</text>
  </g>

  <!-- Bottom factory note -->
  <g>
    <line x1="100" y1="452" x2="840" y2="452" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,3"/>
    <text x="470" y="478" text-anchor="middle" font-size="13" font-weight="600" fill="#1f2937">
      Factory in clients/__init__.py picks one implementation per Protocol from .env
    </text>
    <text x="470" y="500" text-anchor="middle" font-size="12" fill="#475569"
          font-family="ui-monospace, 'SF Mono', Menlo, Monaco, monospace">
      STORAGE_BACKEND · CHAT_PROVIDER · EMBEDDING_PROVIDER
    </text>
  </g>

  <!-- Legend -->
  <g>
    <rect x="100" y="528" width="16" height="14" fill="#dbeafe" stroke="#2563eb" stroke-width="1"/>
    <text x="124" y="540" font-size="11" fill="#4b5563">implemented in this post</text>
    <rect x="290" y="528" width="16" height="14" fill="#f3f4f6" stroke="#9ca3af" stroke-width="1" stroke-dasharray="4,2"/>
    <text x="314" y="540" font-size="11" fill="#4b5563">stub today; filled in later</text>
    <rect x="500" y="528" width="16" height="14" fill="#fef3c7" stroke="#f59e0b" stroke-width="1"/>
    <text x="524" y="540" font-size="11" fill="#4b5563">Protocol (typed interface)</text>
  </g>
</svg>
</div>

*The three Protocols sit in the middle; six leaf implementations sit at the bottom; the factory selects one per Protocol from `.env`. Callers only see the middle row.*

Three things worth noticing in the picture:

- **All the SDK imports happen on the bottom row.** `anthropic`, `httpx`, `aiofiles`, `boto3`, `sentence-transformers` — these names appear *only* inside the leaf implementations in `backend/app/clients/`. Routes and services see Protocols.
- **The factory is the only code that knows which leaves exist.** `clients/__init__.py` reads `Settings` and instantiates one of the leaves. Every other caller asks the factory for a `Storage` or a `ChatClient` and gets back an opaque object whose type is the Protocol.
- **Today there's a hole on the bottom-left.** `R2Storage` is a stub that raises `NotImplementedError` — we fill it in for the cloud deploy in Post 10. The fact that we can ship Post 4–9 with that stub raising is itself a small piece of evidence the abstraction is working: nobody outside `clients/` knows or cares which storage backend is mounted.

Let's build each seam in turn.

---

## Seam 1 — Storage: LocalStorage End to End {#seam-storage}

This is the smallest of the three Protocols, so it's a clean place to start.

### Why store images behind a Protocol at all?

The naive answer is: just write images to disk and have FastAPI serve them. Done. Why dress it up?

The non-naive answer is: in production, images don't live on disk next to the backend. They live in object storage like [Cloudflare R2](https://www.cloudflare.com/developer-platform/products/r2/), [AWS S3](https://aws.amazon.com/s3/), or [Google Cloud Storage](https://cloud.google.com/storage), fronted by a [CDN](https://en.wikipedia.org/wiki/Content_delivery_network) so a reader in Tokyo doesn't have to fetch them from a single server in Frankfurt. The two backends *answer the same questions* — "where can I read this image?", "is this image already uploaded?", "please write these bytes at this key" — but they answer them in totally different ways. Local disk says "your filesystem has it"; R2 says "issue an [S3 `PutObject`](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html) with these headers."

The Protocol is the place where those two backends look the same to the rest of the codebase.

### The interface

```python
# backend/app/clients/storage.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

import aiofiles


class Storage(Protocol):
    async def put(self, key: str, content: bytes, content_type: str) -> None:
        """Write bytes to the backing store at `key`. Idempotent."""
        ...

    async def url_for(self, key: str) -> str:
        """Resolve a relative key to a public URL the frontend can fetch."""
        ...

    async def exists(self, key: str) -> bool: ...
```

> *Plain-English aside: what's an [`async`](https://docs.python.org/3/library/asyncio-task.html) function, and why is every method on this Protocol one?*
>
> A regular Python function runs top to bottom and **blocks** the program until it returns. If it spends 500 ms waiting for a network response or a disk read, the program does *nothing else* during that 500 ms — it sits there. An **async function** (written `async def` instead of plain `def`) is the cooperative alternative: when it hits a slow I/O step, it tells the runtime "I'm waiting for the network now; go do something else and wake me up when the bytes arrive." Multiple async functions take turns on a single thread, and the thread stays busy. Callers wait for an async function to finish with the `await` keyword: `result = await some_async_function(...)`.
>
> Writing files and uploading to object storage — what `Storage.put` and friends actually do — are exactly the kind of slow steps that benefit. For `R2Storage`, every `put()` is an HTTP request to Cloudflare. For `LocalStorage`, every `put()` is a filesystem write that competes with everything else FastAPI's event loop is doing. Marking these methods `async def` means the request handler that called them can keep serving other requests while one is pending.
>
> Once you start using async, it spreads: to `await` a function, the caller must also be `async`. FastAPI's request handlers are async, so they can `await storage.put(...)`. Tests are async (via [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)), so they can call `await client.embed_batch(...)`. This is what people mean by **"async is contagious."** The clean move is to make the *interface* async at the start of the project — that way every implementation, slow-I/O or trivial, looks async to the caller, and the caller writes `await` once and never has to think about it again. That's why even `LocalStorage.url_for`, which is a one-line string concatenation that never blocks, is still declared `async`: the *interface contract* says async, so all implementations are async, so callers always `await`.

Three methods, each chosen because *both* backends need to do them and *neither* call site needs to do more. A handful of design choices in those few lines are worth pausing on:

- **Keys, not paths.** The argument to every method is a `key: str` — a relative path like `episodes/ep01-potion-of-flight/pages/001-display.webp`. Never a full URL, never an absolute filesystem path. That's the same string the database stores in `pages.image_url` (from Post 2's design decision #2). It's *meaningful in the application*, not in the storage backend. `LocalStorage` turns the key into `./data/images/episodes/...`; `R2Storage` turns the same key into an R2 object key — and both compose the same key into a URL via `url_for`. A reader of `pages.image_url` doesn't know or care which backend resolved it.
- **`content_type` is a parameter even though `LocalStorage` ignores it.** Filesystems don't care about MIME types — but R2 does (it goes on the `Content-Type` HTTP response header so browsers render `image/webp` as an image, not a download). Putting the parameter on the Protocol now means R2 doesn't need a wider interface later; `LocalStorage` just discards it. **The Protocol is the union of what implementations *might* need.** This is one of the few places it's worth designing for a use case you don't have yet.
- **`url_for` is async even though `LocalStorage` doesn't need to await anything.** R2's [signed URLs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html) require a network call or at least async-friendly crypto in some configurations; making the whole Protocol async keeps the door open. Async is contagious — fix the contagion at the interface level, not the call site level.
- **`exists` exists.** It's not strictly required for writes (we could just always `put`) — but the ingestion pipeline (Post 4) uses it to skip re-uploading variants that are already in place. R2 round-trips are slower than disk reads; a cheap `HEAD` check is worth defining.

> *Why Protocols and not an Abstract Base Class? An aside for readers new to Python typing.*
>
> If you've seen this pattern before in a Python codebase, it was probably done with an [**Abstract Base Class (ABC)**](https://docs.python.org/3/library/abc.html) instead. An ABC is "a class you can't instantiate directly; it exists to declare a contract that subclasses must implement." The Python idiom looks like this:
>
> ```python
> from abc import ABC, abstractmethod
>
> class BaseStorage(ABC):                       # ← the contract
>     @abstractmethod
>     async def put(self, key: str, content: bytes, content_type: str) -> None: ...
>     @abstractmethod
>     async def url_for(self, key: str) -> str: ...
>     @abstractmethod
>     async def exists(self, key: str) -> bool: ...
>
> class LocalStorage(BaseStorage):              # ← implementations must inherit
>     async def put(self, key, content, content_type): ...
>     async def url_for(self, key): ...
>     async def exists(self, key): ...
> ```
>
> The Protocol version achieves the same effect — declare the three methods every storage backend must support, get static-type-checking that catches mismatched signatures — but does it differently. Three concrete differences tip the choice toward `Protocol` here.
>
> **1. ABCs are *nominal*; Protocols are *structural*.** A nominal type system says "you ARE a `BaseStorage` only if you explicitly inherit from `BaseStorage`." A structural one says "you ARE a `Storage` if you happen to have these methods, regardless of your ancestry." Concretely, `LocalStorage` in this codebase does **not** start with `class LocalStorage(Storage):` — it just defines the three methods, and [`mypy --strict`](https://mypy-lang.org/) recognises it as a `Storage` automatically. A test double can satisfy the Protocol in five lines without inheriting from anything (this is the `FakeChatClient` shape from earlier in this post). The same idea exists in other languages: [Go interfaces](https://go.dev/tour/methods/9) are structural, [TypeScript interfaces](https://www.typescriptlang.org/docs/handbook/2/objects.html) are structural, [Java interfaces](https://docs.oracle.com/javase/tutorial/java/concepts/interface.html) are nominal — Python's `Protocol` is the structural option.
>
> **2. Imports flow only one way.** With an ABC, every implementation has to `import BaseStorage` to inherit from it. With a Protocol, **only the factory** (`clients/__init__.py`, which constructs instances and typehints the return value) imports `Storage`. The implementations don't. If you ever rename or delete the Protocol, only the factory breaks; the implementation files are untouched. That's a cleaner one-way arrow on the import graph, and it shows up as faster mypy runs (fewer files to recheck when the Protocol changes) and easier refactors.
>
> **3. Adapting third-party classes is free.** Suppose you wanted to use an existing third-party class — say a `boto3.S3.Bucket` instance — wherever your code expects a `Storage`. With an ABC, you'd have to write an adapter class that inherits from `BaseStorage` and delegates each call. With a Protocol, you don't strictly need a wrapper: if the third-party class already has matching method names, it *is* a `Storage` for type-checking purposes. (In practice, our `R2Storage` is still its own class because the method names and signatures don't align perfectly with `boto3` — but the type system isn't what forces that.)
>
> **When you'd still pick an ABC.** When the interface needs **shared implementation** — concrete default methods, shared `__init__` logic, validation code that every subclass should run. Protocols are pure shape; they can't ship executable method bodies. None of the Protocols in this project need shared behavior — every implementation's body is genuinely different (HTTP for Ollama, in-process PyTorch for sentence-transformers; filesystem for `LocalStorage`, S3 API for `R2Storage`). So Protocol wins. If a future Protocol needs shared helpers, the move is to add a separate **mixin** class or a free function, not to convert to an ABC.
>
> *Mixin in one paragraph.* A **mixin** is a small class designed to be inherited *alongside* a "main" class purely to add one or two reusable methods — no contract to enforce, no instances of its own, often no `__init__`. It's a Python idiom for "I have helper methods I want to share across these three classes; let me bundle them into a class they can all inherit from."
>
> Here's the shape, using a small example loosely sketched from this codebase. Both `OllamaChatClient` and `AnthropicChatClient` need to fetch image bytes from a URL and base64-encode them for the model's wire format. If we wanted to share that helper instead of duplicating it, we'd write a mixin:
>
> ```python
> import base64
> import httpx
>
> class Base64ImageMixin:
>     """A pure-helper class. Never instantiated on its own —
>     only inherited alongside another class to share this one method."""
>
>     async def _fetch_image_b64(
>         self, url: str, http: httpx.AsyncClient
>     ) -> str:
>         response = await http.get(url)
>         response.raise_for_status()
>         return base64.b64encode(response.content).decode("ascii")
>
>
> class OllamaChatClient(Base64ImageMixin):          # inherits _fetch_image_b64
>     async def stream(self, system, messages, max_tokens=1024):
>         async with httpx.AsyncClient() as http:
>             ...
>             b64 = await self._fetch_image_b64(image_url, http)  # ← shared
>             ...
>
>
> class AnthropicChatClient(Base64ImageMixin):       # also inherits it
>     async def stream(self, system, messages, max_tokens=1024):
>         async with httpx.AsyncClient() as http:
>             ...
>             b64 = await self._fetch_image_b64(image_url, http)  # ← same shared
>             ...
> ```
>
> Three things to notice:
>
> - **`Base64ImageMixin` has no `__init__` and isn't a Protocol or ABC.** It's just a class with one method. You never write `Base64ImageMixin()` to construct one — it exists only to be a parent.
> - **Both clients inherit from it,** so `self._fetch_image_b64(...)` is available inside either client's methods. The underscore prefix is a Python convention for "this is for internal use" — the mixin is sharing *implementation*, not *public API*.
> - **The `ChatClient` Protocol doesn't change.** Callers still see a `ChatClient` with `stream()` and `complete()`. Whether an implementation got `_fetch_image_b64` via inheritance, copy-paste, or magic is not the caller's business. *Mixin shares code among implementations; Protocol declares contract to callers.* Different jobs.
>
> Today the actual codebase duplicates the helper (it's small, the two clients diverge enough that combining them would muddy more than it shares) — but if the helper ever grew, the mixin is the right escape hatch. **The point is that "use a mixin" is a much smaller, more focused move than "convert the Protocol to an ABC and add a default method body."**

### `LocalStorage`, with every line justified

The whole implementation is short enough to fit on one screen. We'll read it top-to-bottom and call out the choices.

```python
class LocalStorage:
    """Filesystem-backed storage. Files are served by the FastAPI app via StaticFiles."""

    def __init__(self, root: Path, url_prefix: str) -> None:
        self._root = root
        self._url_prefix = url_prefix.rstrip("/")
        self._root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        # Defensive: never let a key escape the root via "..".
        target = (self._root / key).resolve()
        if not str(target).startswith(str(self._root.resolve())):
            raise ValueError(f"Refusing to write outside storage root: {key}")
        return target

    _IDEMPOTENCY_COMPARE_LIMIT = 5 * 1024 * 1024  # bytes

    async def put(self, key: str, content: bytes, content_type: str) -> None:
        path = self._path_for(key)
        if (
            len(content) <= self._IDEMPOTENCY_COMPARE_LIMIT
            and path.exists()
            and path.stat().st_size == len(content)
        ):
            async with aiofiles.open(path, "rb") as f:
                existing = await f.read()
            if existing == content:
                return
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(content)

    async def url_for(self, key: str) -> str:
        return f"{self._url_prefix}/{key}"

    async def exists(self, key: str) -> bool:
        path = self._path_for(key)
        return await asyncio.to_thread(path.exists)
```

Four things in this code earn their lines:

**The path-traversal guard.** `_path_for` resolves the requested key against the storage root and then checks the result is still *under* the root. Why? Because in a generic version of "the caller can name any file," an unfriendly caller could pass `../../etc/passwd` and write bytes outside the storage root. In *this* application all the keys come from the ingestion pipeline, not from end users, so it's defense in depth — but it's three lines, and the alternative (forgetting the guard and discovering it the day someone wires this Protocol into a user-facing route) is a bad day. The [OWASP path-traversal cheat sheet](https://owasp.org/www-community/attacks/Path_Traversal) makes the case for this kind of check at every storage boundary.

**The idempotency check.** Ingestion is the kind of pipeline you re-run a lot — when a page description changes, when an image variant gets re-encoded, when you fix a bug. Without idempotency, every re-run rewrites every page image on disk, which is fast but messy (mtimes change, file watchers fire, OS page caches get blown). With it, re-running ingestion against unchanged content is a near-no-op. The "compare bytes" approach is intentionally crude: no hashing, no checksums — just `len(content)` first as a cheap gate, then `read+compare` for files under 5MB. The 5MB cap is because Pepper & Carrot's page images are all in the ~500KB–2MB range; for the few cover images that approach the cap, we fall through to the "just rewrite it" branch and the world doesn't end.

> *Plain-English aside: what does `len(content)` return when `content` is `bytes`?* It returns the **number of bytes** — one element per byte. So `len(b"hello")` is `5`, not 40 (the bit count) or 4.5 (anything character-shaped). A `bytes` object is just a sequence of integers in 0–255, and `len` counts how many of those integers are in it. For non-ASCII text, the count depends on the encoding: `len("é".encode("utf-8"))` is `2`, `len("é".encode("latin-1"))` is `1`. `bytes` objects don't know about characters — only raw bytes.
>
> That's why this check pairs `len(content)` (the in-memory byte count) with `path.stat().st_size` (the on-disk byte count from [`os.stat`](https://docs.python.org/3/library/os.html#os.stat)) — both are measured in bytes, so they're directly comparable. The full condition reads as: *"if the content is under 5 MB, **and** a file already exists at this key, **and** the existing file is the same number of bytes — only then is it worth actually reading the file to do a byte-for-byte compare."* If the sizes don't match, we know the content differs without reading a single byte off disk. The 5 MB cap exists because past that size, comparing bytes becomes expensive (you'd read 5 MB just to maybe skip a 5 MB write). Below the cap, the size check is cheap insurance against unnecessary writes; above it, just rewrite.

> *Plain-English aside: what's `mtime`?* Short for **modification time** — the Unix filesystem metadata field that records when a file's contents were last written. Every file carries three timestamps you can see with `stat`: **atime** (last read), **mtime** (last write), and **ctime** (last metadata change like permissions or rename). The detail that matters here is that **every write bumps mtime to "now" — even if the bytes are identical to what was already on disk**. That trips a lot of downstream tools that subscribe to mtime as a "file changed" signal: [file watchers](https://github.com/emcrisostomo/fswatch) like Vite's hot-reload and IDE auto-rebuilders, build systems like [`make`](https://www.gnu.org/software/make/) and [Bazel](https://bazel.build/) (which compare mtimes to decide what to rebuild), and rsync-style backup tools (default mode uses size + mtime as the "is this different?" check). Rewriting a file with identical content lights all of those up to think something changed. Preserving the original mtime when the content really hasn't changed keeps them quiet. This codebase doesn't plug any watchers into `data/images/` *yet*, but getting the convention right on day one beats remembering to retrofit it the day you do.

**`async` + `aiofiles`.** The FastAPI request handler that ingest-write paths touch is async; if we did blocking disk I/O directly, we'd stall the event loop and starve every other concurrent request. [`aiofiles`](https://github.com/Tinche/aiofiles) wraps file operations in a thread pool so they cooperate with [`asyncio`](https://docs.python.org/3/library/asyncio.html). For a laptop-scale workload this is mild overkill — but it keeps the async story uniform, which matters in Post 6 when we have streaming chat responses and an ingestion run happening in parallel and the whole thing needs to not stutter. (For `exists`, we don't bother with `aiofiles` — just `asyncio.to_thread(path.exists)`, which threads the one-syscall check.)

**`url_for` is dead simple.** Just `f"{self._url_prefix}/{key}"`. The whole point is that `LocalStorage` doesn't *do* any URL signing or routing — it relies on FastAPI to serve the file at that URL, which is the next piece.

### The FastAPI `StaticFiles` mount that closes the loop

`LocalStorage.url_for("episodes/ep01/pages/001.webp")` returns `http://localhost:8000/images/episodes/ep01/pages/001.webp`. That URL has to actually resolve to a file when the browser fetches it. The wiring lives in `backend/app/main.py`:

```python
# Only mount the local image server when STORAGE_BACKEND=local.
# In production (STORAGE_BACKEND=r2) the frontend hits R2's public URL
# directly, never the backend, so this mount becomes dead weight we skip.
if settings.storage_backend == "local":
    mount_path = urlparse(settings.local_image_url_prefix).path or "/images"
    settings.local_image_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        mount_path,
        StaticFiles(directory=settings.local_image_dir),
        name="images",
    )
```

`mount_path` is parsed out of `local_image_url_prefix` instead of hard-coded to `/images` — so if you change the prefix in `.env`, the mount follows. The two strings stay in sync by construction.

> *Plain-English aside: what's [`StaticFiles`](https://fastapi.tiangolo.com/tutorial/static-files/)?* It's a tiny [ASGI](https://asgi.readthedocs.io/en/latest/) sub-app that maps incoming URL paths to files on disk and serves them with the right headers (`Content-Type`, `Content-Length`, conditional 304s on `If-Modified-Since`). In production you'd put a real CDN or [nginx](https://nginx.org/) in front of this — for dev, it's the cheapest possible way to make "an image URL the frontend can `<img src=>`" work.

### Storage data flow in one picture

Two paths share a single file on disk: ingestion writes it once; every browser request reads it back. The relative `key` is the thread that ties them together — it lives in `pages.image_url`, gets composed into a URL on the way out, and resolves to the same path on disk on the way back in.

<div style="margin: 1.5rem 0; overflow-x: auto;">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 540" role="img"
     aria-label="Storage data flow: the ingestion pipeline writes images through LocalStorage.put to a file under ./data/images/; the browser reads the same file via FastAPI StaticFiles serving the same directory. Postgres stores only the relative key, which the storage backend composes into a URL at API response time."
     style="display: block; width: 100%; height: auto; max-width: 920px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">
  <defs>
    <marker id="sd-arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>
    </marker>
  </defs>

  <!-- Path-label headers -->
  <text x="190" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#475569" letter-spacing="0.05em">WRITE PATH</text>
  <text x="190" y="46" text-anchor="middle" font-size="11" fill="#94a3b8" font-style="italic">ingestion: once per page</text>
  <text x="730" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#475569" letter-spacing="0.05em">READ PATH</text>
  <text x="730" y="46" text-anchor="middle" font-size="11" fill="#94a3b8" font-style="italic">browser: every page render</text>

  <!-- Dividing line between paths -->
  <line x1="460" y1="20" x2="460" y2="340" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4,4"/>

  <!-- Actors row -->
  <!-- Ingestion -->
  <g>
    <rect x="100" y="68" width="180" height="62" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="190" y="94" text-anchor="middle" font-size="14" font-weight="600" fill="#1f2937">Ingestion pipeline</text>
    <text x="190" y="115" text-anchor="middle" font-size="11" fill="#92400e" font-style="italic">(Post 4)</text>
  </g>
  <!-- Browser -->
  <g>
    <rect x="640" y="68" width="180" height="62" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="730" y="94" text-anchor="middle" font-size="14" font-weight="600" fill="#1f2937">Browser / frontend</text>
    <text x="730" y="115" text-anchor="middle" font-size="11" fill="#92400e" font-style="italic">(Post 5+)</text>
  </g>

  <!-- Arrows from actors down to middle layer -->
  <line x1="190" y1="130" x2="190" y2="208" stroke="#6b7280" stroke-width="1.5" marker-end="url(#sd-arrow)"/>
  <text x="200" y="160" font-size="11" fill="#475569"
        font-family="ui-monospace, 'SF Mono', Menlo, Monaco, monospace">put(key, bytes,</text>
  <text x="200" y="175" font-size="11" fill="#475569"
        font-family="ui-monospace, 'SF Mono', Menlo, Monaco, monospace">    content_type)</text>

  <line x1="730" y1="130" x2="730" y2="208" stroke="#6b7280" stroke-width="1.5" marker-end="url(#sd-arrow)"/>
  <text x="740" y="160" font-size="11" fill="#475569"
        font-family="ui-monospace, 'SF Mono', Menlo, Monaco, monospace">GET url_for(key)</text>
  <text x="740" y="175" font-size="10" fill="#94a3b8" font-style="italic">composed URL</text>

  <!-- Middle layer -->
  <g>
    <rect x="80" y="210" width="220" height="76" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="190" y="236" text-anchor="middle" font-size="14" font-weight="600" fill="#1f2937">LocalStorage.put()</text>
    <text x="190" y="256" text-anchor="middle" font-size="11" fill="#1e40af">aiofiles · idempotency check</text>
    <text x="190" y="274" text-anchor="middle" font-size="10" fill="#475569" font-style="italic"
          font-family="ui-monospace, 'SF Mono', Menlo, Monaco, monospace">backend/app/clients/storage.py</text>
  </g>
  <g>
    <rect x="620" y="210" width="220" height="76" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="730" y="236" text-anchor="middle" font-size="14" font-weight="600" fill="#1f2937">FastAPI StaticFiles</text>
    <text x="730" y="256" text-anchor="middle" font-size="11" fill="#1e40af">mounted at /images</text>
    <text x="730" y="274" text-anchor="middle" font-size="10" fill="#475569" font-style="italic"
          font-family="ui-monospace, 'SF Mono', Menlo, Monaco, monospace">backend/app/main.py</text>
  </g>

  <!-- Arrows to disk file -->
  <!-- Write: LocalStorage → disk -->
  <path d="M 190 286 Q 190 326 380 326 L 380 358" stroke="#6b7280" stroke-width="1.5" fill="none" marker-end="url(#sd-arrow)"/>
  <text x="240" y="315" font-size="11" fill="#475569" font-style="italic">writes bytes</text>

  <!-- Read: StaticFiles → disk -->
  <path d="M 730 286 Q 730 326 540 326 L 540 358" stroke="#6b7280" stroke-width="1.5" fill="none" marker-end="url(#sd-arrow)"/>
  <text x="600" y="315" font-size="11" fill="#475569" font-style="italic">reads bytes</text>

  <!-- Return-path arrow: StaticFiles back to Browser -->
  <path d="M 840 230 Q 880 230 880 175 L 880 110 Q 880 80 822 80"
        stroke="#6b7280" stroke-width="1.5" stroke-dasharray="6,4" fill="none" marker-end="url(#sd-arrow)"/>
  <text x="886" y="170" font-size="10" fill="#475569" font-style="italic" transform="rotate(90 886 170)">HTTP 200 + bytes</text>

  <!-- Disk file (shared bottom) -->
  <g>
    <rect x="270" y="360" width="380" height="72" rx="8" fill="#d1fae5" stroke="#059669" stroke-width="1.5"/>
    <text x="460" y="385" text-anchor="middle" font-size="14" font-weight="600" fill="#1f2937">./data/images/&lt;key&gt;</text>
    <text x="460" y="404" text-anchor="middle" font-size="11" fill="#065f46">the same file on disk for both paths</text>
    <text x="460" y="421" text-anchor="middle" font-size="10" fill="#475569"
          font-family="ui-monospace, 'SF Mono', Menlo, Monaco, monospace">e.g. episodes/ep01-potion-of-flight/pages/001-display.webp</text>
  </g>

  <!-- Postgres side note -->
  <g>
    <rect x="180" y="460" width="560" height="62" rx="8" fill="#f1f5f9" stroke="#64748b" stroke-width="1.5" stroke-dasharray="5,3"/>
    <text x="460" y="484" text-anchor="middle" font-size="13" font-weight="600" fill="#1f2937">
      Postgres: pages.image_url stores &lt;key&gt; — the relative key only
    </text>
    <text x="460" y="504" text-anchor="middle" font-size="11" fill="#475569" font-style="italic">
      LocalStorage.url_for() composes the full URL at API response time
    </text>
  </g>
</svg>
</div>

*Same key on the way in (write path) and on the way out (read path). The dashed arrow on the right is the HTTP response carrying the bytes back to the browser. The bottom box is the Postgres column that ties everything together — it stores only the relative key, never a full URL.*

### Verifying it end-to-end

The smoke test in [`backend/tests/test_storage.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/backend/tests/test_storage.py) (in the workshop starter) covers:

- `put()` writes the file to disk.
- `put()` with identical content is a no-op (mtime doesn't change).
- `put()` with different content overwrites.
- `url_for()` composes the prefix correctly.
- `exists()` reflects the filesystem.
- A key with `../` is rejected with a clear `ValueError`.

To exercise the *whole loop* — Protocol implementation, FastAPI mount, browser GET — try this:

```bash
# Start the backend (from backend/)
uv run uvicorn app.main:app --reload &

# In another shell:
mkdir -p data/images/test
echo "fake-image-bytes" > data/images/test/hello.txt
curl http://localhost:8000/images/test/hello.txt
# Expected: fake-image-bytes
```

When that prints what you put in, the whole storage seam — `LocalStorage.put` → file on disk → `LocalStorage.url_for` → `StaticFiles` mount → HTTP response — is wired end to end.

### What's *not* in `LocalStorage` (deliberately)

A handful of things you might expect that aren't there:

- **No content hashing.** R2 returns ETags automatically; we don't need to compute one in the app layer. If we ever do, it's a one-method extension to the Protocol.
- **No retry logic.** Disk writes don't fail transiently in any way retries would help. R2 will need retries; that's R2's problem when we build it.
- **No request-time image resizing.** All image variants are pre-computed at ingestion (Post 4) and stored as separate keys. Doing transforms at request time is a different ADR — see [`docs/decisions/0003-storage-abstraction.md`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/docs/decisions/0003-storage-abstraction.md) in the workshop starter.

These omissions are the abstraction earning its keep. Every line that's *not* in `LocalStorage` is a line that doesn't need to be parallelled by an equivalent in `R2Storage` later.

---

## Seam 2 — EmbeddingClient: Two Implementations of the Same Protocol {#seam-embedding}

Now the more interesting seam — interesting because we actually build *two* implementations, both backed by real model code, and the test at the bottom proves they're interchangeable.

> *Plain-English aside: what's an embedding?* An **embedding** is a fixed-length list of numbers — usually 768, 1024, or 1536 of them — that captures the meaning of a piece of text. The trick: two texts with similar meaning land near each other in that high-dimensional space, by some distance metric (usually [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)). Embeddings are what makes "find me the chunks closest in meaning to this question" work in [RAG](https://www.anthropic.com/news/contextual-retrieval) — you embed the question, embed every chunk in your corpus once at ingestion, and the retrieval step is a numerical "which chunk vector is nearest to the question vector?". An **embedding model** like [BGE-M3](https://huggingface.co/BAAI/bge-m3) is the thing that turns text into those numbers.
>
> We use 1024-dim BGE-M3 throughout this project. It's [multilingual](https://huggingface.co/BAAI/bge-m3), runs fine on CPU, and is small enough (~2GB) to ship with the Modal GPU image in Post 10.

### The interface

```python
# backend/app/clients/embedding.py
class EmbeddingClient(Protocol):
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text. Order preserved."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality. Used to validate against Chroma collections."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier used as a tag on Chroma collections (e.g., 'bge-m3')."""
        ...
```

A method and two read-only properties. Three small design notes:

- **Batch by default.** `embed_batch(texts: list[str])` not `embed(text: str)`. Embedding models are *much* faster on a batch than on `N` single calls — the GPU/CPU spends most of its time on per-call overhead otherwise. Even Ollama's HTTP endpoint accepts a batch. We don't expose a single-text convenience method on the Protocol; if you only have one string, pass `[text]` and unpack the single result. One method, no overlap.
- **`dimension` is a property, not a method.** It's a static fact about the model, not a question that needs an `await`. Implementations cache it after first probe.

  > *Plain-English aside: what's `@property`?* `@property` is a [decorator](https://docs.python.org/3/glossary.html#term-decorator) that turns a method into something callers read like an attribute — **no parentheses needed at the call site**. The same getter code runs underneath; the call just looks like a plain field access. Quick example:
  >
  > ```python
  > class Page:
  >     def __init__(self, width: int, height: int) -> None:
  >         self.width = width
  >         self.height = height
  >
  >     @property
  >     def aspect_ratio(self) -> float:
  >         return self.width / self.height
  >
  > page = Page(width=1920, height=1080)
  > page.aspect_ratio         # → 1.777...   (no parens — that's the point)
  > page.width                # → 1920       (a regular attribute)
  > page.aspect_ratio()       # → TypeError: 'float' object is not callable
  > ```
  >
  > Properties fit values that are conceptually *facts* (not actions), that might be cached or computed on demand, and that callers shouldn't *set* directly. `dimension` ticks all three boxes: it's a fact about the embedding model (1024 for bge-m3, always), it's cached after the first probe, and `client.dimension = 768` would be nonsense — the model is what determines the number, not the caller. By contrast, `embed_batch` stays a method because it's an *action*: it takes arguments, does work, and the answer depends on what you ask. Methods get parentheses; properties don't.
- **`model_name` exists so ChromaDB can tag collections.** If you re-ingest with a different embedding model, you don't want the new vectors silently mixed in with the old ones — they're [literally incompatible](https://github.com/chroma-core/chroma/blob/main/docs/docs/architecture/embeddings.md) (different dimensions, different similarity structure). Collections in Chroma are named like `pages_v1_bge-m3`; flipping models means flipping the version suffix, which forces a full re-embed.

### `OllamaEmbeddingClient` — the project default

If you followed Post 2, you already have Ollama running with `bge-m3` pulled. So the default `EmbeddingClient` just talks to it over HTTP — no extra weights to download, no extra process to start.

```python
class OllamaEmbeddingClient:
    """Embeddings via Ollama. Convenient when you're already running Ollama for chat."""

    def __init__(
        self,
        base_url: str,
        model: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        # 180s matches OllamaChatClient — needs to cover serverless-GPU cold
        # starts (Modal: ~30-75s for bge-m3 to load into VRAM after idle).
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(180.0),
            headers=dict(headers) if headers else {},
        )
        self._dimension: int | None = None

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": texts},
        )
        if response.status_code // 100 != 2:
            body = response.text[:500]
            raise RuntimeError(
                f"Ollama /api/embed returned {response.status_code}: {body}"
            )
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama /api/embed returned unexpected payload: {str(data)[:500]}"
            )
        return [list(map(float, vec)) for vec in embeddings]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = await self._embed(texts)
        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])
        return vectors

    async def aclose(self) -> None:
        await self._client.aclose()
```

(The `dimension` property and a lazy-probe helper are omitted from the snippet for brevity — see [`backend/app/clients/embedding.py`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/backend/app/clients/embedding.py) in the workshop starter. The probe issues one short embed call on first access if the dimension hasn't been observed yet, then caches it forever.)

Six things worth pausing on:

**The endpoint is [`/api/embed`](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings) (plural), not `/api/embeddings`.** Ollama has *both*: the new `/api/embed` accepts an `input` array and returns `{"embeddings": [...]}`; the older `/api/embeddings` accepts a single `prompt` and is being phased out. Always use the new one — it batches.

**One [`httpx.AsyncClient`](https://www.python-httpx.org/async/) held on the instance, not per-call.** `httpx` is the [`requests`](https://requests.readthedocs.io/) of the async world. Creating a new client per request would re-pay the TCP+TLS handshake on every embedding call — for a 50-page ingestion run that's 50 needless handshakes. Holding one client on the instance reuses the connection. We expose `aclose()` so the FastAPI shutdown hook can release it cleanly; we don't wire that in this post (no app code calls `get_embedding_client` yet) — that wiring happens in Post 6 where the chat orchestrator first holds a long-lived client.

**The timeout is 180 seconds, which sounds insane until you see Modal cold starts.** On a laptop, Ollama responds to an embed call in ~50ms. In production on Modal, the first call after the GPU has scaled to zero has to (a) allocate a GPU, (b) pull the bge-m3 weights from disk into VRAM, (c) then do the embed — easily 30–75 seconds for a single call. The same client code runs in both environments, so the timeout has to cover the worst case.

**Errors are explicit.** Non-2xx status → raise with the body snippet included. An unexpected payload shape → raise with the shape included. Both errors point at the actual problem (wrong model name, Ollama not running, version mismatch) instead of a `KeyError: 'embeddings'` two stack frames deeper. This is one of those "future-you debugging at 11pm" details that pays for itself the first time it fires.

**Dimension is detected lazily, not declared.** Different embedding models have different dimensions — bge-m3 is 1024, [nomic-embed-text](https://ollama.com/library/nomic-embed-text) is 768, [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is 384. Rather than hard-code the number (and silently break on model swap), we let the first response teach us. If `embed_batch` runs first (the typical path), the size of the first vector becomes the cached dimension. Subsequent reads of `.dimension` are free.

**The injected `headers` dict is the Modal-auth seam.** In local dev it's empty; in production it's `{"Modal-Key": "...", "Modal-Secret": "..."}` — Modal's [proxy-auth](https://modal.com/docs/guide/webhook-urls#auth-tokens) for serverless endpoints. The implementation doesn't care which mode it's in; the factory in `clients/__init__.py` builds the right headers from `Settings` and hands them in. This is dependency injection in its smallest possible form.

### `SentenceTransformersEmbeddingClient` — the zero-network fallback

There are good reasons to have a *second* embedding implementation that doesn't go through Ollama. Maybe Ollama isn't running. Maybe you're in CI without a GPU. Maybe you want a pure-Python path with no HTTP server in the loop, for benchmarking or debugging. Maybe you just want to know that the abstraction is real and not "Ollama by another name."

[`sentence-transformers`](https://www.sbert.net/) is the [Hugging Face](https://huggingface.co/)-backed Python library that loads embedding models directly into your process. Slower to start (a one-time ~2GB model download from [the Hugging Face Hub](https://huggingface.co/BAAI/bge-m3) on first use), faster per call after warmup on CPU, and zero network at runtime.

```python
class SentenceTransformersEmbeddingClient:
    def __init__(self, model: str) -> None:
        self._model_name = model
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(
                "Loading sentence-transformers model %s "
                "(first use; may download ~2GB)", self._model_name
            )
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = await asyncio.to_thread(self._ensure_model)

        def _encode() -> list[list[float]]:
            arr = model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return [vec.tolist() for vec in arr]

        return await asyncio.to_thread(_encode)
```

Four design choices worth flagging:

**Lazy loading, not eager.** The model isn't loaded in `__init__`. Loading is what triggers the 2GB download on first ever run — we don't want `import app.clients.embedding` to fire a download. The `_ensure_model` helper loads on first `embed_batch` and caches the loaded model. Importing the module is a no-op; the first embed call does the work and logs it.

**`asyncio.to_thread(...)` because `model.encode` is synchronous.** [sentence-transformers](https://github.com/UKPLab/sentence-transformers) is built on [PyTorch](https://pytorch.org/) and exposes a blocking API. Calling it on the asyncio thread would freeze the event loop for as long as the encode runs (seconds on CPU, milliseconds on MPS/CUDA). [`asyncio.to_thread`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread) hands the blocking work to a thread-pool worker so the loop stays responsive. Same call shape from the caller's point of view — `await client.embed_batch([...])` — regardless of whether the implementation is HTTP (Ollama) or in-process (sentence-transformers).

**`normalize_embeddings=True`.** [ChromaDB](https://www.trychroma.com/) defaults to [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) for its distance metric; cosine on pre-normalized vectors becomes a plain dot product, which is fast. The Ollama side doesn't have an explicit "normalize" flag, but the [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) recommends L2-normalization for downstream similarity tasks and Ollama applies it by default. Same vectors out of both backends.

**`batch_size=32`.** A sensible mid-range default for the encode call. Too small and the GPU sits idle; too large and you run out of VRAM on machines that have less. 32 lands in the middle for BGE-M3.

### The gotcha: model name differs by provider

There is exactly one subtlety in swapping between these two clients: **the model name is provider-specific**.

| Provider | `EMBEDDING_PROVIDER=` | `EMBEDDING_MODEL=` |
|---|---|---|
| Ollama | `ollama` | `bge-m3` |
| sentence-transformers | `sentence-transformers` | `BAAI/bge-m3` |

Same underlying weights. Same 1024-dimensional vectors. Different naming conventions — Ollama mirrors its [model library naming](https://ollama.com/library), sentence-transformers mirrors [Hugging Face repo names](https://huggingface.co/BAAI/bge-m3).

The factory passes the configured name through unchanged. We could have built a name-mapping layer ("translate `bge-m3` → `BAAI/bge-m3` when the provider flips") but it would mask a real conceptual point: *the model name belongs to the provider, not the abstraction*. If you flip `EMBEDDING_PROVIDER`, flip `EMBEDDING_MODEL` to match. The project's `.env.example` is explicit about this:

```bash
# IMPORTANT: EMBEDDING_MODEL is provider-specific:
#   - ollama:                bge-m3
#   - sentence-transformers: BAAI/bge-m3
# Flip both together when switching providers. Both produce 1024-dim vectors.
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=bge-m3
```

When a configuration choice creeps across two variables, comment them both with the same caveat — and write that caveat next to *both* variables. Documentation in one place is documentation that drifts.

---

## Seam 3 — ChatClient: A Preview of What Post 6 Will Use {#seam-chat}

The third Protocol is the chat client. We won't *use* it in this post — the orchestration that calls `stream(...)` lands in Post 6, and the streaming-to-the-browser SSE plumbing lands in Post 7 — but defining it now serves two purposes: it shows the full set of Protocols the project leans on, and it's the most interesting case for the abstraction itself, because Ollama and Anthropic disagree about almost everything except "stream me some tokens for an assistant turn."

```python
# backend/app/clients/chat.py
class ChatClient(Protocol):
    # Implementations are `async def stream(...) -> AsyncIterator[str]`,
    # i.e. async-generator functions. Calling one returns the generator directly,
    # so the Protocol declares a sync function returning AsyncIterator — that's
    # what `async for token in client.stream(...)` actually consumes.
    def stream(
        self,
        system: str,
        messages: list[Message],
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream text tokens for the next assistant turn."""
        ...

    async def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 256,
        json_format: bool | dict[str, Any] = False,
    ) -> str:
        """One-shot completion. Returns the full assistant text in one call."""
        ...
```

> *Plain-English aside: what's an [async generator](https://peps.python.org/pep-0525/)?* An async generator is a function that produces a sequence of values *asynchronously* — you write it with `async def` and `yield`, and the caller consumes it with `async for`. We use it here because the model returns tokens one at a time over a streaming HTTP connection, and we want to surface each token to the frontend (over [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)) as it arrives, rather than waiting for the full response. The Protocol declares `stream(...) -> AsyncIterator[str]` (sync function returning an async iterator) because calling an async-generator function returns the generator object directly — the body doesn't start running until `async for` is invoked. [PEP 525](https://peps.python.org/pep-0525/) has the full mechanics if this trips you up.

Two interface choices worth flagging:

**`stream` *and* `complete`, not just `stream`.** Streaming is the main act — the page-mode and wiki-mode chat answers stream token-by-token. But the project also generates two follow-up "suggestion chips" after every answer (you saw these in [Post 1's walkthrough]({% post_url 2026-05-09-pepper-carrot-companion-trailer %})). Those don't need streaming — they're tiny, the frontend shows them all at once at the end, streaming would buy nothing. So `complete()` is a one-shot non-streaming call. The two methods cleanly cover both use cases without trying to make `stream()` pretend to be a one-shot via "collect all tokens into a string."

**`json_format` on `complete()`.** This one is interesting and deserves its own paragraph. The follow-up chips need to come back as structured JSON (`{"suggestions": [{"text": ..., "mode": "page" | "wiki"}, ...]}`). Big models like Claude follow [JSON-in-prompt instructions](https://docs.claude.com/en/docs/build-with-claude/json-mode) reliably; small local models (qwen2.5:7b) will happily ignore them and emit a paragraph of prose with maybe-JSON in the middle. To rescue this, Ollama supports [**structured outputs**](https://ollama.com/blog/structured-outputs) — if you pass a [JSON Schema](https://json-schema.org/) as the `format` field, Ollama constrains the *sampling step itself* to match the schema, token by token. The output is *literally not capable* of being malformed JSON.
- `json_format=False` — no constraint.
- `json_format=True` — Ollama's bare `format: "json"`; output is *some* valid JSON but keys/structure are unconstrained.
- `json_format={"type": "object", "properties": ...}` — pass a schema dict; Ollama enforces it during sampling.

The Anthropic implementation ignores `json_format` (Claude is reliable enough at JSON formatting from prompt instructions alone). Same call site; different implementation behavior. The abstraction *accommodates the union of provider capabilities*, and provider-specific knobs are exposed via optional parameters that other implementations are free to no-op.

Both [`OllamaChatClient`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/backend/app/clients/chat.py) and [`AnthropicChatClient`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/backend/app/clients/chat.py) live in `backend/app/clients/chat.py`. We won't walk through them line by line in this post — too much detail for a part that isn't yet wired up — but they're worth a brief tour now:

- `OllamaChatClient.stream` POSTs to [`/api/chat`](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion) with `stream: true` and iterates the response as newline-delimited JSON, yielding `chunk["message"]["content"]` per token.
- `AnthropicChatClient.stream` uses [`anthropic.AsyncAnthropic().messages.stream(...)`](https://github.com/anthropics/anthropic-sdk-python) — and additionally turns on [prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching) via `cache_control={"type": "ephemeral"}`, which ~90%-discounts repeat tokens across multi-turn chat. (Post 8 explains why this is load-bearing for the cost story.)
- Both build the `messages` list from the same internal `Message` model — a small Pydantic class with `role` and a list of `ContentBlock`s (text or image). Either provider's wire format is constructed from this neutral representation immediately before the call. Provider-specific encoding stays inside the provider's file.

The same call site — `async for token in client.stream(system, messages, max_tokens=1024):` — drives both. We'll see this in action in Post 6 and 7.

---

## The Factory in `clients/__init__.py` {#factory}

We have three Protocols and six implementations (one of them a stub). The last piece is the wiring: how does the rest of the codebase get the *right* implementation for the current configuration?

There is exactly one place that knows this: `backend/app/clients/__init__.py`.

```python
# backend/app/clients/__init__.py
from app.clients.chat import AnthropicChatClient, ChatClient, OllamaChatClient
from app.clients.embedding import (
    EmbeddingClient, OllamaEmbeddingClient, SentenceTransformersEmbeddingClient,
)
from app.clients.storage import LocalStorage, R2Storage, Storage
from app.config import Settings


def get_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "local":
        return LocalStorage(
            root=settings.local_image_dir,
            url_prefix=settings.local_image_url_prefix,
        )
    if settings.storage_backend == "r2":
        # ... validate required R2 fields, build R2Storage ...
        return R2Storage(...)
    raise ValueError(f"Unknown storage_backend: {settings.storage_backend}")


def get_chat_client(settings: Settings) -> ChatClient:
    if settings.chat_provider == "ollama":
        return OllamaChatClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            headers=_modal_proxy_headers(settings),
        )
    if settings.chat_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required when chat_provider=anthropic"
            )
        return AnthropicChatClient(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )
    raise ValueError(f"Unknown chat_provider: {settings.chat_provider}")


def get_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformersEmbeddingClient(model=settings.embedding_model)
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
            headers=_modal_proxy_headers(settings),
        )
    raise ValueError(f"Unknown embedding_provider: {settings.embedding_provider}")
```

This file is small on purpose. Three observations:

**Every factory has the same shape.** Read `settings.<provider>`, route to a constructor, fail loudly on unknown values. No clever inheritance, no registry, no plugin system. The cost of being explicit (one `if/elif/else` per Protocol) is much lower than the cost of being clever (a `PROVIDERS = {...}` dict that imports lazily, fails late, and hides which providers actually exist).

**Return type is the Protocol, not the concrete class.** `get_storage(settings) -> Storage`, not `-> LocalStorage`. Callers see a `Storage`; `mypy --strict` won't let them dot into anything that isn't on the Protocol. This is the wall that prevents leaks: a route handler that accidentally writes `storage.upload_to_r2(...)` (which would only exist on `R2Storage`) fails at type-check time.

**The `Settings` object is the only argument.** Not a `db_url`, an `api_key`, a `model_name` — just `Settings`. The factory unpacks what it needs. If we ever add a new field — say `OLLAMA_CHAT_OPTIONS` — only the factory and the affected client change; no caller in routes/services has to grow a new constructor argument. This is *also* why `Settings` lives in `app/config.py` as a single pydantic-settings class (Post 2): one canonical source of truth, typed, loaded from `.env`.

The `_modal_proxy_headers` helper is one tiny additional flourish: when running against a Modal endpoint, the request needs proxy-auth headers; the helper builds them from `Settings.modal_proxy_token_id` + `modal_proxy_token_secret` and fails loudly if *exactly one* of the two is set (a common "I set the secret but forgot the ID" footgun). It's worth showing because the production deploy in Post 10 is the first time those fields get populated, and the friendly error message is the difference between a five-minute fix and a twenty-minute "why am I getting 401s?" chase.

---

## Verification: Prove the Swap Works {#verification}

Three commands. Each one runs the same code path through a different `EMBEDDING_PROVIDER`.

```bash
# A — the default: Ollama on localhost (talks to the daemon from Post 2)
cd backend && uv run python -c "
import asyncio
from app.clients import get_embedding_client
from app.config import get_settings

async def main():
    client = get_embedding_client(get_settings())
    vecs = await client.embed_batch(['Pepper is a witch', 'Carrot is a cat'])
    print(f'{client.model_name}: {len(vecs)} vectors of dim {client.dimension}')

asyncio.run(main())
"
# Expected: bge-m3: 2 vectors of dim 1024

# B — same script, different provider (note the model-name change)
EMBEDDING_PROVIDER=sentence-transformers EMBEDDING_MODEL=BAAI/bge-m3 \
  uv run python -c "
import asyncio
from app.clients import get_embedding_client
from app.config import get_settings

async def main():
    client = get_embedding_client(get_settings())
    vecs = await client.embed_batch(['Pepper is a witch', 'Carrot is a cat'])
    print(f'{client.model_name}: {len(vecs)} vectors of dim {client.dimension}')

asyncio.run(main())
"
# Expected: BAAI/bge-m3: 2 vectors of dim 1024
# (first run may pause for a 2GB model download from HuggingFace)

# C — type-check everything
uv run mypy app/
# Expected: Success: no issues found
```

If all three pass, you've verified the seam. The same `main()` function above — the same `await client.embed_batch([...])` line — drove an HTTP call to Ollama and an in-process [PyTorch](https://pytorch.org/) inference, and `mypy --strict` was happy with both. **That's what "provider abstraction" looks like when it works.**

And the storage smoke test you ran earlier — `curl http://localhost:8000/images/test/hello.txt` printing the file content — is the other half. The full set of seams now stands:

- `Storage` ← `LocalStorage` (✓) | `R2Storage` (stub, fills in Post 10)
- `EmbeddingClient` ← `OllamaEmbeddingClient` (✓) | `SentenceTransformersEmbeddingClient` (✓)
- `ChatClient` ← `OllamaChatClient` (✓ defined; first call site in Post 6) | `AnthropicChatClient` (✓ defined; first call site in Post 6)

---

## The Discipline That Makes This Work {#discipline}

The abstraction is cheap to write. Keeping it cheap to *use* across the rest of the project requires a small amount of ongoing discipline. Four rules that the project's [`CLAUDE.md`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/CLAUDE.md) captures explicitly, and which I'd suggest stealing for your own work:

**1. SDK imports stay in `clients/`.** No `import anthropic` in a route handler. No `import httpx` outside `clients/` *unless* it's for something genuinely client-agnostic (like a webhook caller). If a route needs Anthropic-specific behavior, extend the `ChatClient` Protocol with an optional parameter and have the Ollama side no-op it. The instinct to reach for the SDK directly when a feature seems specific to one provider is the bug — every such case is an invitation to extend the Protocol.

**2. The factory is the only place that knows providers exist.** Callers ask for a Protocol-typed instance; they don't construct one. If you find yourself writing `LocalStorage(...)` outside `clients/`, you've leaked.

**3. New embedding model? Bump the collection version.** Vectors from `bge-m3` and `nomic-embed-text` are not interchangeable — they have different dimensions and different similarity geometries. Mixing them in one ChromaDB collection silently breaks retrieval. The repo's discipline is to name collections like `pages_v1_bge-m3` and re-ingest on model change. Documented in [`docs/decisions/0002-model-provider-abstraction.md`](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop/blob/main/docs/decisions/0002-model-provider-abstraction.md).

**4. Resist provider-specific shortcuts.** The Anthropic SDK has features Ollama doesn't (prompt caching, [tool use](https://docs.claude.com/en/docs/build-with-claude/tool-use), [extended thinking](https://docs.claude.com/en/docs/build-with-claude/extended-thinking)). The temptation is to expose them via Anthropic-specific calls. The discipline is to: (a) extend the Protocol with an optional, well-typed knob; (b) implement it in `AnthropicChatClient`; (c) no-op it in `OllamaChatClient`. The `cache_control={"type": "ephemeral"}` flag in the current `AnthropicChatClient` is a perfect example — it's *internal* to the implementation, not exposed on the Protocol, because Ollama can't honor it and there's no benefit to the caller knowing.

These rules sound restrictive on paper. In practice the codebase has barely felt them — there's roughly one moment per phase where you'd be tempted to break a rule, and the work to extend the Protocol instead is a few minutes. The payoff is that none of the *callers* ever need updating.

---

## Key Takeaways {#key-takeaways}

**1. The Protocol is the contract; the implementation is the choice.** Once `Storage`, `EmbeddingClient`, and `ChatClient` exist, the rest of the codebase is portable by construction. Local ↔ cloud, Ollama ↔ Anthropic, filesystem ↔ R2 — each becomes a one-line change in a single file (`.env`), without any changes to call sites.

**2. Provider abstractions are cheap to build *first*, expensive to retrofit.** A `Protocol` plus a factory adds ~30 lines per seam. Extracting the same shape from a codebase that's been calling `openai.ChatCompletion.create(...)` from twelve places over six months is a multi-day refactor — and the discipline of "is this Anthropic-specific?" gets harder as more places call it. Put the seam in *before* the first caller, not after.

**3. Design the Protocol around the union of provider capabilities, not the intersection.** `Storage.put` takes a `content_type` even though `LocalStorage` ignores it, because `R2Storage` needs it. `ChatClient.complete` takes a `json_format` argument even though `AnthropicChatClient` ignores it, because Ollama's [structured-output mode](https://ollama.com/blog/structured-outputs) is the only way to get reliable JSON out of small local models. *Lowest-common-denominator* Protocols make every implementation worse; *union* Protocols make every implementation possible.

**4. The factory is boring on purpose.** A 30-line `if/elif/raise` is more legible than a 100-line plugin-registry framework, and the only thing it gives up is the ability to load providers you haven't imported — which is exactly the kind of cleverness that produces "why is this string failing at production startup?" puzzles at 11pm. The factory is the one place you want to *see* every concrete class listed.

**5. The same code runs on a laptop and in the cloud. That's not a fluke; it's the design.** When Post 10 sets `STORAGE_BACKEND=r2`, `CHAT_PROVIDER=ollama` with `OLLAMA_BASE_URL` pointing at Modal, and `EMBEDDING_PROVIDER=ollama` pointing at the same Modal — and the deployed backend just works — that's because *the rest of the codebase has been writing against the Protocols this whole time*. The deploy story is short because the architecture work happened in this post.

---

Next up: **Post 4 — Claude Skills as an Ingestion Tool: When the Best Vision Model Is the One Driving Your Editor.** We use Claude Code itself as a one-shot batch vision processor: a `.claude/skills/ingest-from-images/SKILL.md` file walks Claude through reading each page of episode 1 and writing a structured `PageDescription` JSON next to the image on disk. Then `JsonFileVisionClient` — the fourth Protocol implementation, also living in `backend/app/clients/`, also obeying the rule we just built — picks those JSONs up. By the end of that post, episode 1 is fully ingested: images in `LocalStorage`, descriptions in Postgres, embeddings in ChromaDB. The chat layer doesn't run a vision model in production; that's the whole point.

The **workshop starter** that backs this post is at <https://github.com/bearbearyu1223/pepper-carrot-companion-workshop>. The **full source repository** and a public live-demo URL go up alongside Post 10 of this series — the deploy guide — once it's published.

*Pepper & Carrot* is © [David Revoy](https://www.davidrevoy.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All credit to him for the source material that made this project possible.

**All opinions expressed are my own.**
