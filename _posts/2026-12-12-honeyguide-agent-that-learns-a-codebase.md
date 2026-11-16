---
title: "Agentic Workflows [1]: Honeyguide: building an agent that learns a new codebase (and checking that it did)"
date: 2026-12-12 09:00:00 -0800
categories: [Agentic Workflows, Code Understanding]
tags: [agentic-workflow, claude-code, subagents, coding-agents, code-understanding, evaluation]
description: >-
  Coding agents re-read a repository for every question. I built Honeyguide, a
  small team of Claude Code subagents that explores a repo once and writes an
  interactive map where every claim points at real lines of code, and you ask
  follow-ups on the map itself. This post covers how it's built, how it keeps
  what it says honest (anchors a script resolves, evidence labels, a separate
  checker, a runtime trace), and a test of whether the map pays off on two
  repos, against exploring each time, a free outline, and a model with no
  access to the code.
published: false
---

<!-- TODO(draft): first draft. The intro, short version (except its results), Setup, §1-§3 and the appendix are drafted;
§3's Checker receipts and §4's first maps and live questions come from outputs/runs.txt (2026-11-16 runs). §4's
screenshots and video, Han's Checker sample, §5's results and §6 wait for the redesigned page and the eval. No [CITE: ...] markers are left. Honeyguide v1 is still being built: re-read §2 and §3 against the code as built. -->

## Asking the same repo the same questions

When I open a coding agent in a repository I don't know and ask where something happens, it does what I'd do. It searches for likely names, opens a few files, follows an import or two, and answers. Then I ask a second question about the same part of the code, and it searches and opens the same files again. Whatever it worked out the first time lives only in that conversation, and a new session starts from nothing.

The tools do have places to keep what they learn. Claude Code reads [`CLAUDE.md` files](https://code.claude.com/docs/en/memory) from the folder you start it in and the folders above it, editors have built search indexes, and some services generate a whole wiki for a public repo. What I couldn't find in any of them is a check that what they keep still says what the code says. And the one tool I found that measured whether keeping it helps, Cursor, has since stopped using its search-by-meaning index ([§1](#why-a-map)).

The project's name comes from a bird. In Africa, the greater honeyguide leads people to wild bees' nests: the people subdue the bees and open the nest for the honey, and the bird eats the wax ([Spottiswoode and Wood, 2023](https://par.nsf.gov/servlets/purl/10524235)). It knows where the nest is, and you do the work of getting in. That's the job I wanted done for a codebase: something that has already been through the repo and can lead me to the lines that answer my question.

So I built **Honeyguide**. It's a small team of Claude Code agents that explores a repo once and writes an interactive **map** of it: an overview, the modules, and the main flows through the code, where every claim points at the lines it came from. You read the map in a browser, highlight anything that confuses you, and ask. The answer comes back on the map, with its own pointers into the code.

That leaves two questions. Can an agent learn a repo once and keep what it learned honest as the code changes? And is the map worth having, compared with letting the agent explore each time?

### The short version {#the-short-version}

Can an agent learn a codebase once, so I don't pay for it to re-read the repo with every question, and how would I know whether what it learned is right?

**What it is** ([§2](#whats-in-the-build)). Honeyguide is five roles in Claude Code: a 🗺️ **Cartographer** that explores the repo and writes the map, an 🔦 **Investigator** that answers questions, a 🧵 **Tracer** that answers "what happens when..." questions as a path through the code, a ✅ **Checker** that re-reads each claim against the lines it cites, and the 🧭 **Guide**, the Claude Code session you're already in, which hands each question to the right agent. Everything exact (finding line numbers, hashing code, reading git) is a plain script. The page never starts Claude: a question asked on the map waits in an inbox until the session picks it up.

**How it's kept honest** ([§3](#how-its-kept-honest)). An agent never types a line number. It quotes the code it means, and a script finds those lines at a pinned commit, so every pointer on the map resolves to real code. That proves the lines exist, and nothing more, so each claim also says what kind of evidence it rests on (read in the source, asserted by a test, seen when the code ran, inferred, and so on), and a separate Checker reads the cited lines again and marks whether they support the claim. For flows, a script runs the code on a toy input and records which functions actually ran, which catches paths that reading alone gets wrong. When the code changes, claims whose lines changed are marked stale and aren't served as fact until they're re-checked.

**The test** ([§5](#does-the-map-pay-off)). Two repos, each pinned to one commit: Andrej Karpathy's nanochat, mostly written before the model's training cutoff, and waku-agent, a personal assistant agent written after it. The same model answers the same frozen questions four ways: with no access to the code, exploring the repo, exploring with a free script-built outline, and exploring with the map. Splitting the questions by whether their code predates the model's cutoff separates learning the repo from remembering it.

<!-- TODO(draft): the results paragraph, written from outputs/eval_<repo>.txt once the eval has run: the paired
differences map - explore, map - outline, explore - no-access with their 95% CIs; learning vs remembering on the
before/after-cutoff subsets; cost per correct answer and the break-even point, or that it never pays. Bullets of
36-85 words each, each pointing at its section. -->

**[RESULTS: does the map pay off, from the eval capture]**

### Setup

The code is in [Honeyguide](https://github.com/bearbearyu1223/honeyguide), a separate repo from the LLM Architecture Refresh series, but this post follows that series' rule: every number below is printed by a program, and each block of program output, a **receipt**, is pasted byte for byte from a capture in the repo's `outputs/` folder. Next to the receipts is the **recipe**: the exact command that prints them.

<!-- TODO(draft): the Honeyguide repo goes public at publish; check the link resolves then. -->

You'll need Claude Code and [uv](https://docs.astral.sh/uv/), the Python package manager the series uses. Honeyguide's own scripts use only Python's standard library.

```bash
git clone https://github.com/bearbearyu1223/honeyguide
cd honeyguide
uv sync
```

<!-- TODO(draft): install steps for the Claude Code side (the plugin with the agents and skill, how to start the
session watching the inbox, how to open the page). They don't exist yet; write them from the v1 README. -->

**The two repos.** The test runs on [nanochat](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd) at commit `92d63d4` and [waku-agent](https://github.com/ShenSeanChen/waku-agent/tree/2c7e7e68d3d65ac19d428bdde7820844d87dfa19) at commit `2c7e7e6`. nanochat is [MIT-licensed](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/LICENSE); waku-agent's code is [MIT](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/LICENSE) except its `hosted/` directory ([Elastic License 2.0](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/hosted/LICENSE)) and its brand assets ([LICENSE-BRAND](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/LICENSE-BRAND)), none of which this post quotes. Every link into their code in this post is pinned to that commit, so it still shows the same lines after the repos move on. Code quoted from them keeps their licence notice.

**Recipes.** The runtime traces and the eval questions in §3 and §5 come from two commands. The trace script runs a few lines of each repo's own code in that repo's uv environment (its own Python and packages), at its pinned commit:

```bash
uv run python eval/trace/capture.py nanochat > outputs/trace_nanochat.txt
uv run python eval/questions/generate.py nanochat --cutoff 2026-06-30 \
    > outputs/questions_nanochat.txt
```

and the same with `waku-agent`.

<!-- TODO(draft): how a reader fetches the two target repos at their pins and builds their uv environments (today the
scripts expect them in a fixed folder); the eval's runner and scorer commands, the CLI version and the model id once
they exist. -->

**Two kinds of numbers.** Most numbers here are *counted*: how many questions of each type, how many functions a trace saw. Those come out the same on every machine. A few are *measured*, like wall-clock time and, later, tokens and cost, and they move from run to run. I'll say which is which as they come up.

### Table of Contents

[The short version](#the-short-version) above has the findings without the details.

1. [Why a map?](#why-a-map)
2. [What's in the build](#whats-in-the-build)
3. [How it's kept honest](#how-its-kept-honest)
4. [What it's like to use](#what-its-like-to-use)
5. [Does the map pay off?](#does-the-map-pay-off)
6. [What I learned](#what-i-learned)

Then [what's next](#whats-next) and [the glossary](#appendix-glossary) at the end.

---

### 1. Why a map? {#why-a-map}

A **coding agent** is a language model that can act on your code through tools: search for text, read a file, run a command, edit. Claude Code is the one I use. Everything the model knows about your repo during a task is in its **context window**, the text it can see at once, measured in **tokens** (chunks of text of a few characters each; usage and cost are counted in tokens).

Ask an agent a question about an unfamiliar repo and a lot of the work goes into finding things. It searches, reads, follows a call, reads again. The answer can be good. But the reading is thrown away when the conversation ends, and even inside one conversation, a question about another part of the code starts the search over.

<!-- FIGURE F1 (mechanism, no numbers): a plain agent re-reads the repo for every question; a map moves the reading to
the front, once. Left: three questions, each with its own fan of searches and reads into the repo. Right: one
exploration that writes the map, then three questions that each start from it. -->

There are already several ways to keep what an agent learns.

- **`CLAUDE.md` and `/init`.** Claude Code loads `CLAUDE.md` files from the folder you start it in "and every directory above it", and the ones in subfolders when it reads files there ([Claude Code docs, memory](https://code.claude.com/docs/en/memory)). The `/init` command "creates a file with build commands, test instructions, and project conventions it discovers". It's a file of instructions and notes, usually short, and nothing checks it against the code after it's written.
- **Auto-memory.** Claude Code can also save notes on its own and load them into later sessions, up to the first 200 lines or 25KB. The same page says it "skips anything it can derive from the codebase".
- **Editor indexes.** Until mid-2026, Cursor split a codebase into chunks, turned each chunk into an **embedding** (a list of numbers that puts text with similar meaning close together), and kept them in a "specialized vector database", so its agent could search by meaning ([Cursor's semantic-search docs, as archived on 2026-02-25](https://web.archive.org/web/20260225213818/https://cursor.com/docs/context/semantic-search)). In July 2026 Cursor's staff said on its forum that this indexing was "being turned down in favor of grep-based retrieval", meaning search for exact text, the way the classic `grep` tool does ([Cursor forum, 2026-07-22](https://forum.cursor.com/t/what-do-you-think-about-cursor-removing-the-codebase-indexing-settings/165899/10)), and its [search docs](https://cursor.com/docs/agent/tools/search) now say it "does not store embeddings of your codebase for search". It keeps a local index for text search instead, called Instant Grep.
- **Generated wikis.** [DeepWiki](https://docs.devin.ai/work-with-devin/deepwiki), from Cognition, the makers of Devin, "produces wikis with architecture diagrams, links to sources, and summaries" for a repo. Its docs don't say whether those links point at particular lines, or whether a wiki updates itself when the code changes. Google's Code Wiki, launched in public preview for public repos in November 2025, keeps a wiki where every section and answer is "hyper-linked directly to the relevant code files and definitions" ([Google Developers Blog, 2025-11-13](https://developers.googleblog.com/en/introducing-code-wiki-accelerating-your-code-understanding/)). nanochat's own README points readers to DeepWiki: "For questions about the repo, I recommend either using [DeepWiki](https://deepwiki.com/karpathy/nanochat) from Devin/Cognition to ask questions about the repo, or use the [Discussions tab](https://github.com/karpathy/nanochat/discussions), …" ([README, line 8](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/README.md?plain=1#L8)).
- **aider's repo map.** [aider](https://aider.chat/docs/repomap.html) builds a compact map of the repo with tree-sitter, a library that parses source code ([aider's blog, 2023](https://aider.chat/2023/10/22/repomap.html)). It decides what goes in with "a graph ranking algorithm, computed on a graph where each source file is a node and edges connect files which have dependencies", and caps the map's size with a token budget, 1k tokens by default (the `--map-tokens` setting).
- **RepoRabbit.** My own earlier attempt, and the closest to this one. [RepoRabbit](https://github.com/bearbearyu1223/repo-rabbit) is a Claude Code subagent that explores a repo and writes an interactive map it calls an Atlas. Each claim is labelled by the kind of evidence behind it, you highlight text and ask, and a helper script rejects any reference that doesn't resolve to a real file and line. After the code changes, a refresh flags explanations whose code changed.

Each of these keeps knowledge about the code somewhere outside the code. Where their docs say how it stays current, it's by rebuilding: Cursor's index checked for changes "every 5 minutes" (the archived docs above), and Code Wiki "regenerates the documentation after each change" ([Google Developers Blog](https://developers.googleblog.com/en/introducing-code-wiki-accelerating-your-code-understanding/)). None of the docs I read describes checking a kept claim against the lines it cites. RepoRabbit came closest: it rejects any reference to a file or line that doesn't exist ([README, line 115](https://github.com/bearbearyu1223/repo-rabbit/blob/b9e9eae/README.md?plain=1#L115)), and flags an explanation when the lines it cites change ([line 177](https://github.com/bearbearyu1223/repo-rabbit/blob/b9e9eae/README.md?plain=1#L177)). But lines can exist, and stay unchanged, and still not say what the claim says.

As for whether keeping the knowledge pays off, Cursor is the one that measured it. In November 2025 it reported that adding semantic search (the search by meaning above) gave its agent "on average 12.5% higher accuracy in answering questions" than grep alone, on its own set of questions ([Cursor blog, "Improving agent with semantic search"](https://cursor.com/blog/semsearch)). By July 2026 its staff said that path "was rapidly no longer helping in a meaningful way", and that its agents now lean on Instant Grep and plain file reads ([Cursor forum, 2026-07-21](https://forum.cursor.com/t/what-do-you-think-about-cursor-removing-the-codebase-indexing-settings/165899/5)). Paying off is the reason to keep knowledge at all, and for Cursor an index of chunks stopped doing it. So the question this post tests is whether a map of checked claims, rather than an index of chunks, pays off against an agent that searches the repo itself.

Honeyguide is my attempt at both: a map that's checked against the code claim by claim, and a test of whether it pays off.

---

### 2. What's in the build {#whats-in-the-build}

Honeyguide is built from three things Claude Code already provides, plus a page and some scripts.

- A **subagent** is a separate Claude instance that the main conversation starts for one task. In the docs' words, "Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions", and it starts from a "fresh, isolated context window" ([Claude Code docs, subagents](https://code.claude.com/docs/en/sub-agents)). It does the task in its own context "and returns only the summary", so the main conversation gets the report without everything the subagent read.
- A **skill** is a `SKILL.md` file of instructions that Claude Code loads automatically when it's relevant, or that you call by name with `/skill-name`. A **plugin** bundles skills, subagents and a few other kinds of add-on into one package ([Claude Code docs, glossary](https://code.claude.com/docs/en/glossary)).
- **Non-interactive mode** (formerly called headless mode) runs Claude Code without the interactive window: `claude -p "..."` takes a prompt, runs to the end, and prints the result, so a script can call it. `--output-format` chooses plain text, JSON, or a stream of JSON ([glossary](https://code.claude.com/docs/en/glossary); [CLI reference](https://code.claude.com/docs/en/cli-reference)). The eval in §5 uses it; the product doesn't.

#### Five roles, and why each is an agent

The rule I held to: a job gets its own agent only when a fresh context or parallel work earns it. Anything exact is a script.

| Role | What it does | Why it's an agent |
| --- | --- | --- |
| 🧭 Guide | the main Claude Code session: picks up questions from the page, hands each to an agent, records what's running, writes results back | it's the session you're already in, so it has your permissions and you can watch it work |
| 🗺️ Cartographer | explores the repo once and writes the map: overview, modules, flows, every claim pointing at code | reading a whole repo fills a context; in a subagent, that reading doesn't crowd the Guide's |
| 🔦 Investigator | answers one question or follow-up, pointing every claim at code | each question gets a clean context, and several can run at once |
| 🧵 Tracer | answers flow questions ("what happens when...") as a path: entry point, each call in order, where data changes shape, the end | a path is a different kind of answer, and it can confirm hops by running the code (§3) |
| ✅ Checker | re-reads each claim against the lines it cites and marks it supported or not supported; it reports and never edits | a context that didn't write a claim is less inclined to agree with it |

A Claude instance checking another Claude instance's claim is still Claude grading Claude, so its verdict isn't proof on its own. I check a sample of its verdicts by hand, and the agreement is printed in [§3](#the-checker).

The scripts do the work where a model's guess isn't good enough: resolving anchors to lines and hashing them (§3), the runtime trace (§3), validating the map's structure, building the free outline the test compares against (§5), rendering the page and its diagrams, serving the page and its question inbox, the whole eval, and scrubbing transcripts of personal paths before anything is published.

#### Asking a question without leaving the map

The map is a web page served on your own machine. You read it like a document, with the contents on the left. Select any passage, including a passage inside an answer, and a small toolbar offers **Ask**, **Annotate** and **Copy**. Questions and answers open in a discussion panel on the right, so the map itself never shifts under you.

What happens to a question is the part I cared most about getting right:

1. The page writes the question, with the exact text you selected, to an **inbox** on disk.
2. The Guide, in the Claude Code session you started, notices it and decides who should answer: the Investigator, or the Tracer if it's a flow question.
3. That agent works in its own context, with the map and any earlier answers about the same code as a starting point.
4. The answer is written back to disk, and the page shows it next to your question.

The page never starts Claude itself. Everything runs in the one session you opened, under the permissions you gave it, and you can watch it in the terminal. RepoRabbit has an option to launch Claude Code in the background whenever a question was saved; here, nothing runs that you didn't start.

<!-- TODO(draft): confirm against the v1 code how the session notices the inbox (watcher, hook or polling), and say
it in one sentence. -->

While agents work, their avatars (the emoji above) animate in the panel beside the thread they're working on, with one small indicator at the top of the map when the panel is closed. Every answer names the agent that wrote it, and a follow-up on an answer remembers which words you selected, so a thread can branch.

<!-- FIGURE F2 (mechanism; role names and emoji): the page never starts Claude. Page -> inbox -> one Claude Code
session (the Guide) -> Cartographer / Investigator / Tracer / Checker -> answer written back -> page. -->

#### What it remembers

Everything Honeyguide learns about a repo is saved in a folder outside the checkout, `~/.honeyguide/<repo-id>/`, in three layers: the **map**, the **answers** (each with its evidence and whether the Checker has checked it), and your personal **notes**. The repo id is the repo's remote address plus its first commit, so any clone of the same repo, over HTTPS or SSH, finds the same memory, and nothing is written into your working tree.

When the Investigator or Tracer answers a new question, it gets the map plus earlier answers about the same files and functions, matched by where their claims point in the code. Your notes stay private: an agent sees a note only if you mark it "share with agents".

---

### 3. How it's kept honest {#how-its-kept-honest}

A map is only useful if you can trust it, and an agent writing about code it skimmed can be confidently wrong. Honeyguide has four guards: anchors, evidence labels, the Checker, and a runtime trace. A fifth piece, refresh, notices when the code changes under them.

<!-- FIGURE F3 (mechanism; a real claim from the nanochat map, printed by the map capture): one claim's life. Anchor
text -> the script finds the lines at the pinned commit -> the Checker's verdict -> the evidence mark on the page. -->

#### Anchors: every claim points at real lines {#anchors}

An **anchor** is a claim's pointer into the code. The agent writing a claim supplies only a file path and the text of the lines it means, copied from the file. A script then reads that file at the map's commit, finds the quoted text, and fills in the rest: the line range, the commit, the enclosing function or class, and a **hash** of those lines (a short fingerprint that changes if any of the lines change). The agent never types a line number, because a typed line number is a guess.

From the resolved lines the script also builds the two links in the code preview: a GitHub permalink pinned to that exact commit, so it shows the same lines next year, and an "open in editor" link to the line in your local copy. If the quoted text can't be found, or turns up in more than one place, the script refuses to save the claims and lists each problem, so the agent has to fix its quote.

That's a real guarantee, but a narrow one. An anchor proves the cited lines exist at that commit. It doesn't prove that they say what the claim says. A claim that a function retries on a timeout can point at the right function, and resolve perfectly, when the function never retries at all. Catching that takes the next two guards.

#### Evidence labels {#evidence-labels}

Each claim carries a label saying what kind of evidence it rests on:

| Label | Means |
| --- | --- |
| read in source | the agent read the cited lines and they say this |
| asserted by a test | a test in the repo checks this behaviour |
| docs | the repo's documentation says so (which can be out of date) |
| seen at runtime | the runtime trace saw it happen (below) |
| inferred | worked out from the code without a line that says it directly, e.g. which method a call reaches |
| searched, not found | the agent looked for something and it isn't there |
| general knowledge (not from this repo) | true in general, like how a library function behaves, and not a claim about this code |

On the page each label is a small shape beside the claim (a solid dot, a ring, a diamond and so on), so you can tell them apart without relying on colour, with the label's name on hover. Agents mix general knowledge into answers about specific code all the time, and the last label keeps it from passing as something the repo says. It has no anchor, and the Checker skips it.

#### The Checker {#the-checker}

The Checker re-reads each claim next to the lines it cites and gives one verdict: supported or not supported. It works in the background. An answer appears as soon as it's written, its claims gain a small check mark as they're checked, and a claim the Checker doesn't accept gets a quiet flag with the reason, rather than being deleted. The Checker reports; it never edits.

On the first real maps it turned a lot down. In run 2, the second build of each map ([§4](#the-first-maps) has how the first one went), it marked 102 of nanochat's claims not supported, and 47 of waku-agent's. Going by its notes, most of them were true but cited too few lines to show it.

So the Checker's verdicts now feed a **repair pass**. The Cartographer gets back each claim the Checker turned down, with the Checker's note, and does one of three things: cites more lines (it widens the claim's anchors), rewrites the claim to say only what the cited lines show (it narrows it), or retires it. The Checker then re-checks only the claims that changed. That's one round, and there are at most two. A claim that's still not supported after that keeps its flag, and the page lists it under **Needs a look**. Here's what the pass did to both maps, from [`capture.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/runs/capture.py), which reads the saved records of these runs:

<!-- RECEIPT: runs-checker -->
```text
checker
--------------------------------------------------------------------------
Map claims by Checker verdict. Run 2's state before repair and after each
round is derived from each claim's repairs history (a repair at round r
means it was not supported after round r-1) and its current check.
  repo        stage                  supported not supp. unchecked retired
  nanochat    run 1 (no repair)            129        20         0       0
  nanochat    run 2 before repair          158       102         0       0
  nanochat    after repair round 1         253         7         0       0
  nanochat    after repair round 2         260         0         0       0
  waku-agent  run 1 (no repair)            126        12         0       0
  waku-agent  run 2 before repair          219        47         0       0
  waku-agent  after repair round 1         265         1         0       0
  waku-agent  after repair round 2         266         0         0       0

  What repair did, per round (one row per repair action taken):
  repo        round  anchors replaced  cite more lines  rewritten  retired
  nanochat        1                85               85         17        0
  nanochat        2                 5                5          2        0
  waku-agent      1                45               45          2        0
  waku-agent      2                 1                1          0        0
  anchors replaced = the claim's text kept, its anchors replaced; cite
  more lines = of those, the new anchors cover more source lines than the
  old; rewritten = the claim's text changed to say only what the cited
  lines show.
```

Run 1, the first build, had no repair pass, so its flags stayed. In run 2, round 1 cleared nearly all the flags, most of them by citing more lines, and round 2 cleared the rest without retiring anything. Both maps ended with every claim supported.

That turned out to be too good. While I was recording the demo in [§4](#two-live-questions), I asked how each of nanochat's layers gets its attention window, meaning how many of the earlier tokens a layer can look back over. nanochat gives each layer one of two windows: L (long) covers the whole context, and S (short) a quarter of it, rounded up to a multiple of 128 ([`gpt.py`, line 302](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L302)). The Tracer's answer worked the S window out as 512 tokens for a 2048-token context, and pointed out that the comment on that line says 768. I selected those words and asked whether anything else still says 768. The Investigator found that the map did, in a claim the Checker had passed:

<!-- RECEIPT: runs-example-d -->
```text
  (d) c_20b0bb0d778f, the claim the Checker passed
      repo      nanochat
      text
        An S layer attends to a quarter of the context rounded up to a
        multiple of 128 (768 tokens at 2048), and the final layer always
        gets full context.
      evidence  read_in_source
      check     supported (no note)
      repairs   0
      anchor    nanochat/gpt.py:302-302
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3
        tile size (2048 -> 768)
      anchor    nanochat/gpt.py:313-313
        window_sizes[-1] = (long_window, 0)
```

The claim's first anchor is the line that computes the window. The code on it gives 512, and the out-of-date comment at the end of it says 768. The claim repeated the comment, and the Checker, reading the same line, agreed. The anchor couldn't catch this, because the lines exist. Neither could a Checker that only asks whether the lines say what the claim says, since the comment said exactly that.

So the Checker's rule changed to **compute, don't quote**. A claim that states a value (a number, a default, a size, a count, a shape) is supported only if the cited code produces that value, and the Checker works the value out from the code itself. A comment or docstring that states the value doesn't count, except for a claim labelled docs, which only reports what the docs say. Here's the same claim after a check under the new rule and one round of repair, with the Checker's note:

<!-- RECEIPT: runs-example-e -->
```text
  (e) c_20b0bb0d778f after the re-check
      repo      nanochat
      text
        An S layer attends to a quarter of the context rounded up to a
        multiple of 128 (512 tokens for a 2048-token context, although the
        inline comment says 768), and the final layer always gets full
        context.
      check     supported
      note      (empty)
      anchor    nanochat/gpt.py:301-313
        long_window = config.sequence_len
      re-check repairs  1
      round 1 rewrite, the Checker's note:
        nanochat/gpt.py line 302 computes -(-2048 // 4 // 128) * 128 =
        -(-512 // 128) * 128 = -(-4) * 128 = 512, so an S layer gets 512
        tokens at 2048, not 768; the 768 is the stale inline comment on
        that line.
```

Then I re-checked every claim on both maps that states a value, under the new rule, and ran the repair pass on whatever it turned down:

<!-- RECEIPT: runs-recheck -->
```text
recheck
--------------------------------------------------------------------------
After the repair pass, a batch run re-checked every map claim that states
a value under the Checker's new "Compute, don't quote" rule, then ran the
repair pass (at most 2 rounds). Wall clock from the launcher's start/end
files; turns and cost from claude -p's JSON result (cost at API list
price). Claims re-checked: a checked_at later than every one in the map
saved after repair, or a changed check, note, anchors or text. States and
repairs come from the repairs history entries this pass added, numbered by
this pass's rounds (the order of their timestamps); a repair at round r
means not supported after round r-1.
  repo         wall min  turns  cost (list)          exit
  nanochat         19.7     25        $9.19             0
  waku-agent       10.2     21        $4.20             0

  repo         map claims  re-checked  newly not supp.  still not supp.
  nanochat            260         117               39                4
  waku-agent          266          36               13                1
  newly not supported = re-checked claims the re-check found not supported
  (before repair); still not supported = of those, not supported after
  repair round 2.
```

The new rule turned down 39 claims on nanochat and 13 on waku-agent that had passed before. One said a download is "retried up to 5 times". The loop makes 5 attempts, so a download is retried at most 4 times:

<!-- RECEIPT: runs-example-f -->
```text
  (f) c_b9a4c59e2e08, before and after the re-check
      repo      nanochat
      before (after repair):
        text
          Each shard download is retried up to 5 times with exponential
          backoff, written to a .tmp file and renamed when complete.
        check     supported
        note      (empty)
        anchor    nanochat/dataset.py:84-133
          def download_single_file(index):
      after (after re-check):
        text
          Each shard download makes up to 5 attempts (one try plus at most
          4 retries) with a 2**attempt-second backoff, writing to a .tmp
          file that is renamed when complete.
        check     supported
        note      (empty)
        anchor    nanochat/dataset.py:84-133
          def download_single_file(index):
      re-check repairs  1
      round 1 rewrite, the Checker's note:
        max_attempts = 5 with range(1, max_attempts + 1)
        (nanochat/dataset.py lines 99-100) means 5 attempts in total, i.e.
        1 try plus up to 4 retries, not 'retried up to 5 times'; backoff
        2**attempt and .tmp-then-rename are correct.
```

The repair pass fixed most of them. The 4 claims on nanochat and the 1 on waku-agent that it couldn't fix are still flagged under Needs a look.

None of this makes the Checker proof. It's still Claude checking Claude, and the 768 claim got past it once. So I checked a sample of its verdicts myself.

<!-- TODO(draft): Han's hand check hasn't happened yet. Receipt: my agreement with the Checker on a hand-checked sample
of the final maps' verdicts (after the re-check). Anchors resolved / total isn't printed by any capture yet; claims by
evidence label are in the maps block (§4). -->

**[RECEIPT: my agreement with the Checker on a hand-checked sample]**

#### The runtime trace {#the-runtime-trace}

Questions about flow, like "what runs, in order, when the model computes its loss?", are where reading code goes wrong most easily. The reason is **dynamic dispatch**: which function a call reaches is decided when the program runs, not written down at the call. `self.attn(x)` runs whatever object `attn` happens to hold, and from the text alone you often can't tell which class that is. An agent reading the code has to infer it.

So Honeyguide can also watch the code run. Python lets a program register a function that it calls on every function call and return, through [`sys.setprofile`](https://docs.python.org/3.12/library/sys.html#sys.setprofile). It gets a `'call'` event each time a Python function is entered, and separate events (`'c_call'`, `'c_return'`, `'c_exception'`) when Python calls a function written in C. The trace script registers one, runs a small **entry point** (a few lines that build a toy model or a fake client and call into the repo), and keeps only the calls whose code lives in the repo's own package. The result is the ordered list of repo functions that actually ran, each with the file and line where it's defined.

Here's nanochat building a toy two-layer model, from [`capture.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/trace/capture.py). Indentation is call depth, and `x2` means the same call twice in a row:

<!-- RECEIPT: trace-nc-build -->
```text
## build: GPT(config) on meta -> to_empty(cpu) -> init_weights()
   kept calls 16, resumes 0, max depth 3, rows shown 11
GPT.__init__                                                    gpt.py:157
  GPT._compute_window_sizes                                     gpt.py:287
  print0                                                     common.py:118
  Block.__init__ x2                                             gpt.py:145
    CausalSelfAttention.__init__                                 gpt.py:68
      has_ve                                                     gpt.py:53
    MLP.__init__                                                gpt.py:132
  has_ve x2                                                      gpt.py:53
  GPT._precompute_rotary_embeddings                             gpt.py:270
GPT.init_weights                                                gpt.py:203
  GPT._precompute_rotary_embeddings                             gpt.py:270
```

A few things show up even here. `_precompute_rotary_embeddings` runs twice, once while the model is built and again in `init_weights`, which is easy to miss when reading. And `print0`, a logging helper, sits in the middle of the path, so the trace needs filtering before it can serve as an answer key ([§5](#does-the-map-pay-off)).

The trace has blind spots, and it prints them. A call into compiled code shows up as one `'c_call'` event, and as far as I can tell nothing that code does inside it shows up at all, so work done in compiled code is invisible. (Python's docs don't spell that last part out; it's my reading, and the backward step below fits it.) Here are the phases of the nanochat run, counted:

<!-- RECEIPT: trace-nc-phases -->
```text
Phases (run 1)
--------------------------------------------------------------------------
  phase              raw calls   kept  resumes  synthetic
  build                   6223     16        0          7
  optimizer_setup          742      3        0          5
  forward_loss             166     40        0          0
  backward                  99      0        0          0
  optimizer_step          1584     34        0          8
  engine_generate          934    239        4         14
  total                   9748    332        4         34
  raw calls = every Python call event in the phase, any file;
  kept = repo calls on the path; resumes = generator re-entries
  (kept in the path, marked, not counted as calls); synthetic =
  repo lambdas, comprehensions and class bodies dropped.
```

`backward`, the step that computes gradients for training, made 99 Python calls and not one of them was in nanochat's own code, because PyTorch's autograd, the engine that computes gradients, does that work in C++:

<!-- RECEIPT: trace-nc-backward -->
```text
## backward: loss.backward()
   kept calls 0, resumes 0, max depth 0, rows shown 0
   (no repo-owned Python calls)
```

A trace also only sees the path its entry point takes. A toy model on a laptop CPU skips a lot of what nanochat does on a GPU, so the capture lists what this run didn't exercise, each with the line in the code that decides it:

<!-- RECEIPT: trace-nc-skips -->
```text
What this toy run skips (read from the code at this commit)
--------------------------------------------------------------------------
- torch.compile of the fused AdamW/Muon steps: disabled for tracing
  (TORCH_COMPILE_DISABLE=1); same Python calls, eager kernels
    nanochat/optim.py:23 (1st of 2)
- Flash Attention 3 kernels (CUDA sm80/89/90 only); CPU takes the SDPA
  fallback
    nanochat/flash_attention.py:25
- bf16 compute dtype (CUDA SM80+); CPU/MPS default to float32
    nanochat/common.py:25
- sliding-window mask: the short window (128) is longer than T, so every
  layer takes the plain causal branch
    nanochat/flash_attention.py:108
- multi-GPU reduce/gather in the optimizer (world_size == 1 here)
    nanochat/optim.py:284
- torch.compile of the whole model (outside the package filter)
    scripts/base_train.py:246
- FP8 training (H100+, --fp8)
    scripts/base_train.py:165
- fp16 GradScaler path
    scripts/base_train.py:515
- real data loader and grad accumulation; batch here is random ids
    scripts/base_train.py:510
- checkpoint loading (load_model); weights here are fresh init
    nanochat/checkpoint_manager.py:163
- calculator tool use: an untrained model never emits <|python_start|>
    nanochat/engine.py:259
```

A function whose results are cached doesn't run again on a repeat call, so a trace undercounts those calls too. Where the trace can see, it's exact, and it reproduces. Run twice in fresh processes, and a third time with a different hash seed (the number Python uses to randomize its dictionaries), the path of repo calls came out identical; only the wall time is measured rather than counted:

<!-- RECEIPT: trace-nc-repro -->
```text
Reproducibility (fresh process per run)
--------------------------------------------------------------------------
  kept path, run 1 vs run 2:
    identical (336 events)
  kept path, run 1 vs control, PYTHONHASHSEED=1:
    identical (336 events)
  raw calls, runs then control: 9748 / 9748 / 9877
  wall time, runs then control: 1.1 s / 1.1 s / 1.1 s
```

Reading and running catch different things. The eval's "what calls X?" questions (§5) take their answers from both: a parse of the source into its **syntax tree** (the AST, the structure Python builds from the text before running it) gives every call it can resolve, and the trace adds calls the AST can't. For nanochat's ten callers questions:

<!-- RECEIPT: q-nc-callers -->
```text
Callers keys: where each key caller came from
--------------------------------------------------------------------------
  key callers 23: ast only 15, trace only 2, both 6
```

and for waku-agent's ten:

<!-- RECEIPT: q-wk-callers -->
```text
Callers keys: where each key caller came from
--------------------------------------------------------------------------
  key callers 20: ast only 11, trace only 5, both 4
```

Most callers came from reading alone, because the toy runs only exercise part of each repo. But some were seen only when the code ran, and those are exactly the calls an agent reading the code has to infer. In the product, the Tracer labels each hop of a path as read in source, inferred, or seen at runtime, and the page draws them differently.

<!-- FIGURE F3b (mechanism with data; both paths from the capture): where reading the code and running it disagree,
for one nanochat flow. A swimlane diagram, one lane per file or class, time downward; solid arrows seen at runtime,
dashed inferred, dotted where the trace is blind (backward). -->

#### Staying current as the code changes {#staying-current}

A saved map goes stale the moment someone pushes a commit. Honeyguide stores each claim with the commit it was checked at and the hash of its cited lines, so when you come back to a repo it can work out what happened to every claim, using `git diff` from the saved commit to now:

- **unchanged**: the file didn't change, or the cited lines are byte for byte the same in the same place.
- **moved**: the same lines, found in exactly one new place; the line numbers are updated.
- **stale**: the cited lines changed. The claim isn't served as fact, or given to an agent, until the Checker re-reads it, and the Cartographer rewrites it if it no longer holds.
- **retired**: the code is gone. The claim stays on the page, struck through, with the reason.

New code that no claim covers is listed, and the Cartographer maps only that. If the same lines turn up in two places, the claim is marked stale rather than guessed at. On the page, the header says which commit the map is current at, and stale claims lose their evidence mark until they've been re-checked.

<!-- TODO(draft): the refresh measurement, from outputs/refresh_nanochat.txt: a nanochat map built at the last commit
before its July rewrite, refreshed to the pin, against a full rebuild at the pin. Claims unchanged / moved / stale /
retired, new code mapped, refresh vs rebuild tokens and minutes, and how many stale claims the Checker found wrong. -->

<!-- FIGURE F3c (measurement, from the refresh capture): updating a saved map costs a fraction of rebuilding it, and
nothing stale is served as fact. What happened to each saved claim, and refresh vs rebuild cost. -->

**[RECEIPT: refresh across nanochat's rewrite, claims by state, refresh vs rebuild cost]**

---

### 4. What it's like to use {#what-its-like-to-use}

Here's the page on nanochat's map, in the order a reader meets it. The screenshots are of the real page serving the run 2 map described [below](#the-first-maps), and they follow this post's light or dark theme.

The page opens on a card about the repo itself: its link, the description line from its README, its languages, licence, contributors and history, and the commit the map was made at. The line at its foot says whether the map is still current at that commit. A script reads all of this from git and the repo's files, so none of it is a claim and none of it carries an evidence mark. On the left, the outline lists each Overview topic, each module by its short name and each flow, and marks the one you're reading as you scroll. **Find** (press `/`) filters the outline and searches the map's text.

![The top of the Honeyguide page for nanochat. A card names the repo with a link to github.com/karpathy/nanochat, quotes the README's description, shows a language bar for Python and Shell, and lists the licence, contributors, GitHub status and history, with the mapped-at commit in its corner and a line at its foot saying the map is up to date at that commit. On the left, the outline lists the Overview's topics and the modules by name, with a Find box above them](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-repo-card-light.png){: .light width="1000" height="625" }
![The top of the Honeyguide page for nanochat. A card names the repo with a link to github.com/karpathy/nanochat, quotes the README's description, shows a language bar for Python and Shell, and lists the licence, contributors, GitHub status and history, with the mapped-at commit in its corner and a line at its foot saying the map is up to date at that commit. On the left, the outline lists the Overview's topics and the modules by name, with a Find box above them](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-repo-card-dark.png){: .dark width="1000" height="625" }
_The top of nanochat's map: what the repo is, and whether the map is still current, before any claim about its code._

Each module is a card. A module here is a **component**, one or more files that work together; the GPT model is one file, `nanochat/gpt.py`. The card says what the component does, each claim followed by its evidence shape and a chip naming the lines it cites, and then a **Design** paragraph on why it's built that way. A dotted amber underline marks a passage someone has asked about, and the numbered bubble beside it opens that thread.

![The GPT model's card in the Modules section. Under the title "nanochat/gpt.py: the Transformer and its cost model", a paragraph of claims about the model's configuration, each followed by an evidence shape and grey chips citing lines such as nanochat/gpt.py:29-39. Below it, an indented Design paragraph explains why the model is built on the meta device first, and has a sentence beginning "The window pattern is the trap" underlined with amber dots, with a numbered bubble in the margin. The outline on the left marks GPT model as the current module](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-module-card-light.png){: .light width="1000" height="625" }
![The GPT model's card in the Modules section. Under the title "nanochat/gpt.py: the Transformer and its cost model", a paragraph of claims about the model's configuration, each followed by an evidence shape and grey chips citing lines such as nanochat/gpt.py:29-39. Below it, an indented Design paragraph explains why the model is built on the meta device first, and has a sentence beginning "The window pattern is the trap" underlined with amber dots, with a numbered bubble in the margin. The outline on the left marks GPT model as the current module](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-module-card-dark.png){: .dark width="1000" height="625" }
_The GPT model's card. Its Design paragraph is where the map explains the attention windows from [§3](#the-checker), and the underlined sentence is the one the first live question was asked from._

The Overview also has a **component map**: the components and the calls between them, in three bands. **Entry points** are components nothing in the traced flows calls, **stages** are called and call others, and **leaves** call nothing further. A heavier line means more calls. Click a component and the map goes into **focus mode**: the component, its callers and the components it calls stay, and the rest fades. A card underneath lists its calls in and out, each with the lines it's made on and the flow it was traced in, and **Ask about this component** opens the discussion panel with a question about it already written, which you can edit before you send it.

![The component map in focus mode on engine.py, the inference engine and chat command-line interface. Its box has a brown outline; brown curves run to it from tasks/ and from above, and from it to gpt.py, tokenizer.py and common.py, while other components such as dataloader.py, optim.py and flash_attention.py are faded. The Stages and Leaves bands are labelled on the left. Below the map, a card titled engine.py has a brown "Ask about this component" button, an "Open its module" link, and a "Called by" list with each call's code and the flow it appears in](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-component-focus-light.png){: .light width="1000" height="764" }
![The component map in focus mode on engine.py, the inference engine and chat command-line interface. Its box has a brown outline; brown curves run to it from tasks/ and from above, and from it to gpt.py, tokenizer.py and common.py, while other components such as dataloader.py, optim.py and flash_attention.py are faded. The Stages and Leaves bands are labelled on the left. Below the map, a card titled engine.py has a brown "Ask about this component" button, an "Open its module" link, and a "Called by" list with each call's code and the flow it appears in](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-component-focus-dark.png){: .dark width="1000" height="764" }
_Focus mode on `engine.py`: only its callers and the components it calls stay lit, and its card offers to ask about it._

Selecting words anywhere on the map opens the same panel. Here I selected the Design paragraph's sentence about the window pattern and asked where the quarter is computed, the first of the [two live questions](#two-live-questions) below. The panel quotes the passage the question came from, and the answer says which agent wrote it and how it got there: the Tracer, routed by the Guide as a flow question. The underlined words in the answer are where I asked the follow-up that found the stale 768.

![The discussion panel open on the right, beside the GPT model's card, whose window-pattern sentence is highlighted in amber. The panel's Thread tab quotes that sentence under "Modules, nanochat/gpt.py", then shows the question "Where is the quarter computed, and how does each layer's window size reach the attention kernel?" and the Tracer's answer, marked "routed by Guide as a flow question". A phrase in the answer about the 768 in the inline comment is underlined in amber. An Ask box sits at the bottom of the panel](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-answer-panel-light.png){: .light width="1000" height="625" }
![The discussion panel open on the right, beside the GPT model's card, whose window-pattern sentence is highlighted in amber. The panel's Thread tab quotes that sentence under "Modules, nanochat/gpt.py", then shows the question "Where is the quarter computed, and how does each layer's window size reach the attention kernel?" and the Tracer's answer, marked "routed by Guide as a flow question". A phrase in the answer about the 768 in the inline comment is underlined in amber. An Ask box sits at the bottom of the panel](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-answer-panel-dark.png){: .dark width="1000" height="625" }
_A question asked from a sentence on the map and answered beside it, so the map never moves while you read the answer._

Each flow on the map has a **swimlane diagram**: one lane per file or class, the calls numbered in order from top to bottom, each call labelled with its code and the lines it's made on, and each return with the data it hands back. A shaded box marks steps that repeat. Every diagram on the page can be zoomed, dragged and expanded to full screen.

![A swimlane diagram in full screen, titled "What happens when you type a message into scripts/chat_cli.py". Lanes across the top are chat_cli.py loop, checkpoint_manager.py load_model, tokenizer.py RustBPETokenizer and engine.py Engine, with more cut off at the right. Numbered rows down the left run from 1 to 8; arrows go from chat_cli.py to the other lanes, each labelled with a call such as load_model(args.source, device, phase="eval") and its line, and thinner arrows return labelled with what comes back. A shaded box labelled "Loop, per generated token" covers rows 5 to 8. A toolbar at the top right has zoom out, zoom in, Fit and Close, and a key at the bottom explains calls, returns, data and repeats](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-flow-fullscreen-light.png){: .light width="1000" height="625" }
![A swimlane diagram in full screen, titled "What happens when you type a message into scripts/chat_cli.py". Lanes across the top are chat_cli.py loop, checkpoint_manager.py load_model, tokenizer.py RustBPETokenizer and engine.py Engine, with more cut off at the right. Numbered rows down the left run from 1 to 8; arrows go from chat_cli.py to the other lanes, each labelled with a call such as load_model(args.source, device, phase="eval") and its line, and thinner arrows return labelled with what comes back. A shaded box labelled "Loop, per generated token" covers rows 5 to 8. A toolbar at the top right has zoom out, zoom in, Fit and Close, and a key at the bottom explains calls, returns, data and repeats](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-flow-fullscreen-dark.png){: .dark width="1000" height="625" }
_Typing a message into nanochat's chat script, in full screen: the reply is built in a loop that runs once per generated token, and every call names the line it's made on._

And here's the loop in motion, from the demo I recorded on this page: click `gpt.py` on the component map, focus, **Ask about this component**, send. In the terminal in the corner, the session picks the question up and the Guide hands it to the Investigator. The answer lands in the panel, and then the Checker reads its claims and the repair pass from [§3](#the-checker) runs on the ones it turned down, so each claim ends with its Checker mark. Waiting on a model is dull to watch, so those stretches are sped up, and each carries a yellow badge saying by how much.

![A screen recording of the Honeyguide page on nanochat's map, light theme only. The pointer clicks gpt.py on the component map and the map goes into focus mode; Ask about this component opens the panel with the question "What does gpt.py do, and how do the components around it use it?" and it is sent. A terminal in the lower left shows the Claude Code session claiming the question and handing it to the Investigator, while a yellow badge marks the stretch as sped up. The answer appears in the panel with grey chips citing lines of gpt.py, the Checker's marks arrive beside its claims, and a caption at the top reports how many claims pass](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/honeyguide-demo.gif){: width="960" height="600" .shadow }
_One question, from a click on the component map to a checked answer. The recording is in the light theme only._

<!-- TODO(draft): the Honeyguide repo goes public at publish; check the video link resolves then. -->

The [full video](https://github.com/bearbearyu1223/honeyguide/blob/main/demo/honeyguide-demo.mp4) also shows the outline and Find, a code preview, and a flow diagram in full screen, and the [demo's README](https://github.com/bearbearyu1223/honeyguide/blob/main/demo/README.md) lists every stretch that was sped up and by how much.

<!-- TODO(draft): the GIF's live question (q_1a900486f311, Investigator) has its own time, cost and Checker counts, in
demo/README.md only. If the post states them, add that run to a capture (e.g. the runs-demo block in outputs/runs.txt)
and quote them from there; don't type them from the README. -->

#### The first maps {#the-first-maps}

The Cartographer's first real run on the two repos (run 1) saved a map whose claims all pointed at real code, but its Overview read like a list of labels. Here's the start of nanochat's Overview from run 1, and the same place in run 2's:

<!-- RECEIPT: runs-examples-ab -->
```text
  (a) nanochat, run 1: the Overview's first block
      A minimal single-node LLM harness. --depth sets model size.

  (b) nanochat, run 2 after repair: the Purpose block(s)
      nanochat trains a small ChatGPT-style model end to end on one GPU
      node: one script downloads data, trains a byte-pair encoding (BPE)
      tokenizer, pretrains, fine-tunes and evaluates. Everything is built
      around one dial. `--depth` sets the layer count and the width
      follows from it, while batch size, learning rates and weight decay
      are rescaled from a 12-layer reference.

      Precision is explicit, because one global `COMPUTE_DTYPE` replaces
      `torch.amp.autocast` and the custom `Linear` casts full-precision
      (fp32) master weights to the activation dtype. Multi-GPU training
      uses no DistributedDataParallel (DDP) wrapper, since the optimizer
      syncs gradients and shards its own state.
```

Two things made run 1 terse. The Cartographer saved the map through Honeyguide's command line, and the whole map had to fit in a single shell command, so it squeezed the prose to fit. And its instructions gave the Overview no structure to fill.

So I rewrote the Cartographer's instructions, modelled on the structure of RepoRabbit, my earlier project ([§1](#why-a-map)). The Overview now has seven topics, from what the repo is for to which files to read first. A module is now a component, one or more files that work together, with a paragraph on why it's built that way and a few concrete facts. The map is written as draft files that a script checks and saves, so its length is no longer capped. And a **readability check** refuses telegraph text, the clipped style of a telegram: paragraphs too short to explain anything, sentences with no verb, fragments strung together with semicolons, and abbreviations the Overview never spells out. Run 2 built both maps again from scratch with the new instructions. For nanochat:

<!-- RECEIPT: runs-maps-nc -->
```text
maps
--------------------------------------------------------------------------
Run 1 (old Cartographer brief) against run 2 (Cartographer v2, start
over), as saved after the repair pass and after the re-check (the final
state). Checker verdicts are each claim's saved check.
  nanochat, run 1
    sections            3 (overview, modules, flows)
    Overview topics     0 of 7
    module blocks       25 (0 with more than one file)
    flows               2 (8 hops)
    claims              149
    Checker verdicts    supported 129, not supported 20
    evidence labels     read_in_source 136, docs 9, asserted_by_test 2,
                        inferred 2
    Overview words      36
  nanochat, run 2 after repair
    sections            3 (overview, modules, flows)
    Overview topics     7 of 7 (purpose, users, capabilities, commands,
                        entry-points, components, start-here)
    module blocks       13 (8 with more than one file)
    flows               5 (31 hops)
    claims              260
    Checker verdicts    supported 260, not supported 0
    evidence labels     read_in_source 195, docs 36, inferred 11,
                        searched_not_found 10, asserted_by_test 8
    Overview words      323
```

Run 2's map has all seven topics, five flows instead of two, and an Overview long enough to explain something. It has fewer module blocks because related files are grouped into one component. The Checker's verdicts here are after the repair pass in [§3](#the-checker). Here's the readability check on both repos' Overviews:

<!-- RECEIPT: runs-readability -->
```text
readability
--------------------------------------------------------------------------
honeyguide.readability.check_readability(strict=True), the check behind
`map validate --strict`, on the Overview section alone.
  repo        map                   errors  warnings  warnings by kind
  nanochat    run 1                      4        11  abbreviation 2,
                                                    choppy 3, length 1,
                                                    short 2, verbless 3
  nanochat    run 2 after repair         0         0  none
  waku-agent  run 1                      2         5  choppy 1, length 1,
                                                    semicolons 1, short 1,
                                                    verbless 1
  waku-agent  run 2 after repair         0         0  none
```

A warning is one problem in one block; a block with several kinds of problem, or an Overview where many blocks have one, is an error. Both run-1 Overviews fail, and both run-2 Overviews pass with nothing flagged.

#### Two questions, asked live {#two-live-questions}

The demo video shows the loop from [§2](#whats-in-the-build) on the nanochat map, with two real questions. For the recording, each was answered by a non-interactive session (`claude -p`) started as the question was sent, standing in for the session you'd normally have open. The first question was about the attention windows in [§3](#the-checker), and the Guide routed it to the Tracer, because it asks how something gets from one place to another. The second was a follow-up, asked by selecting words in the first answer, and it went to the Investigator:

<!-- RECEIPT: runs-demo -->
```text
demo
--------------------------------------------------------------------------
The two live questions, each answered by one claude -p session (stream-
json transcript; turns, duration_ms and cost from its final result event).
  q1
    question id       q_d7c4fd23c771
    repo              nanochat
    question          Where is the quarter computed, and how does each
                      layer's window size reach the attention kernel?
    route (agent)     tracer
    subagents run     checker 1, tracer 1
    turns             12
    wall clock        180 s (claude -p)
    posted->answered  103 s (answers.json timestamps)
    cost (list)       $1.28
    answer claims     13
    Checker           supported 4, not supported 9
  q2
    question id       q_9b365ac5f974
    repo              nanochat
    question          Does anything else still say 768, or a third of the
                      context: docs, tests, help text, or this map?
    route (agent)     investigator
    subagents run     checker 1, investigator 1
    turns             11
    wall clock        129 s (claude -p)
    posted->answered  75 s (answers.json timestamps)
    cost (list)       $1.08
    answer claims     9
    Checker           supported 8, not supported 1
```

`posted->answered` is the time from asking to the answer appearing on the page. The session ran on after that, because the Checker starts once the answer is up. It passed only 4 of the Tracer's 13 claims, and the other 9 show on the answer with the Checker's notes. At the time, answers had no repair pass; they now get the same one as the map. The second answer is the one that found the map's 768 claim.

---

### 5. Does the map pay off? {#does-the-map-pay-off}

A map costs something to build, so the question is whether answers from it are better, or cheaper, than answers from an agent that explores the repo each time. And an agent that answers well about a famous repo might be remembering it from training rather than reading it.

#### Four conditions

The same model answers the same questions in four conditions. Every run gets the same instructions, the same answer format, and the same tools; the only difference is what extra file it's given.

| Condition | Working folder | Tools | Extra context | Passes per question |
| --- | --- | --- | --- | --- |
| no access (control) | empty | none | none | 1 |
| explore | a fresh copy of the repo | Read, Grep, Glob | none | 2 |
| outline | a fresh copy | Read, Grep, Glob | a script-built outline (files, classes, functions, imports), cut to the same size in tokens as the map | 2 |
| map | a fresh copy | Read, Grep, Glob | one of three maps, built independently | 3 (one per map) |

<!-- TODO(draft): confirm passes per question and the model id against the pilot and the full run's settings. -->

<!-- FIGURE F5 (mechanism; the condition table above): the four conditions differ only in the extra file they get. -->

The **no-access** run is the control: if the model answers a question with no code at all, it knew the answer already. The **outline** is the free baseline. It costs nothing to build, so if the map can't beat it, the thing paying off is an index, and the learning isn't. **Read, Grep and Glob** are Claude Code's tools for reading a file, searching file contents, and finding files by name.

The maps for the test are built by the Cartographer's instructions with the same three tools, with no shell and no runtime trace. So the test measures the instructions and the map they produce. It doesn't measure the page, the Guide, or the Tracer agent; flow questions are answered with the same instructions as every other question.

**Isolation.** Each run must see only what its condition gives it. Every run starts in a fresh configuration folder with no settings files, no MCP servers (add-on tool servers Claude Code can connect to), no saved sessions and no auto-memory, works on a fresh copy of the pinned commit with no `.git` folder, and has the repos' own agent files (nanochat's `.claude/`, waku-agent's `AGENTS.md`, `CLAUDE.md` and `.claude/`) removed. Maps, outlines, questions and answer keys live outside every working folder. Claude Code's `--bare` mode would have done much of this in one flag, but it "doesn't use your subscription login": it needs an API key, set in `ANTHROPIC_API_KEY` or supplied by an `apiKeyHelper` script ([Claude Code docs, non-interactive mode](https://code.claude.com/docs/en/headless)). These runs use a subscription, so they set each piece separately (`--setting-sources ""`, `--strict-mcp-config`, `--no-session-persistence`, and auto-memory switched off). They also pass `--exclude-dynamic-system-prompt-sections`, which doesn't remove the per-machine parts of Claude Code's system prompt (the working folder, environment details, memory paths and git status); it moves them into the first user message ([CLI reference](https://code.claude.com/docs/en/cli-reference)). So the runs can still see those details, and a pilot checked what they contain: each condition recited its own context, and a script scanned that for anything personal. The tools each condition gets are named in `--tools`, because the default set on macOS and Linux leaves out Glob and Grep.

<!-- TODO(draft): a receipt from the isolation pilot (leak scans clean per run, no user, memory or git in the recited
context, including the sections moved into the first user message) once its capture is printed without
machine-specific paths; also the CLI version and model id. -->

#### The questions and their answer keys

The questions were generated by a script and frozen, with their answer keys, in a commit made before any map existed. A manifest of their hashes proves they haven't changed since. There are four types:

| Type | Asked as | Answer key from | Scored by |
| --- | --- | --- | --- |
| locate | a behaviour, never a function name ("where is ... computed?") | the AST | right file and function |
| callers | "what calls X directly?" | the AST plus the runtime trace; I settle the disputed cases | overlap with the key's set |
| flow | an entry point and a config: "which repo functions run, in order?" | the runtime trace, filtered to repo code; I mark which hops are required | required hops found, their order, and hops that appear in neither the trace nor the AST |
| why/how | written by me, about ten per repo, with a rubric | my rubric | graded by me blind, with the condition hidden and the order shuffled |

Two examples from nanochat, from [`generate.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/questions/generate.py). The tag in brackets is explained below:

<!-- RECEIPT: q-nc-examples -->
```text
  nc-loc-01 [after] Where does the code look up a GPU's peak memory
    bandwidth from its device name, the ceiling that decode speed is
    compared against?
    key: nanochat/common.py get_peak_bandwidth lines 281-328
```

<!-- RECEIPT: q-nc-flow -->
```text
  nc-flo-02 [before] Same toy GPTConfig on CPU (n_layer=2, n_embd=32,
    float32, torch.compile off): loss = model(x, y) with x and y random
    token ids of shape (2, 16). Which repo functions run, in order, from
    GPT.forward until the loss is returned?
    candidates: GPT.forward > norm > Linear.forward > Block.forward >
      CausalSelfAttention.forward > apply_rotary_emb > flash_attn_func >
      _sdpa_attention > MLP.forward
```

A question can give its answer away by naming it. The generator checks for that and prints the count. For nanochat:

<!-- RECEIPT: q-nc-leak -->
```text
Leakage (key names or file names verbatim in the text)
--------------------------------------------------------------------------
  locate: key function, class or file named        0 of 12
  locate: all words of a key name present (loose)   0 of 12
  callers: target named (by design)                10 of 10
  callers: a key caller or its file named            0 of 10
  flow: a hop's function named (not the entry's)    0 of 5
```

and for waku-agent, where one flow question names one of its own hops:

<!-- RECEIPT: q-wk-leak -->
```text
Leakage (key names or file names verbatim in the text)
--------------------------------------------------------------------------
  locate: key function, class or file named        0 of 12
  locate: all words of a key name present (loose)   0 of 12
  callers: target named (by design)                10 of 10
  callers: a key caller or its file named            0 of 10
  flow: a hop's function named (not the entry's)    1 of 8
    wk-flo-04: save_note
```

Callers questions name their target by design; that's the question.

#### Learning, or remembering?

A model's **training cutoff** is the date after which it saw no training data. Anthropic gives it only to the month: the two models I'm choosing between for these runs, Claude Opus 5.5 and Claude Sonnet 5.5, both list "Reliable knowledge cutoff: Jun 2026", and the same month for their training data ([Anthropic, models overview](https://platform.claude.com/docs/en/about-claude/models/overview)). Code written before the cutoff may be in the model's training data, so a good answer about it could be **memorization**, the model recalling text it was trained on, rather than reading. Code written after the cutoff can only be answered by reading it.

There's evidence that this happens with code. Given only the text of a bug report, "without access to repository structure", models from the Claude 4 generation and earlier named the file with the bug up to 76% of the time on SWE-Bench Verified, a widely used benchmark built from bug reports in public repos, against at most 53% on repos outside it ([SWE-Bench Illusion](https://arxiv.org/abs/2506.12286)). The pattern held on reports that don't mention a file path (§4.2). On competition programming problems, [LiveCodeBench](https://arxiv.org/abs/2403.07974) scored models on "problems released after the cutoff date" and found a "stark drop" for one DeepSeek coding model on problems from after August 2023, while Claude 3 showed no such drop (§5.1). Neither study tested the models used here, or questions about how a repo's code fits together, so this test checks for it directly.

So each question is tagged by when its answer's code was written, using `git blame` (which records the commit that last changed each line): a question counts as *after* when at least half of its key's lines changed after the cutoff. The generator takes the cutoff as a parameter, and these captures use 2026-06-30. The docs give the cutoff only as a month, June 2026, so code changed during June could fall on either side of it. None of the code these questions are about was changed in June 2026.

<!-- TODO(draft): the generator doesn't yet exclude June-dated code; once it does, a receipt line prints how many key
lines were last changed in June 2026 (expected 0), and this sentence cites it. -->

nanochat splits:

<!-- RECEIPT: q-nc-types -->
```text
Questions by type and subset (cutoff 2026-06-30)
--------------------------------------------------------------------------
  type        total  before  after  none
  locate         12       6      6     0
  callers        10       7      3     0
  flow            5       5      0     0
  all            27      18      9     0
```

waku-agent was created on 2026-07-10, after the cutoff (its [first commit](https://github.com/ShenSeanChen/waku-agent/commit/427d2ad) is from that day), and has grown in public since, so every question lands after it:

<!-- RECEIPT: q-wk-types -->
```text
Questions by type and subset (cutoff 2026-06-30)
--------------------------------------------------------------------------
  type        total  before  after  none
  locate         12       0     12     0
  callers        10       0     10     0
  flow            8       0      8     0
  all            30       0     30     0
```

The two repos play opposite parts. If the no-access control does well on nanochat's before-cutoff questions, the model already knew that code, and any gain from the map should be judged on the questions it couldn't have seen.

#### How the results are read

Results are paired by question: each question is answered in every condition, and the comparison is the difference on the same question. The headline differences are map minus explore, map minus outline, and explore minus no access, pooled over each repo's questions, with 95% intervals from resampling the questions (a **bootstrap**). Results per question type are descriptive only, since no type has more than a dozen questions in a repo. A run that hits its limit or fails counts as wrong, and those runs are reported, never quietly re-run.

Cost is counted in tokens of four kinds: **input**, **output**, **cache writes** and **cache reads**. An agent sends the same start of the prompt (its instructions, the files it has already read) with every request, so that start, the **prefix**, can be cached. Saving a prefix is a cache write, priced at 1.25x the base input price if it's kept for 5 minutes or 2x if it's kept for an hour, and Claude Code on a subscription keeps it for an hour. A cache read is input read from a prefix that an earlier request cached, priced at 0.05x the base input price on Claude Opus 5.5 and Sonnet 5.5, not the 0.1x most models use ([Anthropic, prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), Pricing; [cost tracking](https://code.claude.com/docs/en/agent-sdk/cost-tracking)). Claude Code also reports a dollar figure per run, `costUSD`, but that's its own estimate at API list price (its `costBasis` is `"list"`); on a subscription, it isn't what was paid ([Claude Code docs, costs](https://code.claude.com/docs/en/costs)). The map's own build cost is included, spread over the questions asked of it. The **break-even point** is the number of questions after which the map's answers cost less per correct answer than exploring, if there is one.

<!-- TODO(draft): results, from outputs/eval_nanochat.txt and outputs/eval_waku-agent.txt: correct / total per condition
and repo; the paired differences with CIs; per-type results; capped and errored runs; before / after-cutoff accuracy
for no access and map; tokens by kind, tool calls and minutes per run; map build cost per seed; cost per correct answer
and the break-even count, or that it never pays. Plus the why/how grades and my agreement where relevant. -->

<!-- FIGURE F6 (measurement, from the eval capture): paired differences with 95% CIs, one panel per repo. -->

<!-- FIGURE F7 (measurement, from the eval capture): learning vs remembering. No-access and map accuracy, before vs
after cutoff, per repo. -->

<!-- FIGURE F8 (measurement, from the eval capture): cost per correct answer as questions accumulate, map build
included; break-even marked, or "never pays". -->

**[RESULTS: the eval, per repo and condition]**

---

### 6. What I learned {#what-i-learned}

<!-- TODO(draft): written from the results. What paid off and what didn't; the honest limits (toy entry points, so
the trace sees part of each repo; two repos; one model; maps built by the same model that answers; the Checker is
Claude checking Claude, with my sample as the only outside check); and whether the headline is "the map pays",
"an index pays", or "it doesn't pay". -->

**[WHAT I LEARNED: from the results]**

---

### What's next {#whats-next}

This post starts a new series, Agentic Workflows. The next one goes behind the scenes: the team of agents that helped me write this post.

A map that explains a repo is a start. After that, the series adds a 💡 **Scout**, an agent that proposes changes worth making to the repo, each with its evidence, and tests it against history: given a repo as it was at some earlier commit, does it propose what the maintainers went on to build, more often than a control does?

### Appendix: glossary {#appendix-glossary}

The terms this post leans on; anything not listed is defined where it first appears.

| Term | Means | In this post |
| --- | --- | --- |
| coding agent | a language model that acts on code through tools: search, read, run, edit | Claude Code |
| context window | the text a model can see at once | each agent has its own |
| token | a chunk of text of a few characters; models read and bill in tokens | §1, §5 |
| map | Honeyguide's document about a repo: overview, modules, flows, every claim pointing at code | written by the Cartographer |
| subagent | a separate Claude instance the main session starts for one task, with its own context, instructions and tools | Cartographer, Investigator, Tracer, Checker |
| skill | a `SKILL.md` file of instructions Claude Code loads when relevant, or that you call with `/skill-name` | §2 |
| plugin | a bundle of skills, subagents and other add-ons, as one package | §2 |
| non-interactive mode | running Claude Code from a script with `claude -p`, no interactive window (formerly called headless mode) | the eval, §5 |
| Guide | the main Claude Code session, which routes questions from the page | §2 |
| inbox | where the page leaves questions for the session to pick up | §2 |
| anchor | a claim's pointer into the code: path and quoted lines, resolved by a script to a line range, commit and hash | §3 |
| hash | a short fingerprint of some text that changes if the text changes | one per anchor |
| permalink | a link to a file at one fixed commit, so it always shows the same lines | the GitHub link in the preview |
| evidence label | what kind of evidence a claim rests on | seven labels, §3 |
| Checker verdict | supported or not supported, after re-reading the cited lines | §3 |
| dynamic dispatch | which function a call reaches is decided when the program runs | why flows are hard to read |
| runtime trace | the ordered list of repo functions that ran for one entry point | §3 |
| entry point | a few lines that call into the repo with a toy input, for the trace | one per flow |
| AST | abstract syntax tree: the structure Python builds from source text before running it | callers keys, §3 and §5 |
| hash seed | the number Python uses to randomize its dictionaries | changed once to check the trace |
| stale, moved, retired | what a refresh finds: the cited lines changed, moved, or are gone | §3 |
| repo id | the repo's remote address plus its first commit; where its memory is saved | §2 |
| training cutoff | the date after which the model saw no training data; given only as Jun 2026 for the models here | §5 |
| memorization | a model recalling text it was trained on rather than reading it | §5 |
| `git blame` | git's record of which commit last changed each line | the before/after tags |
| no access, explore, outline, map | the four conditions of the test | §5 |
| bootstrap | estimating an interval by resampling the questions many times | §5 |
| prefix | the start of a prompt that repeats from request to request, so it can be cached | §5 |
| cache write | saving a prefix for later requests; 1.25x the base input price for 5 minutes, 2x for an hour (Claude Code on a subscription uses the hour) | §5 |
| cache read | input read from a prefix an earlier request cached; 0.05x the base input price on Claude Opus 5.5 and Sonnet 5.5 (0.1x on most models) | §5 |
| `costUSD` | Claude Code's own cost estimate at API list price, not what a subscription pays | §5 |
| break-even | the number of questions after which the map costs less per correct answer than exploring | §5 |
| uv environment | a project's own Python and packages, managed by uv | each target repo has one |

### References

Sources checked on 2026-11-15; Claude Code's docs are for CLI 2.1.325.

- Anthropic, Claude Code docs, [Subagents](https://code.claude.com/docs/en/sub-agents) — "Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions" (introduction), and it starts from a "fresh, isolated context window" (What loads at startup); the subagent does its work in its own context "and returns only the summary" (introduction). Cited in §2.
- Anthropic, Claude Code docs, [Glossary](https://code.claude.com/docs/en/glossary) — Skill: "A SKILL.md file …", where Claude Code "loads a skill automatically when relevant, or you invoke it directly with /skill-name"; Plugin: a "bundle of skills, hooks, subagents, and MCP servers"; "Non-interactive mode … Formerly called headless mode". Cited in §2.
- Anthropic, Claude Code docs, [Memory](https://code.claude.com/docs/en/memory) — `CLAUDE.md` loads from the working folder "and every directory above it", and from subfolders on demand; auto memory loads its first 200 lines or 25KB and "skips anything it can derive from the codebase"; `/init` "creates a file with build commands, test instructions, and project conventions it discovers". Cited in the introduction and §1.
- Anthropic, Claude Code docs, [CLI reference](https://code.claude.com/docs/en/cli-reference) — the rows for `--output-format` (text, json, stream-json), `--setting-sources`, `--strict-mcp-config`, `--no-session-persistence`, `--exclude-dynamic-system-prompt-sections` (moves the per-machine sections into the first user message) and `--tools` (the default set on macOS, Linux and WSL leaves out Glob and Grep). Cited in §2 and §5.
- Anthropic, Claude Code docs, [Non-interactive mode](https://code.claude.com/docs/en/headless), the `--bare` section — it "never reads OAuth credentials or the system keychain", takes an API key from `ANTHROPIC_API_KEY` or an `apiKeyHelper`, and "doesn't use your subscription login". Cited in §5.
- Anthropic, Claude Code docs, [Costs](https://code.claude.com/docs/en/costs) — Claude Code "computes the dollar figure locally from token counts at list price", which for subscribers "isn't relevant for billing purposes"; and [Agent SDK, cost tracking](https://code.claude.com/docs/en/agent-sdk/cost-tracking) — subscription turns use the 1-hour cache, and the reported costs are "client-side estimates, not authoritative billing data". Cited in §5.
- Anthropic, [Models overview](https://platform.claude.com/docs/en/about-claude/models/overview) — Claude Opus 5.5 and Claude Sonnet 5.5: "Reliable knowledge cutoff | Jun 2026" and "Training data cutoff | Jun 2026", with no day given. Cited in §5.
- Anthropic, [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), Pricing — cache writes cost 1.25x the base input price for 5 minutes and 2x for 1 hour; "Cache hits and refreshes on Claude Opus 5.5 and Claude Sonnet 5.5 are priced at 0.05x the base input price", against 0.1x for other models. Cited in §5.
- aider, [Repository map](https://aider.chat/docs/repomap.html) — "a graph ranking algorithm, computed on a graph where each source file is a node and edges connect files which have dependencies", with the map's size set by `--map-tokens` (1k by default); and aider's [blog post of 2023-10-22](https://aider.chat/2023/10/22/repomap.html): "aider uses tree sitter to build the map". Cited in §1.
- Cognition, [DeepWiki](https://docs.devin.ai/work-with-devin/deepwiki) (Devin docs) — it "produces wikis with architecture diagrams, links to sources, and summaries". The docs don't say whether links point at line ranges or whether a wiki updates itself. Cited in §1.
- Fergus Hurley, Pedro Rodriguez, Rafael Marques and Omar Shams, ["Introducing Code Wiki: Accelerating your code understanding"](https://developers.googleblog.com/en/introducing-code-wiki-accelerating-your-code-understanding/), Google Developers Blog, 2025-11-13 — "Code Wiki scans the full codebase and regenerates the documentation after each change"; "Every wiki section and chat answer is hyper-linked directly to the relevant code files and definitions"; launched in public preview, for public repositories. Cited in §1.
- Cursor, [Search](https://cursor.com/docs/agent/tools/search) docs — Cursor "does not store embeddings of your codebase for search" and uses a local index, Instant Grep. Cursor's forum thread ["What do you think about Cursor removing the codebase indexing settings?"](https://forum.cursor.com/t/what-do-you-think-about-cursor-removing-the-codebase-indexing-settings/165899): Colin (Community Support Engineer), [2026-07-22](https://forum.cursor.com/t/what-do-you-think-about-cursor-removing-the-codebase-indexing-settings/165899/10), "Semantic/embeddings indexing is being turned down in favor of grep-based retrieval"; Maxime (Cursor staff), [2026-07-21](https://forum.cursor.com/t/what-do-you-think-about-cursor-removing-the-codebase-indexing-settings/165899/5), "the older dedicated semantic search path was rapidly no longer helping in a meaningful way", and agents "now lean on" Instant Grep "(and normal file reads)". [Cursor's semantic-search docs, as archived on 2026-02-25](https://web.archive.org/web/20260225213818/https://cursor.com/docs/context/semantic-search): chunks turned into embeddings, stored in a "specialized vector database", with "periodic checks every 5 minutes". Cited in §1.
- Heule, Jia and Jain, ["Improving agent with semantic search"](https://cursor.com/blog/semsearch), Cursor blog, 2025-11-06 — semantic search gave "on average 12.5% higher accuracy in answering questions (6.5%–23.5% depending on the model)" than grep alone, on Cursor's own question set. Cited in §1.
- Liang, Garg and Zilouchian Moghaddam, [SWE-Bench Illusion](https://arxiv.org/abs/2506.12286), arXiv preprint 2506.12286v4 (2025-12-01) — models given only the issue text, "without access to repository structure", identify the buggy file's path up to 76% of the time on SWE-Bench Verified against up to 53% on 245 tasks from 7 popular repos outside SWE-Bench; the pattern persists on issues that name no path (§4.2). The models tested are from the Claude 4 generation and earlier. Cited in §5.
- Jain, Han, Gu, Li, Yan, Zhang, Wang, Solar-Lezama, Sen and Stoica, ["LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code"](https://arxiv.org/abs/2403.07974), arXiv 2403.07974v2, ICLR 2025 (cited from v2; the ICLR version differs) — competition problems; §5.1 evaluates "problems released after the cutoff date" and reports a "stark drop" for DS-Ins-33B after Aug. 2023, while Claude 3 shows no drastic variation. Cited in §5.
- Python docs, [`sys.setprofile`](https://docs.python.org/3.12/library/sys.html#sys.setprofile) (3.10 to 3.12) — the events `'call'`, `'return'`, `'c_call'`, `'c_return'` and `'c_exception'`; the profile function is "thread-specific", so it "does not make sense to use this in the presence of multiple threads" ([`threading.setprofile`](https://docs.python.org/3.12/library/threading.html#threading.setprofile) covers threads started from `threading`). That compiled code's own work is invisible is my inference, not the docs'. Cited in §3.
- Spottiswoode and Wood, "Culturally determined interspecies communication between humans and honeyguides", *Science* 382(6675):1155–1158, 2023-12-08, [doi:10.1126/science.adh4129](https://doi.org/10.1126/science.adh4129) ([open copy](https://par.nsf.gov/servlets/purl/10524235)) — p. 1: honeyguides "lead people in Africa to wild bees' nests … Honeyguides naturally locate bees' nests and eat wax. Humans can subdue the bees and open their nests, thus exposing beeswax for the honeyguides and honey for themselves". Cited in the introduction.
- [nanochat](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd) at `92d63d4` — [LICENSE](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/LICENSE): MIT, "Copyright (c) 2025 Andrej Karpathy"; [README](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/README.md?plain=1#L8), line 8, the DeepWiki pointer quoted in §1. Cited in Setup, §1, §3 and §5.
- [waku-agent](https://github.com/ShenSeanChen/waku-agent/tree/2c7e7e68d3d65ac19d428bdde7820844d87dfa19) at `2c7e7e6` (2026-11-15, 05:09 UTC) — [LICENSE](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/LICENSE): MIT, "Copyright (c) 2026 Sean Chen (ShenSeanChen)"; [`hosted/LICENSE`](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/hosted/LICENSE): Elastic License 2.0; [LICENSE-BRAND](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/LICENSE-BRAND): brand assets and names aren't MIT (copyright AutoManus Technologies, Inc.); [README](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/README.md?plain=1#L129-L133), lines 129 to 133: "Code is MIT … except `hosted/`, which is Elastic License 2.0". Repo created 2026-07-10; [first commit](https://github.com/ShenSeanChen/waku-agent/commit/427d2ad) `427d2ad` the same day. Cited in Setup and §5.
- [RepoRabbit](https://github.com/bearbearyu1223/repo-rabbit), my earlier Claude Code subagent for exploring a repo. README at `b9e9eae`: [line 115](https://github.com/bearbearyu1223/repo-rabbit/blob/b9e9eae/README.md?plain=1#L115), a reference that "doesn't resolve to a real file and line is rejected"; [line 177](https://github.com/bearbearyu1223/repo-rabbit/blob/b9e9eae/README.md?plain=1#L177), "Each reference stores a hash of the lines it cites … If they changed, that explanation is flagged for the agent to review". Cited in §1 and §2.
