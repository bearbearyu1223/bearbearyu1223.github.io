---
title: "Agentic Workflows [1]: Honeyguide: who maintains the understanding when agents write the code?"
date: 2026-11-16 09:00:00 -0800
categories: [Agentic Workflows, Code Understanding]
tags: [agentic-workflow, claude-code, subagents, coding-agents, code-understanding, evaluation]
description: >-
  Coding agents make code easier to write and change, but what an agent works
  out about a repository usually stays in one private conversation. Honeyguide
  keeps that understanding as a shared, checkable map: every claim cites its
  source lines, a separate Checker reads those lines against the claim, and a
  refresh keeps the map current as the code changes. On the repo I tested, the
  map didn't measurably help an agent answer more accurately. This post shows
  how the checking and the upkeep work and what they cost. The shared team
  workflow isn't built yet, and whether it saves engineering time is open.
published: false
---

## Understanding that stays in one chat

Coding agents are making it easier to write and change code. But understanding that code, checking how it works, and keeping that understanding current remain hard. Each time an engineer asks an agent about an unfamiliar repository, the agent explores the code and produces an answer. That answer may be correct, but it usually stays in a private conversation, where others can't easily review, reuse, or maintain it.

I built **Honeyguide** to explore a different approach: treating codebase understanding as a shared, checkable artifact. It builds an interactive map of a repository, ties each claim to the source lines it comes from, has a separate Checker read those lines against the claim, and refreshes the map as the code changes.

I first expected the map to help coding agents answer questions more accurately. On the repository I tested, it [didn't measurably improve accuracy](#does-an-agent-need-the-map) over exploring the code directly. That removed my first reason for building a map. What's left is a different case for it: keeping checked, source-grounded knowledge instead of reconstructing it every time someone asks.

This post looks at how Honeyguide builds, checks, and updates that knowledge, with its limits and costs. The shared team workflow isn't built yet, and whether it saves engineering time is still an open question.

The name comes from a bird. In Africa, the greater honeyguide leads people to wild bees' nests: the people subdue the bees and open the nest for the honey, and the bird eats the wax ([Spottiswoode and Wood, 2023](https://par.nsf.gov/servlets/purl/10524235)). That's the job I wanted done for a codebase: something that has already been through the repo and can lead me to the lines that answer my question.

### The short version {#the-short-version}

**The hypothesis: does an agent need the map?** ([§4](#does-an-agent-need-the-map)) In a 20-question pilot on Andrej Karpathy's nanochat, a small codebase that trains a ChatGPT-style language model end to end, the same model answered with no access to the code, by exploring it, from a script-built outline, or with the map. Every condition that could read the repo, map or no map, scored at or near the ceiling on lookups (where is X, what calls X, what runs in order). On code written after the model's training cutoff, the model with no access to the code scored 0.200 and every condition with the repo 1.000. On five why/how questions, graded blind by another model, the map scored highest: a hint at most. Map answers cost about twice as much per run as exploring, so in the pilot's agent-cost model the map never pays back.

**Are ad-hoc answers worse?** ([§5](#are-ad-hoc-answers-worse)) Not measurably. A Checker that finds its own evidence judged 2.0% of the claims in the pilot's ad-hoc answers not supported or wrong, against 1.0% of a map's unchecked claims, and the interval on the difference includes zero. The errors that do occur are subtle.

**What that leaves** ([§1](#the-team-problem)). Of three problems a map could address, finding code is already done well by an agent searching it, repeated reconstruction across a team is argued but unmeasured, and trustworthy, maintained knowledge is where this post's evidence is. Other tools already write docs about a repo or keep a team's memory; Honeyguide's difference is narrow, a check of each claim against the lines it cites.

**What Honeyguide is** ([§2](#what-honeyguide-is)). Five roles in Claude Code: a 🗺️ **Cartographer** writes the map, an 🔦 **Investigator** and a 🧵 **Tracer** answer questions on it, a ✅ **Checker** re-reads each claim against its cited lines, and the 🧭 **Guide**, the session you're in, routes the questions.

**How it keeps itself honest** ([§3](#how-it-keeps-itself-honest)). A script, never the agent, finds the lines each claim cites. Before any repair, the Checker turned down 102 of the 260 claims on nanochat's final map, and repair fixed most by citing more lines while leaving the words alone: what it mostly catches is citations too narrow to show the claim. It also passed one wrong value (768 where the code gives 512), and its rule changed to "compute, don't quote".

**What a shared map costs to keep** ([§6](#what-a-shared-map-costs-to-keep)). Refreshing nanochat's map across a 60-commit rewrite cost about half of a full checked build of the older, larger code; a refresh after an ordinary day on waku-agent cost about a fifth to a quarter of a rebuild. Refresh caught every stale claim that no longer held. Its gap is claims whose cited lines didn't change: mapping the new code later retired 3 of the 216 such claims, and 6 of the 12 that cite no code.

**Limits** ([§7](#limits-and-large-codebases)). One person, one model, two repos: nanochat is small, and waku-agent is larger but most of it lies outside what the questions cover. Whether a team reuses the map, adopts it or saves time with it is untested, and nothing here shows how this works on a large codebase.

### Setup

The code is in [Honeyguide](https://github.com/bearbearyu1223/honeyguide), a separate repo from the LLM Architecture Refresh series, but this post follows that series' rule: every number below is printed by a program, and each block of program output, a **receipt**, is pasted byte for byte from a capture in the repo's `outputs/` folder. Next to the receipts is the **recipe**: the exact command that prints them.

To try Honeyguide on a repo of your own, you'll need Claude Code and [uv](https://docs.astral.sh/uv/), the Python package manager the series uses. Install the `honeyguide` command and the plugin, then ask the Guide to explore from inside the repo you want mapped:

```bash
git clone https://github.com/bearbearyu1223/honeyguide ~/honeyguide
uv tool install --editable ~/honeyguide
```

```
/plugin marketplace add bearbearyu1223/honeyguide
/plugin install honeyguide@honeyguide
/honeyguide:honeyguide explore
```

The [README](https://github.com/bearbearyu1223/honeyguide/blob/main/README.md) has the permission rules that keep Claude Code from asking before every step.

**The two repos.** Honeyguide mapped two repos, each pinned to one commit: Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd) at `92d63d4`, mostly written before June 2026, the training cutoff of the Claude model used here, and [waku-agent](https://github.com/ShenSeanChen/waku-agent/tree/2c7e7e68d3d65ac19d428bdde7820844d87dfa19) at `2c7e7e6`, a personal assistant agent created after it. nanochat is [MIT-licensed](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/LICENSE); waku-agent's code is [MIT](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/LICENSE) except its `hosted/` directory ([Elastic License 2.0](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/hosted/LICENSE)) and its brand assets ([LICENSE-BRAND](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/LICENSE-BRAND)), none of which this post quotes. Every link into their code is pinned to that commit, and code quoted from them keeps their licence notice.

**What ran where.** Not every experiment ran on both repos:

| Experiment | Repo | Section |
| --- | --- | --- |
| the maps, the Checker and the repair pass | both | §3 |
| the four-way test, run as a 20-question pilot | nanochat only | §4 |
| E1: checking the pilot's ad-hoc answers | nanochat only | §5 |
| E2: refreshing a map across a 60-commit rewrite | nanochat | §6 |
| E3: refreshing a map after an ordinary day | waku-agent | §6 |

The four-way test was planned for both repos. I ran it as a 20-question pilot on nanochat, and once the story became about a shared map rather than a smarter agent, I kept the pilot as the test instead of scaling it up. §4 reports it as what it is: one repo, 20 questions, a small sample.

**Recipes.** Every receipt comes from one of five commands, run in the Honeyguide repo:

```bash
uv run python eval/runs/capture.py > outputs/runs.txt
uv run python eval/cutoff/june_check.py > outputs/june_check.txt
uv run python eval/score/evalscore.py --run-id pilot-nanochat-2 \
    > outputs/eval_pilot_nanochat.txt
uv run python eval/adhoc/adhoc.py --run-id pilot-nanochat-2 --capture \
    > outputs/adhoc_check.txt
uv run python eval/refresh/capture.py > outputs/refresh.txt
```

All but the June check read the saved records of the runs, which aren't in the repo, so the files in `outputs/` are what you get. What you can check without the records is the code the receipts quote: every claim's pointer into the code (its **anchor**, [§3](#anchors)) names a file and its lines at the pinned commit, so you can open line 302 of nanochat's `gpt.py` and work out the attention window yourself.

The June check (`june_check.py`, which dates the code each question's answer rests on against the model's June 2026 training cutoff, [§4](#learning-or-remembering)) needs only the two repos and git. The scripts look for them in a fixed folder, `~/Documents/GitHub/honeyguide-targets/`, and read the pinned commits with `git show` and `git blame`, never the working tree, so a full clone of each is enough, whatever branch it has checked out:

```bash
mkdir -p ~/Documents/GitHub/honeyguide-targets
cd ~/Documents/GitHub/honeyguide-targets
git clone https://github.com/karpathy/nanochat
git clone https://github.com/ShenSeanChen/waku-agent
```

A shallow clone won't do: `git blame` needs the history, and the script stops if a pinned commit isn't there ([`repo.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/questions/repo.py)).

**Two kinds of numbers.** Most numbers here are *counted*: claims, verdicts, questions. Some are *measured*, like minutes, tokens (the chunks of text a model reads and is billed in, [§1](#the-team-problem)) and dollars, and they'd move if the runs were repeated. Every dollar figure is Claude Code's own estimate at API list price; these runs used a subscription, so it isn't what was paid ([§4](#what-an-answer-costs)).

### Table of Contents

[The short version](#the-short-version) above has the findings without the details, in the order I ran into them. The sections below start from the problem and the tool, then the tests.

1. [The team problem](#the-team-problem)
2. [What Honeyguide is](#what-honeyguide-is)
3. [How it keeps itself honest](#how-it-keeps-itself-honest)
4. [Does an agent need the map?](#does-an-agent-need-the-map)
5. [Are ad-hoc answers worse?](#are-ad-hoc-answers-worse)
6. [What a shared map costs to keep](#what-a-shared-map-costs-to-keep)
7. [Limits, and what a large codebase needs](#limits-and-large-codebases)

Then [what I learned](#what-i-learned), [what's next](#whats-next) and [the glossary](#appendix-glossary) at the end.

---

### 1. The team problem {#the-team-problem}

A **coding agent** is a language model that can act on your code through tools: search for text, read a file, run a command, edit. Claude Code is the one I use. Everything the model knows about your repo during a task is in its **context window**, the text it can see at once, measured in **tokens** (chunks of text of a few characters each; usage and cost are counted in tokens).

Ask an agent about an unfamiliar repo and most of the work goes into finding things. It searches, reads, follows a call, reads again. For one person on a small repo that's cheap: in the pilot in [§4](#does-an-agent-need-the-map), an agent searching nanochat with plain text search answered a question in 0.2 minutes for $0.074 on average, and scored 1.000 on the questions that ask where something happens.

Where it falls short is across a team. Picture five engineers who join a project in the same month, each asking their own agent how training is set up or what happens when a checkpoint loads. That's five explorations of the same code and five private answers. If one is subtly wrong, the person who got it carries the mistake into their next change, and the next person with the same question pays to explore again. A shared map is meant to be the one place those answers are checked and kept.

![Two panels. Before: five engineers each send their own agent to search the code again, and each gets a private answer, checked by no one, seen by one person, gone when the chat ends. The aim: one shared map, built once from the code and refreshed as the code changes, where every claim cites its lines and is checked; dashed arrows, marked as the untested aim, lead from it to all five engineers, and questions asked on it, with their checked answers, are added to it](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/team-before-after-light.png){: .light width="1000" height="482" }
![Two panels. Before: five engineers each send their own agent to search the code again, and each gets a private answer, checked by no one, seen by one person, gone when the chat ends. The aim: one shared map, built once from the code and refreshed as the code changes, where every claim cites its lines and is checked; dashed arrows, marked as the untested aim, lead from it to all five engineers, and questions asked on it, with their checked answers, are added to it](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/team-before-after-dark.png){: .dark width="1000" height="482" }
_Five private answers that nobody checks and nobody keeps, against one map that's checked and kept current. The checking and the refresh are what this post measures; a team reading one map is the aim, still untested._

This matters more as agents write more of the code. By GitHub's own count, which it calls a floor, more than 40% of pull requests on GitHub in September 2026 started with an agent's first draft, up from 17% in March ([Octoverse 2026](https://github.blog/news-insights/octoverse/octoverse-commits-pushes-reach-new-highs-in-2026-as-developers-build-with-agents/)). A model trained to spot AI-written code, run over GitHub commits, estimates that AI writes 29% of Python functions in the US ([Daniotti et al., *Science*, 2026](https://arxiv.org/abs/2506.08945)), and open-source projects that adopted Cursor added about 28.6% more lines, a transient rise, with no significant change in their number of commits ([He et al., MSR '26](https://arxiv.org/abs/2511.04427), §4.1.1). Whether teams ship faster is less clear: in a randomized trial, 16 developers working in mature projects took 19% longer with early-2025 AI tools ([METR](https://arxiv.org/abs/2507.09089)), and Google's DORA survey finds AI adoption improving delivery throughput (how quickly and how often changes ship) but still increasing delivery instability (DORA, *2025 State of AI-assisted Software Development*, [p. 4](https://services.google.com/fh/files/misc/2025_state_of_ai_assisted_software_development.pdf)). None of this needs AI-written code to be worse: more change is enough to make shared understanding harder to keep current.

Written knowledge about code already falls behind it. In 1,500 Java projects, only 13% to 20% of code changes came with an update to the code's comments; the authors note this doesn't mean the other changes left comments wrong, only that they could have ([Wen et al., ICPC 2019](https://doi.org/10.1109/ICPC.2019.00019), §IV-A). Among the most popular projects on GitHub, 28.9% have documentation that names code which has since been deleted, 82.3% have had it at some point, and such references "were typically outdated for years before they were noticed" ([Tan, Wagner and Treude, 2024](https://arxiv.org/abs/2212.01479), §1). That counts only the narrowest kind of drift, a name whose code is gone. And in the documentation problems people report, being out of date makes up 39% of the issues about content (Aghajani et al., ICSE 2019, [§IV-A](https://doi.org/10.1109/ICSE.2019.00122)). nanochat has an instance: a comment in its `gpt.py` gives 768 for a value the code on the same line computes as 512, and in [§3](#the-claim-it-passed) it fooled the Checker.

A map could address three problems, and the evidence for each is very uneven:

| Problem | What the evidence says | Where |
| --- | --- | --- |
| A. Code discovery: finding the code that answers a question | little to gain: an agent searching with plain text already does it | [§4](#does-an-agent-need-the-map) |
| B. Repeated reconstruction: each engineer's agent working out the same code again | argued, not measured: exploring is cheap for one person ($0.074 a question in the pilot) but redone by everyone who asks; whether engineers' questions overlap, and whether they'd read each other's answers, is untested | this section, [§4](#does-an-agent-need-the-map) |
| C. Trustworthy, maintained knowledge: claims checked against the source and kept current | measured: the Checker turned down 102 of 260 claims, most for citing too few lines, and passed one wrong value; ad-hoc answers weren't measurably less accurate (2.0% of claims not supported or wrong, against 1.0% for a map); a refresh costs about half a rebuild across a rewrite and a fifth to a quarter after an ordinary day | [§3](#how-it-keeps-itself-honest), [§5](#are-ad-hoc-answers-worse), [§6](#what-a-shared-map-costs-to-keep) |

This post shows C and argues B, and the argument has assumptions. It assumes the five engineers need the same knowledge, when the overlap may be partial, and that reading a map can stand in for reading the code, when working the code out yourself builds an understanding a summary doesn't. A shared map's worth depends on **reuse** (do people read what others asked), **trust** (can they believe it), **change** (does it stay current) and **adoption** (does a team use it at all). This post measures trust ([§3](#how-it-keeps-itself-honest), [§5](#are-ad-hoc-answers-worse)) and change ([§6](#what-a-shared-map-costs-to-keep)). Reuse and adoption are the big unknowns: I'm one person ([§7](#limits-and-large-codebases)).

#### What already exists

Several tools already keep knowledge about a codebase outside the code. Here's what each one's own docs say it does; "not stated" means I found nothing either way, and the quotes behind each cell are in the [References](#references).

| Tool | Writes docs or a map | Search index | Updates when code changes | Shared with a team | Checks each claim against its lines |
| --- | --- | --- | --- | --- | --- |
| Claude Code `CLAUDE.md` | notes for the agent; `/init` drafts them | no, loaded as is | audit on request (`/doctor prompt-audit`) | if committed | no; the audit checks that files and commands exist |
| Cursor | not stated | by meaning until mid-2026, now a local text index | changed files only (archived docs) | not stated; today's index is local | not stated |
| DeepWiki (Cognition) | wikis with diagrams, summaries, links to sources | indexes repos automatically | auto-update not stated; regenerate on request | by link ([nanochat's README](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/README.md?plain=1#L8) gives one) | not stated |
| Google Code Wiki | wiki and chat, linked to files and definitions | not stated | regenerated after each change | public repos (preview, 2025) | not stated |
| aider repo map | a map of each file's key symbols | not stated; ranks files on a dependency graph | not stated | not stated | not stated |
| RepoAgent (research) | docs for every Python class and function | not stated | affected objects, from a pre-commit hook | not stated | not stated |
| Sourcegraph (Deep Search) | answers, with the searches run and files read | text and symbols | indexes kept current; answers not stated | answers, by link on an instance | not stated |
| Augment (Context Engine, Cosmos memory) | team memory as Markdown | code, commit history, docs, tickets | index on push (local copy: real time); memory flags conflicts when loaded | memory, with an organization | flags conflicts with current evidence; line citations not stated |
| Honeyguide | a map whose every claim cites lines | no | refresh, claim by claim ([§6](#what-a-shared-map-costs-to-keep)) | not yet ([§2](#what-honeyguide-is)) | yes, the Checker ([§3](#the-checker)) |

Augment comes closest. Its Cosmos memory can be shared across an organization, is usually kept per repository, and "flags discrepancies when current evidence conflicts with a remembered rule"; its docs don't say a remembered rule cites source lines or is re-checked when the code changes. Sourcegraph's Deep Search answers can be shared, and each lists the searches it ran and the files it read, but nothing in its docs says an answer is checked or kept current. My own earlier [RepoRabbit](https://github.com/bearbearyu1223/repo-rabbit) rejects any reference that doesn't resolve to a real file and line ([README, line 115](https://github.com/bearbearyu1223/repo-rabbit/blob/b9e9eae/README.md?plain=1#L115)) and flags an explanation when its cited lines change ([line 177](https://github.com/bearbearyu1223/repo-rabbit/blob/b9e9eae/README.md?plain=1#L177)), but lines can exist, and stay unchanged, and still not say what the claim says. None of the docs I read describes checking a kept claim against the lines it cites. That's Honeyguide's difference, and it's narrow: a separate check of each claim against its lines, and a refresh that works claim by claim. On sharing, it's behind several of these tools.

Cursor is also the one tool I found that measured whether keeping knowledge helps. In November 2025 it reported that search by meaning gave its agent "on average 12.5% higher accuracy in answering questions" than grep alone (exact-text search, as the classic `grep` tool does), on its own question set ([Cursor blog, "Improving agent with semantic search"](https://cursor.com/blog/semsearch)). In July 2026 its staff said that path "was rapidly no longer helping in a meaningful way" ([Cursor forum, 2026-07-21](https://forum.cursor.com/t/what-do-you-think-about-cursor-removing-the-codebase-indexing-settings/165899/5)) and that the indexing was "being turned down in favor of grep-based retrieval" ([Cursor forum, 2026-07-22](https://forum.cursor.com/t/what-do-you-think-about-cursor-removing-the-codebase-indexing-settings/165899/10)). My pilot points the same way on a small repo: for lookups, grep was enough ([§4](#does-an-agent-need-the-map)). That's problem A, and it's why this post's case for a map rests on B and C.

---

### 2. What Honeyguide is {#what-honeyguide-is}

Honeyguide is one Claude Code **plugin**, a bundle of skills, subagents and other add-ons in one package ([Claude Code docs, glossary](https://code.claude.com/docs/en/glossary)). Its **skill**, a `SKILL.md` file of instructions that Claude Code loads when it's relevant or that you call by name, turns the session you're in into the Guide. The other four roles are **subagents**: separate Claude instances the main conversation starts for one task. "Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions", and it does the task in its own context "and returns only the summary" ([Claude Code docs, subagents](https://code.claude.com/docs/en/sub-agents)).

The rule I held to: a job gets its own agent only when a fresh context or parallel work earns it. Anything exact is a script: resolving citations to lines, the runtime trace, validating the map, serving the page, and the eval.

Honeyguide's own code was written with an agent. All 69 commits on its main branch, up to [`f1b52e9`](https://github.com/bearbearyu1223/honeyguide/commit/f1b52e9), carry a Claude co-author trailer, a line that marks a commit made with the agent's help; it doesn't say what share of the lines the agent wrote.

🗺️ **Cartographer.** Explores the repo and writes the first map: an Overview of seven topics, from what the repo is for to which files to read first; modules, each a **component** (one or more files that work together) with a Design paragraph on why it's built that way; and the main flows through the code. In a subagent, reading a whole repo doesn't crowd the Guide's context. It also repairs claims the Checker turns down ([§3](#how-it-keeps-itself-honest)) and maps new code when the repo moves ([§6](#what-a-shared-map-costs-to-keep)), so the map is kept current without a rebuild.

🔦 **Investigator.** Answers one question or follow-up asked on the map, starting from the map and any earlier answers about the same code, with every claim pointing at code. Each question gets a clean context, and several can run at once.

🧵 **Tracer.** Answers "what happens when..." questions as a path: where it starts (its **entry point**), each call along it in order (a **hop**), and where it ends. Which function a call reaches is often decided only when the program runs (`self.attn(x)` runs whatever object `attn` holds), so the Tracer can confirm hops by running the code: a script hooks Python's [`sys.setprofile`](https://docs.python.org/3.12/library/sys.html#sys.setprofile), calls into the repo with a toy input, and records which repo functions ran, in order (a **runtime trace**). Each hop says whether it was read, inferred, or seen at runtime.

✅ **Checker.** Re-reads each claim next to the lines it cites and says supported or not supported. It reports and never edits, and it works in a context that didn't write the claim, so it's less inclined to agree with it.

🧭 **Guide.** The Claude Code session you're already in. It picks up questions from the page, hands each to the Investigator or the Tracer, and writes the answer back, under your permissions, where you can watch it.

What these agents produce is a set of claim records. Each holds what the map believes about the code (the claim), why (the lines it cites, its evidence label from [§3](#evidence-labels) and the Checker's verdict), and when to reconsider it (whether it's current, stale because its cited lines changed, or retired because its code is gone, [§6](#how-refresh-works)). That's the record a team would review. Generated documentation mostly keeps the first part; the other two are what let a reader check a claim and know when to look at it again.

#### What a teammate sees

The map is a web page. You read it like a document, with an outline on the left. Each module card says what its component does, each claim followed by a small shape for its kind of evidence and a chip naming the lines it cites; click the chip and you see the code.

![The GPT model's module card on nanochat's map: claims about the model, each followed by an evidence shape and grey chips citing lines of nanochat/gpt.py, one paragraph underlined in amber dots with a numbered bubble beside it, and a Design paragraph below that gives one attention window as 512 tokens of 2048, not the 768 its code comment claims](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-module-card-light.png){: .light width="1000" height="625" }
![The GPT model's module card on nanochat's map: claims about the model, each followed by an evidence shape and grey chips citing lines of nanochat/gpt.py, one paragraph underlined in amber dots with a numbered bubble beside it, and a Design paragraph below that gives one attention window as 512 tokens of 2048, not the 768 its code comment claims](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-module-card-dark.png){: .dark width="1000" height="625" }
_nanochat's GPT model, as a teammate reads it: every claim cites its lines, and the dotted underline marks a passage someone has already asked about. The 512 in the Design paragraph is a value the map once got wrong ([§3](#the-checker))._

Select any passage, even inside an answer, and choose **Ask**. The page writes the question and your selection to an **inbox** on disk; the Guide picks it up and hands it to an agent, and the answer appears beside the passage. The page never starts Claude itself: the session runs a watcher, `honeyguide watch --follow`, under Claude Code's Monitor tool, and each new question in the inbox reaches it as a notification ([`SKILL.md`](https://github.com/bearbearyu1223/honeyguide/blob/main/skills/honeyguide/SKILL.md), section 3, Watch).

![The discussion panel beside the GPT model's card. It quotes an earlier sentence of the map, marked "not found in the current map", then the question "Where is the quarter computed, and how does each layer's window size reach the attention kernel?" and the Tracer's answer, in which the phrase "At 2048 that is 512, not the 768 the inline comment says" is underlined in amber](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-answer-panel-light.png){: .light width="1000" height="625" }
![The discussion panel beside the GPT model's card. It quotes an earlier sentence of the map, marked "not found in the current map", then the question "Where is the quarter computed, and how does each layer's window size reach the attention kernel?" and the Tracer's answer, in which the phrase "At 2048 that is 512, not the 768 the inline comment says" is underlined in amber](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/tour-answer-panel-dark.png){: .dark width="1000" height="625" }
_A question asked from a sentence on the map and answered beside it, so it stays on the map for the next reader. The quoted sentence has since been rewritten, so the panel marks it as no longer found._

A question that loses its place when its claim is rewritten is a real gap for a shared map, where answers are meant to accumulate into the repo's documentation ([what's next](#whats-next)).

Here's the loop in motion, from the demo I recorded on nanochat's map: click `gpt.py` on the component map (the components and the calls between them), **Ask about this component**, send. The session in the corner picks the question up, the Guide hands it to the Investigator, and the answer lands in the panel; then the Checker reads its claims and the repair pass from [§3](#the-checker) runs on the ones it turned down. Stretches spent waiting on a model are sped up, each with a yellow badge.

![A screen recording on nanochat's map: a click on gpt.py in the component map, Ask about this component, and the question sent; a terminal in the corner shows the session handing it to the Investigator, sped-up stretches carry a yellow badge, and the answer appears with chips citing gpt.py, followed by the Checker's marks](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/honeyguide-demo.gif){: width="960" height="600" .shadow }
_One question, from a click on the component map to a checked answer. The recording is in the light theme only._

The [full video](https://github.com/bearbearyu1223/honeyguide/blob/main/demo/honeyguide-demo.mp4) also shows a code preview and a flow diagram, and the [demo's README](https://github.com/bearbearyu1223/honeyguide/blob/main/demo/README.md) lists every sped-up stretch.

**Where the map lives.** Everything Honeyguide learns about a repo is saved outside the checkout, in `~/.honeyguide/<repo-id>/`: the map, the answers with their evidence and verdicts, and your private notes. The repo id is the repo's remote address plus its first commit, so any clone finds the same map. Today Honeyguide is single-user: the map lives on your machine and the page is served only there. A team can't share it yet; sharing is first on the list in [what's next](#whats-next).

---

### 3. How it keeps itself honest {#how-it-keeps-itself-honest}

A claim on a shared map can be trusted at three levels: its citation points at real lines, those lines support it, and it's correct. Anchors make the first a script's job. Most of what the Checker first turned down was true but cited too little, so it failed the second. And one claim gave a value as 768 where the code gives 512: a comment on its cited line supported it, and it was still wrong. A map a team relies on needs all three, because the citation is how a reader checks a claim without searching. Honeyguide's guards are anchors, evidence labels, and the Checker with its repair pass.

#### Anchors: every claim points at real lines {#anchors}

An **anchor** is a claim's pointer into the code. The agent writing a claim supplies only a file path and the text of the lines it means. A script reads that file at the map's commit, finds the quoted text, and fills in the line range, the commit, the enclosing function and a **hash** of those lines (a short fingerprint that changes if any of the lines change). The agent never types a line number, because a typed line number is a guess. If the quote can't be found, or turns up twice, the script refuses the claim.

That's a real guarantee, but a narrow one. An anchor proves the cited lines exist at that commit. It doesn't prove that they say what the claim says.

#### Evidence labels {#evidence-labels}

Each claim carries a label saying what kind of evidence it rests on:

| Label | Shape on the page | Means |
| --- | --- | --- |
| read in source | filled dot | the agent read the cited lines and they say this |
| asserted by a test | filled square | a test in the repo checks this behaviour |
| docs | filled triangle | the repo's documentation says so (which can be out of date) |
| seen at runtime | small filled dot inside a ring | the runtime trace saw it happen |
| inferred | hollow circle | worked out from the code without a line that says it directly, e.g. which method a call reaches |
| searched, not found | dotted hollow circle | the agent looked for something and it isn't there |
| general knowledge (not from this repo) | hollow diamond | true in general, like how a library function behaves, and not a claim about this code |

On the page each label is that small shape beside the claim ([`page.css`](https://github.com/bearbearyu1223/honeyguide/blob/main/src/honeyguide/page/page.css)). The last label keeps general knowledge, which agents mix into answers all the time, from passing as something the repo says.

#### The Checker {#the-checker}

The Checker re-reads each claim next to the lines it cites and gives one verdict: supported or not supported. It works in the background, so an answer appears as soon as it's written, and a claim the Checker doesn't accept gets a quiet flag with the reason rather than being deleted.

I built each map twice. Run 1 used the Cartographer's first instructions and produced an Overview of terse labels; run 2 started over with rewritten instructions, modelled on RepoRabbit's structure, and produced the map described in [§2](#what-honeyguide-is). In run 2 the Checker marked 102 of nanochat's 260 claims not supported, and 47 of waku-agent's 266.

So the Checker's verdicts feed a **repair pass**. The Cartographer gets back each claim the Checker turned down, with the Checker's note, and does one of three things: cites more lines (it widens the claim's anchors), rewrites the claim to say only what the cited lines show (it narrows it), or retires it. The Checker then re-checks only the claims that changed. That's one round, and a pass has at most two. A claim still not supported after that keeps its flag, and the page lists it under **Needs a look**.

![A loop diagram of the Checker and repair pass on nanochat's run-2 map. The Checker re-reads each of 260 claims and its cited lines; supported claims get their evidence shape, and the 102 of 260 not supported go to the Cartographer, which cites more lines, narrows the claim or retires it. Round 1: 85 anchors replaced (all 85 cite more lines), 17 rewritten, 0 retired; round 2: 5 anchors replaced (all 5 cite more lines), 2 rewritten, 0 retired. The Checker re-checks only the claims that changed, at most 2 rounds per pass: 7 not supported after round 1, 0 after round 2. Supported: 158 at once, +95 after round 1, +7 after round 2, so 260 of 260 at the end; a claim still not supported after round 2 would be flagged under Needs a look, and none was on nanochat's run 2. The not-supported path, the repair box and the Flagged box are drawn in berry](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/repair-loop-light.png){: .light width="1000" height="426" }
![A loop diagram of the Checker and repair pass on nanochat's run-2 map. The Checker re-reads each of 260 claims and its cited lines; supported claims get their evidence shape, and the 102 of 260 not supported go to the Cartographer, which cites more lines, narrows the claim or retires it. Round 1: 85 anchors replaced (all 85 cite more lines), 17 rewritten, 0 retired; round 2: 5 anchors replaced (all 5 cite more lines), 2 rewritten, 0 retired. The Checker re-checks only the claims that changed, at most 2 rounds per pass: 7 not supported after round 1, 0 after round 2. Supported: 158 at once, +95 after round 1, +7 after round 2, so 260 of 260 at the end; a claim still not supported after round 2 would be flagged under Needs a look, and none was on nanochat's run 2. The not-supported path, the repair box and the Flagged box are drawn in berry](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/repair-loop-dark.png){: .dark width="1000" height="426" }
_A claim the Checker turns down goes back for repair at most twice per pass. On nanochat's run 2, 102 of 260 claims went back, and two rounds brought all of them back supported, most by citing more lines._

Here's what the pass did to both maps, from [`capture.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/runs/capture.py), which reads the saved records of these runs:

<!-- RECEIPT: runs-checker -->
```text
checker
--------------------------------------------------------------------------
Map claims by Checker verdict. Run 2's state before repair and after each
round is derived from each claim's repairs history in the repair pass (a
repair at round r means it was not supported after round r-1) and its
current check.
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

Look at the second table. On nanochat, round 1 replaced the anchors of 85 claims and left their words alone, and every one of those 85 now cites more source lines than before; only 17 claims were rewritten, and none retired. waku-agent's round 1 is even more lopsided. Most of what the Checker turned down was true and cited too narrowly.

[§5](#are-ad-hoc-answers-worse) comes at this from the other side: a Checker that had to find the evidence itself judged only 1 of 100 claims not supported, on a different nanochat map that no Checker had seen (the one built for the pilot in [§4](#does-an-agent-need-the-map)), against 102 of 260 here. Claims mostly hold when the Checker looks for the evidence itself, so a citation-reading Checker mostly judges the second level, whether the cited code shows the claim, which is the one a reader clicking a chip cares about.

#### The claim it passed

The Checker also let one wrong claim through. While recording an earlier cut of the demo, I asked on the map how each of nanochat's layers gets its attention window, meaning how many of the earlier tokens a layer can look back over. nanochat gives each layer one of two windows: L (long) covers the whole context, and S (short) a quarter of it, rounded up to a multiple of 128 ([`gpt.py`, line 302](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L302)). The default pattern, `SSSL`, repeats across the layers, and the last layer always gets L. The Tracer's answer worked the S window out as 512 tokens for a 2048-token context, and pointed out that the comment on that line says 768, which is what the same formula gives for a third of the context: the line divided by 3 until a later commit changed it to 4 and left the comment alone. I selected those words and asked whether anything else still says 768. The Investigator found that the map did, in a claim the Checker had passed ([`capture.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/runs/capture.py)):

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

So the Checker's rule changed to **compute, don't quote**. A claim that states a value (a number, a default, a size, a count, a shape) is supported only if the cited code produces that value, and the Checker works the value out from the code itself. A comment or docstring that states the value doesn't count, except for a claim labelled docs, which only reports what the docs say. Here's the same claim after a check under the new rule and one round of repair ([`capture.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/runs/capture.py)):

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

Then I re-checked every claim on both maps that states a value under the new rule, with a repair pass after ([`capture.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/runs/capture.py)):

<!-- RECEIPT: runs-recheck -->
```text
  repo         wall min  turns  cost (list)          exit
  nanochat         19.7     25        $9.19             0
  waku-agent       10.2     21        $4.20             0

  repo         map claims  re-checked  newly not supp.  still not supp.
  nanochat            260         117               39                4
  waku-agent          266          36               13                1
  newly not supported = re-checked claims the re-check found not supported
  (before repair); still not supported = of those, not supported when this
  pass's repair ended (tries each got: the last table below).
```

The new rule turned down 39 claims on nanochat and 13 on waku-agent that had passed before; one said a download is "retried up to 5 times", when the loop makes 5 attempts in all. Repair left 4 on nanochat and 1 on waku-agent under Needs a look. Here's how many tries (one repair each) those leftovers got ([`capture.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/runs/capture.py)):

<!-- RECEIPT: runs-recheck-tries -->
```text
  Still not supported when this pass ended, by the repairs each claim got
  (one repair = one try): in this pass (entries it added) and in earlier
  passes (entries saved before it).
  repo         tries in this pass    tries before it  claims
  nanochat                      2                  0       2
  nanochat                      1                  1       2
  waku-agent                    2                  0       1
```

Two of nanochat's 4 got only one try in this pass, after one try in the earlier repair pass, because the two-round limit then counted every repair a claim had ever had. Since [`74e0c69`](https://github.com/bearbearyu1223/honeyguide/commit/74e0c69) it counts two rounds per pass.

None of this makes the Checker proof. It's Claude checking Claude, and the 768 claim got past it once. The only outside look at a Checker's verdicts so far is the spot-check in [§5](#are-ad-hoc-answers-worse), by a different model working from a sheet of the verdicts and the lines they cite, without the source.

---

### 4. Does an agent need the map? {#does-an-agent-need-the-map}

This is the test the project started with. If a map made each answer better or cheaper, the case for it wouldn't need a team. So the same model answers the same questions with and without the map, against two controls.

#### What ran

This is the pilot from Setup: 20 of nanochat's questions, every condition, Claude Opus 5.5 at medium effort. Each run is Claude Code's **non-interactive mode** (formerly called headless mode), `claude -p "..."`, which takes a prompt, runs to the end and prints the result, so a script can call it ([glossary](https://code.claude.com/docs/en/glossary)). From [`evalscore.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/score/evalscore.py), which scores the saved runs:

<!-- RECEIPT: pilot-setup -->
```text
Honeyguide eval: nanochat (run pilot-nanochat-2)
==========================================================================
  commit      92d63d4e8bb4 (https://github.com/karpathy/nanochat)
  frozen set  MANIFEST.txt sha256 9e707c3e2a59; cutoff 2026-06-30
  size        pilot: 20 questions (locate 6, callers 5, flow 4,
              why/how 5); before cutoff 10, after 5, untagged 5
  model       requested claude-opus-5-5, effort medium
              used: claude-opus-5-5 x161
  runs        161 planned (1 map builds); finished 161 (ok 161, capped 0,
              invalid answer 0, error 0); not run 0; re-run after a
              failure 0; re-run after an interruption or usage limit 0
```

Every run gets the same instructions and the same answer format, and the three conditions that can see the repo get the same tools:

| Condition | Working folder | Tools | Extra context | Passes per question |
| --- | --- | --- | --- | --- |
| no access (control) | empty | none | none | 1 |
| explore | a fresh copy of the repo | Read, Grep, Glob | none | 2 |
| outline | a fresh copy | Read, Grep, Glob | a script-built outline (files, classes, functions, imports), cut to the same size in tokens as the map | 2 |
| map | a fresh copy | Read, Grep, Glob | the map | 3 |

**Read, Grep and Glob** are Claude Code's tools for reading a file, searching file contents, and finding files by name. The map and the outline go into the prompt in the same wording. Answers vary from run to run, so the conditions that read the repo answer each question more than once. The plan was three independent maps, one pass on each; the pilot built one, which the Checker never checked, and all three map passes used it.

The **no-access** run is the control: if the model answers a question with no code at all, it knew the answer already. The **outline** is the free baseline. It costs nothing to build, so if the map can't beat it, what pays off is an index, and the Cartographer's learning isn't.

**Isolation.** Each run sees only what its condition gives it: a fresh configuration folder with no settings, add-on tool servers, saved sessions or auto-memory, and a fresh copy of the pinned commit with no `.git` folder and without nanochat's own `.claude/` folder. (Claude Code's `--bare` mode would do much of this, but it "doesn't use your subscription login" ([non-interactive mode](https://code.claude.com/docs/en/headless)), so each piece is set separately, and tools are named in `--tools` because the default set on macOS and Linux leaves out Glob and Grep ([CLI reference](https://code.claude.com/docs/en/cli-reference)).) A scan of every saved transcript for personal strings came back clean (the [capture](https://github.com/bearbearyu1223/honeyguide/blob/main/outputs/eval_pilot_nanochat.txt)'s last line).

#### The questions

The locate, callers and flow questions were generated by a script and committed, with their answer keys, before any map existed ([`4224c0d`](https://github.com/bearbearyu1223/honeyguide/commit/4224c0d)); a code review later corrected some keys ([`343ac4b`](https://github.com/bearbearyu1223/honeyguide/commit/343ac4b)) without changing any question. A [manifest](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/questions/MANIFEST.txt) of the files' hashes lets you check they haven't changed since. The pilot takes the first questions of each type in that frozen order.

| Type | Asked as | Answer key from | Scored by |
| --- | --- | --- | --- |
| locate | a behaviour, never a function name ("where is ... computed?") | the parsed source | right file and function |
| callers | "what calls X directly?" | the parsed source plus the runtime trace | overlap with the key's set |
| flow | an entry point and a config: "which repo functions run, in order?" | the runtime trace, filtered to repo code | required hops found, in order |
| why/how | a design question, with a rubric | the rubric | a blind grade by another model |

The parsed source is the **AST** (abstract syntax tree), the structure Python builds from the text before running it. Claude drafted the disputed callers and the required hops from the pinned code, and I confirmed them. I wrote the why/how questions and their rubrics myself, after I'd seen the first maps. [`generate.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/questions/generate.py) prints every generated question with its key.

#### Learning, or remembering? {#learning-or-remembering}

A model's **training cutoff** is the date after which it saw no training data. Anthropic gives it only to the month: Claude Opus 5.5 gives a "Reliable knowledge cutoff" of "Jun 2026", and the same month for its "Training data cutoff" ([Anthropic, models overview](https://platform.claude.com/docs/en/about-claude/models/overview)). Code written before the cutoff may be in the model's training data, so a good answer about it could be **memorization**, the model recalling text it was trained on, rather than reading.

There's evidence this happens with code. Given only a repo's name and a bug report, "without access to repository structure", models up to o3 and Claude 4 named the file with the bug up to 76% of the time on SWE-Bench Verified, a widely used benchmark built from public repos, against at most 53% on repos outside it ([SWE-Bench Illusion](https://arxiv.org/abs/2506.12286)). Neither that study nor [LiveCodeBench](https://arxiv.org/abs/2403.07974), which compares problems released before and after a model's cutoff (§5.1), tested the model used here.

So each question is tagged by when its answer's code was written, using `git blame` (which records the commit that last changed each line): a question counts as *after* when at least half of its **key lines**, the source lines its answer key points at, changed after 2026-06-30. Because the cutoff is given only as a month, code changed during June could fall on either side. [`june_check.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/cutoff/june_check.py) dates every key line the tags count, for all of both repos' frozen questions (the pilot uses 20 of nanochat's 27):

<!-- RECEIPT: june-check -->
```text
  repo         questions  key lines  before June  in June  after June
  nanochat            27       1366         1157        0         209
  waku-agent          30       2421            0        0        2421

  questions with any key line last changed in June 2026:
    nanochat    0 of 27
    waku-agent  0 of 30
  key lines in June 2026 if dated by authored commit (no --first-parent):
    nanochat    0
    waku-agent  0

  check against the frozen JSON (sums of lines_counted, lines_after):
    nanochat    counted 1366, after 2026-06-30 209: match
    waku-agent  counted 2421, after 2026-06-30 2421: match
```

Not one key line falls in June, whichever commit date is used. In the pilot, 10 questions are tagged before the cutoff and 5 after; the 5 why/how questions carry no tag.

#### Results {#pilot-results}

A score is between 0 and 1 for each run: right or wrong for locate, the overlap with the key's set for callers (the usual F1 score, which counts both naming the right callers and not naming wrong ones), the share of required hops found in the right order for flow, and a grade for why/how. Mean score per run, by question type ([`evalscore.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/score/evalscore.py)):

<!-- RECEIPT: pilot-scores -->
```text
by type (descriptive; mean per run)
--------------------------------------------------------------------------
  type      questions   no-access     explore     outline         map
  locate            6       0.500       1.000       1.000       1.000
  callers           5       0.733       0.933       1.000       0.933
  flow              4       0.817       0.958       0.958       0.942
  whyhow            5       0.500       0.900       0.900       1.000
```

**Lookups are at the ceiling.** Every condition that could read the repo found every locate answer, and callers and flow scores sit between 0.933 and 1.000 whether or not the run had a map. These questions are too easy, for this model on a repo this size, to show what a map adds. The no-access row is the only one with room below it. Exploring agents do well on questions like these elsewhere too: on SWE-QA, 720 questions about 15 Python repos scored by a model acting as judge, agents generally beat retrieval, which hands the model the chunks of code that best match the question, though not with every model ([Peng et al., arXiv 2509.14635v2](https://arxiv.org/abs/2509.14635), §3.2, Table 3).

<!-- RECEIPT: pilot-paired -->
```text
paired differences by question
--------------------------------------------------------------------------
  per question: mean over its passes; then the mean difference over
  questions with both conditions, 95% CI from 10000 bootstrap resamples of
  questions (seed 0)
  map - explore           +0.022  [-0.010, +0.075]  n=20
  map - outline           +0.005  [-0.050, +0.072]  n=20
  explore - no-access     +0.328  [+0.140, +0.530]  n=20

learning vs remembering: mean score by cutoff subset
--------------------------------------------------------------------------
  subset    questions   no-access     explore     outline         map
  before           10       0.893       0.950       0.983       0.943
  after             5       0.200       1.000       1.000       1.000
  map - explore, before           -0.007  [-0.020, +0.000]  n=10
  map - explore, after            +0.000  [+0.000, +0.000]  n=5
  explore - no-access, before     +0.057  [+0.000, +0.113]  n=10
  explore - no-access, after      +0.800  [+0.400, +1.000]  n=5
  why/how questions carry no cutoff tag and are in neither subset.
```

Each difference ([`evalscore.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/score/evalscore.py)) is taken question by question and then averaged, and the interval in brackets is a 95% interval from resampling the questions (a **bootstrap**). Map minus explore over all 20 questions is +0.022, and its interval runs from below zero, so the pilot can't tell the two apart.

**Looking up beats remembering on changed code.** On the 5 questions whose code was written after the cutoff, the model with no access scored 0.200, and every condition with the repo scored 1.000. Before the cutoff, no access already scored 0.893: the model knows a lot of nanochat. That's a case for letting the agent read the code, with or without a map.

![Dot plot of mean score per run for four conditions in two rows, split by whether the questions' code last changed before or after the training cutoff (June 2026). Before the cutoff, 10 questions: no access 0.89, explore 0.95, outline 0.98, map 0.94. After the cutoff, 5 questions: no access 0.20, and explore, outline and map all 1.00, with a bracket marking explore minus no access, +0.800 [+0.400, +1.000]. A footnote says why/how questions carry no cutoff tag and are in neither row](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/cutoff-split-light.png){: .light width="1000" height="541" }
![Dot plot of mean score per run for four conditions in two rows, split by whether the questions' code last changed before or after the training cutoff (June 2026). Before the cutoff, 10 questions: no access 0.89, explore 0.95, outline 0.98, map 0.94. After the cutoff, 5 questions: no access 0.20, and explore, outline and map all 1.00, with a bracket marking explore minus no access, +0.800 [+0.400, +1.000]. A footnote says why/how questions carry no cutoff tag and are in neither row](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/cutoff-split-dark.png){: .dark width="1000" height="541" }
_Without the repo, the model knows nanochat's older code and not its newer code; with the repo, every condition gets the newer code right, map or no map._

**Why/how is a hint.** On the five why/how questions the map's answers scored 1.000, explore and outline 0.900 and no access 0.500. They were graded blind by ChatGPT (GPT-6), working from a sheet of answers only ([`evalscore.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/score/evalscore.py)):

<!-- RECEIPT: pilot-grader -->
```text
why/how (graded blind)
--------------------------------------------------------------------------
  grader: ChatGPT (GPT-6); from sheet.md only: blind to the condition, no
    access to the repository source
  graded 40 of 40 finished answers; questions awaiting grades (left out of
    every score above): 0
```

That's five questions, one grader that couldn't open the source, and rubrics I wrote after seeing the maps. The result points the map's way, on the kind of question its Design paragraphs are written for, and it can't carry weight on its own.

![Two panels. Left: mean score per run by question type for no access, explore, outline and map. On locate, callers and flow the three conditions that read the code sit between 0.93 and 1.00 and no access sits lower; on why/how, graded blind by GPT-6 on 5 questions, the map scores 1.00 against 0.90 for explore and outline. Right: paired differences over 20 questions with 95% intervals: map minus explore +0.022 and map minus outline +0.005, both intervals crossing zero, and explore minus no access +0.328, well above zero](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/pilot-scores-light.png){: .light width="1000" height="562" }
![Two panels. Left: mean score per run by question type for no access, explore, outline and map. On locate, callers and flow the three conditions that read the code sit between 0.93 and 1.00 and no access sits lower; on why/how, graded blind by GPT-6 on 5 questions, the map scores 1.00 against 0.90 for explore and outline. Right: paired differences over 20 questions with 95% intervals: map minus explore +0.022 and map minus outline +0.005, both intervals crossing zero, and explore minus no access +0.328, well above zero](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/pilot-scores-dark.png){: .dark width="1000" height="562" }
_With the repo in reach, the three conditions tie near the ceiling on lookups, and the only clear gap is against no access. Why/how leans to the map, on five questions._

#### What an answer costs {#what-an-answer-costs}

Cost is counted in tokens of four kinds: **input**, **output**, **cache writes** and **cache reads**. An agent sends the same start of the prompt (its instructions, the files it has read) with every request, so that start, the **prefix**, can be cached. Saving it is a cache write, at 2x the base input price for an hour's keeping, which is what Claude Code on a subscription uses within the plan's included usage; reading it back is a cache read, at 0.05x on Claude Opus 5.5 ([Anthropic, prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), Pricing; [cost tracking](https://code.claude.com/docs/en/agent-sdk/cost-tracking)). Claude Code's dollar figure per run is an estimate at API list price, not what a subscription pays ([Claude Code docs, costs](https://code.claude.com/docs/en/costs)). The means per run, from [`evalscore.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/score/evalscore.py):

<!-- RECEIPT: pilot-cost -->
```text
tokens (thousands), time and cost per answering run (means)
--------------------------------------------------------------------------
  condition   input cache rd cache wr  output tools  min  $ list $ nocache
  no-access     0.0      1.6      0.9     1.4   1.0  0.2   0.037     0.039
  explore       0.0     23.6      5.1     1.4   4.5  0.2   0.074     0.144
  outline       0.0     52.3     14.4     1.4   3.0  0.2   0.154     0.295
  map           0.0     77.5     12.3     1.7   3.7  0.3   0.147     0.393
  price check: list cost rebuilt from tokens with the config prices is
    within 0.0% of the CLI's total_cost_usd on every run (161 runs)
  cost per exact answer (list, with cache):
    no-access      $0.081  (9 exact of 20 runs)
    explore        $0.087  (34 exact of 40 runs)
    outline        $0.171  (36 exact of 40 runs)
    map            $0.167  (53 exact of 60 runs)
  map build: $7.22 list per map (mean of 1)
    break-even vs explore: never (map answers are not cheaper per exact
      answer)
    break-even vs outline: 1803 questions asked (map answers $0.0045
      cheaper per exact answer)
```

Token columns are in thousands; `$ nocache` prices every input token at the full input price. A map run cost $0.147 on average against $0.074 for exploring, about twice as much. The map rides along in every request, and the cache reads show it: 77.5 thousand tokens per map run against 23.6 thousand. And map runs still made 3.7 tool calls on average, against 4.5 without it. Per exact answer (a run that scored 1) the map costs $0.167 and exploring $0.087. The **break-even point** is the number of questions after which the map, its $7.22 build included, costs less per exact answer than exploring. Since each map answer already costs more, there isn't one: the receipt says "never". Against the outline, at $0.171 per exact answer, the map's answers are $0.0045 cheaper, less than half a cent, so it takes 1803 questions for that saving to cover the build.

So by the pilot's cost model, which counts only agent dollars for one person asking lookup questions about a small repo, the map doesn't pay. A team's cost is wider, `team cost = build + refresh + questions + human review`, set against every engineer exploring alone, their own time included. This post measures the build and the questions here and refresh in [§6](#what-a-shared-map-costs-to-keep), but not human review or engineer time.

---

### 5. Are ad-hoc answers worse? {#are-ad-hoc-answers-worse}

There's a version of the team argument that doesn't need the map to win on cost: if ad-hoc answers were often wrong, a checked map would be worth having for accuracy alone. So I checked the pilot's ad-hoc answers. Call it **E1**.

The answers are the explore condition's answers to the flow and why/how questions: 18 answers to 9 questions, from an agent with Read, Grep and Glob and no map, checked by nobody. A script split them into 520 **units** (sentences and list items). The control is 100 of the 224 claims on the pilot's never-checked map, sampled across evidence labels (leaving out the 9 labelled searched, not found), with their anchors stripped. Both went to the same Checker brief in isolated runs on a fresh copy of the repo, so for both the Checker had to find the evidence itself. That's the third level from [§3](#how-it-keeps-itself-honest), as far as a Checker can judge it: is the claim correct? Each unit is judged supported, not supported, wrong, or not a claim (a heading, or "I didn't run it"). From [`adhoc.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/adhoc/adhoc.py):

<!-- RECEIPT: adhoc-comparison -->
```text
  arm              units claims supp. not supp. wrong no verdict    ns+w
  ------------------------------------------------------------------------
  ad-hoc answers     520    489   479         5     5          0   0.020
  map claims         100    100    99         1     0          0   0.010
  claims = units judged supported, not supported or wrong; ns+w = (not
    supported + wrong) / claims. not_a_claim and no verdict are left out
    of claims.
  difference in ns+w, ad-hoc minus map: +0.010, 95% CI [-0.016, +0.033]
    (bootstrap, 10000 resamples; ad-hoc units clustered by answer (18
    answers), map claims resampled one by one (100; a random sample, so
    its batches are arbitrary); seed 0; units without a verdict left out)
```

The difference, +0.010 (1.0 percentage point), has an interval running from -0.016 to +0.033, so this can't tell them apart. Repeating the check on 3 answers gave the same verdict on 75 of 76 units.

![Two panels. Left: the share of claims the Checker judged not supported or wrong, 2.0% (10 of 489) for ad-hoc answers and 1.0% (1 of 100) for map claims sampled before any Checker pass. Right: the difference, +1.0 percentage points, with a 95% interval from -1.6 to +3.3 that crosses zero](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/adhoc-vs-map-light.png){: .light width="1000" height="516" }
![Two panels. Left: the share of claims the Checker judged not supported or wrong, 2.0% (10 of 489) for ad-hoc answers and 1.0% (1 of 100) for map claims sampled before any Checker pass. Right: the difference, +1.0 percentage points, with a 95% interval from -1.6 to +3.3 that crosses zero](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/adhoc-vs-map-dark.png){: .dark width="1000" height="516" }
_Ad-hoc answers and a map's unchecked claims are wrong about equally rarely on nanochat; the interval on the difference crosses zero._

The errors that do occur are the kind a reader wouldn't catch. Here are the five the Checker judged wrong, with its notes (some cut short by the capture; [`adhoc.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/adhoc/adhoc.py)):

<!-- RECEIPT: adhoc-wrong -->
```text
the 10 answer units judged wrong or not supported (truncated)
--------------------------------------------------------------------------
  [WRONG] nc-flo-01__explore__p1#s4: With the 'SSSL' pattern and 2 layers,
      layer 0 would get the short window and layer 1 the long one.
      note: The pattern tiling gives layer 1 pattern[1] = 'S', so the
        pattern alone makes both layers short. Layer 1 only becomes long
        through the final-layer override at line 313.
  [WRONG] nc-why-01__explore__p1#s22: In `GPT.setup_optimizer`
      (nanochat/gpt.py:432-433), the AdamW learning rates are also scaled
      by (model_dim/768)^-0.5.
      note: dmodel_lr_scale = (model_dim/768)**-0.5 is defined at gpt.py
        432-433, but it is applied only to the lm_head, embedding and
        value-embed AdamW groups (lines 439-441). The resid, x0 and smear
        AdamW gr...
  [WRONG] nc-why-02__explore__p1#s36: It then builds GPTConfig and GPT
      inside `with torch.device("meta")`.
      note: GPTConfig is built at lines 136-140, outside the with block.
        Only GPT(config) is built inside `with torch.device("meta")`, at
        lines 141-142.
  [WRONG] nc-why-02__explore__p1#s38: Loading a checkpoint: build_model in
      nanochat/checkpoint_manager.py rebuilds
      GPTConfig(**meta_data["model_config"]), adding any missing keys
      first.
      note: _patch_missing_config_keys adds only one missing key,
        window_pattern (set to 'L'), not any missing key. It is then used
        to build GPTConfig(**model_config_kwargs).
  [WRONG] nc-why-05__explore__p1#s42: Time from the first 10 steps is left
      out of `total_training_time`.
      note: Line 555 adds dt only when step > 10, so steps 0 through 10
        are left out. That is 11 steps on a fresh run, not 10; the 'first
        10 steps' wording is the comment's.
```

The AdamW one is the sort of thing that bites later: the scale is real, but it applies only to the lm_head, embedding and value-embedding groups, and an engineer tuning the residual, x0 or smear groups would be misled. The meta-device one (PyTorch's meta device creates tensors with shapes but no data) puts a line inside a `with` block that it sits just outside of. None of them would look wrong to a reader without the source open.

How far can the Checker's verdicts be trusted? A different model, ChatGPT (GPT-6), reviewed a sample of 30 of them from a sheet, the sentence, verdict, note and cited lines as quoted, without the source ([`adhoc.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/adhoc/adhoc.py)):

<!-- RECEIPT: adhoc-review -->
```text
spot-check review of the Checker's verdicts
--------------------------------------------------------------------------
  reviewer: ChatGPT (GPT-6); from spot_check.md only: the sentence,
    verdict, note and cited lines as quoted; no access to the repository
    source
  items 30; agree 24, would change 2, borderline 4
  verdict          items  agree  change  borderline
  supported           14     14       0           0
  not_supported        6      2       0           4
  wrong                5      3       2           0
  not_a_claim          5      5       0           0
  by arm (the spot-check sample each item came from):
  arm             verdict          items  agree  change  borderline
  ad-hoc answers  supported            5      5       0           0
  ad-hoc answers  not_supported        5      2       0           3
  ad-hoc answers  wrong                5      3       2           0
  ad-hoc answers  not_a_claim          5      5       0           0
  map claims      supported            9      9       0           0
  map claims      not_supported        1      0       0           1
  #6 not_supported (borderline): reasonable inference, presented as
      rationale
  #10 not_supported (borderline): reasonable inference; the mechanism is
      supported
  #11 wrong (change): wrong -> supported with qualification: the outcome
      is right, the causal explanation is wrong (keep wrong if the
      mechanism is graded)
  #15 wrong (change): wrong -> supported with minor imprecision: step > 10
      excludes 11 steps; the unit follows the source comment
  #18 not_supported (borderline): keep not_supported under a strict
      source-evidence policy
  #22 not_supported (borderline): keep not_supported under a strict
      source-evidence policy (inference, not a measurement)
```

Of the 479 ad-hoc claims the Checker judged supported, the reviewer saw 5 (the by-arm rows), and it saw 9 supported map claims; it agreed with all of them, so no disagreement said the Checker was too lenient. Its disagreements were all with verdicts against a claim: it would soften two of the five ad-hoc "wrong" verdicts to supported with a qualification (#11 is the SSSL one above; #15 is the steps one), and it called four "not supported" verdicts borderline (three ad-hoc, one map), all inferences the source doesn't state; for two of those it said a strict policy would keep the verdict. What the spot check can't do is bound the errors the Checker missed. Five supported verdicts out of 479 is far too few for that, so 2.0% is the Checker's count, and I can't say from this whether the true rate is higher or lower.

So on nanochat, ad-hoc answers are not measurably worse. The case for a shared map can't rest on accuracy, then. What it buys is exact citations a reader can click, one reviewed record instead of many private ones, and fixing an error once, for everyone, instead of in whichever chat it happened to turn up.

---

### 6. What a shared map costs to keep {#what-a-shared-map-costs-to-keep}

A saved map goes stale the moment someone pushes a commit. Keeping it current is exploration too, of the code that changed, and if it cost as much as a rebuild there'd be little point keeping a map. So I measured a refresh twice: across a large rewrite (**E2**, nanochat) and across an ordinary day (**E3**, waku-agent).

#### How refresh works

Honeyguide stores each claim with the commit it was checked at and the hash of its cited lines. A **refresh** uses `git diff` from that commit to the new one to classify every claim:

- **unchanged**: the file didn't change, or the cited lines are byte for byte the same in the same place.
- **moved**: the same lines, found in exactly one new place; the line numbers are updated.
- **stale**: the cited lines changed. The claim isn't served as fact, or given to an agent, until the Checker re-reads it, and the Cartographer rewrites it if it no longer holds.
- **retired**: the code is gone. The claim stays on the page, struck through.
- **unanchored**: the claim cites no code, so the hashes can't say anything about it.

New code that no claim covers is listed, and the Cartographer maps only that: the **re-map**. It can also retire older claims it finds no longer hold. The comparison for each refresh is a **rebuild**: the full pipeline (explore, Checker, repair) from scratch.

E2 spans nanochat's July rewrite: from `dc54a1a` to the pin, 60 commits that changed 33 Python files, +1070 and -2498 lines. E3 is waku-agent's next 2 commits after its pin, which changed 3 files, +155 and -5 lines: the kind of change a team sees every day. All of the receipts below come from [`eval/refresh/capture.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/refresh/capture.py), starting with the commits and which Honeyguide code each run used:

<!-- RECEIPT: refresh-commits -->
```text
  E2  nanochat: nanochat across its July rewrite
    base                dc54a1a  2026-05-05 03:17 UTC
    target              92d63d4  2026-07-03 22:54 UTC
    days apart          59.8
    commits in between  60
    .py files changed   33 (+1070 / -2498 lines)
    all files changed   42 (+1194 / -4079 lines)
    .py lines at base   9,203 in 36 files
    .py lines at target 7,775 in 36 files
  E3  waku-agent: waku-agent, an ordinary day
    base                2c7e7e6  2026-11-15 05:09 UTC
    target              21dcd56  2026-11-16 23:08 UTC
    days apart          1.7
    commits in between  2
    .py files changed   3 (+155 / -5 lines)
    all files changed   3 (+155 / -5 lines)
    .py lines at base   87,084 in 386 files
    .py lines at target 87,234 in 387 files

  Honeyguide code at each run's start: the commit the honeyguide-run
  worktree's HEAD was on (its reflog, git reflog --date=iso-strict) at the
  time in the run's start file. The reflog records HEAD moves, not
  uncommitted edits. Read from eval/refresh/run_commits.json, frozen from
  that reflog at 2026-11-17 05:07 UTC (git expires reflog entries).
    run                   start (UTC)         commit   HEAD on it since
    E2 build              2026-11-16 23:34:10 e855908  2026-11-16 23:23:33
    E2 refresh            2026-11-17 00:55:27 e855908  2026-11-16 23:23:33
    E3 refresh, attempt 1 2026-11-16 23:23:49 e855908  2026-11-16 23:23:33
    E3 refresh, attempt 2 2026-11-16 23:25:19 e855908  2026-11-16 23:23:33
    all 4 runs started on the same commit, e855908
```

#### What it cost

<!-- RECEIPT: refresh-cost-e2 -->
```text
  E2  nanochat: nanochat across its July rewrite
    run                      $ list  wall min  session  turns  exit
    build                    $21.03      52.5     52.5     27     0
    refresh                  $10.69      25.2     25.2     28     0

    tokens                    input   cache read  cache write     output
    build                       434   33,908,276    1,265,160    389,268
    refresh                     298   16,803,122      719,709    180,164

    build: models claude-opus-5-5 ($21.03)
      subagents cartographer 3, checker 3; result success; permission
      denials 0
    refresh: models claude-opus-5-5 ($10.69)
      subagents cartographer 2, checker 3; result success; permission
      denials 1 (Bash 1)

    rebuild: the build above (same Honeyguide commit as the refresh,
    e855908, by the reflog; full Explore + Checker + repair at the base
    commit).
    refresh / rebuild:  $ 0.51, wall minutes 0.48
```

For E2 the rebuild is a full checked build of nanochat's map at the old commit, run on the same Honeyguide commit as the refresh, `e855908` ([`capture.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/refresh/capture.py)). The old commit is the bigger repo, 9,203 lines of Python against the 7,775 the refresh ended on, so the build had more code to read than a rebuild at the new commit would. A rebuild at the pin would probably cost somewhat less, and the ratio would come out above 0.51.

E3's comparison is weaker, and the capture ([`capture.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/refresh/capture.py)) says why:

<!-- RECEIPT: refresh-cost-e3 -->
```text
  E3  waku-agent: waku-agent, an ordinary day
    run                      $ list  wall min  session  turns  exit
    refresh, attempt 1        $0.67       1.3      1.3     16     0
    refresh, attempt 2        $3.25       8.4      8.4     27     0
    refresh total             $3.92       9.7      9.7

    tokens                    input   cache read  cache write     output
    refresh, attempt 1           42      637,196       57,864      7,769
    refresh, attempt 2          150    3,096,300      256,770     60,533

    refresh, attempt 1: models claude-opus-5-5 ($0.67)
      subagents checker 1; result success; permission denials 3 (Write 3)
    refresh, attempt 2: models claude-opus-5-5 ($3.25)
      subagents cartographer 2, checker 3; result success; permission
      denials 0
    refresh, attempt 1: denied writes into the store:
      drafts/checker/stale-1.json x2, drafts/guide/stale-1.json x1

    rebuild: the waku-agent full pipeline at its pin, from runs.txt's runs
    block (run 2 + repair + re-check). It ran on older Honeyguide code
    (runs.txt does not record which commit) and was a different
    invocation: three separate claude -p passes, not one.
    rebuild pass             $ list  wall min  turns
    run 2                    $12.43      35.2     2*
    repair                    $2.76       6.3     19
    re-check                  $4.20      10.2     21
    rebuild total            $19.39      51.7
    rebuild less re-check    $15.19      41.5
    rebuild tokens        not recorded (runs.txt holds none)
    (* in runs.txt: turns of the last result segment only)
    refresh (both attempts) / rebuild:  $ 0.20, wall minutes 0.19
    last attempt only / rebuild:  $ 0.17, wall minutes 0.16
    The re-check pass has no counterpart in a one-run build (E2's:
    Explore, Checker, repair); the ratio without it:
    both attempts / rebuild less re-check:  $ 0.26, wall minutes 0.23
```

E3's refresh took two attempts: the permission rules denied three of the first attempt's writes into the map's folder, and that attempt alone cost $0.67. E3's rebuild also has a pass that E2's doesn't: the re-check of every claim that states a value, from [§3](#the-claim-it-passed), run on its own after repair. E2's rebuild is one run of Explore, Checker and repair, so the capture prints E3's ratio both ways. Counted with both attempts, an ordinary day cost 0.20 of a rebuild with the re-check and 0.26 without it (0.19 and 0.23 in wall minutes): about a fifth to a quarter. With the last attempt alone it's a little less.

![Two panels of paired bars, rebuild against refresh, for waku-agent after an ordinary day and nanochat across its July rewrite, under the title: keeping the map current costs about 1/5 to 1/4 of a rebuild for an ordinary day, and about 1/2 across a 60-commit rewrite. Cost at list price: waku-agent $19.39 against $3.92 (0.20 to 0.26), nanochat $21.03 against $10.69 (0.51). Wall-clock minutes: 51.7 against 9.7 (0.19 to 0.23), and 52.5 against 25.2 (0.48). A dashed line across waku-agent's rebuild bars marks off its re-check pass, which nanochat's one-run rebuild doesn't have; the lower ratio counts the re-check and the higher one leaves it out](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/refresh-vs-rebuild-light.png){: .light width="1000" height="686" }
![Two panels of paired bars, rebuild against refresh, for waku-agent after an ordinary day and nanochat across its July rewrite, under the title: keeping the map current costs about 1/5 to 1/4 of a rebuild for an ordinary day, and about 1/2 across a 60-commit rewrite. Cost at list price: waku-agent $19.39 against $3.92 (0.20 to 0.26), nanochat $21.03 against $10.69 (0.51). Wall-clock minutes: 51.7 against 9.7 (0.19 to 0.23), and 52.5 against 25.2 (0.48). A dashed line across waku-agent's rebuild bars marks off its re-check pass, which nanochat's one-run rebuild doesn't have; the lower ratio counts the re-check and the higher one leaves it out](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/refresh-vs-rebuild-dark.png){: .dark width="1000" height="686" }
_Keeping a map current costs about half a rebuild across a rewrite and about a fifth to a quarter after an ordinary day._

#### What it did

E2's map isn't the 260-claim map from [§3](#the-checker): it's the build above, made at `dc54a1a`, with 310 claims. Here's how the refresh classified them before it started, its **plan**, and what happened after ([`capture.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/refresh/capture.py)):

<!-- RECEIPT: refresh-plan-e2 -->
```text
  E2  nanochat: nanochat across its July rewrite
    plan                dc54a1a -> 92d63d4
    claims              310
      unchanged            88
      moved               128
      stale                50
      retired              32
      unanchored           12
    uncovered units     60 (new 54, changed 6)
    changed files with no outline (not Python)  7
    applied 2026-11-17T00:55:37Z: dc54a1a -> 92d63d4; unchanged 88, moved
      128, stale 50, retired 32, unanchored 12; uncovered 60
```

<!-- RECEIPT: refresh-did-e2 -->
```text
  E2  nanochat: nanochat across its July rewrite
    map commit after    92d63d4 (= target)
    claims after        397: current 319, stale 0, retired 78
    retired in the refresh  78
      by refresh apply (plan: retired)      32
      by the re-map                         46
      by the repair pass (retire)            0
      re-map retirements by plan class: unchanged 0, moved 3, stale 37,
        retired 0, unanchored 6, new 0
    new claims          87
    uncovered units cited by a new claim (anchor at target, lines
      overlap): 43 of 60 (new 37 of 54, changed 6 of 6)
    uncovered units cited by any current claim (anchor at target, lines
      overlap): 44 of 60 (new 38 of 54, changed 6 of 6)
    stale claims        50, re-checked in the refresh 50
      Checker on re-check              claims  then re-map retired
      still held (supported)               15                    2
      no longer held (not supported)       35                   35
      left stale after the refresh          0
    new claims, Checker    supported  not supp.   other
      first check                 83          4       0
      after round 1               87          0       0
      after round 2               87          0       0
    repair in the refresh: 1 pass(es), 1 round(s)
      round 1: anchor 3, rewrite 1
    older claims repaired in the refresh  0
```

All 50 stale claims were re-checked, and nothing was left stale ([`capture.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/refresh/capture.py)). The re-map wrote 87 new claims, which cite 43 of the 60 new or changed functions, classes and methods no claim had covered (the receipt's uncovered units); 4 of the new claims failed their first check and one repair round fixed them.

![A flow from 310 claims before the refresh, by the plan's class, stacked from the top as stale 50, retired 32, moved 128, unanchored 12 and unchanged 88, to 397 claims after it, none stale. All 50 stale claims go through the Checker's re-check: 35 no longer held, all retired, and 15 still held, of which the re-map retired 2. Retired 32 flow straight to retired. Moved 128 and unchanged 88 flow to current, except 3 moved claims the re-map retired; of the 12 unanchored claims, the re-map retired 6. 87 new claims from the re-map join current. Retired claims are drawn in berry. After: retired 78 (32 by refresh apply, 46 by the re-map), stacked as 35 stale and no longer held, 2 stale and still held, 32 the plan retired, 3 moved (the gap) and 6 unanchored (the gap); and current 319 (232 kept, 87 new). The 3 moved and 6 unanchored bands are outlined and hatched where they enter retired, with a callout at the right: the gap, 3 moved and 6 unanchored claims whose hashes matched (or there were none), so the refresh kept them current; the re-map retired them later](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/refresh-flow-light.png){: .light width="1000" height="659" }
![A flow from 310 claims before the refresh, by the plan's class, stacked from the top as stale 50, retired 32, moved 128, unanchored 12 and unchanged 88, to 397 claims after it, none stale. All 50 stale claims go through the Checker's re-check: 35 no longer held, all retired, and 15 still held, of which the re-map retired 2. Retired 32 flow straight to retired. Moved 128 and unchanged 88 flow to current, except 3 moved claims the re-map retired; of the 12 unanchored claims, the re-map retired 6. 87 new claims from the re-map join current. Retired claims are drawn in berry. After: retired 78 (32 by refresh apply, 46 by the re-map), stacked as 35 stale and no longer held, 2 stale and still held, 32 the plan retired, 3 moved (the gap) and 6 unanchored (the gap); and current 319 (232 kept, 87 new). The 3 moved and 6 unanchored bands are outlined and hatched where they enter retired, with a callout at the right: the gap, 3 moved and 6 unanchored claims whose hashes matched (or there were none), so the refresh kept them current; the re-map retired them later](/assets/picture/2026-12-12-honeyguide-agent-that-learns-a-codebase/refresh-flow-dark.png){: .dark width="1000" height="659" }
_Every stale claim was re-checked and none was left stale; the claims the hash check couldn't flag are the 3 moved and 6 unanchored ones the re-map later retired._

E3 had little to do: 7 moved claims, 1 stale claim that still held, and 6 new claims citing 10 of the 12 new or changed pieces of code, with nothing retired.

#### Its gap

Look at the re-map's retirements by plan class in E2: moved 3, unanchored 6. Those are claims the refresh had kept current because their hashes matched (or, for unanchored claims, because there was nothing to hash), and that the re-map then retired: 3 of the 216 unchanged or moved claims, and 6 of the 12 unanchored ones. The capture doesn't read why each was retired, because the reason is the agent's own text, so I'll say "retired by the re-map" and not "false". What I can say is how it's possible. A claim can depend on code it doesn't cite: a claim about the repo as a whole, or that something isn't in it, keeps matching hashes while the code around it changes. Half the unanchored claims were retired in E2, and the hash check can't flag any of them. That's the refresh's gap, and it isn't new: RepoAgent, which updates the documentation of code objects a commit changes or whose callers change, says its "update is not triggered when an object’s reference objects change" ([Luo et al., 2024](https://arxiv.org/abs/2402.16667), §2.3). [What's next](#whats-next) has the fix I want to try.

---

### 7. Limits, and what a large codebase needs {#limits-and-large-codebases}

**What this doesn't show.** The team benefit, engineers saving time by reading one map instead of each exploring, is argued here, not measured: I'm one person, and nobody else has used these maps, so reuse and adoption are untested. The four-way test is a 20-question pilot on one repo. Everything ran on one model, Claude Opus 5.5, which also built the maps it was tested with. The why/how grades and the review of the Checker's verdicts were done by another model from sheets, without the source. The Checker is Claude checking Claude. The runtime traces start from toy entry points, so they see part of each repo.

**nanochat is small; waku-agent is larger, but most of it is outside what the questions cover.** Here's the Python at each pin, all of it and the part the questions' answer keys may cite (their **key scope**), from [`june_check.py`](https://github.com/bearbearyu1223/honeyguide/blob/main/eval/cutoff/june_check.py):

<!-- RECEIPT: june-check-pylines -->
```text
  Python at the pinned commit: every .py file in the commit's tree (git
  ls-tree -r), lines counted as wc -l counts them; key scope = the .py
  files the questions' keys may cite (below).
                              every .py file           in key scope
  repo        commit        files      lines       files      lines
  nanochat    92d63d4          36      7,775          30      6,715
  waku-agent  2c7e7e6         386     87,084          99     21,564
  key scope:
    nanochat: every tracked .py outside tests/, test/, examples/, docs/
    waku-agent: every tracked .py under waku/ (hosted/, evals/, lab/,
      scripts/, examples/, docs/ excluded)
```

nanochat has 7,775 lines of Python, and the map build reads files one by one. On a repo that size, Grep plus Read finds almost anything in a few calls, which is a large part of why lookups hit the ceiling in [§4](#does-an-agent-need-the-map), and the pilot ran only on nanochat. waku-agent has 87,084 lines, but 21,564 in its key scope, the `waku/` package; the questions don't cover the rest. Neither is what I'd call a large codebase, and on one I'd expect three things to change. Exploring costs more per question, because a search returns hundreds of hits. Building one map by reading files stops scaling. And one map of a monorepo would be too long to read. These are expectations, not measurements. The one outside measurement I know of that varies with codebase size is Cursor's: with search by meaning available, the code its agent wrote was kept 0.3% more often overall, and 2.6% more often "on large codebases with 1,000 files or more" ([Cursor blog](https://cursor.com/blog/semsearch)). That's code retention, not cost or accuracy, and Cursor has since dropped that search ([§1](#what-already-exists)), but it suggests that size changes what helps.

What I think a large codebase needs, as directions and not results:

- **Per-package maps**, each owned by the team that owns the code and refreshed on its own schedule, with a thin map on top that links them.
- **An index under the map.** A symbol table and a call graph, built by a parser rather than by an agent reading files, so the Cartographer starts from structure instead of from Grep. aider's repo map ([§1](#what-already-exists)), its symbols ranked on a graph of file dependencies, is prior art here. Claude Code can already use one kind of index: a code-intelligence plugin connects it to a **language server** (the program that gives an editor go-to-definition), so it "finds definitions and references by symbol instead of by text search" ([Claude Code docs, code intelligence](https://code.claude.com/docs/en/plugins/code-intelligence)); the pilot's runs didn't use it. In research, LocAgent parses a Python repo into a graph of directories, files, classes and functions joined by contain, import, invoke and inherit edges, and has an agent search it to find the code an issue needs changed ([Chen et al., ACL 2025](https://arxiv.org/abs/2503.09089), §3.1–3.2).
- **Search over claims.** An agent answering a question should retrieve the few claims that bear on it. The pilot's map runs read the whole map every time, which is part of why they cost twice as much.

---

### What I learned {#what-i-learned}

I started out to show that a prebuilt map makes an agent answer questions about code more accurately. On the repo and questions I tested, it didn't measurably. Exploring was already very good, the map's one lead (five why/how questions, graded by GPT-6) was a hint at most, and the map added cost.

So who maintains the understanding when agents write the code? When each engineer asks their own agent, nobody does. Each conversation starts another investigation, and Claude Code's own auto memory, the notes it saves for itself between sessions, "skips anything it can derive from the codebase, such as architecture, file paths, or debugging fixes", and "Auto memory is machine-local" ([Claude Code docs, memory](https://code.claude.com/docs/en/memory)). Honeyguide's answer is a map that's kept under checks: claims tied to their source lines, a Checker that reads those lines and works values out from the code instead of trusting a comment, and a refresh that re-checks what changed. That part I've shown, on two repos and at a measured cost: about half a rebuild across a rewrite, and a fifth to a quarter after an ordinary day. The Checker still passed one wrong value before its rule changed.

What I haven't shown is whether a team gains from it. The map can't be shared yet, and nobody but me has used it, so whether it saves engineers time, cuts duplicated investigation, or improves anyone's understanding of the code is still open. Those are the next questions.

### What's next {#whats-next}

First, **sharing**: export the map, with its citations, checks and answers, as a static page a team can commit next to the code or host. Second, **reviewable knowledge updates**: when a pull request changes code, Honeyguide proposes the matching change to the map's claims (old claim, new claim, the source lines) for review alongside the code. Third, a **team study**, as its own post and the real test of the team value: four to six engineers new to a repo, on overlapping tasks, with and without Honeyguide, measuring repeated investigation time, reuse, verification effort, stale knowledge and understanding.

Then two smaller ones: a **map health check** that generates locate and callers questions with keys from the code, answers them from the map and scores them by script, to show where a map is thin; and closing the refresh gap from [§6](#its-gap) by re-checking unanchored and repo-wide claims on every refresh, and keeping a question attached when its claim is rewritten.

This post starts a new series, Agentic Workflows. The next one goes behind the scenes: the team of agents that helped me write this post. After that, a 💡 **Scout**, an agent that proposes changes worth making to a repo, tested against history: given a repo at an earlier commit, does it propose what the maintainers went on to build?

### Appendix: glossary {#appendix-glossary}

The terms this post leans on; anything not listed is defined where it first appears.

| Term | Means | In this post |
| --- | --- | --- |
| coding agent | a language model that acts on code through tools: search, read, run, edit | Claude Code |
| context window, token | the text a model can see at once; the chunks of a few characters it reads and bills in | §1, §4 |
| map | Honeyguide's document about a repo: overview, modules, flows, every claim pointing at code | written by the Cartographer |
| subagent | a separate Claude instance the main session starts for one task, with its own context, instructions and tools | Cartographer, Investigator, Tracer, Checker |
| skill, plugin | a `SKILL.md` file of instructions Claude Code loads when relevant; a bundle of skills, subagents and other add-ons | §2 |
| entry point, hop | where a flow starts; one call on its path | §2, §4 |
| runtime trace | the ordered list of repo functions that ran for one entry point | §2, §4 |
| anchor, hash | a claim's pointer into the code, resolved by a script to a line range, commit and a fingerprint of the lines | §3, §6 |
| evidence label | what kind of evidence a claim rests on | seven labels, §3 |
| Checker verdict | supported or not supported, after re-reading the cited lines; in E1, also wrong or not a claim | §3, §5 |
| repair pass | the Cartographer's fix for a claim the Checker turned down: cite more lines, narrow the claim, or retire it; at most two rounds per pass | §3, §6 |
| non-interactive mode | running Claude Code from a script with `claude -p` (formerly called headless mode) | §4 |
| pilot | the four-way test on 20 nanochat questions, before any full run | §4 |
| no access, explore, outline, map | the four conditions of the test | §4 |
| bootstrap | estimating an interval by resampling the questions (or claims) many times | §4, §5 |
| cache write, cache read | saving the repeated start of a prompt for later requests; reading it back at a fraction of the input price | §4 |
| break-even | the number of questions after which the map costs less per exact answer than exploring | §4 |
| E1, E2, E3 | the ad-hoc answer check; the refresh across nanochat's rewrite; the refresh after an ordinary day on waku-agent | §5, §6 |
| refresh, rebuild | updating a saved map to a new commit; building it again from scratch | §6 |
| unchanged, moved, stale, retired, unanchored | what a refresh finds for each claim | §6 |
| re-map | the part of a refresh where the Cartographer maps new code and retires claims that no longer hold | §6 |

### References

Sources checked on 2026-11-15 and 2026-11-16; Claude Code's docs are for CLI 2.1.325. Quotes are given where the post cites them.

- Anthropic, Claude Code docs: [Subagents](https://code.claude.com/docs/en/sub-agents) (introduction; §2); [Glossary](https://code.claude.com/docs/en/glossary), Skill, Plugin and "Non-interactive mode … Formerly called headless mode" (§2, §4); [Memory](https://code.claude.com/docs/en/memory): `/init` "creates a file with build commands, test instructions, and project conventions it discovers"; §Audit your instruction files, `/doctor prompt-audit` looks for "instructions written for older models, references to files or commands that don't exist, and files that contradict each other" (§1); auto memory (what I learned); [CLI reference](https://code.claude.com/docs/en/cli-reference), the `--tools` row: the default set on macOS, Linux and WSL leaves out Glob and Grep (§4); [Non-interactive mode](https://code.claude.com/docs/en/headless), the `--bare` section (§4); [Costs](https://code.claude.com/docs/en/costs), which "computes the dollar figure locally from token counts at list price", and [Agent SDK, cost tracking](https://code.claude.com/docs/en/agent-sdk/cost-tracking), the 1-hour cache within a plan's included usage (Setup, §4); [Code intelligence plugins](https://code.claude.com/docs/en/plugins/code-intelligence), "finds definitions and references by symbol instead of by text search", through an `LSP` tool that is read-only (§7).
- Anthropic, [Models overview](https://platform.claude.com/docs/en/about-claude/models/overview) — Claude Opus 5.5: "Reliable knowledge cutoff | Jun 2026" and "Training data cutoff | Jun 2026", with no day given. Cited in Setup and §4.
- Anthropic, [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), Pricing — 1-hour cache writes at 2x the base input price; "Cache hits and refreshes on Claude Opus 5.5 and Claude Sonnet 5.5 are priced at 0.05x the base input price". Cited in §4.
- aider, [Repository map](https://aider.chat/docs/repomap.html) — "the key symbols which are defined in each file"; "a graph ranking algorithm, computed on a graph where each source file is a node and edges connect files which have dependencies", sized by `--map-tokens`. Cited in §1 and §7.
- Cognition, [DeepWiki](https://docs.devin.ai/work-with-devin/deepwiki) (Devin docs) — "Devin now automatically indexes your repos and produces wikis with architecture diagrams, links to sources, and summaries of your codebase"; after editing the wiki's configuration, "Commit the file and regenerate your wiki". The docs don't say whether links point at line ranges, or that a wiki updates itself. Cited in §1.
- Hurley, Rodriguez, Marques and Shams, ["Introducing Code Wiki: Accelerating your code understanding"](https://developers.googleblog.com/en/introducing-code-wiki-accelerating-your-code-understanding/), Google Developers Blog, 2025-11-13 — "regenerates the documentation after each change"; "hyper-linked directly to the relevant code files and definitions"; launched "in public preview … It ingests public repositories". Cited in §1.
- Cursor: [semantic-search docs, as archived on 2026-02-25](https://web.archive.org/web/20260225213818/https://cursor.com/docs/context/semantic-search) ("specialized vector database"; "The system intelligently updates only changed files, removing old embeddings and creating new ones as needed"); Heule, Jia and Jain, ["Improving agent with semantic search"](https://cursor.com/blog/semsearch), Cursor blog, 2025-11-06 ("on average 12.5% higher accuracy in answering questions (6.5%–23.5% depending on the model)", on Cursor's own question set); the same post's A/B test: "agent code retention increases by 0.3% when semantic search is available. This effect increases to 2.6% on large codebases with 1,000 files or more" (code retention, not cost; §7); forum thread ["What do you think about Cursor removing the codebase indexing settings?"](https://forum.cursor.com/t/what-do-you-think-about-cursor-removing-the-codebase-indexing-settings/165899), Maxime (Cursor staff) [2026-07-21](https://forum.cursor.com/t/what-do-you-think-about-cursor-removing-the-codebase-indexing-settings/165899/5) and Colin (Community Support Engineer) [2026-07-22](https://forum.cursor.com/t/what-do-you-think-about-cursor-removing-the-codebase-indexing-settings/165899/10), whose Instant Grep index is "built per-repo and stored locally only". Cited in §1 and §7.
- Luo et al., ["RepoAgent: An LLM-Powered Open-Source Framework for Repository-level Code Documentation Generation"](https://arxiv.org/abs/2402.16667), arXiv 2402.16667v1, EMNLP 2024 System Demonstrations ([Anthology 2024.emnlp-demo.46](https://aclanthology.org/2024.emnlp-demo.46/)) — §2.1: Python files, with "all Classes and Functions" documented; §2.3: from a Git pre-commit hook, "RepoAgent only updates the documentation of affected objects"; "the update is not triggered when an object’s reference objects change". Cited in §1 and §6.
- Sourcegraph docs (undated pages): [Deep Search](https://sourcegraph.com/docs/deep-search), "an agentic code search tool that understands natural language questions about your codebase"; "Every Deep Search response includes a detailed list of sources contributing to the answer. These sources show exactly which searches were performed and which files were read"; "Once you share a conversation, any user on your instance can view it with the link" (an admin turns sharing on). [Code navigation](https://sourcegraph.com/docs/code-navigation): search-based navigation "uses a mix of text search and syntax-level heuristics"; [precise code navigation](https://sourcegraph.com/docs/code-navigation/precise-code-navigation) "relies on the open source SCIP, which is a language-agnostic protocol for indexing source code", and auto-indexing creates indexes "keeping precise code navigation available and up-to-date". Cited in §1.
- Augment Code docs: [Context Engine](https://docs.augmentcode.com/context-services/mcp/overview), "Indexes commit history, codebase patterns, external sources (docs, tickets), and tribal knowledge"; the local server picks up file changes immediately, the remote one updates "Automatically when commits are pushed to the default branch" (§How Indexing Works). [Cosmos Experts Memory](https://docs.augmentcode.com/cosmos/experts-memory): "Memory can be shared with an organization"; "Repository-based Experts typically use one scope per repository"; the Expert "writes the information as readable Markdown", and on load "flags discrepancies when current evidence conflicts with a remembered rule". Cited in §1.
- GitHub Staff, [Octoverse 2026](https://github.blog/news-insights/octoverse/octoverse-commits-pushes-reach-new-highs-in-2026-as-developers-build-with-agents/), GitHub blog, 2026-10-28 (updated 2026-11-06) — "By September 2026, more than 40% of pull requests on GitHub started with an agent's first draft, up from 17% in March"; methodology: the agent signal "is a floor, since not all agent use may be detected". GitHub's own telemetry. Cited in §1.
- Daniotti, Wachs, Feng and Neffke, "Who is using AI to code? Global diffusion and impact of generative AI", *Science* 391(6787):831–835, 2026, [doi:10.1126/science.adz9311](https://doi.org/10.1126/science.adz9311) (read as [arXiv 2506.08945v2](https://arxiv.org/abs/2506.08945)) — abstract: "Today, AI writes an estimated 29% of Python functions in the US", from a classifier run over GitHub commits; the supplement says its design "is not optimized for identifying causal effects". Cited in §1.
- He, Miller, Agarwal, Kästner and Vasilescu, ["How Cursor AI Increases Short-Term Velocity and Long-Term Complexity in Open-Source Projects"](https://arxiv.org/abs/2511.04427), arXiv 2511.04427v3, MSR '26 — §4.1.1: "Lines added increase by about 28.6% (Table 2). There is no statistically significant effect for the volume of commits"; the abstract calls the increase "large, but transient". 806 open-source repos (§3.5). Cited in §1.
- METR, randomized controlled trial, [arXiv 2507.09089v2](https://arxiv.org/abs/2507.09089) — "16 developers with moderate AI experience complete 246 tasks in mature projects"; "allowing AI actually increases completion time by 19%"; early-2025 tools. Cited in §1.
- DORA, *2025 State of AI-assisted Software Development* (v. 2025.2, Google), [report PDF](https://services.google.com/fh/files/misc/2025_state_of_ai_assisted_software_development.pdf) — p. 3: "survey responses from nearly 5,000 technology professionals"; p. 4: "AI adoption now improves software delivery throughput, a key shift from last year. However, it still increases delivery instability." Self-reported survey data. Cited in §1.
- Wen, Nagy, Bavota and Lanza, "A Large-Scale Empirical Study on Code-Comment Inconsistencies", ICPC 2019, pp. 53–64, [doi:10.1109/ICPC.2019.00019](https://doi.org/10.1109/ICPC.2019.00019) — §III-A: "1,500 open source Java projects hosted on GitHub"; §IV-A: "between 13% and 20% of code changes trigger comment updates. This does not imply that in the remaining ∼80% of cases code-comment inconsistencies are introduced, but they represent a possibility". Cited in §1.
- Tan, Wagner and Treude, "Detecting outdated code element references in software repository documentation", *Empirical Software Engineering* 29(1):5, 2024, [doi:10.1007/s10664-023-10397-6](https://doi.org/10.1007/s10664-023-10397-6) (read as [arXiv 2212.01479v1](https://arxiv.org/abs/2212.01479)) — §1: "28.9% of the most popular projects on GitHub currently contain at least one outdated reference, with 82.3% of the projects being outdated at least once during the project's history. These references were typically outdated for years before they were noticed". An outdated reference names a code element that's been deleted from the source (§3). Cited in §1.
- Aghajani et al., "Software Documentation Issues Unveiled", ICSE 2019, [doi:10.1109/ICSE.2019.00122](https://doi.org/10.1109/ICSE.2019.00122) — §IV-A: "Up-to-dateness problems account for 39% of issues related to documentation content", counted over 878 documentation-related artifacts from mailing lists, Stack Overflow, issues and pull requests. Cited in §1.
- Chen et al., ["LocAgent: Graph-Guided LLM Agents for Code Localization"](https://arxiv.org/abs/2503.09089), arXiv 2503.09089v2, ACL 2025 — §3.1: a graph of {directory, file, class, function} nodes with {contain, import, invoke, inherit} edges, built by parsing each Python file's AST ("we focus our study on Python repositories"); §3.2: an agent searches the graph to localize the code an issue needs changed. Cited in §7.
- Liang, Garg and Zilouchian Moghaddam, [SWE-Bench Illusion](https://arxiv.org/abs/2506.12286), arXiv preprint 2506.12286v4 (2025-12-01) — given the repo's name and the issue text, "without access to repository structure", models identify the buggy file up to 76% of the time on SWE-Bench Verified (o3) against up to 53% on 245 tasks from 7 repos outside it; models of the Claude 4 generation and earlier. Cited in §4.
- Peng et al., [SWE-QA](https://arxiv.org/abs/2509.14635), arXiv 2509.14635v2 (2026-04-26) — 720 questions on 15 Python repos, scored out of 100 by a model acting as judge; Table 3's caption: "In general, RAG-enhanced methods outperform direct inference, and agent-based methods further improve upon RAG methods"; §3.2: smaller models "benefit less from agent frameworks, with agent-based methods performing on par with or worse than RAG". Cited in §4.
- Jain et al., ["LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code"](https://arxiv.org/abs/2403.07974), arXiv 2403.07974v2, ICLR 2025 (cited from v2) — §5.1 evaluates "problems released after the cutoff date". Cited in §4.
- Python docs, [`sys.setprofile`](https://docs.python.org/3.12/library/sys.html#sys.setprofile) (3.10 to 3.12). Cited in §2.
- Spottiswoode and Wood, "Culturally determined interspecies communication between humans and honeyguides", *Science* 382(6675):1155–1158, 2023-12-08, [doi:10.1126/science.adh4129](https://doi.org/10.1126/science.adh4129) ([open copy](https://par.nsf.gov/servlets/purl/10524235)), p. 1. Cited in the introduction.
- [nanochat](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd) at `92d63d4` — [LICENSE](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/LICENSE): MIT, "Copyright (c) 2025 Andrej Karpathy"; [README, line 8](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/README.md?plain=1#L8), which sends readers with questions to DeepWiki; [`gpt.py`, line 302](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L302); 7,775 lines of Python at this commit (printed by `june_check.py`). Cited in Setup, §1, §3 and §7.
- [waku-agent](https://github.com/ShenSeanChen/waku-agent/tree/2c7e7e68d3d65ac19d428bdde7820844d87dfa19) at `2c7e7e6` (2026-11-15, 05:09 UTC) — [LICENSE](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/LICENSE): MIT, "Copyright (c) 2026 Sean Chen (ShenSeanChen)"; [`hosted/LICENSE`](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/hosted/LICENSE): Elastic License 2.0; [LICENSE-BRAND](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/LICENSE-BRAND): brand assets and names aren't MIT (copyright AutoManus Technologies, Inc.); [README, lines 129 to 133](https://github.com/ShenSeanChen/waku-agent/blob/2c7e7e68d3d65ac19d428bdde7820844d87dfa19/README.md?plain=1#L129-L133). Repo created 2026-07-10. Cited in Setup, §6 and §7.
- [RepoRabbit](https://github.com/bearbearyu1223/repo-rabbit), my earlier Claude Code subagent for exploring a repo. README at `b9e9eae`: [line 115](https://github.com/bearbearyu1223/repo-rabbit/blob/b9e9eae/README.md?plain=1#L115), a reference that "doesn't resolve to a real file and line is rejected"; [line 177](https://github.com/bearbearyu1223/repo-rabbit/blob/b9e9eae/README.md?plain=1#L177), "Each reference stores a hash of the lines it cites … If they changed, that explanation is flagged for the agent to review". Cited in §1 and §3.
