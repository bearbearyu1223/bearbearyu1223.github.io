---
name: post-desk
description: Han's review and approval gate for the LLM Architecture Refresh blog series, plus its status board and agent workflow. Serves a phone page (over Tailscale) with a live preview of unpublished posts where Han reads a post, approves it or requests changes; runs the receipt-auditor, first-reader and promo-writer agents; gates every git push of the blog and the demo repo. Use when the user asks where the series stands, wants to review, approve, publish or push a post, gives feedback on a post, wants a LinkedIn promo for a post, or before any push.
---

# Post desk

Script: `python3 .claude/skills/post-desk/scripts/desk.py <cmd>` (stdlib only, run from the blog repo).

| Command | What it does |
|---|---|
| `status [--slug S]` | Every series post: live / scheduled / hidden, review state, unpushed and uncommitted files, next steps. |
| `serve [--port 8766]` | Phone page. `/_desk/` is the desk; every other path is the preview site, unpublished and future posts included. Han reaches it at `https://hans-macbook-pro-3.taila13321.ts.net:8443/_desk/` (`tailscale serve --bg --https=8443 8766`, tailnet only, never Funnel). |
| `preview` | Rebuild the preview site into `.desk/site` (about 3 s). Han can also tap "Rebuild preview". |
| `receipts <slug>` | Every ```text block byte-exact against `llm-architectures-refresher/outputs/*.txt`, fence balance, and prose decimals no capture contains. |
| `feedback <slug> [--all]` | Han's open notes, with ids. |
| `addressed <slug> <id> "what changed"` | Marks a note handled. Do this after each revision. |
| `check <slug> <item>` | Exit 0 only if `post` or `linkedin` is approved at its current content. |

Items: `post` = the post file (its `published:` line is ignored, so flipping it on publish day does not void the
approval) plus its `assets/picture/...` folders; `linkedin` = `.desk/<slug>/linkedin.md`. `.desk/` is gitignored.

## The gate

- Approvals live in `.desk/<slug>/approvals.json`. **Only the desk writes them, when Han taps Approve.** Never
  write, copy or POST to them; `.claude/hooks/blog_gate.py` denies it.
- `git push` of the blog is **denied** while the demo repo has unpushed commits, while a series post in the
  pushed commits isn't approved at its current content, or has uncommitted edits. Otherwise it is a permission
  prompt naming what goes live and what stays hidden. Han answers it in the terminal or on his phone.
- `git push` of the demo repo is always a permission prompt (the demo repo's `.claude/settings.json` uses the
  same hook).
- Still true from before the desk: **push only when Han says push, each push its own ask** (the prompt is that
  ask), demo repo first, then verify the deploy and the live page.

## Agent workflow for a post

1. **Write or revise** in the main session, following the series memory
   (`~/.claude/projects/-Users-hanyu/memory/MEMORY.md`, the post's resume-point memory).
2. **Long demo runs** (hours): launch detached with `nohup caffeinate -i uv run demoNN ... > outputs/... 2> outputs/....err &`,
   watch with Monitor, and send Han a PushNotification when it finishes or fails. Recapture before re-checking.
3. **Review agents, in parallel**, before Han sees it: `receipt-auditor` and `first-reader` (both report only).
   Fix what they find in the main session; facts and numbers never change for style. Re-run `desk.py receipts`.
4. `desk.py preview`, then tell Han it's on the desk: `https://hans-macbook-pro-3.taila13321.ts.net:8443/_desk/p/<slug>`.
5. **Feedback loop:** read `desk.py feedback <slug>`. One-off edit: revise, re-check, `addressed`. A lasting rule
   (applies to future posts too): propose the memory or agent change to Han and apply it only when he agrees.
   His questions are defect reports about the post (see working-with-han-on-revisions).
6. **Approved:** commit. On the publish date, set `published: true` (or drop the key), commit, push the demo repo,
   then the blog (both prompt). Then verify the deploy and the live page (post-revision-checklist "After publishing").
7. **Promo:** `promo-writer` drafts `.desk/<slug>/linkedin.md`; Han approves it on the desk, posts by hand, and
   sends the URL. Record it in the post's memory.

## Files

- `scripts/desk.py`: the desk. `scripts/sync_post07.py`: post 7's anchor-based receipt and figure paster
  (`<!-- RECEIPT: id -->` / `<!-- FIGURE: name -->` markers); copy and adapt it for later posts.
- `../../hooks/blog_gate.py`: the gate. `../../agents/`: receipt-auditor, first-reader, promo-writer.

Gaps: the gate catches `git push`; a push through another tool (GitHub web UI, a different git client) isn't
caught. The permission prompt is the real lock.
