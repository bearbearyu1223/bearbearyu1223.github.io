#!/usr/bin/env python3
"""PreToolUse gate for the LLM Architecture Refresh series: nothing goes live without Han.

Used by the blog repo AND the demo repo (llm-architectures-refresher/.claude/settings.json points here).

- `git push` from the blog repo:
    * denied if the demo repo still has unpushed commits (push code first, so the post's links resolve);
    * denied if a series post (or its figures) in the pushed commits is not approved on the post desk at
      its current content, or has uncommitted edits (the approval would not match what gets pushed);
    * otherwise a permission prompt that says which posts go LIVE and which stay hidden.
- `git push` from the demo repo: always a permission prompt listing the commits.
- Force pushes, `published:` flips are not special-cased beyond the above: any push asks.
- Agents may not write .desk/*/approvals.json or call the desk's approve/revoke endpoints.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/post-desk/scripts"))
import desk  # noqa: E402

BLOG, DEMO = desk.BLOG.resolve(), desk.DEMO.resolve()


def decide(decision: str, reason: str) -> None:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                             "permissionDecision": decision,
                                             "permissionDecisionReason": reason}}))
    sys.exit(0)


def toplevel(path: str) -> Path | None:
    r = subprocess.run(["git", "-C", path, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    return Path(r.stdout.strip()).resolve() if r.returncode == 0 else None


def push_targets(cmd: str, cwd: str) -> list[tuple[Path | None, list[str]]]:
    """(repo, push args) for each `git ... push` in the command, following `cd X &&` and `git -C X`."""
    out = []
    here = cwd
    for part in re.split(r"&&|\|\||;|\n", cmd):
        try:
            toks = shlex.split(part)
        except ValueError:
            toks = part.split()
        if not toks:
            continue
        if toks[0] == "cd" and len(toks) > 1:
            p = Path(toks[1]).expanduser()
            here = str(p if p.is_absolute() else Path(here) / p)
            continue
        if "git" in toks and "push" in toks:
            i = toks.index("git")
            repo = here
            if "-C" in toks[i:]:
                j = toks.index("-C", i)
                p = Path(toks[j + 1]).expanduser()
                repo = str(p if p.is_absolute() else Path(here) / p)
            out.append((toplevel(repo), toks[toks.index("push") + 1:]))
    return out


def gate_blog_push(args: list[str]) -> None:
    demo_left = desk.unpushed(DEMO)
    if demo_left:
        decide("deny", f"blog gate: the demo repo has {len(demo_left)} unpushed commit(s). Push "
                       "llm-architectures-refresher first so the post's code links and receipts resolve.")
    changed = set(desk.git(BLOG, "diff", "--name-only", "@{u}..HEAD").splitlines())
    if not changed:
        decide("ask", "git push (blog): nothing new to push according to the upstream; allow?")
    going_live, hidden, problems = [], [], []
    for slug in desk.slugs():
        files = {str(f.relative_to(BLOG)) for f in desk.item_files(slug, "post")}
        if not files & changed:
            continue
        st = desk.post_state(slug)
        state, h = desk.approval_state(slug, "post")
        if st["dirty"]:
            problems.append(f"{slug} has uncommitted edits ({len(st['dirty'])} file(s)); commit them so "
                            "what Han approved is what gets pushed")
        elif state != "approved":
            why = {"pending": "hasn't been reviewed", "changed": "changed since Han approved it"}.get(state, state)
            problems.append(f"{slug} {why}")
        committed = desk.front(desk.git(BLOG, "show", f"HEAD:{st['file']}"))
        (going_live if desk.visible(committed) else hidden).append(f"[{st['num']}] {slug}" + (f" ({h[:10]})" if h else ""))
    if problems:
        decide("deny", "blog gate: " + "; ".join(problems) + ". Put it on the post desk "
                       "(desk.py serve, /_desk/p/<slug>) and wait for Han's Approve; see `desk.py feedback <slug>`.")
    n = len(desk.unpushed(BLOG))
    msg = [f"Push {n} commit(s) to the blog. GitHub Pages deploys on push."]
    if going_live:
        msg.append("Goes LIVE: " + ", ".join(going_live) + ".")
    if hidden:
        msg.append("Pushed but hidden (published: false or future date): " + ", ".join(hidden) + ".")
    if "--force" in args or "-f" in args or any(a.startswith("--force") for a in args):
        msg.append("This is a FORCE push.")
    decide("ask", " ".join(msg))


def gate_bash(cmd: str, cwd: str) -> None:
    if "approvals.json" in cmd or re.search(r"/_desk/api/[\w-]+/\w+/(approve|revoke)", cmd):
        decide("deny", "blog gate: approvals are written only by the post desk, when Han taps Approve.")
    for repo, args in push_targets(cmd, cwd):
        if repo == BLOG:
            gate_blog_push(args)
        elif repo == DEMO:
            log = desk.unpushed(DEMO)
            decide("ask", f"Push {len(log)} commit(s) to llm-architectures-refresher (public code): "
                          + "; ".join(l[:60] for l in log[:6]) + (" …" if len(log) > 6 else ""))
        else:
            decide("ask", f"git push from {repo or cwd}: allow?")


def main() -> None:
    try:
        ev = json.load(sys.stdin)
    except ValueError:
        sys.exit(0)
    tool, inp = ev.get("tool_name", ""), ev.get("tool_input") or {}
    if tool == "Bash":
        gate_bash(inp.get("command", ""), ev.get("cwd") or str(BLOG))
    elif tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = str(inp.get("file_path") or inp.get("notebook_path") or "")
        if path.endswith("approvals.json") and "/.desk/" in path:
            decide("deny", "blog gate: approvals are written only by the post desk, when Han taps Approve.")
    sys.exit(0)


if __name__ == "__main__":
    main()
