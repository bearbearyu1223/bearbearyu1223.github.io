#!/usr/bin/env python3
"""PreToolUse gate for the blog's reviewed posts (the LLM Architecture Refresh series and the posts registered in
publish.py's EXTRA_POSTS): nothing goes live without Han.

Lives in the blog repo, not in the research plugin, so disabling the plugin can't switch it off. It imports
publish.py from the plugin (research-content-assistant) and fails closed: if that import fails, every push and
every approval write is denied.

Used by the blog repo AND the demo repo (llm-architectures-refresher/.claude/settings.json points here).

- `git push` from the blog repo:
    * denied if the demo repo still has unpushed commits (push code first, so the post's links resolve);
    * denied if a pushed post's companion repo (publish.companions: honeyguide for the Honeyguide post) is
      missing, has no upstream, or has unpushed commits;
    * only the commits actually pushed count: `git push origin <commit>:main` checks @{u}..<commit>, and a post
      whose files differ between <commit> and HEAD is denied (the approval is of HEAD's version);
    * denied if a reviewed post (or its figures) in the pushed commits is not approved on the publish page at
      its current content, or has uncommitted edits (the approval would not match what gets pushed);
    * otherwise a permission prompt that says which posts go LIVE and which stay hidden.
- `git push` from the demo repo: always a permission prompt listing the commits.
- Force pushes, `published:` flips are not special-cased beyond the above: any push asks.
- Agents may not write .publish/*/approvals.json or call the publish page's approve/revoke endpoints.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

PUBLISH_SCRIPTS = Path(os.environ.get(
    "RESEARCH_PLUGIN_DIR", Path.home() / "Documents/GitHub/research-content-assistant/plugins/research"
)) / "skills/publish/scripts"
APPROVE_URL = re.compile(r"/_(desk|publish)/api/[\w-]+/\w+/(approve|revoke)")


def decide(decision: str, reason: str) -> None:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                             "permissionDecision": decision,
                                             "permissionDecisionReason": reason}}))
    sys.exit(0)


try:
    sys.path.insert(0, str(PUBLISH_SCRIPTS))
    import publish as desk  # noqa: E402
    BLOG, DEMO = desk.BLOG.resolve(), desk.DEMO.resolve()
except Exception as e:  # fail closed: without the publish state, nothing may be pushed or approved
    _raw = sys.stdin.read()
    if re.search(r"\bpush\b", _raw) or "approvals.json" in _raw or APPROVE_URL.search(_raw):
        decide("deny", f"blog gate: can't load publish.py from {PUBLISH_SCRIPTS} ({type(e).__name__}: {e}), "
                       "so pushes and approval writes are blocked. Restore the research-content-assistant "
                       "repo or set RESEARCH_PLUGIN_DIR.")
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
            rest = []
            for t in toks[toks.index("push") + 1:]:
                if t in ("|", "&", ">", ">>", "<") or t.startswith("|"):
                    break  # a pipe or redirection ends the push's own arguments
                if re.fullmatch(r"\d*[<>]+&?\d*\S*", t):
                    continue  # 2>&1, >/dev/null, >out.txt
                rest.append(t)
            out.append((toplevel(repo), rest))
    return out


PUSH_OPTS_WITH_VALUE = {"-o", "--push-option", "--repo", "--receive-pack", "--exec"}


def pushed_commit(args: list[str]) -> tuple[str, str]:
    """(commit sha, how it was named) that this push sends to the upstream branch.

    `git push` / `git push origin` send HEAD; `git push origin <src>:<dst>` or `git push origin <src>` send <src>.
    Several refspecs, --all, --mirror, --tags or an unresolvable <src> fall back to HEAD (checks everything
    unpushed, the strict reading).
    """
    if any(a in ("--all", "--mirror", "--tags") for a in args):
        return desk.git(BLOG, "rev-parse", "HEAD").strip(), "HEAD"
    pos, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in PUSH_OPTS_WITH_VALUE:
            skip = True
            continue
        if a.startswith("-"):
            continue
        pos.append(a)
    refspecs = pos[1:]
    if len(refspecs) != 1:
        return desk.git(BLOG, "rev-parse", "HEAD").strip(), "HEAD"
    src = refspecs[0].lstrip("+").split(":", 1)[0] or "HEAD"
    sha = desk.git(BLOG, "rev-parse", "--verify", "--quiet", f"{src}^{{commit}}").strip()
    if not sha:
        return desk.git(BLOG, "rev-parse", "HEAD").strip(), "HEAD"
    return sha, src


def gate_blog_push(args: list[str]) -> None:
    demo_left = desk.unpushed(DEMO)
    if demo_left:
        decide("deny", f"blog gate: the demo repo has {len(demo_left)} unpushed commit(s). Push "
                       "llm-architectures-refresher first so the post's code links and receipts resolve.")
    sha, named = pushed_commit(args)
    head = desk.git(BLOG, "rev-parse", "HEAD").strip()
    partial = sha != head
    if partial and subprocess.run(["git", "-C", str(BLOG), "merge-base", "--is-ancestor", sha, "HEAD"],
                                  capture_output=True).returncode != 0:
        decide("deny", f"blog gate: {named} is not an ancestor of HEAD; push from the branch it is on.")
    changed = set(desk.git(BLOG, "diff", "--name-only", f"@{{u}}..{sha}").splitlines())
    if not changed:
        decide("ask", "git push (blog): nothing new to push according to the upstream; allow?")
    going_live, hidden, problems = [], [], []
    for slug in desk.slugs():
        files = {str(f.relative_to(BLOG)) for f in desk.item_files(slug, "post")}
        if not files & changed:
            continue
        st = desk.post_state(slug)
        state, h = desk.approval_state(slug, "post")
        for repo in desk.companions(slug):
            if repo.resolve() == DEMO:
                continue  # checked above, for every blog push
            if not (repo / ".git").exists():
                problems.append(f"{slug}'s companion repo {repo} is missing")
            elif not desk.git(repo, "rev-parse", "--abbrev-ref", "@{u}"):
                problems.append(f"{slug}'s companion repo {repo.name} has no upstream; push it first")
            elif desk.unpushed(repo):
                problems.append(f"{slug}'s companion repo {repo.name} has {len(desk.unpushed(repo))} unpushed "
                                "commit(s); push it first so the post's code links and receipts resolve")
        if partial and desk.git(BLOG, "diff", "--name-only", sha, "HEAD", "--", *sorted(files)).strip():
            problems.append(f"{slug} differs between the pushed commit {named} and HEAD; the approval is of "
                            "HEAD's version, so push HEAD or approve what is pushed")
        elif st["dirty"]:
            problems.append(f"{slug} has uncommitted edits ({len(st['dirty'])} file(s)); commit them so "
                            "what Han approved is what gets pushed")
        elif state != "approved":
            why = {"pending": "hasn't been reviewed", "changed": "changed since Han approved it"}.get(state, state)
            problems.append(f"{slug} {why}")
        committed = desk.front(desk.git(BLOG, "show", f"{sha}:{st['file']}"))
        (going_live if desk.visible(committed) else hidden).append(f"{st.get('label', '[' + st['num'] + ']')} {slug}" + (f" ({h[:10]})" if h else ""))
    if problems:
        decide("deny", "blog gate: " + "; ".join(problems) + ". Put it on the publish page "
                       "(publish.py serve, /_publish/p/<slug>) and wait for Han's Approve; see `publish.py feedback <slug>`.")
    n = len(desk.git(BLOG, "rev-list", f"@{{u}}..{sha}").split())
    msg = [f"Push {n} commit(s) to the blog" + (f" (up to {named}; {len(desk.unpushed(BLOG)) - n} later "
           "commit(s) stay local)" if partial else "") + ". GitHub Pages deploys on push."]
    if going_live:
        msg.append("Goes LIVE: " + ", ".join(going_live) + ".")
    if hidden:
        msg.append("Pushed but hidden (published: false or future date): " + ", ".join(hidden) + ".")
    if "--force" in args or "-f" in args or any(a.startswith("--force") for a in args):
        msg.append("This is a FORCE push.")
    decide("ask", " ".join(msg))


def gate_bash(cmd: str, cwd: str) -> None:
    if "approvals.json" in cmd or APPROVE_URL.search(cmd):
        decide("deny", "blog gate: approvals are written only by the publish page, when Han taps Approve.")
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
        if path.endswith("approvals.json") and ("/.publish/" in path or "/.desk/" in path):
            decide("deny", "blog gate: approvals are written only by the publish page, when Han taps Approve.")
    sys.exit(0)


if __name__ == "__main__":
    main()
