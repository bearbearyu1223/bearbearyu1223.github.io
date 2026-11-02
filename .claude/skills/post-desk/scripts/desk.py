#!/usr/bin/env python3
"""Post desk: Han's review and approval gate for the LLM Architecture Refresh series.

Stdlib only, so the PreToolUse hook can import it with plain python3.

  desk.py status [--slug S]            every series post: live / hidden / date, review state, unpushed commits
  desk.py serve [--port 8766]          phone page on 127.0.0.1: /_desk/ for review, everything else is the preview site
                                       (tailnet: tailscale serve --bg --https=8443 8766)
  desk.py preview                      rebuild the preview site (unpublished + future posts) into .desk/site
  desk.py receipts <slug>              byte-exact check of every ```text block against the demo captures,
                                       plus decimals in the prose that no capture contains
  desk.py feedback <slug> [--all]      Han's open "Request changes" notes
  desk.py addressed <slug> <id> MSG    mark a note handled, saying what changed
  desk.py check <slug> <item>          exit 0 only if the item is approved at its current content

Items per post: `post` (the post file, ignoring its `published:` line, plus its asset folders) and
`linkedin` (.desk/<slug>/linkedin.md). Approvals live in .desk/<slug>/approvals.json and are written
ONLY by the review page when Han taps Approve. .claude/hooks/blog_gate.py enforces it on `git push`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

BLOG = Path(__file__).resolve().parents[4]
DEMO = Path(os.environ.get("LLM_DEMO_REPO", BLOG.parent / "llm-architectures-refresher"))
POSTS = BLOG / "_posts"
DESK = BLOG / ".desk"
SITE = DESK / "site"
SERIES_GLOB = "*-llm-architectures-*.md"
ITEMS = ("post", "linkedin")
LABEL = {"post": "Blog post", "linkedin": "LinkedIn"}
LIVE_BASE = "https://bearbearyu1223.github.io/posts/"
PROMO_FROM = "2026-11-01"  # promos for posts before the desk existed are not nagged about


# ---------------------------------------------------------------- posts and files

def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def post_files() -> dict[str, Path]:
    out = {}
    for p in sorted(POSTS.glob(SERIES_GLOB)):
        out[re.sub(r"^\d{4}-\d{2}-\d{2}-", "", p.stem)] = p
    return out


def slugs() -> list[str]:
    return list(post_files())


def front(text: str) -> dict[str, str]:
    m = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    out = {}
    for line in (m.group(1).splitlines() if m else []):
        k = re.match(r"^(\w+):\s*(.*)$", line)
        if k:
            out[k.group(1)] = re.sub(r"\s+#.*$", "", k.group(2)).strip().strip('"')
    return out


def asset_dirs(text: str) -> list[Path]:
    names = sorted(set(re.findall(r"/assets/picture/([^/)\s\"']+)/", text)))
    return [BLOG / "assets/picture" / n for n in names if (BLOG / "assets/picture" / n).is_dir()]


def item_files(slug: str, item: str) -> list[Path]:
    if item == "linkedin":
        p = DESK / slug / "linkedin.md"
        return [p] if p.is_file() else []
    post = post_files().get(slug)
    if not post:
        return []
    files = [post]
    for d in asset_dirs(post.read_text(encoding="utf-8")):
        files += sorted(f for f in d.rglob("*") if f.is_file() and f.name != ".DS_Store")
    return files


def _content(p: Path) -> bytes:
    data = p.read_bytes()
    if p.parent == POSTS:  # flipping `published:` doesn't void Han's approval of the content
        data = re.sub(rb"(?m)^published:.*\n", b"", data, count=1)
    return data


def item_hash(slug: str, item: str) -> str | None:
    files = item_files(slug, item)
    if not files:
        return None
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(BLOG)}\0{hashlib.sha256(_content(p)).hexdigest()}\n".encode())
    return h.hexdigest()


def load_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def approvals(slug: str) -> dict:
    return load_json(DESK / slug / "approvals.json", {})


def approval_state(slug: str, item: str) -> tuple[str, str | None]:
    """('approved' | 'changed' | 'pending' | 'none', current hash)."""
    cur = item_hash(slug, item)
    if cur is None:
        return "none", None
    a = approvals(slug).get(item)
    if not a:
        return "pending", cur
    return ("approved" if a.get("hash") == cur else "changed"), cur


def feedback(slug: str) -> list[dict]:
    p = DESK / slug / "feedback.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = {r["ref"]: r for r in rows if r.get("kind") == "addressed"}
    return [dict(r, addressed=done.get(r["id"])) for r in rows if r.get("kind") == "changes"]


def append_feedback(slug: str, row: dict) -> dict:
    d = DESK / slug
    d.mkdir(parents=True, exist_ok=True)
    row = {"id": uuid.uuid4().hex[:8], "at": now(), **row}
    with open(d / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def write_approval(slug: str, item: str, h: str | None) -> None:
    d = DESK / slug
    d.mkdir(parents=True, exist_ok=True)
    a = approvals(slug)
    if h is None:
        a.pop(item, None)
    else:
        a[item] = {"hash": h, "at": now(), "files": [str(p.relative_to(BLOG)) for p in item_files(slug, item)]}
    (d / "approvals.json").write_text(json.dumps(a, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- git and publish state

def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def unpushed(repo: Path, *paths: str) -> list[str]:
    out = git(repo, "log", "--oneline", "@{u}..HEAD", *(["--", *paths] if paths else []))
    return out.splitlines() if out else []


def dirty(repo: Path, *paths: str) -> list[str]:
    out = git(repo, "status", "--porcelain", *(["--", *paths] if paths else []))
    return [l for l in out.splitlines() if not l.endswith(".DS_Store")] if out else []


def post_date(fm: dict) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(fm.get("date", "")[:25], "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        try:
            return dt.datetime.strptime(fm.get("date", "")[:10], "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            return None


def visible(fm: dict, at: dt.datetime | None = None) -> bool:
    """Would a production build include this post right now?"""
    if fm.get("published", "true").lower() == "false":
        return False
    d = post_date(fm)
    return d is None or d <= (at or dt.datetime.now(dt.timezone.utc))


def post_state(slug: str) -> dict:
    p = post_files()[slug]
    text = p.read_text(encoding="utf-8")
    fm = front(text)
    rel = str(p.relative_to(BLOG))
    on_remote = bool(git(BLOG, "cat-file", "-t", f"@{{u}}:{rel}"))
    remote_fm = front(git(BLOG, "show", f"@{{u}}:{rel}")) if on_remote else {}
    live = on_remote and visible(remote_fm)
    files = [str(f.relative_to(BLOG)) for f in item_files(slug, "post")]
    num = re.search(r"\[(\d+)\]", fm.get("title", ""))
    return {
        "slug": slug, "file": rel, "title": fm.get("title", slug), "num": num.group(1) if num else "?",
        "date": fm.get("date", ""), "published": fm.get("published", "true"),
        "hidden_locally": not visible(fm, dt.datetime.max.replace(tzinfo=dt.timezone.utc)),
        "future": (post_date(fm) or dt.datetime.min.replace(tzinfo=dt.timezone.utc)) > dt.datetime.now(dt.timezone.utc),
        "live": live, "unpushed": unpushed(BLOG, *files), "dirty": dirty(BLOG, *files),
    }


def next_steps(slug: str, st: dict) -> list[str]:
    s = []
    open_fb = [f for f in feedback(slug) if not f.get("addressed")]
    if open_fb:
        s.append(f"{len(open_fb)} open feedback note(s) for the agents")
    pstate = approval_state(slug, "post")[0]
    if not st["live"]:
        if pstate in ("pending", "changed"):
            s.append("Post: review on the desk" + (" (changed since approval)" if pstate == "changed" else ""))
        elif pstate == "approved":
            if st["hidden_locally"]:
                s.append("Post approved: on its date, set published: true, commit, push demo repo then blog")
            elif st["dirty"]:
                s.append("Post approved: commit it, then push")
            else:
                s.append("Post approved: push demo repo, then blog")
    elif st["unpushed"] or st["dirty"]:
        s.append("Live post has local changes: review, then push")
    li = approval_state(slug, "linkedin")[0]
    if st["live"] and li == "none" and st["date"][:10] >= PROMO_FROM:
        s.append("LinkedIn: draft a promo (promo-writer agent)")
    elif li in ("pending", "changed"):
        s.append("LinkedIn: review the promo on the desk")
    elif li == "approved":
        s.append("LinkedIn: approved, post it by hand and send me the URL")
    return s


def cmd_status(a) -> int:
    for slug in ([a.slug] if a.slug else slugs()):
        st = post_state(slug)
        where = ("✅ live" if st["live"] else "🗓 future" if st["future"] and not st["hidden_locally"]
                 else "🙈 hidden")
        settled = st["live"] and not st["unpushed"] and not st["dirty"]
        gates = " ".join(f"{LABEL[i]} {'·' if (i == 'post' and settled) else GATE[approval_state(slug, i)[0]] or '·'}"
                         for i in ITEMS)
        extra = []
        if st["unpushed"]:
            extra.append(f"{len(st['unpushed'])} unpushed")
        if st["dirty"]:
            extra.append(f"{len(st['dirty'])} uncommitted")
        print(f"[{st['num']}] {slug}: {where} · {st['date'][:16]} · {gates}" + (f" · {', '.join(extra)}" if extra else ""))
        for n in next_steps(slug, st):
            print(f"    → {n}")
    d = unpushed(DEMO)
    if d:
        print(f"\ndemo repo: {len(d)} unpushed commit(s), push it before the blog")
    print("\nlegend: review ✔ approved ⚠ changed since approval ? not reviewed · nothing to review")
    return 0


GATE = {"approved": "✔", "changed": "⚠", "pending": "?", "none": ""}


def cmd_feedback(a) -> int:
    rows = feedback(a.slug) if a.all else [r for r in feedback(a.slug) if not r.get("addressed")]
    if not rows:
        print("no open feedback")
    for r in rows:
        tag = f" [addressed: {r['addressed']['text']}]" if r.get("addressed") else ""
        print(f"{r['id']}  {r['at']}  {r['item']}: {r['text']}{tag}")
    return 0


def cmd_addressed(a) -> int:
    if not any(r["id"] == a.id for r in feedback(a.slug)):
        print(f"no feedback {a.id} for {a.slug}", file=sys.stderr)
        return 1
    append_feedback(a.slug, {"kind": "addressed", "ref": a.id, "text": a.message})
    print("recorded")
    return 0


def cmd_check(a) -> int:
    state, h = approval_state(a.slug, a.item)
    print(f"{a.slug} {a.item}: {state}" + (f" ({h[:12]})" if h else ""))
    return 0 if state == "approved" else 1


# ---------------------------------------------------------------- receipts

FENCE = re.compile(r"^```text[^\n]*\n(.*?)^```", re.M | re.S)


def receipts(slug: str) -> dict:
    """Every non-blank line of every ```text block must be a whole line of some demo capture."""
    text = post_files()[slug].read_text(encoding="utf-8")
    caps = sorted((DEMO / "outputs").glob("*.txt"))
    lines: set[str] = set()
    for c in caps:
        lines.update(c.read_text(encoding="utf-8", errors="replace").splitlines())
    bad = []
    blocks = list(FENCE.finditer(text))
    for i, m in enumerate(blocks, 1):
        miss = [l for l in m.group(1).splitlines() if l.strip() and l not in lines]
        if miss:
            lineno = text[:m.start()].count("\n") + 1
            bad.append({"block": i, "line": lineno, "missing": miss[:3], "n": len(miss)})
    prose = FENCE.sub("", re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S))
    prose = re.sub(r"^\|.*$|^#.*$|\$\$.*?\$\$|\$[^$\n]+\$", "", prose, flags=re.M | re.S)
    blob = "\n".join(lines)
    decimals = sorted(set(d for d in re.findall(r"(?<![\w.])\d+\.\d{2,4}(?![\w.])", prose) if d not in blob))
    return {"blocks": len(blocks), "bad": bad, "untraced": decimals,
            "fences_balanced": text.count("```") % 2 == 0, "captures": [c.name for c in caps]}


def cmd_receipts(a) -> int:
    r = receipts(a.slug)
    ok = r["blocks"]
    print(f"{a.slug}: {ok - len(r['bad'])}/{ok} text blocks byte-exact against {len(r['captures'])} captures"
          + ("" if r["fences_balanced"] else "  ⚠ UNBALANCED ``` FENCES"))
    for b in r["bad"]:
        print(f"  ✗ block {b['block']} (post line {b['line']}): {b['n']} line(s) not in any capture, e.g.")
        for l in b["missing"]:
            print(f"      {l!r}")
    if r["untraced"]:
        print(f"  prose decimals not found in any capture (trace each to another post or a paper): "
              + ", ".join(r["untraced"]))
    return 0 if not r["bad"] and r["fences_balanced"] else 1


# ---------------------------------------------------------------- preview build

def build_preview() -> tuple[bool, str]:
    env = dict(os.environ, JEKYLL_ENV="development")
    r = subprocess.run(["bundle", "exec", "jekyll", "build", "--unpublished", "--future", "--quiet",
                        "-d", str(SITE)], cwd=BLOG, env=env, capture_output=True, text=True)
    stamp = DESK / "site.built"
    if r.returncode == 0:
        DESK.mkdir(exist_ok=True)
        stamp.write_text(now())
    return r.returncode == 0, (r.stdout + r.stderr)[-2000:]


def cmd_preview(a) -> int:
    ok, log = build_preview()
    print("preview built into .desk/site" if ok else "preview build FAILED\n" + log)
    return 0 if ok else 1


def preview_age(slug: str) -> str:
    stamp = DESK / "site.built"
    if not stamp.exists():
        return "never built"
    built = stamp.stat().st_mtime
    newest = max((f.stat().st_mtime for f in item_files(slug, "post")), default=0)
    return "up to date" if built >= newest else "OLDER than the post: rebuild"


# ---------------------------------------------------------------- review page

CSS = """
:root{--ink:#1d1b26;--mut:#6b6880;--bg:#f6f4ef;--card:#fff;--ok:#1f7a4d;--warn:#b5651d;--acc:#3a6ea5}
*{box-sizing:border-box}body{margin:0;font:16px/1.45 -apple-system,system-ui,sans-serif;color:var(--ink);background:var(--bg)}
header{position:sticky;top:0;background:var(--ink);color:#fff;padding:12px 16px;z-index:5}
header a{color:#fff;text-decoration:none}main{padding:12px;max-width:760px;margin:auto}
.card{background:var(--card);border-radius:14px;padding:14px;margin:12px 0;box-shadow:0 1px 3px #0001}
h2{margin:0 0 8px;font-size:19px}h3{margin:14px 0 6px;font-size:15px;color:var(--mut)}
.badge{display:inline-block;font-size:13px;padding:2px 8px;border-radius:99px;background:#eee;margin-left:6px}
.approved{background:#dff3e8;color:var(--ok)}.changed{background:#fde9d6;color:var(--warn)}.pending{background:#e6eef7}
pre{white-space:pre-wrap;word-wrap:break-word;background:#faf9f6;border:1px solid #eee;border-radius:8px;padding:10px;font:14px/1.45 ui-monospace,Menlo,monospace;max-height:60vh;overflow:auto}
textarea{width:100%;min-height:90px;font:16px system-ui;border-radius:8px;border:1px solid #ccc;padding:8px}
button,.btn{font:600 16px system-ui;border:0;border-radius:10px;padding:12px 16px;margin:6px 6px 0 0;display:inline-block;text-decoration:none}
.ok{background:var(--ok);color:#fff}.chg{background:#e8752a;color:#fff}.ghost{background:#eee;color:var(--ink)}.go{background:var(--acc);color:#fff}
.row{display:flex;flex-wrap:wrap;align-items:center}.mut{color:var(--mut);font-size:13px}
.fb{border-left:3px solid #e8752a;padding:4px 8px;margin:6px 0}.fb.done{border-color:var(--ok);opacity:.7}
a.post{display:block;color:inherit;text-decoration:none}li{margin:2px 0}.bad{color:#b3261e}
"""


def esc(s) -> str:
    return html.escape(str(s))


def page(title: str, body: str) -> bytes:
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><style>{CSS}</style></head><body><header><a href="/_desk/">◀ Post desk</a> · {esc(title)}</header>'
            f"<main>{body}</main></body></html>").encode()


def index_html() -> bytes:
    rows = []
    for slug in reversed(slugs()):
        st = post_state(slug)
        where = "live" if st["live"] else "scheduled " + st["date"][:10] if st["future"] else "hidden"
        steps = next_steps(slug, st)
        rows.append(f'<a class="post card" href="/_desk/p/{slug}"><h2>[{st["num"]}] {esc(slug.replace("llm-architectures-", ""))}'
                    f'<span class="badge">{esc(where)}</span></h2><div class="mut">{esc(st["title"])}</div>'
                    + ("<ul>" + "".join(f"<li>{esc(s)}</li>" for s in steps) + "</ul>" if steps else "") + "</a>")
    d = unpushed(DEMO)
    note = f'<p class="mut">Demo repo: {len(d)} unpushed commit(s).</p>' if d else ""
    return page("LLM Architecture Refresh", note + "".join(rows))


def post_html(slug: str) -> bytes:
    st = post_state(slug)
    fb = feedback(slug)
    r = receipts(slug)
    cards = []
    head = (f'<section class="card"><h2>{esc(st["title"])}</h2>'
            f'<div class="mut">{esc(st["file"])} · date {esc(st["date"])} · published: {esc(st["published"])}'
            f'{" · LIVE" if st["live"] else ""}</div>'
            f'<div class="row"><a class="btn go" href="/posts/{slug}/" target="_blank">Read the preview ↗</a>'
            f'<form method="post" action="/_desk/api/{slug}/post/rebuild"><button class="ghost">Rebuild preview</button></form></div>'
            f'<p class="mut">Preview: {esc(preview_age(slug))}'
            + (f' · live page: <a href="{LIVE_BASE}{slug}/">{LIVE_BASE}{slug}/</a>' if st["live"] else "") + "</p>"
            f'<h3>Receipts</h3><p>{r["blocks"] - len(r["bad"])}/{r["blocks"]} output blocks match the demo captures'
            + ("" if not r["bad"] else f' <span class="bad">· {len(r["bad"])} do not</span>')
            + ("" if r["fences_balanced"] else ' <span class="bad">· unbalanced fences</span>') + "</p>"
            + (f'<p class="mut">Unpushed commits:</p><pre>{esc(chr(10).join(st["unpushed"]))}</pre>' if st["unpushed"] else "")
            + (f'<p class="mut">Uncommitted:</p><pre>{esc(chr(10).join(st["dirty"]))}</pre>' if st["dirty"] else "")
            + "</section>")
    cards.append(head)
    for item in ITEMS:
        state, h = approval_state(slug, item)
        if state == "none":
            continue
        badge = {"approved": "approved", "changed": "changed since approval", "pending": "needs review"}[state]
        notes = "".join(
            f'<div class="fb{" done" if f.get("addressed") else ""}"><span class="mut">{esc(f["at"][:16])}</span> {esc(f["text"])}'
            + (f'<br><span class="mut">✔ {esc(f["addressed"]["text"])}</span>' if f.get("addressed") else "") + "</div>"
            for f in fb if f["item"] == item)
        body = ""
        if item == "linkedin":
            txt = re.sub(r"\A---\n.*?\n---\n", "", (DESK / slug / "linkedin.md").read_text(encoding="utf-8"), flags=re.S)
            body = f"<pre>{esc(txt)}</pre>"
        else:
            body = '<p class="mut">Read the preview above, then approve or request changes. Flipping published: later does not void the approval; any other edit does.</p>'
        act = f"/_desk/api/{slug}/{item}"
        form = (f'<form method="post" action="{act}/feedback"><textarea name="text" placeholder="What should change? Quote the passage if you can."></textarea>'
                f'<div class="row"><button class="chg">Request changes</button></div></form>'
                f'<form method="post" action="{act}/approve" class="row"><input type="hidden" name="hash" value="{h}">'
                f'<button class="ok">Approve this version</button>'
                + (f'<button class="ghost" formaction="{act}/revoke">Revoke</button>' if state in ("approved", "changed") else "")
                + "</form>")
        cards.append(f'<section class="card" id="{item}"><h2>{LABEL[item]}<span class="badge {state}">{esc(badge)}</span></h2>'
                     f'<div class="mut">version {h[:10]}</div>{body}{"<h3>Feedback</h3>" + notes if notes else ""}{form}</section>')
    return page(f"[{st['num']}] {slug.replace('llm-architectures-', '')}", "".join(cards))


class Handler(BaseHTTPRequestHandler):
    server_version = "PostDesk/1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (now(), fmt % args))

    def send(self, code: int, body: bytes, ctype="text/html; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/_desk", "/_desk/"):
            return self.send(200, index_html())
        m = re.fullmatch(r"/_desk/p/([\w-]+)", path)
        if m and m.group(1) in slugs():
            return self.send(200, post_html(m.group(1)))
        if path.startswith("/_desk"):
            return self.send(404, b"not found", "text/plain")
        return self.static(path)

    def static(self, path: str):
        p = (SITE / path.lstrip("/")).resolve()
        if p.is_dir():
            p = p / "index.html"
        if not p.is_relative_to(SITE.resolve()) or not p.is_file():
            if path == "/":
                return self.send(302, b"", extra={"Location": "/_desk/"})
            return self.send(404, b"not found (rebuild the preview?)", "text/plain")
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript", ".json": "application/json",
                 ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".svg": "image/svg+xml",
                 ".webp": "image/webp", ".ico": "image/x-icon", ".woff2": "font/woff2", ".woff": "font/woff",
                 ".xml": "application/xml", ".txt": "text/plain; charset=utf-8", ".webmanifest": "application/manifest+json"
                 }.get(p.suffix.lower(), "application/octet-stream")
        self.send(200, p.read_bytes(), ctype)

    def same_origin(self) -> bool:
        origin = self.headers.get("Origin") or self.headers.get("Referer") or ""
        host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host") or ""
        return bool(origin) and urlparse(origin).netloc == host

    def do_POST(self):
        m = re.fullmatch(r"/_desk/api/([\w-]+)/(\w+)/(approve|feedback|revoke|rebuild)", urlparse(self.path).path)
        if not m or m.group(1) not in slugs() or m.group(2) not in ITEMS:
            return self.send(404, b"not found", "text/plain")
        if not self.same_origin():
            return self.send(403, b"cross-origin request refused", "text/plain")
        slug, item, action = m.groups()
        n = int(self.headers.get("Content-Length") or 0)
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(n).decode()).items()}
        state, cur = approval_state(slug, item)
        if action == "rebuild":
            ok, log = build_preview()
            if not ok:
                return self.send(500, page("Build failed", f"<pre>{esc(log)}</pre>"))
        elif action == "approve":
            if not cur or form.get("hash") != cur:
                return self.send(409, page("Changed", "<p>This changed after you opened the page. "
                                                      f'<a href="/_desk/p/{slug}">Reload</a> and review the new version.</p>'))
            write_approval(slug, item, cur)
            append_feedback(slug, {"kind": "approve", "item": item, "hash": cur, "text": "approved"})
        elif action == "revoke":
            write_approval(slug, item, None)
            append_feedback(slug, {"kind": "revoke", "item": item, "text": "approval revoked"})
        else:
            text = form.get("text", "").strip()
            if text:
                append_feedback(slug, {"kind": "changes", "item": item, "hash": cur, "text": text})
        self.send(303, b"", extra={"Location": f"/_desk/p/{slug}#{item}"})


def cmd_serve(a) -> int:
    if not (DESK / "site.built").exists():
        build_preview()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"post desk on http://127.0.0.1:{a.port}/_desk/  (phone: tailscale serve --bg --https=8443 {a.port})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--slug")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8766)
    sub.add_parser("preview")
    s = sub.add_parser("receipts")
    s.add_argument("slug")
    s = sub.add_parser("feedback")
    s.add_argument("slug")
    s.add_argument("--all", action="store_true")
    s = sub.add_parser("addressed")
    s.add_argument("slug")
    s.add_argument("id")
    s.add_argument("message")
    s = sub.add_parser("check")
    s.add_argument("slug")
    s.add_argument("item", choices=ITEMS)
    a = ap.parse_args()
    return {"status": cmd_status, "serve": cmd_serve, "preview": cmd_preview, "receipts": cmd_receipts,
            "feedback": cmd_feedback, "addressed": cmd_addressed, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
