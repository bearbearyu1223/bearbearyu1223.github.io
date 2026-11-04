"""Paste post 7's receipts and figures from the demo's capture, by anchor.

Receipts: each <!-- RECEIPT: id --> marker is followed by a ```text fence holding
the lines of outputs/demo07.txt from the first line matching SPEC[id][0] to the
first later line matching SPEC[id][1], inclusive, byte-exact. Figures: each
<!-- FIGURE: name --> marker is followed by the light/dark image pair, sized from
the PNG, and the PNGs are copied into the blog's assets.
"""

import re
import shutil
import struct
import sys
from pathlib import Path

DEMO = Path.home() / "Documents/GitHub/llm-architectures-refresher"
BLOG = Path.home() / "Documents/GitHub/bearbearyu1223.github.io"
POST = BLOG / "_posts/2026-11-03-llm-architectures-multimodal.md"
FIGS = DEMO / "figures/07-multimodal"
ASSETS = BLOG / "assets/picture/2026-10-11-llm-architectures-multimodal"
URL = "/assets/picture/2026-10-11-llm-architectures-multimodal"

# id: (section header in the capture, first line regex, last line regex)
SPEC = {
    "s1": ("1. From pixels", r"^  step\s+result$", r"^     place of a token id\.$"),
    "s1b": ("1. From pixels", r"^\s+vision encoder  language model$", r"^  causal mask\s+no\s+yes$"),
    "s1c": ("1. From pixels", r"^  Change only the last position", r"changes; the language model's first word can't see the last\.$"),
    "s2a": ("2. The connector", r"^  step\s+shape$", r"^    without the merge"),
    "s2b": ("2. The connector", r"^  piece\s+positions", r"^  TOTAL"),
    "s3": ("3. Stage 1", r"^  trained: the connector only", r"^  time per step"),
    "s3r": ("3. Stage 1", r"^  Captions exactly right, of 120", r"^    held-out pairs"),
    "s3b": ("3. Stage 1", r"^  truth:", None),  # three truth/model pairs
    "s4a": ("4. Stage 2", r"^  batch ", r"^  connector \+ LoRA\s+stage 1"),
    "s4b": ("4. Stage 2", r"^  task\s+model\s+seen pairs", r"^\s+connector \+ LoRA\s+\d+.*[\d)]$", "last"),
    "s5": ("5. Is it looking", r"^  task\s+model  right  blank", r"^\s+connector \+ LoRA", "last"),
    "s6a": ("6. Pairs it never saw", r"^  never in a training picture", None, "table"),
    "sm": ("FILE:demo07-misses.txt", r"^  Wrong color answers", None, "table"),
    "s6b": ("6. Pairs it never saw", r"^  Binding on held-out pairs", r"^  connector \+ LoRA\s+\d+ of"),
    "s6c": ("6. Pairs it never saw", r"^  Some wrong answers", r"^$", "before"),
    "f1": ("FILE:demo07-followup.txt", r"^  Median of 5 seeds", r"^  Seed by seed, held-out pairs", "before"),
    "f2": ("FILE:demo07-followup.txt", r"^  Seed by seed, held-out pairs", None, "table"),
    "f3": ("FILE:demo07-followup.txt", r"^  read from", r"^  tested on"),
    "f4": ("FILE:demo07-followup.txt", r"^  How often each shape came", None, "table"),
    "f5": ("FILE:demo07-followup.txt", r"^  Wrong held-out binding answers, all 5", None, "table"),
    "f6": ("FILE:demo07-followup.txt", r"^  Of the wrong colors that were", None, "table"),
    "s7t": ("7. Text, a picture", r"^  input\s+positions", r"bought|can change it\.$"),
    "s8a": ("8. What an image costs", r"^  KV cache per token", r"^  caption: "),
    "s8b": ("8. What an image costs", r"^\s+49 tokens  196 tokens$", r"^  448 x 448 px"),
}

NEW_ALT = {
    "experiment": "The whole experiment in three columns: generated captions and questions, none containing a held-out pair; stage 1 trains the connector on captions, then three stage-2 runs of three seeds each; tests on seen and held-out pairs, each question asked with the right picture, a blank one, another scene's picture and no picture",
    "four-controls": "One question asked four ways: with the right picture, with a blank white picture, with another scene's picture, and with no picture at all, only the question's words",
    "held-out-split": "A grid of six colors by three shapes; fifteen pairs appear in training pictures and three, a green circle, a blue square and a red triangle, are held out for the test",
    "held-out-results": "For each question and each trained model, the score on pairs seen in training beside the score on held-out pairs, with a mark for the best control on the held-out questions",
    "training-loss": "Training loss on a log scale: stage 1 with 49 and with 196 image tokens, and stage 2 for the three kinds of run",
    "stage-two-results": "Stage 2 scores on pairs seen in training for three questions, each model asked with the right picture, a blank picture, another scene's picture and no picture at all",
    "paired-seeds": "Five seeds, each drawn as a line joining the same seed trained three ways, no stage 1, connector only and connector plus captions; left, binding on held-out pairs, medians 36, 71 and 69 percent; right, leftmost on held-out pairs, medians 62, 80 and 93 percent",
    "failures": "Four held-out test pictures the connector-only model answered wrong, each with its question, the right answer and what the model said; every wrong color is the color of another shape in the same picture",
}


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def section(capture: list[str], title: str) -> list[str]:
    if title.startswith("FILE:"):
        return (DEMO / "outputs" / title[5:]).read_text().splitlines()
    start = next(i for i, l in enumerate(capture) if l.startswith(title))
    end = next((i for i in range(start + 2, len(capture)) if re.match(r"^\d+\. |^Figures$", capture[i])),
               len(capture))
    return capture[start:end]


def extract(capture: list[str], spec) -> list[str]:
    title, first, last, *mode = spec
    lines = section(capture, title)
    i = next(k for k, l in enumerate(lines) if re.search(first, l))
    if mode and mode[0] == "table":  # through the end of the first table after `first`
        d = next(k for k in range(i, len(lines)) if lines[k].startswith("  ---"))
        j = next((k for k in range(d, len(lines)) if not lines[k].strip()), len(lines))
        return lines[i:j]
    if last is None:  # s3b: three truth/model pairs, each possibly wrapped
        out, pairs = [], 0
        for l in lines[i:]:
            if l.startswith("  truth:"):
                pairs += 1
                if pairs == 4:
                    break
            out.append(l)
        while out and (not out[-1].strip() or set(out[-1].strip()) == {"-"}):
            out.pop()  # trailing blanks, and the next section's rule
        return out
    if mode and mode[0] == "before":
        j = next(k for k in range(i + 2, len(lines)) if re.search(last, lines[k]))
        while j > i and not lines[j - 1].strip():
            j -= 1
        return lines[i:j]
    js = [k for k in range(i, len(lines)) if re.search(last, lines[k])]
    if mode and mode[0] == "last":
        # the last match inside the first contiguous block after `first`
        j = i
        for k in range(i, len(lines)):
            if not lines[k].strip():
                break
            if re.search(last, lines[k]):
                j = k
        return lines[i:j + 1]
    return lines[i:js[0] + 1]


def main() -> None:
    capture = (DEMO / "outputs/demo07.txt").read_text().splitlines()
    post = POST.read_text().splitlines()
    out: list[str] = []
    k = 0
    used = set()
    while k < len(post):
        line = post[k]
        out.append(line)
        k += 1
        m = re.match(r"<!-- RECEIPT: (\w+) -->", line)
        if m:
            rid = m.group(1)
            used.add(rid)
            block = extract(capture, SPEC[rid])
            if k < len(post) and post[k].startswith("```text"):
                k += 1
                while not post[k].startswith("```"):
                    k += 1
                k += 1
            out += ["```text", *block, "```"]
            continue
        m = re.match(r"<!-- FIGURE: ([\w-]+) -->", line)
        if m:
            name = m.group(1)
            alt = NEW_ALT.get(name)
            while k < len(post) and post[k].startswith("!["):
                alt = alt or re.match(r"!\[(.*?)\]\(", post[k]).group(1)
                k += 1
            ASSETS.mkdir(parents=True, exist_ok=True)
            for theme in ("light", "dark"):
                src = FIGS / f"{name}-{theme}.png"
                shutil.copy2(src, ASSETS / src.name)
            w, h = png_size(FIGS / f"{name}-light.png")
            hh = round(1000 * h / w)
            for theme in ("light", "dark"):
                out.append(f"![{alt}]({URL}/{name}-{theme}.png){{: .{theme} width=\"1000\" height=\"{hh}\" }}")
    POST.write_text("\n".join(out) + "\n")
    missing = set(SPEC) - used
    print("receipts pasted:", len(used), "unused specs:", sorted(missing) or "none")
    print("figures:", sorted(set(re.findall(r"<!-- FIGURE: ([\w-]+) -->", POST.read_text()))))


if __name__ == "__main__":
    main()
