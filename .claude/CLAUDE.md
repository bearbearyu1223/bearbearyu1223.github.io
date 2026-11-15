# bearbearyu1223.github.io

Han's Jekyll (Chirpy) blog. Pushing `main` deploys the live site through GitHub Pages.

- **LLM Architecture Refresh series** (`_posts/*-llm-architectures-*.md`, code in
  `~/Documents/GitHub/llm-architectures-refresher`): read `~/.claude/projects/-Users-hanyu/memory/MEMORY.md`
  first. All series memories live there, not in this project's memory folder. Status and plans are in
  `~/Documents/GitHub/research-content-assistant/plans/`; the map of where every kind of context lives is that
  repo's `CONTEXT.md`.
- **Review and publishing go through `/research:publish`**, from the `research` plugin in
  `~/Documents/GitHub/research-content-assistant` (enabled in `.claude/settings.json`; its agents are
  `research:auditor`, `research:first-reader`, `research:promo-writer`). Han approves each post on his phone
  (`/_publish/`) before it is pushed, and `.claude/hooks/blog_gate.py` enforces it. The gate stays in this
  repo on purpose and blocks every push if it can't load the plugin's `publish.py`.
- Commit freely; push only when Han asks, demo repo first.
- Never embed Pepper & Carrot course videos or YouTube links in posts here.
- Preview builds need both `--unpublished` and `--future`.
