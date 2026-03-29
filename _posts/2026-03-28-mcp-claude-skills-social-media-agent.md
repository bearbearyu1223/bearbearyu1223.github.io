---
title: "Building a Social Media Content Agent with MCP and Claude Skills"
date: 2026-03-28 00:00:00 -0800
categories: [MCP, Agent]
tags: [mcp, claude, xiaohongshu, content-creator, agent, claude-skills, claude-code]
description: >-
  An engineer's hands-on experiment connecting Claude to Xiaohongshu
  via MCP — and what it reveals about the future of AI-powered content creation.
pin: true
---

An engineer's hands-on experiment connecting Claude to 小红书 (Xiaohongshu/RedNote)
via MCP — and what it reveals about the future of AI-powered content creation.

---

## Table of Contents

1. [What is MCP?](#what-is-mcp)
2. [Why MCP Matters: The Tool Problem](#why-mcp-matters)
3. [Xiaohongshu MCP: A Real-World Example](#xiaohongshu-mcp)
4. [What are Claude Skills?](#what-are-claude-skills)
5. [Building the Agent: Project Walkthrough](#building-the-agent)
6. [Putting It All Together: The Full Workflow](#putting-it-all-together)
7. [What This Means for Content Creators](#what-this-means)
8. [Key Takeaways](#key-takeaways)

---

## What is MCP? {#what-is-mcp}

If you've been following the AI agent space, you've probably seen the acronym **MCP** appearing everywhere. But what does it actually mean, and why does it matter?

**MCP stands for Model Context Protocol** — an open standard introduced by Anthropic in late 2024 that defines how AI models like Claude can connect to external tools, data sources, and services.

Think of MCP as **USB for AI**. Just as USB gave computers a standard way to connect to any peripheral device — keyboards, mice, storage drives, cameras — MCP gives AI models a standard way to connect to any external capability. Before MCP, every AI integration was a custom one-off: bespoke API wrappers, hand-rolled tool definitions, fragile glue code. With MCP, any service that implements the protocol becomes instantly available to any MCP-compatible AI client.

### The Protocol in Plain English

At its core, MCP defines a simple client-server relationship:

```
┌─────────────────────────────────────────────────┐
│                  MCP Architecture               │
│                                                 │
│   ┌──────────┐         ┌──────────────────────┐ │
│   │  Claude  │ ◄─────► │    MCP Server        │ │
│   │ (client) │  MCP    │  (your tool/service) │ │
│   └──────────┘ protocol└──────────────────────┘ │
│                              │                  │
│                    ┌─────────┴──────────┐       │
│                    │  External World    │       │
│                    │  - APIs            │       │
│                    │  - Databases       │       │
│                    │  - Browser         │       │
│                    │  - File system     │       │
│                    └────────────────────┘       │
└─────────────────────────────────────────────────┘
```

The MCP server exposes a set of **tools** — named functions with defined inputs and outputs. Claude discovers these tools at runtime, understands what they do from their descriptions, and calls them when relevant to complete a task. The entire exchange happens over a standard transport (stdio for local servers, HTTP for remote ones).

### What an MCP Tool Looks Like

Each tool has three things: a name, a description (which Claude reads to understand when to use it), and a schema defining the inputs. Here's a simplified example of what a `publish_content` tool definition looks like:

```json
{
  "name": "publish_content",
  "description": "Publish an image+text post to Xiaohongshu (小红书).
                  Requires at least one image. Title must be ≤20 Chinese
                  characters. Returns publish status.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "title":      { "type": "string", "description": "Post title, ≤20 chars" },
      "content":    { "type": "string", "description": "Post body text" },
      "images":     { "type": "array",  "description": "Local image paths or URLs" },
      "tags":       { "type": "array",  "description": "Hashtag list without #" },
      "visibility": { "type": "string", "description": "公开可见 | 仅自己可见" }
    },
    "required": ["title", "content", "images"]
  }
}
```

Claude reads the description and schema, then knows: "when the user wants to post something to Xiaohongshu, I can call this tool with these parameters." No hard-coding required on the model side — it figures this out dynamically from the tool definition.

---

## Why MCP Matters: The Tool Problem {#why-mcp-matters}

To appreciate why MCP is significant, it helps to understand the problem it solves.

### Before MCP: Every Integration Was Custom

Before MCP, connecting an LLM to an external service required:

1. Writing a custom API wrapper in whatever language your app used
2. Defining tool schemas in a format specific to your LLM provider
3. Writing glue code to parse model outputs and call your functions
4. Rebuilding all of this if you switched LLM providers

This meant most LLM integrations lived in large, monolithic applications. There was no way to reuse a "Gmail tool" built for one app in a different app — everything was tightly coupled.

### After MCP: Composable, Reusable Servers

With MCP, a service is built once as an MCP server and works everywhere:

| Without MCP | With MCP |
|---|---|
| Custom integration per app | Build once, works in any MCP client |
| Provider-specific tool format | Standard protocol across all LLMs |
| Fragile glue code | Structured, typed tool calls |
| No discoverability | Claude auto-discovers available tools |
| Hard to share | Public MCP server registry growing daily |

The result is a composable ecosystem: developers publish MCP servers, users connect them to their AI clients, and the AI figures out how to use them from the tool descriptions. This is the same network effect that made npm and pip so powerful for software — a shared ecosystem of reusable components.

### MCP Transport Options

MCP supports two transport modes, suited for different deployment scenarios:

```
Local (stdio)                    Remote (HTTP/SSE)
─────────────────────            ──────────────────────────
Claude Code ──► local binary     Claude Desktop ──► hosted URL
                │                                    │
          runs on your                         runs on a server
          machine                              (a service)

Best for:                        Best for:
- Personal automation            - Shared team tools
- Sensitive credentials          - SaaS integrations
- Low-latency tools              - Multi-user workflows
```

The xiaohongshu-mcp server we use in this project runs locally over HTTP, exposing itself at `http://localhost:18060/mcp`. Claude Code connects to it at session start and discovers all available tools automatically.

---

## Xiaohongshu MCP: A Real-World Example {#xiaohongshu-mcp}

[`xpzouying/xiaohongshu-mcp`](https://github.com/xpzouying/xiaohongshu-mcp) is an open-source MCP server written in Go that bridges Claude to 小红书 (Xiaohongshu), one of China's largest social media platforms — roughly equivalent to a hybrid of Instagram and Pinterest with ~300 million monthly active users.

### What It Does

The server uses a headless browser (Playwright under the hood) to automate the Xiaohongshu web interface. Once you authenticate via QR code scan, it maintains your session in a local cookie file and exposes the following tools over MCP:

| Tool | Purpose |
|---|---|
| `check_login_status` | Verify current authentication state |
| `publish_content` | Post image+text note (title, body, images, tags, visibility) |
| `publish_with_video` | Post video note with local file |
| `search_feeds` | Keyword search with filters (sort, type, date, location) |
| `list_feeds` | Fetch homepage recommendation feed |
| `get_feed_detail` | Full post detail including comments and engagement metrics |
| `post_comment_to_feed` | Post a comment on a note |
| `reply_comment_in_feed` | Reply to a specific comment |
| `like_feed` | Like or unlike a note |
| `favorite_feed` | Favorite or unfavorite a note |
| `user_profile` | Fetch a user's profile and post list |
| `get_login_qrcode` | Generate QR code for fresh login |
| `delete_cookies` | Reset authentication state |

### Architecture: How It Works

```
┌──────────────────────────────────────────────────┐
│              xiaohongshu-mcp Architecture        │
│                                                  │
│  Claude Code                                     │
│      │                                           │
│      │  HTTP MCP (localhost:18060)               │
│      ▼                                           │
│  ┌─────────────────────┐                         │
│  │  xiaohongshu-mcp    │                         │
│  │  (Go binary)        │                         │
│  │                     │                         │
│  │  MCP Server  ──────►│  Headless Browser       │
│  │  (HTTP/SSE)         │  (Playwright/Chromium)  │
│  └─────────────────────┘         │               │
│                                  │  HTTPS        │
│                                  ▼               │
│                       xiaohongshu.com            │
│                       (web interface)            │
│                                                  │
│  Local storage:                                  │
│  ~/.cookies/xiaohongshu.json  (auth session)     │
└──────────────────────────────────────────────────┘
```

### Setup and Installation

The project provides pre-compiled binaries for all platforms. For Apple Silicon (M1/M2/M3/M4):

```bash
# Download the latest release tarball
curl -L -o xiaohongshu-mcp.tar.gz \
  https://github.com/xpzouying/xiaohongshu-mcp/releases/latest/download/xiaohongshu-mcp-darwin-arm64.tar.gz

# Extract
tar xzf xiaohongshu-mcp.tar.gz

# Make executables
chmod +x xiaohongshu-mcp-darwin-arm64
chmod +x xiaohongshu-login-darwin-arm64

# If macOS security blocks execution:
xattr -d com.apple.quarantine xiaohongshu-mcp-darwin-arm64
xattr -d com.apple.quarantine xiaohongshu-login-darwin-arm64

# Step 1: Login (one-time, opens browser for QR scan)
./xiaohongshu-login-darwin-arm64

# Step 2: Start the MCP server
./xiaohongshu-mcp-darwin-arm64
# Server now running at http://localhost:18060/mcp
```

### Connecting to Claude Code

```bash
# Add the MCP server to Claude Code
claude mcp add --transport http xiaohongshu-mcp http://localhost:18060/mcp

# Verify connection
claude mcp list
# xiaohongshu-mcp: http://localhost:18060/mcp (HTTP) - ✓ Connected
```

### Important Operating Notes

A few practical considerations worth knowing before you use this:

- **Single web session limit**: Xiaohongshu does not allow the same account to be logged into multiple web sessions simultaneously. Once the MCP server is authenticated, don't open Xiaohongshu in a browser tab — it will invalidate the MCP session. The mobile app is safe to use in parallel.
- **Cookie expiry**: Sessions expire periodically. When they do, re-run the login binary and scan the QR code again.
- **New account verification**: New or non-verified accounts may be prompted for identity verification on first post. Real-name verification resolves this.
- **Content guidelines**: Avoid pure content re-posting (搬运) without original perspective — the platform actively de-promotes it.

---

## What are Claude Skills? {#what-are-claude-skills}

MCP gives Claude the ability to *act* — to call tools and interact with external services. But MCP alone doesn't tell Claude *when* to act, in what *order*, or following what *rules*. That's where **Claude Skills** come in.

A Claude Skill is a plain Markdown file that defines a reusable, trigger-activated workflow for Claude Code. Think of it as an SOP (Standard Operating Procedure) that Claude reads and follows when a specific situation arises.

### Anatomy of a Skill

```markdown
# Skill: [Name]

Trigger: [natural language phrases that activate this skill]

## Context
[Who the user is, what this skill is for]

## Workflow

### Step 1 — [Action Name]
[Detailed instructions for this step]

### Step 2 — [Action Name]
[...]

## Rules
[Constraints Claude must follow]
```

The key insight is that **skills are instructions, not code**. You don't write functions or define APIs — you write English (or any language) describing what Claude should do, in what order, and under what conditions. Claude reads the skill at session start and follows it autonomously.

### Skills vs CLAUDE.md

These two files serve different but complementary purposes:

| | `CLAUDE.md` | `.claude/skills/*.md` |
|---|---|---|
| **Purpose** | Project-wide context and global rules | Specific repeatable workflow |
| **When used** | Every session, automatically | When trigger phrase is detected |
| **Scope** | "Here's everything about this project" | "Here's how to do this specific task" |
| **Analogy** | Employee onboarding doc | Step-by-step job SOP |

In practice: `CLAUDE.md` tells Claude what tools are available, where files live, and what rules always apply. Skills tell Claude how to orchestrate those tools into coherent multi-step workflows.

### Where Skills Live

```
your-project/
├── CLAUDE.md                          ← project context (always loaded)
└── .claude/
    └── skills/
        └── rednote-lifestyle.md       ← skill: personal photos → post
```

To load skills at session start, reference them from `CLAUDE.md` using `@import` syntax:

```markdown
## Skills Available
@.claude/skills/rednote-lifestyle.md
```

---

## Building the Agent: Project Walkthrough {#building-the-agent}

I built a personal content creation agent that converts personal photos into Xiaohongshu lifestyle posts using Claude Code + xiaohongshu-mcp + a Claude Skill. Here's how it's structured and what each piece does.

### Project Structure

```
rednote-agent/
├── CLAUDE.md                          ← global context, MCP config, rules
├── .claude/
│   └── skills/
│       └── rednote-lifestyle.md       ← personal photos → lifestyle post
└── posts/
    └── published/                     ← auto-created post logs
```

### CLAUDE.md: The Project Brain

`CLAUDE.md` serves as the persistent context Claude reads at the start of every session. For this project, it covers:

```markdown
# Project: Photo-to-RedNote Agent

## What This Project Does
Converts personal photos into 小红书 lifestyle posts.
Personal photos → English lifestyle posts for 小红书's international community.

## MCP Servers
- xiaohongshu-mcp at http://localhost:18060/mcp

## Before Every Publish Action
1. Call check_login_status first — every time, no exceptions
2. If not logged in, stop and tell user to run login binary
3. Always show full preview and wait for explicit "yes"
4. Always default visibility to 仅自己可见

## Skills Available
@.claude/skills/rednote-lifestyle.md
```

The global rules section (`Before Every Publish Action`) is critical — it ensures Claude never silently publishes, always checks authentication first, and always defaults to private visibility for safety.

### The Skill: Personal Photos → English Lifestyle Post

This skill creates personal lifestyle posts from photos taken during free time — hiking, food, travel, everyday moments. Posts are in English to reach 小红书's growing international community.

**Trigger phrases:** `share my photos`, `lifestyle post`, `my photos at`, `create a personal post`

**Workflow summary:**

```
Step 1: Pre-flight login check
Step 2: Gather intent — vibe, photo paths, any memory to share
Step 3: Visually analyze each photo using Claude's vision
         └── Subject, mood, lighting, composition, impact score
Step 4: Generate post
         ├── Title: ≤50 chars, warm and personal, 1 emoji
         ├── Body: 100–300 chars, conversational English
         └── Tags: 5–7 English hashtags
Step 5: Order images — suggest cover, propose narrative sequence
Step 6: Show preview, wait for "yes"
Step 7: Publish via publish_content MCP tool
Step 8: Save record to ./posts/published/
```

**Photo analysis with vision:** Claude Code can read image files directly using its built-in vision capability. The skill instructs Claude to inspect each photo and describe what it sees:

```markdown
### Step 3 — Analyze the Photos
Use the Read tool to visually inspect each selected photo. For each one:
- Describe: subject, setting, mood, lighting, colors, time of day
- Note interesting details: expressions, textures, background
- Infer the emotion or story the photo conveys
- Rate visual impact 1–3 (3 = strongest scroll-stopper)
```

This is more powerful than it might seem. When you give Claude a folder of photos, it doesn't just see filenames — it sees the actual images. A photo named `IMG_4823.jpg` might be a misty forest trail at dawn; Claude can describe this even without any filename hints. Combined with the user's brief description of the memory behind the photo, it produces post copy that feels genuinely personal.

**HEIC handling:** iPhone photos export as HEIC by default, which xiaohongshu-mcp handles better as JPG. The skill auto-converts using `sips`, macOS's built-in image processing tool:

```bash
sips -s format jpeg photo.heic --out /tmp/rednote-images/photo.jpg
```

No third-party tools needed — `sips` ships with every Mac.

---

## Putting It All Together: The Full Workflow {#putting-it-all-together}

Here's the complete end-to-end sequence for creating a post, from terminal command to published note. I'll walk through a real example: sharing cherry blossom photos from a visit to the University of Washington.

### Step 1: Start the MCP Server

```bash
# Terminal 1 — keep running throughout the session
./xiaohongshu-mcp-darwin-arm64

# Output:
# [INFO] Server started at http://localhost:18060/mcp
# [INFO] Headless browser initialized
```

### Step 2: Launch Claude Code

```bash
# Terminal 2 — your agent
cd ~/rednote-agent
claude
```

At session start, Claude Code reads `CLAUDE.md` and loads the skill. The xiaohongshu-mcp tools are automatically available.

### Step 3: Trigger the Workflow

```
I want to share some photos from my life on xiaohongshu
```

Claude first checks login status via the MCP server, then asks a few questions to understand the intent:

```
You're logged in. Let's get your post ready!

A few quick questions before I draft this:

1. Where are your photos? (folder path or individual file paths)
2. What's the general vibe? (hiking / food / travel / city / home / other)
3. Any specific memory, feeling, or detail you want captured?
```

I pointed Claude to my cherry blossom photos and shared the mood:

```
Photos are at ~/Desktop/test_photos, they are my photos when visit
University of Washington for the Cherry Blossom season; it is a warm
and relaxed afternoon since I was working very hard and under pressure
for a while
```

### Step 4: Claude Analyzes Your Photos

Claude reads each image file using its vision capability and produces a detailed analysis:

```
Found 5 photos! Here's my analysis:

1. IMG_0016.jpeg (impact: ⭐⭐⭐)
→ Looking straight up into a canopy of cherry blossoms — branches
  spreading in every direction, blossoms so thick they almost block
  out the sky. Dreamy, immersive.

2. IMG_0018.jpeg (impact: ⭐⭐⭐)
→ The iconic UW Quad scene — cherry blossom trees in full bloom
  framing a Gothic-style building, people scattered on the lawn.
  Captures the lively yet relaxed energy perfectly.

3. IMG_0020.jpeg (impact: ⭐⭐)
→ A row of massive cherry trees in full bloom, the historic building
  peeking through behind them. Shows the sheer scale of the trees.

4. IMG_0023.jpeg (impact: ⭐⭐)
→ UW's Gothic architecture up close with cherry blossoms in the
  foreground. Nice contrast between grand stone and soft pink petals.

5. IMG_0025.jpeg (impact: ⭐⭐)
→ People picnicking and hanging out under the cherry trees on the
  quad lawn. Blankets, friends, a relaxed weekend vibe.

Suggested cover: IMG_0018.jpeg
Reason: It has everything — blossoms, the iconic UW building,
people enjoying the moment — a classic scroll-stopper.
```

Notice how Claude doesn't just list files — it describes composition, mood, and visual impact, then recommends a cover image with reasoning. This is the vision capability combined with the skill's instructions working together.

When I added that I wanted to convey "always try to visit nature, life is so short, every day is a gift," Claude incorporated that sentiment into the draft.

### Step 5: Review and Approve

Claude presents a structured preview before any publish action:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIFESTYLE POST PREVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TITLE:
Cherry blossom therapy at UW 🌸

BODY:
Took a long overdue afternoon off and drove up to the UW Quad
for cherry blossom season 🌸 After weeks of nonstop pressure,
standing under those massive pink canopies felt like the world
finally hit pause.

People were just sitting on the grass, no rush, nowhere to be.
That energy was contagious 🍃

Life is so short. Every single day is a gift. Go outside, find
the nearest blooming tree, and just breathe 💕

What's your favorite spring spot? 🌷

TAGS:
#CherryBlossomSeason #UWCherryBlossoms #Seattle #SpringVibes
#PNWLife #NatureHeals #LifeIsShort

IMAGES (5 photos):
1. 01-cover.jpg ← cover (full quad scene)
2. 02-architecture.jpg (Gothic building + blossoms)
3. 03-canopy.jpg (looking up into canopy)
4. 04-trees.jpg (massive trees in bloom)
5. 05-picnic.jpg (people on the lawn)

VISIBILITY: 仅自己可见
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Publish? (yes / edit title / edit body / reorder images / cancel)
```

Type `yes` to publish, or request specific edits — Claude will regenerate and re-present the preview.

And here's what the final published post looks like on 小红书:

![Final published post on Xiaohongshu — cherry blossom photos from UW](/assets/picture/2026-03-28-mcp-claude-skills-social-media-agent/rednote-cherry-blossom-post.png){: width="600" }
_The finished post on 小红书 — from photos on my desktop to a published note, all through Claude Code._

### Step 6: Check and Go Public

After publishing, check your 小红书 profile on the mobile app. The post appears under private visibility. When you're satisfied with how it looks:

```
republish the last post as 公开可见
```

Claude calls `publish_content` again with `visibility: 公开可见`.

### Session Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Session Flow                          │
│                                                                 │
│  User                Claude Code              xiaohongshu-mcp  │
│   │                      │                          │           │
│   │  "share my photos    │                          │           │
│   │   on xiaohongshu"    │                          │           │
│   │─────────────────────►│                          │           │
│   │                      │  check_login_status      │           │
│   │                      │─────────────────────────►│           │
│   │                      │  ✓ logged in             │           │
│   │                      │◄─────────────────────────│           │
│   │                      │                          │           │
│   │  "Where are photos?" │                          │           │
│   │◄─────────────────────│                          │           │
│   │                      │                          │           │
│   │  "~/Desktop/photos,  │                          │           │
│   │   cherry blossoms"   │                          │           │
│   │─────────────────────►│                          │           │
│   │                      │  [read & analyze photos] │           │
│   │                      │  [draft post copy]       │           │
│   │                      │                          │           │
│   │  "PREVIEW..."        │                          │           │
│   │◄─────────────────────│                          │           │
│   │                      │                          │           │
│   │  "yes"               │                          │           │
│   │─────────────────────►│                          │           │
│   │                      │  publish_content(...)    │           │
│   │                      │─────────────────────────►│           │
│   │                      │  ✓ published             │           │
│   │                      │◄─────────────────────────│           │
│   │                      │                          │           │
│   │  "Published!"        │  [save ./posts/...]      │           │
│   │◄─────────────────────│                          │           │
└─────────────────────────────────────────────────────────────────┘
```

---

## What This Means for Content Creators {#what-this-means}

This experiment is a small, concrete example of something larger happening across the content creation space. Let me share what I think it reveals about where things are heading.

### Today: Automation of the Repetitive

The most immediate value of MCP + AI agents for content creators is eliminating the repetitive, low-creativity parts of the workflow:

- **Format conversion**: Long-form blog → punchy social post → short video script
- **Platform adaptation**: Same idea reformatted for Xiaohongshu, Twitter, LinkedIn, WeChat
- **Image sourcing**: Finding and downloading relevant cover images
- **Scheduling logistics**: Posting at optimal times across platforms

These tasks consume a disproportionate amount of a creator's time relative to their creative value. A travel photographer shouldn't spend three hours reformatting captions for six platforms — they should spend those three hours taking photos.

### Near-Term: The Personal Media Machine

As MCP ecosystems mature, the logical next step is full personal media pipelines — systems where a single creative input (a photo album, a voice note, a blog post) automatically fans out across all relevant channels with platform-appropriate formatting.

```
Creator's original content
         │
         ├──► Xiaohongshu (image-forward, Chinese, short)
         ├──► LinkedIn (professional, English, longer)
         ├──► Twitter/X (punchy, English, thread format)
         ├──► WeChat (conversational, Chinese, article)
         └──► Newsletter (long-form, curated, weekly digest)
```

Each channel's MCP server handles the posting mechanics. An AI layer handles the reformatting. The creator handles the original creative work — which is where their unique value actually lives.

### The Emerging Role: Prompt Engineer as Editor

As these systems get more capable, the creator's role shifts. Less time executing; more time directing and editing. The workflow evolves from:

```
[Write] → [Format] → [Source images] → [Post] → [Repeat for each platform]
```

To:

```
[Create original content] → [Review AI-generated adaptations] → [Approve/Refine] → [Done]
```

This is analogous to what happened in photography with digital processing — photographers still make the creative decisions, but the darkroom work is automated. The skill of the photographer didn't become less valuable; it became more focused on the irreducibly human parts.

### Risks Worth Naming

This experiment also surfaces some things worth being honest about:

**Authenticity at scale is a tension.** When an AI drafts your 小红书 captions, the content is technically accurate but the voice may drift from yours. Audiences on personal platforms like Xiaohongshu are sensitive to this — they follow people, not posts. Over-automating the human voice is a real risk.

**Platform terms of service.** Browser automation tools like xiaohongshu-mcp operate in a gray area with respect to platform ToS. The risk is low at personal usage volumes, but worth understanding before scaling up.

**Filter bubbles get faster.** AI tools that optimize for engagement may gradually narrow what creators produce, converging toward what performs rather than what's genuine. The antidote is intentional — using AI to handle mechanics while preserving your own editorial judgment.

### What Stays Human

The most durable creative advantages are the ones AI can't replicate from an instruction file:

- **Original experiences**: The hike you took, the bug you debugged, the insight that surprised you
- **Earned perspective**: The opinions that come from years of working in a field
- **Authentic voice**: The specific way you notice and describe things
- **Taste and curation**: The judgment about what's worth sharing and what isn't

MCP and AI agents are powerful amplifiers of creative output — but they amplify what's already there. The best use of these tools is to spend less time on the mechanics and more time generating the original experiences and ideas worth sharing in the first place.

---

## Key Takeaways {#key-takeaways}

**1. MCP is infrastructure, not a product.**
Like HTTP or USB, it's a protocol that enables an ecosystem. The value comes from what gets built on top of it — the servers, the clients, the composable workflows.

**2. Skills are the glue between tools and intent.**
MCP gives Claude the ability to act. Skills give Claude the judgment about when and how to act. Together they form a complete agent: tools + workflow = useful automation.

**3. Plain Markdown is a surprisingly powerful programming language.**
Claude Skills are just `.md` files with structured English. There's no DSL to learn, no runtime to configure. The "code" is readable by anyone, editable without tooling, and remarkably effective at directing complex multi-step behavior.

**4. The split between local and remote MCP is meaningful.**
Local MCP servers (like xiaohongshu-mcp) keep credentials on your machine, handle session state in local files, and work without cloud infrastructure. For personal automation involving sensitive accounts, this matters.

**5. Private-first publishing is the right default.**
Always publish to `仅自己可见` first. Check on mobile. Go public when satisfied. This simple discipline prevents embarrassing mistakes during the iterative tuning of a new workflow.

**6. The best AI workflows preserve human judgment at the decision points that matter.**
Automate the mechanics. Keep humans in the loop for creative and editorial decisions. In this project: Claude drafts, human approves. That balance is worth preserving intentionally as these systems get more capable.

**All opinions expressed are my own**.