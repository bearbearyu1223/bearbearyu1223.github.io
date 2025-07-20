---
layout: post
title: "Study Notes: Stanford CS336 Language Modeling from Scratch [1]"
categories: cs336
author: 
- Han Yu
---

## Introduction

It’s been a while since my last blog post about my generative AI learning journey (January 1, 2024). My day job has demanded significant time and energy, but I don’t want that to derail my passion and curiosity for deep-diving into this field.

This week, I had a realization: instead of endlessly reflecting, thinking, or planning, it’s time to act. Even with just one hour of focused study after work each day, I can make meaningful progress by year-end.

I’ve decided to tackle Stanford’s [**CS336: Language Modeling from Scratch**](https://stanford-cs336.github.io/spring2025/) at my own pace. I’m grateful that Stanford makes their [**Lecture Videos**](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&ab_channel=StanfordOnline), [**Lecture Notes & Assignments**](https://github.com/stanford-cs336) freely available online—a perfect fit for someone like me who can’t commit to fixed class schedules but still craves structured learning.

To give back to the community, I’ll document and share my learning notes as I progress. I’m curious to see how much I can accomplish by dedicating just an hour a day (or ~10 hours per week) through the end of the year. Can I complete all the lectures and assignments? I know I’ll move slowly, but I’m excited to test my consistency and see what’s possible.

This first post covers **setting up the local development environment**—a small but necessary step to begin engaging with the lecture materials.


## Setting Up the Local Dev Environment

### Prerequisites

Make sure you have Git, Python 3.11, and Node.js installed on your system. You’ll also need [`uv`](https://github.com/astral-sh/uv), a fast Python package manager.


### Step 1: Clone the CS336 Repository

```bash
git clone https://github.com/stanford-cs336/spring2025-lectures
cd spring2025-lectures
```


### Step 2: Set Up Python Virtual Environment with UV

Install `uv` (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment using Python 3.11:

```bash
uv venv --python=3.11
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate
```


### Step 3: Install Python Dependencies

> 💡 Note: I'm using a MacBook Pro (M4). The `triton` package doesn't support Apple Silicon, so I excluded it during the initial setup.  
> More info: [Triton GitHub](https://github.com/triton-lang/triton)

```bash
grep -v "triton" requirements.txt > requirements_no_triton.txt
uv pip install -r requirements_no_triton.txt
```


### Step 4: Generate Executable Lecture Content

Compile a lecture:

```bash
python execute.py -m lecture_01
```

This will generate a trace file at:

```
var/traces/lecture_01.json
```


### Step 5: Build the Local React Web App to View Lectures

Install Node.js if needed:

```bash
brew install node
```

Then build and serve the trace viewer:

```bash
cd trace-viewer
npm install
npm run dev
```

Open the viewer in your browser:

```
http://localhost:<PORT>?trace=var/traces/lecture_01.json
```

It should be something like ![this](/assets/picture/2025_07_20_cs336_note_get_started/cs336_lecture_view.png).


Then enjoy going through the lecture notes! 
