---
title: "LLM Architecture Refresh [1]: Inside a Transformer Block — Attention, Heads, and the FFN"
date: 2026-08-06 00:00:00 -0700
categories: [LLM Architecture Refresh, Transformers]
tags: [attention, multi-head-attention, transformer, ffn, swiglu, rmsnorm, rope, positional-encoding, softmax, flops, pytorch]
description: >-
  Taking a transformer block apart and measuring every piece — what attention
  computes and what it costs, why heads are a reshape rather than extra
  machinery, why the FFN is where a model keeps what it knows, and why the
  1/sqrt(d_k) factor and RoPE are there at all.
math: true
pin: true
---

## Taking a transformer block apart, one measurement at a time

I can draw a transformer block from memory. For a long time I still couldn't have told you which parts of it are actually expensive, why there's a feed-forward network in there at all, or what that $1/\sqrt{d_k}$ is really doing. I'd absorbed the names without the reasons — and the usual explanations hand you more names.

So this post takes one block apart and puts a number on every piece.

### The short version {#the-short-version}

If you read nothing else, these are the things this post establishes — each one measured rather than asserted:

- **Attention is a weighted average, nothing more exotic.** Each token looks at the others and pulls in a blend of what they offer, weighted by how relevant each one is. It is the *only* step in the entire model where tokens see each other at all.
- **Extra heads are almost free.** Splitting attention into 32 parallel copies costs the same parameters and the same arithmetic as running it once — it's a reshape of a fixed budget, not extra machinery. What you buy is several opinions at once instead of one blurred compromise. The one thing that *does* grow with head count is the score matrix, which is the tensor nobody wants to store anyway.
- **What makes attention expensive is model width and context length — not head count.** And the famous quadratic term is smaller than its reputation: at a 1,024-token sequence it's 6% of the layer, while plain matrix multiplies are 89%.
- **Attention can average, but it cannot conclude.** Blending two facts never produces a third. That's why every block also has a feed-forward network — the only part that transforms a token on its own, shaped like a lookup table, and where a model's facts actually live.
- **The $\sqrt{d_k}$ is not about overflow.** It keeps the softmax in a range where a gradient still flows back, so how wide you make the heads stays a free choice instead of silently breaking training.
- **RoPE gets relative position out of absolute rotation.** Spin each token's query and key by an angle set by its position, and the score between any two tokens ends up depending only on the gap between them — with no learned parameters at all.
- **Training and serving are different jobs.** Same model, 3× the arithmetic and 9× the memory to train it.

Every one of those has a **receipt** behind it — a small program that prints the number, so you can check it rather than take my word. The code lives in a companion repo and runs unchanged on Apple Silicon or a Linux + NVIDIA box:

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo01
```

Every number and figure below came out of that command on my M-series Mac. The Python shown alongside each result is the part that matters, trimmed of setup — the runnable version is in [`demos/d01_attention.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d01_attention.py).

### Table of Contents

Skip to [the short version](#the-short-version) for the findings without the derivations.

1. [What a language model actually does](#what-a-language-model-does)
2. [What attention is for](#what-attention-is-for)
3. [The formula, symbol by symbol](#the-formula-symbol-by-symbol)
4. [What a "head" is](#what-a-head-is)
5. [What attention costs](#what-attention-costs)
6. [Following the shapes](#following-the-shapes)
7. [The rest of the block](#where-attention-sits-in-the-model)
8. [The FFN: where the model knows things](#the-ffn)
9. [The implementation, in five lines](#the-implementation-in-five-lines)
10. [Why divide by sqrt(d_k)?](#why-divide-by-sqrt-dk)
11. [Causal masking](#causal-masking)
12. [RoPE: absolute rotation, relative score](#rope-absolute-rotation-relative-score)
13. [A silent MPS bug that deleted RoPE](#a-silent-mps-bug-that-deleted-rope)
14. [Sidebar: the probe](#sidebar-the-probe)
15. [Appendix: all notation](#appendix-all-notation)

---

### 1. What a language model actually does {#what-a-language-model-does}

Before attention, the thing attention is part of.

A language model does one thing: **given some tokens, predict the next one.** To write a sentence it does that over and over — predict a token, stick it on the end, feed the whole thing back in, predict again. All the sophistication is in how that single prediction gets computed.

That computation is a **transformer**: a stack of $L$ identical **blocks** — 32 of them in an 8-billion-parameter model — that a token's numbers pass through in order, edited a little by each. Every block does exactly two things:

![Where attention sits in a decoder block](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/block-anatomy-light.png){: .light width="700" height="845" }
![Where attention sits in a decoder block](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/block-anatomy-dark.png){: .dark width="700" height="845" }

1. **Attention** — each token looks at the other tokens and pulls in whatever it needs. This is the *only* step where tokens see each other.
2. **A feed-forward network (FFN)** — each token, now holding some context, gets transformed on its own. No looking around.

Gather, then think. Thirty-two times over. At the end, one more layer turns the final token's numbers into a probability for every word in the vocabulary, and you pick one.

**This post is about step 1**, where most of the interesting design lives. [§8](#the-ffn) comes back for the FFN, and [§7](#where-attention-sits-in-the-model) for the norms and residuals, once the vocabulary is in place.

Numbers throughout this post come from one real model, **Llama-3-8B**, so they're a checkable config rather than placeholders:

| | Llama-3-8B | What it is |
| --- | --- | --- |
| $d_{model}$ | **4096** | how many numbers describe one token |
| $n_{heads}$ | **32** | how many attention heads (§4) |
| $d_{head}$ | **128** | width of one head's slice |
| $n_{kv\_heads}$ | 8 | key/value heads — fewer than query heads (§6) |
| $d_{ff}$ | 14336 | width of the FFN's middle layer (§8) |
| $L$ | 32 | how many blocks are stacked |

These are one model's choices, not universal constants — GPT-2 XL used $d_{model} = 1600$ with 25 heads of 64. One relationship *is* universal, and it does a lot of work later:

$$
d_{model} = n_{heads} \times d_{head} \qquad 4096 = 32 \times 128
$$

For simplicity, §4 and §5 assume every query head has its own key and value head — 32 of each. That's **multi-head attention**, or **MHA**. Llama-3-8B actually shares 8 key/value heads across its 32 query heads; [§6](#following-the-shapes) covers what changes.

### 2. What attention is for {#what-attention-is-for}

A language model reads a sequence of tokens and builds a representation of each one. The hard part: a token's meaning depends on other tokens, sometimes far away.

> *"The trophy didn't fit in the suitcase because **it** was too big."*

Resolving **it** means looking back at two candidate nouns and picking one. A model that processes each token in isolation cannot do this at all.

Attention lets a token **go and get information from other tokens**, and learn which ones to get it from. For each token it computes a weight over every other token, then returns a weighted average of what those tokens offer. Relevant tokens get large weights; the rest get weights near zero.

Each token produces three vectors for this, from three learned projections:

| | The question it answers | Its role |
| --- | --- | --- |
| **Query** $Q$ | "What am I looking for?" | the token doing the looking |
| **Key** $K$ | "What am I?" | how a token advertises itself |
| **Value** $V$ | "What do I contribute if chosen?" | the content actually retrieved |

A useful analogy: attention is a **Python dictionary with fuzzy matching**. A normal lookup tests keys for equality and returns one value. Attention replaces equality with a dot product (how *aligned* are they?) and replaces one value with a weighted average of all of them. That softness is the whole trick — a hard lookup has no gradient, so nothing could be learned.

### 3. The formula, symbol by symbol {#the-formula-symbol-by-symbol}

Before the equation, the symbols in it:

| Symbol | What it is | Shape |
| --- | --- | --- |
| $n$ | how many tokens are in the sequence | — |
| $d_k$ | how many numbers describe one query or key | — |
| $Q$ | the queries, one row per token | $(n, d_k)$ |
| $K$ | the keys, one row per token | $(n, d_k)$ |
| $V$ | the values, one row per token | $(n, d_k)$ |
| $K^\top$ | $K$ flipped, so the matrix multiply lines up | $(d_k, n)$ |
| $M$ | the causal mask, added before the softmax (see below) | $(n, n)$ |

With those in hand:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Read it from the inside out:

1. $QK^\top$ — every query dotted with every key, giving an $n \times n$ grid of relevance scores. Row $i$ holds "how relevant is each token to token $i$?"
2. $\div \sqrt{d_k}$ — a constant that keeps those scores in a sane range. [§10](#why-divide-by-sqrt-dk) is devoted to why.
3. $\text{softmax}$ — turns each row of scores into weights that are positive and sum to 1.
4. $\times V$ — uses those weights to average the values, giving each token its answer.

#### The piece that formula is missing: the mask

As written, that equation lets every token see every other token — including the ones that come *after* it. For an encoder like BERT that's exactly right. For a model that generates text it's cheating: predicting token 5 while looking at token 6 isn't prediction.

So decoder-only models add one more term, a **mask** $M$:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

$M$ is an $n \times n$ matrix holding 0 where a position is allowed and $-\infty$ where it's forbidden:

$$
M_{ij} = \begin{cases} 0 & j \le i \quad \text{(token } j \text{ is at or before } i\text{)} \\[2pt] -\infty & j > i \quad \text{(token } j \text{ is in the future)} \end{cases}
$$

It's rarely shown, so here it is for a 6-token sequence:

```text
        key0     key1     key2     key3     key4     key5
  q0    0        -inf     -inf     -inf     -inf     -inf
  q1    0        0        -inf     -inf     -inf     -inf
  q2    0        0        0        -inf     -inf     -inf
  q3    0        0        0        0        -inf     -inf
  q4    0        0        0        0        0        -inf
  q5    0        0        0        0        0        0
```

Two details make this work, and both are the reason it's written as an *addition* rather than a multiplication:

- **$-\infty$, not a large negative number.** Softmax exponentiates, and $e^{-\infty} = 0$ exactly. Forbidden positions get precisely zero weight, not merely a small one.
- **It's added *before* the softmax.** So the surviving weights renormalize among themselves and each row still sums to 1. Masking after the softmax would leave rows summing to less than 1 — a classic bug.

Everything else in the formula is unchanged. [§11](#causal-masking) measures the result.

Two consequences to carry forward:

- **The dot product is the only place tokens interact.** Everything else in a transformer processes each token alone.
- **The equation has no idea what order the tokens came in.** Shuffle them and you get the same outputs, shuffled. Position must be injected separately — that's [§12](#rope-absolute-rotation-relative-score).

### 4. What a "head" is {#what-a-head-is}

Running attention once has a limitation: **a softmax produces exactly one set of weights** — one opinion about which tokens matter. But a token usually needs several unrelated things at once: what noun this pronoun refers to, which verb governs this subject, which adjective modifies this noun. One weighting blurs them together.

The fix is to run attention several times in parallel, on different slices of the same vector — each copy free to form its own opinion.

#### What a head actually is

A **head** is one independent copy of attention, working on a 128-number slice of $Q$, $K$ and $V$. The order of operations matters, and it's the detail most explanations blur:

1. The token's full 4,096-number vector is projected into $Q$, $K$, $V$. Each projection is $4096 \times 4096$, so **$Q$, $K$ and $V$ are each 4,096 wide**, and every number in them combines *all* 4,096 inputs.
2. *Then* $Q$, $K$, $V$ are each cut into 32 slices of 128.
3. Head $h$ takes slice $h$ of each and runs a complete attention pass.
4. The 32 results are concatenated back to 4,096 and mixed by a final matrix $W_o$.

![Multi-head attention: splitting the vector across heads](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/multi-head-light.png){: .light width="1000" height="899" }
![Multi-head attention: splitting the vector across heads](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/multi-head-dark.png){: .dark width="1000" height="899" }

**What gets sliced is $Q$/$K$/$V$, never the input.** Two misreadings to head off, one per axis:

- **Not the sequence.** Every head sees every token. Head 3 doesn't get "the last quarter of the sentence."
- **Not the input vector.** Head 3 doesn't get input dimensions 384–511 either. It gets dimensions 384–511 *of $Q$, $K$, $V$*, each computed from all 4,096 inputs.

Equivalently: head $h$ owns the 128 columns of $W_q$, $W_k$, $W_v$ that produce its slice, and those columns read the whole vector. That's why you'll see the mechanism described both ways — "project everything, then reshape" is what the code does; "each head has its own smaller projections" is what the math says.

$W_o$ at the end is not bookkeeping. Without it, 32 heads' findings would sit in 32 disjoint stretches of the vector, unable to influence one another.

So what does all that buy? Several attention patterns at once, instead of one averaged compromise:

![Four heads, four attention patterns on the same input](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/head-patterns-light.png){: .light width="1000" height="453" }
![Four heads, four attention patterns on the same input](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/head-patterns-dark.png){: .dark width="1000" height="453" }

Each triangle is one head's weights: row $i$ is where query $i$ looks, and the staircase edge is causal masking ([§11](#causal-masking)). Four visibly different patterns from the same input.

One caveat I want to be exact about: **these are random weights.** The heads differ because they were initialized differently, which shows they aren't redundant copies — it is *not* specialization. In trained models specialization is real and catalogued: "previous-token heads" that look one step back, "induction heads" that spot a repeated pattern and predict its continuation. You can't see that in a figure like this.

There is a trade-off in *how many* heads to use: more heads means more distinct patterns, but each one gets narrower — 64 heads leaves each only 64 dimensions to describe a query with. Models land at 32–64 because both extremes are bad.

### 5. What attention costs {#what-attention-costs}

The natural worry is that 32 heads means 32× the work. It doesn't — but "heads are free" would be the wrong lesson, because attention is *not* cheap. It's just that the head count isn't what makes it expensive.

Two things do: **how wide the model is** ($d_{model}$) and **how long the context is** ($seq$). Here is where every cost in an attention layer comes from:

| Cost | Kind | Formula | Grows with | Set by head count? |
| --- | --- | --- | --- | --- |
| the projection matrices | **memory** | $4 d_{model}^2$ | width, **squared** | no |
| applying them | **compute** | $2 \cdot seq \cdot d_{model}^2$ | length × width² | no |
| computing the scores | **compute** | $2 \cdot seq^2 \cdot d_{model}$ | **length squared** × width | no |
| holding the scores | **memory** | $n_{heads} \cdot seq^2$ | length squared × head count | **yes** |

Two costs of each kind, and it's worth keeping them apart — **compute** is work the chip has to do, **memory** is bytes it has to hold, and on real hardware they run out at different times. Three of the four don't mention $n_{heads}$ at all, which is what makes the head count a free architectural choice; the fourth is a genuine exception.

**Parameters.** Start with a question: how wide does $W_q$'s output need to be?

It has to hand every head one slice of queries. With 32 heads of 128 that's $32 \times 128 = 4096$ numbers — the same width as the input it started from. So $W_q$ takes 4,096 numbers in and produces 4,096 out: a $4096 \times 4096$ matrix.

Now change the head count and watch what happens. With 64 heads, each slice is only 64 numbers wide — but $64 \times 64 = 4096$ again. With 8 heads, each slice is 512 wide, and $8 \times 512 = 4096$. The output width never moves, because the head count and the head width always multiply back to $d_{model}$.

**So $W_q$ is $4096 \times 4096$ no matter how many heads you choose**, and the same holds for $W_k$, $W_v$ and $W_o$:

```text
  matrix                         maps         shape  params
  -----------------------------------------------------------
  W_q     d_model -> n_heads x d_head  (4096, 4096)   16.8M
  W_k     d_model -> n_heads x d_head  (4096, 4096)   16.8M
  W_v     d_model -> n_heads x d_head  (4096, 4096)   16.8M
  W_o     n_heads x d_head -> d_model  (4096, 4096)   16.8M
  total                                               67.1M
```

$4096^2 = 16.8\text{M}$ per matrix, four matrices, $67.1$M per block — for **any** head count. Nothing in that arithmetic mentions $n_{heads}$. Note also that every matrix has 4,096 rows — the shape-level version of the point from §4, that each head reads the whole vector.

#### An aside: what a FLOP is, and how to count one

The other half of the cost is measured in FLOPs, so before using the word:

A **FLOP** is one **fl**oating-point **op**eration — a single add or a single multiply on decimal numbers. "FLOPs" (lowercase s) counts operations; "FLOP/s" is a *rate*, operations per second. They look almost identical and mean different things: a model needs so many FLOPs, a GPU delivers so many FLOP/s, and dividing one by the other estimates time.

Counting them in a transformer is easy, because nearly all the arithmetic is matrix multiplication, and one rule covers every matmul:

$$
(a, b) \times (b, c) \;\longrightarrow\; 2 \cdot a \cdot b \cdot c \;\text{ FLOPs}
$$

Where that comes from: the result has $a \times c$ entries, and each is a dot product of two length-$b$ vectors — $b$ multiplies and $b$ adds, so $2b$ operations. Multiply out and you get $2abc$. In words:

> **FLOPs = 2 × (number of output entries) × (length of the dimension being summed away).**

The 2 is the multiply-and-add pair. Hardware runs them as one fused instruction, but the convention counts both.

That's the whole method. It also gives a rule of thumb worth carrying: since each weight is used in exactly one multiply-add per token, **a forward pass costs about $2N$ FLOPs per token**, where $N$ is the parameter count.

**Score FLOPs.** This is the part that really looks like it should scale with the head count — 32 heads each doing their own matrix multiply. It doesn't, and it fails to for the same reason.

Inside one head, $QK^\top$ multiplies a $(seq, d_{head})$ matrix by a $(d_{head}, seq)$ one. Applying the rule from above — 2 × output entries × the dimension summed away:

$$
\text{one head} \;=\; 2 \times seq^2 \times d_{head}
$$

Then there are $n_{heads}$ of them, so multiply through:

$$
\text{all heads} \;=\; n_{heads} \times 2 \times seq^2 \times d_{head}
\;=\; 2 \times seq^2 \times \underbrace{\big(n_{heads} \times d_{head}\big)}_{=\; d_{model}}
\;=\; 2 \times seq^2 \times d_{model}
$$

The head count cancels, exactly as it did for the parameters — and for the same reason, that the two always multiply back to $d_{model}$. At $seq = 1024$:

```text
  n_heads  d_head  n_heads x d_head  FLOPs per head  x n_heads = total
  ----------------------------------------------------------------------
  1          4096              4096          8.59 G             8.59 G
  8           512              4096          1.07 G             8.59 G
  32          128              4096          0.27 G             8.59 G
  64           64              4096          0.13 G             8.59 G
```

Read the last two columns together: **halving $d_{head}$ halves the per-head cost, and doubling the head count multiplies it straight back.** Going from 1 head to 64 makes each head 64× cheaper and there are 64× as many — exactly inverse, so the final column never moves.

So heads are a *reshape of a fixed budget*, not extra machinery. You're only choosing whether to read the same 4,096 numbers as one wide space or many narrow ones.

#### Except in one place: the score matrix

That table had a fourth row, and it's the honest exception. The scores live in a tensor of shape $(n_{heads}, seq, seq)$ — one $seq \times seq$ grid *per head*. Narrower heads make each dot product cheaper, which is why the FLOPs cancel; they do **not** make each grid smaller.

```text
  n_heads  d_head  score FLOPs  score matrix (fp32)
  ---------------------------------------------------
  1          4096       8.59 G                4 MiB
  8           512       8.59 G               32 MiB
  32          128       8.59 G              128 MiB
  64           64       8.59 G              256 MiB
```

**Identical arithmetic, 64× the activation memory.** So the accurate statement isn't "heads are free" — it's that heads cost nothing in *parameters or arithmetic*, and cost linearly in *the one tensor nobody wants to store anyway*. [Post 3](/posts/llm-architectures-flash-attention/) is entirely about not storing it.

#### Where the FLOPs actually go

That table counted only $QK^\top$. Applying the same rule to every matmul in one attention layer gives the full picture — and it's not what people expect:

```text
  step                       shapes (32 heads)    FLOPs  share
  --------------------------------------------------------------
  W_q/W_k/W_v/W_o  4 x (1024,4096)@(4096,4096)  137.4 G    89%
  Q @ K.T           32 x (1024,128)@(128,1024)    8.6 G     6%
  weights @ V      32 x (1024,1024)@(1024,128)    8.6 G     6%
  total                                         154.6 G
```

Note the `32 ×` on the attention rows: those are **thirty-two small matmuls, one per head**, summed — not one big one. It happens not to change the total, because $32 \times 128 = 4096$ makes the arithmetic identical either way, but it is what actually runs.

Worth stating while we're here, since it's a natural assumption: **GQA doesn't shrink these.** Its 8 key/value heads are broadcast back up to 32 right before the matmul, so every query head still scores against a full-width key. GQA saves cache and parameters, not attention FLOPs.

**The quadratic term is the smallest item here.** At a 1,024-token sequence, attention's famous $n^2$ cost is 6% of the layer; the four projections are 89%. The $n^2$ term only takes over once $seq$ grows past $d_{model}$ — below that, a transformer is mostly big dense matrix multiplies, and "attention is quadratic" describes the *asymptote*, not the regime most models run in. [Post 3](/posts/llm-architectures-flash-attention/) is about what happens when you do cross that line.

The rule of thumb checks out too:

```text
  projection params N                67.1M
  2 x N x tokens                     137.4 G
  equals the measured projection cost yes
```

$2 \times 67.1\text{M} \times 1024$ lands exactly on the projection cost — because every weight is used in exactly one multiply-add per token. (Everything above is head-count independent for the same reason as before.)

#### What this costs on real hardware

Everything so far is per layer. Scaled to a whole model, the interesting thing is that **training and inference are not the same job** — and they differ far more in memory than in arithmetic.

**Compute.** A backward pass costs about twice a forward one, because every weight needs two gradients computed for it — one flowing back to its input, one for the weight itself. So:

```text
  per token        FLOPs                            why
  -------------------------------------------------------
  inference  2N = 16.1 G               one forward pass
  training   6N = 48.2 G  forward, then backward at ~2x
```

Three times the arithmetic. That part is mild.

**Memory** is where they separate. Inference needs the weights and nothing else structural. Training has to keep the optimizer's state beside them:

```text
  what                   cost    size  running total
  ----------------------------------------------------
  fp16 weights      2 B/param  15 GiB         15 GiB
  fp32 master copy  4 B/param  30 GiB         45 GiB
  fp32 gradients    4 B/param  30 GiB         75 GiB
  Adam moment m     4 B/param  30 GiB        105 GiB
  Adam moment v     4 B/param  30 GiB        135 GiB
```

**About 2 bytes per parameter to serve, about 18 to train** — and that's before activations, which training must also keep because the backward pass needs them.

So an 8B model is 15 GiB to serve and roughly 135 GiB to train: **3× the compute, 9× the memory.** That's the first thing to check when a run won't fit — it's almost never the arithmetic.

#### What that means in actual GPUs

Gigabytes are easier to judge against hardware you can rent. Two common accelerators, at spec-sheet peaks:

```text
  GPU        memory  bandwidth  BF16 compute
  --------------------------------------------
  A100 80GB   80 GB  2.04 TB/s   312 TFLOP/s
  H100 80GB   80 GB  3.35 TB/s   990 TFLOP/s
```

**Serving** an 8B model is comfortable. The weights are 16.1 GB, so one card holds them with ~64 GB to spare — worth roughly **488,000 tokens of KV cache**, which is post 2's currency.

**Training** the same model doesn't fit at all: ~145 GB of optimizer state against an 80 GB card, before any activations. You need several GPUs and a sharding strategy for a model that serves happily on one. Same weights, different job.

Then the number that decides how fast you can generate. A decode step reads every weight exactly once, so you can time it two ways — by the arithmetic it does, or by the bytes it moves:

```text
  GPU        if compute-bound  if bandwidth-bound   gap
  -------------------------------------------------------
  A100 80GB      19,427 tok/s           127 tok/s  153x
  H100 80GB      61,644 tok/s           209 tok/s  296x
```

**Bandwidth wins by two orders of magnitude.** The chip finishes the multiplying and then sits waiting for the next weights to arrive. All those TFLOP/s are unreachable for this workload.

And it's getting *worse*, not better: the H100 has 3.2× the compute of an A100 but only 1.6× the bandwidth, so the gap roughly doubles between generations. Buying a faster chip mostly buys compute you can't use.

*(These are peak numbers; real kernels reach a fraction of them, and batching improves the picture a lot. But the ratio is what matters here, and it survives the discount.)*

It also sets up the two posts that follow, which are both about memory rather than math. [Post 2](/posts/llm-architectures-kv-cache/) is about a cost that doesn't appear in any table here at all — generating text needs to *keep* every key and value it has computed, and that cache can outgrow the weights. [Post 3](/posts/llm-architectures-flash-attention/) is about the fourth row above: the $seq \times seq$ score grid, and how to get attention's answer without ever writing it down.


### 6. Following the shapes {#following-the-shapes}

Shapes are where most confusion about attention lives. Rather than write a table by hand — easy to get subtly wrong — the demo runs a real attention module on a 10-token prompt and prints what PyTorch reports. This is the code it runs:

```python
q, k, v = x @ w_q, x @ w_k, x @ w_v                    # (seq, 4096) each

# split into heads: view, then move the head axis to the front
qh, kh, vh = (t.view(seq, n_heads, d_head).transpose(0, 1)
              for t in (q, k, v))                      # (32, seq, 128)

scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(d_head)   # (32, seq, seq)
weights = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
ctx = weights @ vh                                     # (32, seq, 128)

merged = ctx.transpose(0, 1).reshape(seq, d_model)     # (seq, 4096)
out = merged @ w_o                                     # (seq, 4096)
```

The `.transpose(0, 1)` on line 4 is worth noticing — it's what makes the head axis come *first*. And what PyTorch reports for each step:

```text
  classic multi-head attention (MHA): one K and V head per query head
  seq=10, d_model=4096, n_heads=32, d_head=128

  tensor                             shape                          note
  ------------------------------------------------------------------------
  x   (the token vectors)       (10, 4096)             one row per token
  W_q, W_k, W_v               (4096, 4096)     each reads all of d_model
  Q, K, V  after projection     (10, 4096)              still full width
  Q, K, V  split into heads  (32, 10, 128)               4096 = 32 x 128
  one head's Q                   (10, 128)  what the formula operates on
  scores = Q Kt / sqrt(d_k)   (32, 10, 10)    per head, quadratic in seq
  weights  after softmax      (32, 10, 10)            each row sums to 1
  weights @ V                (32, 10, 128)               per-head answer
  concatenated heads            (10, 4096)            back to full width
  output  after W_o             (10, 4096)       same shape as the input

  input and output shapes match      yes
  attention weights row sum          1.0000
```

Three things fall out:

- **The module is shape-preserving.** In at `(10, 4096)`, out at `(10, 4096)` — which is exactly what makes blocks stackable.
- **Only the score matrix depends on $n^2$.** At `seq=10` it's a nuisance; at `seq=128000` it's the whole problem, which is post 3.
- **$W_q$ is square.** It shrinks nothing. The reshape does the splitting; the matrix only mixes.

#### How to picture `(32, 10, 128)`

Three-axis shapes are where notation stops being readable. The image that works: **a deck of 32 sheets, each sheet a 10 × 128 table** — 10 rows, one per token; 128 columns, that head's features.

![Reading a (32, 10, 128) tensor](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/tensor-3d-light.png){: .light width="960" height="650" }
![Reading a (32, 10, 128) tensor](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/tensor-3d-dark.png){: .dark width="960" height="650" }

Resist drawing it as a solid cuboid. A cuboid suggests the three axes are interchangeable directions, and they aren't — **each axis has a different job**:

| Axis | Size | What happens along it |
| --- | --- | --- |
| heads | 32 | **Nothing.** Pure parallelism; sheets never interact until $W_o$ |
| tokens | 10 | **Attention mixes along this axis** — the only one that gets mixed |
| features | 128 | **Dot products contract this axis** — it disappears in $QK^\top$ |

That last row explains the shape changes people find most confusing. One rule does all the work:

> A matrix multiply **contracts the shared inner dimension**. $(a, b) \times (b, c)$ gives $(a, c)$, and $b$ is summed away.

Attention applies it twice, and the two are mirror images:

```text
  step              left      right     output  axis summed away
  ----------------------------------------------------------------
  Q @ K.T      (10, 128)  (128, 10)   (10, 10)    128 (features)
  weights @ V   (10, 10)  (10, 128)  (10, 128)         10 (keys)
```

$$
(10, 128) \;\xrightarrow{\;Q K^\top\;}\; (10, 10) \;\xrightarrow{\;\times V\;}\; (10, 128)
$$

**First matmul:** the 128 features are summed away and a *second* token axis appears. You've turned "each token's features" into "each token's relevance to each other token."

**Second matmul:** the 10 keys are summed away and V's 128 features come back. This is the step whose output shape surprises people: `weights` is $(10, 10)$ and `V` is $(10, 128)$, so the shared axis is *tokens*, leaving one row per query and 128 columns from V. Written out, output row $i$ is literally a weighted sum of V's rows:

$$
\text{out}_i = \sum_{j} w_{ij} \cdot V_j
$$

Ten weights, ten value-vectors of 128 numbers each, one 128-number result. That's why the answer is 128 wide rather than 10 wide — **the weights say *how much* of each token to take, and V says *what* to take.**

Worth checking that claim rather than taking it. Pick any single output row — say head 0, query position 3 — and compute it two ways: once as the library does it, and once literally as the sum above, looping over tokens.

```python
from_matmul = (weights @ V)[0, 3]                                  # the matmul
by_hand = sum(weights[0, 3, j] * V[0, j] for j in range(seq))      # the definition
```

Both should be the same 128 numbers. Their first four:

```text
    from the matmul                  +0.5989  -0.0095  -0.3333  -0.4037
    from the by-hand sum             +0.5989  -0.0095  -0.3333  -0.4037

  largest disagreement, all 128      1.788e-07
```

Agreement to `1.8e-07` across all 128 — float noise, not a real difference. **The matmul *is* the weighted sum**, written as one operation instead of a loop. Which is the point: `weights @ V` looks like opaque linear algebra, and it's doing exactly the "take 70% of this token, 20% of that one" averaging from §1.

#### What changes with grouped-query attention

Here's the simplification promised in §3. Everything above is **classic multi-head attention**, where every query head gets its own key and value head. Llama-3-8B instead uses **grouped-query attention**: 32 query heads sharing only 8 key/value heads, so its $K$ and $V$ projections are a quarter as wide:

```text
  tensor                  MHA (32 kv heads)  Llama-3-8B (8 kv heads)
  --------------------------------------------------------------------
  W_q                          (4096, 4096)             (4096, 4096)
  W_k, W_v                     (4096, 4096)             (4096, 1024)
  Q  split into heads         (32, 10, 128)            (32, 10, 128)
  K, V  split into heads      (32, 10, 128)             (8, 10, 128)
```

Each projection still follows one rule — it is $(d_{model},\; \text{heads it feeds} \times d_{head})$ — so 8 key heads of 128 gives $(4096, 1024)$. That drops the block's attention parameters from 67.1M to **41.9M**.

Note the asymmetry: **$Q$ keeps full width; only $K$ and $V$ shrink.** That's deliberate — $K$ and $V$ are the tensors generation has to *cache*, so shrinking them shrinks the memory that limits how many users you can serve. $Q$ is recomputed every step and never cached. [Post 2](/posts/llm-architectures-kv-cache/) is largely about this.

Read the rest of this post as classic MHA; the mechanism is identical either way.

### 7. The rest of the block {#where-attention-sits-in-the-model}

Back to the diagram from [§1](#what-a-language-model-does). Attention was one box in it. Here are the others — and the wrapping around all of them.

#### Inside the attention box

Start with the box this post has been about. Here's the data path inside it, with the tensor's shape down the right margin:

![Inside the multi-head attention module](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/attention-zoom-light.png){: .light width="900" height="964" }
![Inside the multi-head attention module](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/attention-zoom-dark.png){: .dark width="900" height="964" }

Two steps in it are worth naming. **One input, three projections** — $Q$, $K$ and $V$ are three learned *views of the same vector*, which is what makes self-attention "self." And **RoPE rotates $Q$ and $K$, never $V$** — position belongs in the *matching* step; $V$ is the content you retrieve once matching is done, so rotating it would corrupt the payload. [§12](#rope-absolute-rotation-relative-score) covers RoPE properly.

#### The other pieces, in plain English

**Token embeddings** — a lookup table. Every token in the vocabulary owns a list of 4,096 numbers, and that list *is* the model's representation of it. At the start it encodes only "which token is this." Each block edits it toward "which token is this, *in this context*." The word *bank* enters generic and leaves nudged toward *riverbank* or *savings account*.

**RMSNorm** — as a vector passes through dozens of layers, its numbers drift, growing until they overflow or shrinking until they vanish. Normalization rescales the whole vector back to a standard size, like setting every track on a mixing desk to a consistent level. Relative proportions survive; only the overall magnitude is standardized.

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{n}\sum x_i^2}} \cdot g
$$

where $g$ is a learned per-dimension scale. The original transformer used **LayerNorm**, which also subtracts the mean first. RMSNorm's contribution is a *discovery, not an invention*: that mean-subtraction turned out not to matter. Dropping it costs no quality and saves a pass over the data.

#### The residual, and what "pre-norm" means

**The residual is a real bypass.** Each sublayer's input branches off, skips both the norm and the sublayer, and is added back at the $\oplus$. So a sublayer never computes its output — it computes a *correction*:

$$
x \leftarrow x + \text{Attention}(\text{Norm}(x))
$$

If a sublayer learns nothing useful, the block degrades to the identity rather than to noise. That unbroken path is also the road the gradient travels back down undiminished, which is what lets you stack 80 of these.

**"Pre-norm"** describes where the norm sits relative to that bypass. The 2017 original normalized *after* adding the residual, putting a norm on the trunk itself. Modern decoders moved it *inside* the branch, so the residual highway stays unnormalized end to end — which is why deep models train stably without the learning-rate warmup gymnastics the original recipe needed.

That leaves one box in the diagram unexplained, and it's the biggest one.

### 8. The FFN: where the model knows things {#the-ffn}

**FFN** stands for **feed-forward network**. It processes each token entirely on its own — that's what "position-wise" means; token 5 has no idea token 6 exists. Next to attention it looks like filler. It isn't, and it's worth three questions: what it is, why it's there, and why facts end up inside it.

#### What it's made of

Not one matrix — **three matrices with an elementwise gate between them**. For Llama-3-8B, per block:

```text
  matrix          shape  params                          role
  -------------------------------------------------------------
  W_gate  (4096, 14336)   58.7M     opens or closes each slot
  W_up    (4096, 14336)   58.7M  the content each slot offers
  W_down  (14336, 4096)   58.7M        writes the result back
  total                  176.2M        per block, x 32 blocks
```

![What an FFN is made of](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/ffn-anatomy-light.png){: .light width="960" height="768" }
![What an FFN is made of](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/ffn-anatomy-dark.png){: .dark width="960" height="768" }

The bars are drawn to scale, so the widening is real: **4096 → 14336 → 4096**. Widen, act, narrow.

That gate is **SwiGLU**, and it's why there are three matrices rather than two. The original FFN used **ReLU** — keep positives, zero the negatives — one fixed rule applied to everything, with a single up-projection. SwiGLU replaces the rule with something learned:

$$
\text{SwiGLU}(x) = W_{\text{down}}\big(\,\text{SiLU}(W_{\text{gate}}\,x) \;\odot\; W_{\text{up}}\,x\,\big)
$$

Two up-projections run in parallel. One produces content; the other is squashed by a smooth S-curve (SiLU) into a set of dimmers, and the two are multiplied element by element. The network learns, per feature and per input, how much signal to let through — a dimmer it controls rather than a hard on/off switch. The third matrix is paid for by shrinking the expansion from $4d$ to about $\tfrac{8}{3}d$, which is why $d_{ff}$ is 14,336 rather than a rounder 16,384.

Worth knowing for its honesty: the paper introducing SwiGLU tested a family of gated variants, found they worked better, offered no theory, and closed by attributing their success "to divine benevolence." Much of a modern transformer is there because it measured better, not because someone derived it.

**Why more than one matrix at all?** Because two stacked matrices with nothing between them *are* one matrix:

Multiply the two matrices together *first*, then apply the product. If that gives the same answer, the pair was never doing more than a single matrix could:

```text
    two steps:  (x @ W_up) @ W_down  -11.776383  ...
    one matrix: x @ (W_up @ W_down)  -11.776383  ...

  largest disagreement               3.94e-13
  shape of that single matrix        (64, 64)
```

Identical. Without a nonlinearity you could pre-multiply $W_{\text{up}}$ and $W_{\text{down}}$ into a single $4096 \times 4096$ matrix and all that widening would buy exactly nothing. **The nonlinearity is the only thing stopping them from collapsing into one** — a good hint that it's carrying the weight here.

#### Why an FFN is needed at all

**Because attention can only average.** Look at what a softmax guarantees: the weights are non-negative and sum to 1. So every output is a *blend* of the value rows — and a blend of things can never be anything but a mixture of those things.

```python
w = torch.softmax(q @ k.T / math.sqrt(d_model), dim=-1)
attn = w @ v

(w >= 0).all()                          # every weight non-negative
w.sum(-1)                               # every row sums to 1
torch.allclose(w @ (2 * v), 2 * attn)   # exactly linear in V
```

```text
  softmax weights are all >= 0       yes
  each row of weights sums to        1.0000
  so every output is a blend of V's rows yes
  attn(2V) == 2 x attn(V)            yes
```

Two consequences. The output can never leave the range of the values it was handed. And holding the weights fixed, attention is exactly **linear** in $V$ — that last row is the test, and we'll come back to it in a moment.

That's a real limitation. Attention can bring "this is a plural noun" and "this sentence is about France" into the same vector, but it cannot compute anything *from* them. Averaging two facts doesn't produce a conclusion. "If A and B are both present, then C" is not something a weighted average can express — no matter how many attention layers you stack.

The FFN is the only place in the block where a token's own features get transformed nonlinearly. Here's a test that shows the difference — the simplest possible one:

> **The doubling test.** A linear function must obey $f(2x) = 2f(x)$: feed it twice the input and you get exactly twice the output. That's what "linear" means.

Run it on both:

```python
def ffn(t):
    return F.silu(t @ w_up) @ w_down

ffn(2 * x)   # what the function actually returns for a doubled input
2 * ffn(x)   # what a linear function would have returned
```

```text
  function                    f(2x)  2 x f(x)  off by  linear?
  --------------------------------------------------------------
  attention (weights fixed)  23.542    23.542    0.0%      yes
  FFN                        14.641    13.375   21.0%       no
```

The first two columns are the two things being compared, measured by vector length: what the function *actually* returns when you double its input, versus what doubling its output would have given. **Attention matches exactly. The FFN is 21% off** — so it is not a linear function.

Easier to hold onto with a single number from the output:

```text
    ffn(x) gives                     -0.1197
    2 x that = what linear predicts  -0.2394
    ffn(2x) actually gives           -0.3333
```

Double the input and a linear function would have moved that number to $-0.2394$. The FFN returns $-0.3333$ instead. It responds to *how much* signal arrives, not just proportionally — which is exactly the freedom attention doesn't have.

And that's the entire point. **Attention decides what to look at; the FFN decides what it means.** A meeting where everyone shares information, then the work you actually do with what you heard.

#### Why knowledge ends up there

The second surprise is that the FFN's *shape* is a lookup table. Write it out one slot at a time:

$$
\text{FFN}(x) = \sum_{i=1}^{d_{ff}} \underbrace{a_i(x)}_{\text{did slot } i \text{ match?}} \cdot \underbrace{W_{\text{down}}[i]}_{\text{what slot } i \text{ says}}
$$

where $a_i(x)$ is the activation of the $i$-th middle neuron. Each of the 14,336 middle dimensions is one **memory slot** with three parts:

| Piece | Role |
| --- | --- |
| column $i$ of $W_{\text{up}}$ | the **pattern** slot $i$ looks for |
| $a_i$, its activation | **how strongly** this token matched it |
| row $i$ of $W_{\text{down}}$ | the **content** slot $i$ adds if it matched |

Match against stored patterns, then add up the content of whatever matched, weighted by match strength. That's the same soft-lookup shape as attention itself — except attention looks up *other tokens*, while the FFN looks up *stored weights*.

The claim is that the ordinary matmul and the slot-by-slot sum are the same thing. Same check as before — take one token's output and compute it both ways:

```python
acts = F.silu(x @ w_up)                          # one score per slot
from_matmul = (acts @ w_down)[row]               # what the model runs
by_hand = sum(acts[row, i] * w_down[i] for i in range(d_ff))   # slot by slot
```

```text
    from the matmul                  -0.2740  +0.1003  +0.0076  -0.1433
    from the slot-by-slot sum        -0.2740  +0.1003  +0.0076  -0.1433

  largest disagreement, all 64       8.941e-08
```

So "the model knows Paris is in France" has a concrete home: some slot whose up-projection fires on *the Eiffel Tower is in*, whose down-projection nudges the output toward *Paris*. Llama-3-8B has 14,336 slots per layer across 32 layers — **458,752 of them**. This isn't only a metaphor: the ROME and MEMIT lines of work locate specific factual associations in specific FFN weights and *edit* them, changing what a model believes by writing to a handful of numbers.

Two caveats so the picture isn't too tidy. Slots aren't cleanly one-fact-each — facts spread across many, and a slot participates in many facts. And with the random weights measured above, most slots respond to anything; trained FFNs are far sparser.

#### Who gets the parameters?

Counting attention against FFN in a single block, across three real models:

```text
  block shape                  attention params  FFN params  FFN share
  ----------------------------------------------------------------------
  GPT-2 style (MHA, ReLU FFN)             10.2M       20.5M        67%
  Llama-3-8B (GQA, SwiGLU)                41.9M      176.2M        81%
  Llama-3-70B (GQA, SwiGLU)              151.0M      704.6M        82%
```

Attention gets the name and the diagrams, but it's the **minority of the weights**. The classic two-thirds figure comes from the original shapes — attention $4d^2$, FFN $2 \cdot d \cdot 4d = 8d^2$. Modern decoders push further from both ends: GQA shrinks $K$/$V$ while SwiGLU adds a third FFN matrix.

Which is the parameter-count version of the point above: **routing lives in attention, knowledge lives in the FFN.** If the FFN is the model's memory, it needs to be big. It's also why LoRA on attention alone underperforms (post 5), and why Mixture-of-Experts replaces the *FFN* (post 9).

### 9. The implementation, in five lines {#the-implementation-in-five-lines}

```python
def scaled_dot_product_attention(q, k, v, *, causal=False, scale=None):
    d_k = q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(d_k)

    scores = (q @ k.transpose(-2, -1)) * scale  # (..., seq_q, seq_k)

    if causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        # True above the diagonal = "this key is in the future" = forbidden.
        mask = torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights
```

Shapes are `(..., seq, head_dim)` — the leading dimensions carry batch and heads, since multi-head attention is this same function applied in parallel. I return the `weights` alongside the output because they're exactly what an optimized kernel throws away, and they're the thing worth looking at.

Checking it against PyTorch's fused `F.scaled_dot_product_attention`:

```text
  max |ours - torch|  (causal=False) 5.364e-07
  max |ours - torch|  (causal=True)  4.768e-07
```

Agreement to `5e-07` in float32 — floating-point reassociation noise, not an algorithmic difference. Worth internalizing early, because it's the same fact that makes Flash Attention work: **a fast kernel is an optimization, not an approximation.** Post 3 returns to this.

### 10. Why divide by sqrt(d_k)? {#why-divide-by-sqrt-dk}

The argument is short. If $q$ and $k$ have independent components with mean 0 and variance 1, then

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

is a sum of $d_k$ independent mean-zero terms. So $\operatorname{Var}(q \cdot k) = d_k$, and the standard deviation is $\sqrt{d_k}$.

Logits with a standard deviation of 32 aren't "large numbers" — they're a **temperature setting**. A softmax over scores that spread that wide puts essentially all its mass on the single largest one.

So let's measure it. Sample random queries and keys at several $d_k$, and report two diagnostics: the average largest weight (1.0 = fully one-hot) and the entropy in nats (0.0 = one-hot; $\ln 8 = 2.08$ = uniform over our 8 keys).

```text
  d_k   logit std  max w unscaled  H unscaled  max w scaled  H scaled
  ---------------------------------------------------------------------
  4        1.9642          0.5222      1.2986        0.3401    1.7548
  16       3.9493          0.7428      0.6825        0.3523    1.7350
  64       7.9399          0.8738      0.3223        0.3506    1.7407
  256     15.9159          0.9326      0.1691        0.3533    1.7318
  1024    32.4533          0.9735      0.0674        0.3701    1.7103
```

![Softmax saturation without the sqrt(d_k) scale](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/softmax-saturation-light.png){: .light width="1000" height="696" }
![Softmax saturation without the sqrt(d_k) scale](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/softmax-saturation-dark.png){: .dark width="1000" height="696" }

Read the `logit std` column first: `1.96, 3.95, 7.94, 15.92, 32.45` — exactly $\sqrt{4}, \sqrt{16}, \sqrt{64}, \sqrt{256}, \sqrt{1024}$. The theory isn't approximately right, it's exactly right.

Then the consequence. Unscaled, entropy collapses from 1.30 nats to **0.067**; at $d_k = 1024$ the average largest weight is 0.97, so attention has stopped averaging and become a hard `argmax`. Scaled, entropy holds at **~1.74 across a 256× range of $d_k$**. That flat blue line is the entire justification for the constant.

Why does saturation hurt? Two reasons, and the second is what actually kills training:

1. **It stops being attention.** A near-one-hot distribution ignores all but one token.
2. **The gradient vanishes.** The softmax Jacobian is $\operatorname{diag}(p) - pp^\top$. As $p \to$ one-hot, every entry goes to zero. No gradient reaches $Q$ and $K$, and the model can't learn to attend differently.

That second point reframes it. $1/\sqrt{d_k}$ is not about overflow — softmax handles large logits fine by subtracting the row max. It's there to **keep the softmax in a regime where it still has a gradient**, whatever width you make the heads.

### 11. Causal masking {#causal-masking}

[§3](#the-formula-symbol-by-symbol) introduced the mask $M$. Here is what it does to the weights:

```text
        key0    key1    key2    key3    key4    key5
  q0    1.000    0.000    0.000    0.000    0.000    0.000
  q1    0.180    0.820    0.000    0.000    0.000    0.000
  q2    0.366    0.477    0.157    0.000    0.000    0.000
  q3    0.717    0.176    0.072    0.035    0.000    0.000
  q4    0.238    0.015    0.342    0.345    0.060    0.000
  q5    0.126    0.348    0.240    0.160    0.040    0.086
```

![Causal attention weight matrix](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/causal-mask-light.png){: .light width="820" height="726" }
![Causal attention weight matrix](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/causal-mask-dark.png){: .dark width="820" height="726" }

Note row `q0`: weight exactly 1.000 on itself. The first token has nothing else to attend to, so a softmax over a single unmasked score returns 1.

This one matrix is why the same architecture serves both BERT (unmasked, good for understanding) and GPT (masked, good for generation).


### 12. RoPE: absolute rotation, relative score {#rope-absolute-rotation-relative-score}

Back to the gap from §2: attention has no idea what order the tokens came in. Something must inject position.

The original transformer added a position vector to each embedding. That encodes *absolute* position, but language mostly cares about *relative* — "the adjective two tokens back" is a useful pattern at position 5 and at position 5000.

**Rotary Position Embedding** gets relative position out of absolute rotation, using one fact of geometry: rotate two vectors, and their dot product depends only on the *difference* of the angles.

$$
\langle R_m q,\; R_n k \rangle = \langle R_{m-n}\, q,\; k \rangle
$$

Here $R_m$ is the rotation applied at position $m$, and $\theta_i$ below is the rotation speed for the $i$-th pair of dimensions. Split the head dimension into 2D subspaces and rotate the $i$-th by $\text{position} \times \theta_i$:

```python
def rope_frequencies(head_dim, base=10_000.0):
    i = torch.arange(0, head_dim, 2, dtype=torch.float64)
    return 1.0 / (base ** (i / head_dim))

def rotate_half(x):
    """[x1, x2] -> [-x2, x1]: a 90-degree rotation in each 2D subspace."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, positions, base=10_000.0):
    head_dim = x.shape[-1]
    pos = positions.cpu().to(torch.float64)[:, None]
    angles = torch.cat([pos * rope_frequencies(head_dim, base)] * 2, dim=-1)
    cos = angles.cos().to(torch.float32).to(x.device).to(x.dtype)
    sin = angles.sin().to(torch.float32).to(x.device).to(x.dtype)
    return x * cos + rotate_half(x) * sin
```

Low $i$ rotates fast (resolving nearby tokens), high $i$ rotates slowly (carrying long-range position). Every context-extension method you've heard of — position interpolation, NTK-aware scaling, YaRN — is some way of **rescaling that frequency vector**, so rotations learned at 4k still mean something at 128k.

Now the receipt. Take one query and one key, place them at wildly different absolute positions but always 3 apart:

```text
  query pos m  key pos n  offset m-n  q_m . k_n
  -----------------------------------------------
  0                    3          -3    -1.1905
  5                    8          -3    -1.1905
  105                108          -3    -1.1905
  4093              4096          -3    -1.1905

  spread across absolute positions   1.073e-06
```

Position 5 and position 4093 give the same score to seven digits. The model never has to relearn "3 tokens back" for each position in the window — it gets that invariance from geometry, with **zero learned parameters**.

![RoPE scores are invariant to absolute position, sensitive to relative offset](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/rope-relative-light.png){: .light width="1000" height="544" }
![RoPE scores are invariant to absolute position, sensitive to relative offset](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/rope-relative-dark.png){: .dark width="1000" height="544" }

Two panels, same y-axis. Left: slide both vectors along 128 positions, holding the gap at +3 — a flat line. Right: pin the query and sweep the offset — structure everywhere. Flat where you want invariance, expressive where you want sensitivity.

One consequence that returns in post 12: because the rotation happens *after* the $Q$/$K$ projections, RoPE doesn't commute with tricks that absorb those projections into neighbouring matrices — exactly the complication DeepSeek's MLA has to work around.

### 13. A silent MPS bug that deleted RoPE {#a-silent-mps-bug-that-deleted-rope}

While building this demo, the RoPE section printed a score that was *identical for every offset*. Not wrong-looking — suspiciously perfect.

The angle table needs float64: at position 4096 the fastest frequency has accumulated ~4096 radians, and float32 has about 7 digits for an argument that large. So I moved positions to the CPU and cast in one call — the idiomatic thing to write:

```python
pos = positions.to("cpu", torch.float64)     # looks fine. is not.
```

On an **int64 MPS** tensor, torch 2.13 reinterprets the bits instead of converting them:

```python
p = torch.tensor([105], device='mps')
p.to('cpu', torch.float64)    # tensor([5.1877e-322])   <- 105's bit pattern read as a float
p.cpu().to(torch.float64)     # tensor([105.])          <- correct
```

`5.19e-322` is a denormal indistinguishable from zero. Every angle became 0, so `cos = 1`, `sin = 0`, and `x * 1 + rotate_half(x) * 0` returned `x` unchanged. **RoPE had silently degraded into the identity function** — no exception, no warning, just a model with no positional information that would have trained to a mediocre loss and left me blaming hyperparameters.

The fix is to separate the device move from the dtype cast. The lesson generalizes: when a numerical result looks *too* clean, verify it on a second device. `LLMR_DEVICE=cpu uv run demo01` takes two seconds and would have caught this immediately.

### Sidebar: the probe {#sidebar-the-probe}

> **"Why is there a $1/\sqrt{d_k}$ in the attention equation?"**

**A weak answer:** "For numerical stability — it keeps the dot products from getting too large."

Not wrong, but it doesn't survive a follow-up. Softmax is already robust to large logits: every implementation subtracts the row max first, so `[1000, 1001, 1002]` and `[-2, -1, 0]` give identical output. Overflow isn't the problem being solved.

**A stronger answer:** "The dot product of two $d_k$-dimensional unit-variance vectors has variance $d_k$, so without the scale the logits grow like $\sqrt{d_k}$ and the softmax sharpens as you widen the heads. Once it saturates toward one-hot, the Jacobian $\operatorname{diag}(p) - pp^\top$ goes to zero and no gradient reaches $Q$ and $K$ — the model can't learn what to attend to. The scale holds the temperature roughly constant, so head width stays a free architectural choice instead of something that silently changes training dynamics."

The difference is that the second answer says what breaks, and where.

### What's next

Post 2 takes the same measure-it-yourself approach to the **KV cache** — the thing that decides what you can actually serve. We time generation with and without it, work out why decode is memory-bandwidth-bound while prefill is compute-bound, and see why that one distinction explains batching, quantization and speculative decoding at once.

### Appendix: all notation {#appendix-all-notation}

| Symbol | Means | Llama-3-8B |
| --- | --- | --- |
| $n$, `seq` | tokens in the sequence | varies with input |
| $d_{model}$ | width of a token's vector | 4096 |
| $h$, `n_heads` | number of attention heads | 32 |
| $d_{head}$, $d_k$ | width of one head's slice | 128 |
| $d_{ff}$ | width of the FFN's middle layer | 14336 |
| $L$ | blocks stacked | 32 |
| $x$ | the input, one vector per token | $(n, 4096)$ |
| $W_q, W_k, W_v$ | projections producing $Q$, $K$, $V$ | $(4096, 4096)$ |
| $Q, K, V$ | queries, keys, values **before** the head split | $(n, 4096)$ |
| $Q, K, V$ | the same tensors **after** the head split | $(32, n, 128)$ |
| $W_o$ | output projection | $(4096, 4096)$ |
| $\oplus$ | residual addition | — |
| $\odot$ | elementwise multiply (SwiGLU's gate) | — |
| $R_m$ | RoPE's rotation for position $m$ | — |
| $\theta_i$ | RoPE's rotation frequency for dimension pair $i$ | — |
| FLOP | one floating-point add or multiply; a matmul $(a,b)\times(b,c)$ costs $2abc$ | — |
| MHA | multi-head attention — one key/value head per query head | 32 of each |
| GQA | grouped-query attention — several query heads share one key/value head | 32 query, 8 kv |

$Q$, $K$ and $V$ get two rows because they have two shapes — full width leaving the projection, regrouped into heads immediately after. Nothing is added or discarded between them ($32 \times 128 = 4096$); papers write $Q$ for both and leave you to infer which is meant.

Two naming traps, because papers are inconsistent about both:

- **$d_k$ almost always means $d_{head}$, not $d_{model}$.** Substituting 4096 for 128 makes the whole scaling argument come out wrong.
- **Keys and values have separate widths in principle** ($d_k$ and $d_v$); the original paper distinguishes them. Every model sets them equal, so this post writes $d_{head}$ for both.

### References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — the scale factor is justified in a footnote in §3.2.1.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021).
- Chen et al., [Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/abs/2306.15595) (2023).
- Peng et al., [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) (2023).
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (2020) — the SwiGLU paper, and the divine-benevolence line.
- Geva et al., [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) (2021) — where the lookup-table reading comes from.
- Meng et al., [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (2022) — ROME, editing a fact by writing to FFN weights.
- Code for this post: [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher), `uv run demo01`.
