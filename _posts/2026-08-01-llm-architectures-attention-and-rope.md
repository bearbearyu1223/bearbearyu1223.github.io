---
title: "LLM Architectures [1]: Attention, the sqrt(d_k) Scale, and RoPE"
date: 2026-08-01 00:00:00 -0700
categories: [LLM Architectures, Transformers]
tags: [attention, rope, transformer, pytorch, positional-encoding, softmax, mps]
description: >-
  Building scaled dot-product attention from scratch and checking it against
  PyTorch's fused kernel — then measuring why the 1/sqrt(d_k) factor exists,
  and watching RoPE turn absolute rotations into relative positions.
math: true
---

## Understanding attention by measuring it, not by reading about it

I have read the attention equation many times. I can write it from memory. But when someone asks *why* the $1/\sqrt{d_k}$ is there, "for numerical stability" is the kind of answer that sounds fine and explains nothing — a phrase I'd absorbed rather than a thing I'd seen happen.

So this series gives every claim a **receipt**: a small program that prints the number the claim asserts. The code lives in a companion repo and runs unchanged on Apple Silicon or a Linux + NVIDIA box:

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo01
```

Every number and figure below came out of that command on my M-series Mac.

### Table of Contents

1. [What attention is for](#what-attention-is-for)
2. [The formula, symbol by symbol](#the-formula-symbol-by-symbol)
3. [What a "head" is](#what-a-head-is)
4. [Following the shapes](#following-the-shapes)
5. [Where attention sits in the model](#where-attention-sits-in-the-model)
6. [The implementation, in five lines](#the-implementation-in-five-lines)
7. [Why divide by sqrt(d_k)?](#why-divide-by-sqrt-dk)
8. [Causal masking](#causal-masking)
9. [RoPE: absolute rotation, relative score](#rope-absolute-rotation-relative-score)
10. [A silent MPS bug that deleted RoPE](#a-silent-mps-bug-that-deleted-rope)
11. [Sidebar: the probe](#sidebar-the-probe)
12. [Appendix: all notation](#appendix-all-notation)

---

### 1. What attention is for {#what-attention-is-for}

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

### 2. The formula, symbol by symbol {#the-formula-symbol-by-symbol}

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
2. $\div \sqrt{d_k}$ — a constant that keeps those scores in a sane range. [§7](#why-divide-by-sqrt-dk) is devoted to why.
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

Everything else in the formula is unchanged. [§8](#causal-masking) measures the result.

Two consequences to carry forward:

- **The dot product is the only place tokens interact.** Everything else in a transformer processes each token alone.
- **The equation has no idea what order the tokens came in.** Shuffle them and you get the same outputs, shuffled. Position must be injected separately — that's [§9](#rope-absolute-rotation-relative-score).

### 3. What a "head" is {#what-a-head-is}

Running attention once has a limitation: **a softmax produces exactly one set of weights** — one opinion about which tokens matter. But a token usually needs several unrelated things at once: what noun this pronoun refers to, which verb governs this subject, which adjective modifies this noun. One weighting blurs them together.

A **head** is one independent copy of attention, working on a 128-number slice of $Q$, $K$ and $V$. The order of operations matters, and it's the detail most explanations blur:

1. The token's full 4,096-number vector is projected into $Q$, $K$, $V$. Each projection is $4096 \times 4096$, so **$Q$, $K$ and $V$ are each 4,096 wide**, and every number in them combines *all* 4,096 inputs.
2. *Then* $Q$, $K$, $V$ are each cut into 32 slices of 128.
3. Head $h$ takes slice $h$ of each and runs a complete attention pass.
4. The 32 results are concatenated back to 4,096 and mixed by a final matrix $W_o$.

$$
d_{model} = n_{heads} \times d_{head} \qquad 4096 = 32 \times 128
$$

![Multi-head attention: splitting the vector across heads](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/multi-head-light.png){: .light width="1000" }
![Multi-head attention: splitting the vector across heads](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/multi-head-dark.png){: .dark width="1000" }

**What gets sliced is $Q$/$K$/$V$, never the input.** Two misreadings to head off, one per axis:

- **Not the sequence.** Every head sees every token. Head 3 doesn't get "the last quarter of the sentence."
- **Not the input vector.** Head 3 doesn't get input dimensions 384–511 either. It gets dimensions 384–511 *of $Q$, $K$, $V$*, each computed from all 4,096 inputs.

An equivalent way to say it: head $h$ owns the 128 columns of $W_q$, $W_k$, $W_v$ that produce its slice, and those columns read the whole vector. That's why you'll see the mechanism described both ways — "project everything, then reshape" is what the code does; "each head has its own smaller projections" is what the math says.

$W_o$ at the end is not bookkeeping. Without it, 32 heads' findings would sit in 32 disjoint stretches of the vector, unable to influence one another.

#### Heads are free

The natural worry is that 32 heads means 32× the work. It doesn't — and the reason is one line of algebra worth doing slowly, because it's the thing that makes multi-head attention obviously worth doing.

**The invariant:** the head count and the head width always multiply back to the same total.

$$
n_{heads} \times d_{head} = d_{model} \qquad 32 \times 128 = 4096
$$

Choosing a head count doesn't add anything. It only decides how a fixed 4,096 gets partitioned. Both costs follow from that.

**Parameters.** $W_q$ maps $d_{model}$ into $n_{heads} \times d_{head}$ — which *is* $d_{model}$. So it's a $4096 \times 4096$ matrix whatever the head count, and so are the other three:

```text
  matrix                         maps         shape  params
  -----------------------------------------------------------
  W_q     d_model -> n_heads x d_head  (4096, 4096)   16.8M
  W_k     d_model -> n_heads x d_head  (4096, 4096)   16.8M
  W_v     d_model -> n_heads x d_head  (4096, 4096)   16.8M
  W_o     n_heads x d_head -> d_model  (4096, 4096)   16.8M
  total                                               67.1M
```

$4096^2 = 16.8\text{M}$ per matrix, four matrices, $67.1$M total — for **any** head count. Nothing in that arithmetic mentions $n_{heads}$.

**Score FLOPs.** Now the part that looks like it should scale. Inside one head, computing $QK^\top$ is a $(seq, d_{head})$ matrix times a $(d_{head}, seq)$ matrix. Every one of the $seq \times seq$ outputs costs $d_{head}$ multiply-adds, and the usual convention counts a multiply-add as 2 FLOPs:

$$
\text{per head} = 2 \times seq^2 \times d_{head}
\qquad
\text{all heads} = n_{heads} \times 2 \times seq^2 \times d_{head}
$$

And $n_{heads} \times d_{head}$ is $d_{model}$ again, so the total is $2 \times seq^2 \times d_{model}$ with **the head count cancelled out**. At $seq = 1024$:

```text
  n_heads  d_head  n_heads x d_head  FLOPs per head  x n_heads = total
  ----------------------------------------------------------------------
  1          4096              4096          8.59 G             8.59 G
  8           512              4096          1.07 G             8.59 G
  32          128              4096          0.27 G             8.59 G
  64           64              4096          0.13 G             8.59 G
```

Read the last two columns together: **halving $d_{head}$ halves the per-head cost, and doubling the head count multiplies it straight back.** Going from 1 head to 64 makes each head 64× cheaper and there are 64× as many. The two changes are exactly inverse, which is why the final column never moves.

So heads are a *reshape of a fixed budget*, not extra machinery — you're only choosing whether to read the same 4,096 numbers as one wide space or many narrow ones. What you buy is several attention patterns at once instead of one averaged compromise.

*(Two caveats on the numbers. The FLOP column counts only $QK^\top$; multiplying by $V$ costs the same again, and the four projections cost $2 \cdot seq \cdot d_{model}^2$ each — larger than the scores until the sequence gets long. All of them are head-count independent for the same reason, so the conclusion holds.)*

So heads cost nothing and buy several attention patterns at once. Running one input through four heads:

![Four heads, four attention patterns on the same input](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/head-patterns-light.png){: .light width="1000" }
![Four heads, four attention patterns on the same input](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/head-patterns-dark.png){: .dark width="1000" }

Each triangle is one head's weights: row $i$ is where query $i$ looks, and the staircase edge is causal masking ([§8](#causal-masking)). Four visibly different patterns from the same input.

One caveat I want to be exact about: **these are random weights.** The heads differ because they were initialized differently, which shows they aren't redundant copies — it is *not* specialization. In trained models specialization is real and catalogued: "previous-token heads" that look one step back, "induction heads" that spot a repeated pattern and predict its continuation. You can't see that in a figure like this.

There's a trade-off in the head count: more heads means more distinct patterns but narrower ones — 64 heads leaves each only 64 dimensions. Models land at 32–64 because both extremes are bad.

### 4. Following the shapes {#following-the-shapes}

Shapes are where most confusion about attention lives. Rather than write a table by hand — easy to get subtly wrong — the demo runs a real attention module on a 10-token prompt and prints what PyTorch reports:

```text
  classic MHA - seq=10, d_model=4096, n_heads=32, d_head=128

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

![Reading a (32, 10, 128) tensor](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/tensor-3d-light.png){: .light width="960" }
![Reading a (32, 10, 128) tensor](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/tensor-3d-dark.png){: .dark width="960" }

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

So the whole per-head shape story is:

$$
(10, 128) \;\xrightarrow{\;Q K^\top\;}\; (10, 10) \;\xrightarrow{\;\times V\;}\; (10, 128)
$$

**First matmul:** the 128 features are summed away, and a *second* token axis appears. You've turned "each token's features" into "each token's relevance to each other token."

**Second matmul:** the 10 keys are summed away, and V's 128 features come back. This is the step whose output shape surprises people, so it's worth being concrete: `weights` is $(10, 10)$ and `V` is $(10, 128)$; the shared 10 is the *token* axis, so it contracts, leaving one row per query and 128 columns from V.

Written out, output row $i$ is literally a weighted sum of V's rows:

$$
\text{out}_i = \sum_{j} w_{ij} \cdot V_j
$$

Ten weights, ten value-vectors of 128 numbers each, one 128-number result. That's why the answer is 128 wide rather than 10 wide — **the weights say *how much* of each token to take, and V says *what* to take.** The demo checks this against the matmul directly:

```text
  out[h=0, q=3] via matmul           (128,)
  same, as sum_j w[3,j] * V[j]       (128,)
  max abs difference                 1.788e-07
```

Same answer to float noise. Stacking all 32 heads back on, that's `(32, 10, 128) → (32, 10, 10) → (32, 10, 128)`.

#### A correction: Llama-3-8B isn't quite this shape

Everything above is **classic multi-head attention**, where every query head gets its own key and value head. But I've been quoting Llama-3-8B's dimensions, and it actually uses **grouped-query attention** — 32 query heads sharing only 8 key/value heads:

```text
  tensor                  MHA (32 kv heads)  Llama-3-8B (8 kv heads)
  --------------------------------------------------------------------
  W_q                          (4096, 4096)             (4096, 4096)
  W_k, W_v                     (4096, 4096)             (4096, 1024)
  Q  split into heads         (32, 10, 128)            (32, 10, 128)
  K, V  split into heads      (32, 10, 128)             (8, 10, 128)
```

Note the asymmetry: **$Q$ keeps full width; only $K$ and $V$ shrink.** That's deliberate — $K$ and $V$ are the tensors generation has to *cache*, so shrinking them shrinks the memory that limits how many users you can serve. $Q$ is recomputed every step and never cached. Post 2 is largely about this.

Read the rest of this post as classic MHA; the mechanism is identical either way.

### 5. Where attention sits in the model {#where-attention-sits-in-the-model}

Attention is a *component*, not the model. A decoder stacks $N$ identical blocks, each doing two things: attention, then a feed-forward network, each wrapped in a residual connection with a normalization layer.

![Where attention sits in a decoder block](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/block-anatomy-light.png){: .light width="700" }
![Where attention sits in a decoder block](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/block-anatomy-dark.png){: .dark width="700" }

The names in that diagram are all jargon. In plain English:

**Token embeddings** — a lookup table. Every token in the vocabulary owns a list of 4,096 numbers, and that list *is* the model's representation of it. At the start it encodes only "which token is this." Each block edits it toward "which token is this, *in this context*." The word *bank* enters generic and leaves nudged toward *riverbank* or *savings account*.

**RMSNorm** — as a vector passes through dozens of layers, its numbers drift, growing until they overflow or shrinking until they vanish. Normalization rescales the whole vector back to a standard size, like setting every track on a mixing desk to a consistent level. Relative proportions survive; only the overall magnitude is standardized.

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{n}\sum x_i^2}} \cdot g
$$

where $g$ is a learned per-dimension scale. The original transformer used **LayerNorm**, which also subtracts the mean first. RMSNorm's contribution is a *discovery, not an invention*: that mean-subtraction turned out not to matter. Dropping it costs no quality and saves a pass over the data.

**FFN** (feed-forward network) — the plainest thing in the architecture. Widen the vector, apply a nonlinear function, narrow it back:

$$
\text{FFN}(x) = W_{\text{down}}\big(\,\text{nonlinearity}(W_{\text{up}}\,x)\,\big)
$$

Typically 4,096 → 16,384 → 4,096. The wide middle gives the model room to test many specialized questions at once: each intermediate dimension can learn to respond to some pattern ("is this a plural noun?"), the nonlinearity keeps what fires, and the down-projection recombines the survivors.

Two things make the FFN matter more than it looks. **It is where knowledge lives** — interpretability work traces specific facts to specific FFN weights, and it reads naturally as a key–value memory: the up-projection matches patterns, the down-projection writes the associated content back. And **it is most of the model**, as the parameter count below shows.

"Position-wise" just means each token goes through the FFN alone — token 5 has no idea token 6 exists.

**SwiGLU** — a better nonlinearity. The original FFN used **ReLU**: keep positives, zero the negatives. SwiGLU replaces that fixed rule with a **learned gate**. Run two up-projections in parallel; squash one through a smooth S-curve (SiLU) into a set of dimmer switches, and multiply:

$$
\text{SwiGLU}(x) = W_{\text{down}}\big(\,\text{SiLU}(W_{\text{gate}}\,x) \;\odot\; W_{\text{up}}\,x\,\big)
$$

The network learns, per feature and per input, how much signal to let through — a dimmer it controls rather than a hard on/off switch. The cost is a third matrix, paid for by shrinking the expansion from $4d$ to about $\tfrac{8}{3}d$, which is why Llama-3-8B's $d_{ff}$ is 14,336 rather than a rounder 16,384.

Worth knowing for its honesty: the paper introducing it tested a family of gated variants, found they worked better, offered no theory, and closed by attributing their success "to divine benevolence." Much of a modern transformer is there because it measured better, not because someone derived it.

#### The residual, and what "pre-norm" means

**The residual is a real bypass.** Each sublayer's input branches off, skips both the norm and the sublayer, and is added back at the $\oplus$. So a sublayer never computes its output — it computes a *correction*:

$$
x \leftarrow x + \text{Attention}(\text{Norm}(x))
$$

If a sublayer learns nothing useful, the block degrades to the identity rather than to noise. That unbroken path is also the road the gradient travels back down undiminished, which is what lets you stack 80 of these.

**"Pre-norm"** describes where the norm sits relative to that bypass. The 2017 original normalized *after* adding the residual, putting a norm on the trunk itself. Modern decoders moved it *inside* the branch, so the residual highway stays unnormalized end to end — which is why deep models train stably without the learning-rate warmup gymnastics the original recipe needed.

#### Who gets the parameters?

```text
  block shape                  attention params  FFN params  FFN share
  ----------------------------------------------------------------------
  GPT-2 style (MHA, ReLU FFN)             10.2M       20.5M        67%
  Llama-3-8B (GQA, SwiGLU)                41.9M      176.2M        81%
  Llama-3-70B (GQA, SwiGLU)              151.0M      704.6M        82%
```

Attention gets the name and the diagrams, but it's the **minority of the weights**. The classic two-thirds figure comes from the original shapes — attention $4d^2$, FFN $2 \cdot d \cdot 4d = 8d^2$. Modern decoders push further from both ends: GQA shrinks $K$/$V$ while SwiGLU adds a third FFN matrix.

The summary worth keeping: **routing lives in attention, knowledge lives in the FFN.** It's why LoRA on attention alone underperforms (post 5), and why Mixture-of-Experts replaces the *FFN* (post 9).

#### Inside the attention box

The block diagram draws attention as one box. Here's the data path inside it, with the tensor's shape down the right margin:

![Inside the multi-head attention module](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/attention-zoom-light.png){: .light width="900" }
![Inside the multi-head attention module](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/attention-zoom-dark.png){: .dark width="900" }

Three steps in it deserve a note:

1. **One input, three projections.** $Q$, $K$, $V$ are three learned *views of the same vector* — which is what makes self-attention "self."
2. **RoPE rotates $Q$ and $K$, never $V$.** Position belongs in the *matching* step. $V$ is the content you retrieve once matching is done; rotating it would corrupt the payload.
3. **Mask, then softmax — in that order.** Masking sets forbidden scores to $-\infty$ *before* normalizing, so surviving weights renormalize among themselves and each row still sums to 1. Masking afterwards is a classic bug.

### 6. The implementation, in five lines {#the-implementation-in-five-lines}

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

### 7. Why divide by sqrt(d_k)? {#why-divide-by-sqrt-dk}

The argument is short. If $q$ and $k$ have independent components with mean 0 and variance 1, then

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

is a sum of $d_k$ independent mean-zero terms. So $\operatorname{Var}(q \cdot k) = d_k$, and the standard deviation is $\sqrt{d_k}$.

Logits with a standard deviation of 32 aren't "large numbers" — they're a **temperature setting**. A softmax over scores that spread that wide puts essentially all its mass on the single largest one.

So let's measure it. Sample random queries and keys at several $d_k$, and report two diagnostics: the average largest weight (1.0 = fully one-hot) and the entropy in nats (0.0 = one-hot; $\ln 8 = 2.08$ = uniform over our 8 keys).

```text
  d_k   logit std  max w (raw)  H (raw)  max w (/sqrt)  H (/sqrt)
  -----------------------------------------------------------------
  4        1.9642       0.5222   1.2986         0.3401     1.7548
  16       3.9493       0.7428   0.6825         0.3523     1.7350
  64       7.9399       0.8738   0.3223         0.3506     1.7407
  256     15.9159       0.9326   0.1691         0.3533     1.7318
  1024    32.4533       0.9735   0.0674         0.3701     1.7103
```

![Softmax saturation without the sqrt(d_k) scale](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/softmax-saturation-light.png){: .light width="1000" }
![Softmax saturation without the sqrt(d_k) scale](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/softmax-saturation-dark.png){: .dark width="1000" }

Read the `logit std` column first: `1.96, 3.95, 7.94, 15.92, 32.45` — exactly $\sqrt{4}, \sqrt{16}, \sqrt{64}, \sqrt{256}, \sqrt{1024}$. The theory isn't approximately right, it's exactly right.

Then the consequence. Unscaled, entropy collapses from 1.30 nats to **0.067**; at $d_k = 1024$ the average largest weight is 0.97, so attention has stopped averaging and become a hard `argmax`. Scaled, entropy holds at **~1.74 across a 256× range of $d_k$**. That flat blue line is the entire justification for the constant.

Why does saturation hurt? Two reasons, and the second is what actually kills training:

1. **It stops being attention.** A near-one-hot distribution ignores all but one token.
2. **The gradient vanishes.** The softmax Jacobian is $\operatorname{diag}(p) - pp^\top$. As $p \to$ one-hot, every entry goes to zero. No gradient reaches $Q$ and $K$, and the model can't learn to attend differently.

That second point reframes it. $1/\sqrt{d_k}$ is not about overflow — softmax handles large logits fine by subtracting the row max. It's there to **keep the softmax in a regime where it still has a gradient**, whatever width you make the heads.

### 8. Causal masking {#causal-masking}

[§2](#the-formula-symbol-by-symbol) introduced the mask $M$. Here is what it actually does to the weights:

```text
        key0    key1    key2    key3    key4    key5
  q0    1.000    0.000    0.000    0.000    0.000    0.000
  q1    0.180    0.820    0.000    0.000    0.000    0.000
  q2    0.366    0.477    0.157    0.000    0.000    0.000
  q3    0.717    0.176    0.072    0.035    0.000    0.000
  q4    0.238    0.015    0.342    0.345    0.060    0.000
  q5    0.126    0.348    0.240    0.160    0.040    0.086
```

![Causal attention weight matrix](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/causal-mask-light.png){: .light width="820" }
![Causal attention weight matrix](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/causal-mask-dark.png){: .dark width="820" }

Note row `q0`: weight exactly 1.000 on itself. The first token has nothing else to attend to, so a softmax over a single unmasked score returns 1 — the mask makes the earliest tokens progressively less interesting, not broken.

This one line is why the same architecture serves both BERT (unmasked, good for understanding) and GPT (masked, good for generation).

### 9. RoPE: absolute rotation, relative score {#rope-absolute-rotation-relative-score}

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

![RoPE scores are invariant to absolute position, sensitive to relative offset](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/rope-relative-light.png){: .light width="1000" }
![RoPE scores are invariant to absolute position, sensitive to relative offset](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/rope-relative-dark.png){: .dark width="1000" }

Two panels, same y-axis. Left: slide both vectors along 128 positions, holding the gap at +3 — a flat line. Right: pin the query and sweep the offset — structure everywhere. Flat where you want invariance, expressive where you want sensitivity.

One consequence that returns in post 12: because the rotation happens *after* the $Q$/$K$ projections, RoPE doesn't commute with tricks that absorb those projections into neighbouring matrices — exactly the complication DeepSeek's MLA has to work around.

### 10. A silent MPS bug that deleted RoPE {#a-silent-mps-bug-that-deleted-rope}

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
- Code for this post: [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher), `uv run demo01`.
