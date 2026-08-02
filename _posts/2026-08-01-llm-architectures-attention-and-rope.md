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

I have read the attention equation many times. I can write it from memory. But when someone asks *why* the $1/\sqrt{d_k}$ is there, "for numerical stability" is the kind of answer that sounds fine and explains nothing — it's a phrase I'd absorbed rather than a thing I'd seen happen.

So this series takes a different approach. Every claim gets a **receipt**: a small program that prints the number the claim is asserting. Nothing here is bigger than it needs to be — the demos use toy tensors (`d_k=64`, 8 tokens, 6 positions) because a synthetic tensor makes the point as well as a 7B checkpoint, in one second instead of ten minutes.

All the code lives in a companion repo, [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher), managed with `uv`. It runs unchanged on Apple Silicon and on a Linux + NVIDIA box:

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync
uv run demo01
```

Every number and figure in this post came out of that command on my M-series Mac.

### First, what attention is {#first-what-attention-is}

Before measuring anything, the thirty-second version.

A language model reads a sequence of tokens and has to build a representation of each one. The hard part is that a token's meaning depends on other tokens, sometimes far away. In *"the trophy didn't fit in the suitcase because **it** was too big,"* resolving **it** requires looking back at two candidate nouns and picking one. A model that processes each token in isolation cannot do this at all.

Attention is the mechanism that lets a token **go and get information from other tokens**, and — crucially — *learn which ones to get it from*. For each token it computes a set of weights over all the other tokens, then returns a weighted average of what those tokens have to offer. Relevant tokens get large weights; irrelevant ones get weights near zero.

Every token produces three vectors for this, each from its own learned projection:

| | Question it answers | Role |
| --- | --- | --- |
| **Query** $Q$ | "What am I looking for?" | the token doing the looking |
| **Key** $K$ | "What am I?" | how a token advertises itself |
| **Value** $V$ | "What do I contribute if chosen?" | the content actually retrieved |

Compare every query against every key by dot product (large dot product = aligned = relevant), turn those scores into weights that sum to 1 with a softmax, and use the weights to average the values. That is the entire mechanism:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Reading it right to left: $QK^\top$ is every query scored against every key (an $n \times n$ matrix of relevance scores); $\sqrt{d_k}$ is a scaling constant we'll spend a whole section justifying; $\text{softmax}$ turns each row of scores into weights summing to 1; multiplying by $V$ takes the weighted average.

#### What a "head" is {#what-a-head-is}

You'll see "multi-head attention" everywhere, so it's worth being concrete about what a head is.

The problem with running attention just once is that **a softmax produces exactly one set of weights**. One opinion about which tokens matter. But a token usually needs several unrelated things at the same time — what noun this pronoun refers to, which verb governs this subject, which adjective modifies this noun. Forcing all of that through a single weighting gives you a blurry average of all three.

A **head** is one independent copy of the attention mechanism, working on a *slice* of the vector. Rather than run attention once on all 4,096 numbers, you cut the vector into (say) 32 slices of 128 numbers each. Each slice gets its own Q, K and V projections, computes its own scores, and produces its own answer about where to look. The 32 results are glued back together into a 4,096-vector and mixed by one final matrix.

$$
d_{model} = n_{heads} \times d_{head} \qquad 4096 = 32 \times 128
$$

The natural worry is that this must be 32× the work. It isn't — and that's the part worth seeing, because it explains why every model does it:

```text
  n_heads  head_dim (= d_k)  projection params  score FLOPs
  -----------------------------------------------------------
  1                    4096              67.1M        8.6 G
  8                     512              67.1M        8.6 G
  32                    128              67.1M        8.6 G
  64                     64              67.1M        8.6 G
```

**Identical, every row.** Heads are a *reshape of a fixed budget*, not extra machinery. The four projection matrices are the same size regardless — you're just deciding whether to treat their output as one wide space or many narrow ones. The score computation works out the same too: $n_{heads} \times n^2 \times d_{head} = n^2 \times d_{model}$, and the head count cancels.

So multiple heads are free, and they buy several attention patterns at once instead of one averaged compromise. Running the same input through four heads, here's where each one's final query places its attention:

```text
  head  pos0  pos1  pos2  pos3  pos4  pos5  entropy
  ---------------------------------------------------
  0     0.04  0.04  0.06  0.50  0.02  0.02     1.72
  1     0.09  0.18  0.08  0.04  0.20  0.07     2.28
  2     0.01  0.33  0.03  0.01  0.18  0.03     1.86
  3     0.05  0.08  0.12  0.13  0.00  0.16     2.19
```

Four genuinely different distributions — head 0 concentrates half its weight on position 3 while head 1 spreads out. (These are random weights, so the differences here only show heads aren't redundant copies; they aren't *specialization*.) In trained models the specialization is real and has been catalogued: interpretability researchers have found "previous-token heads" that consistently look one step back, and "induction heads" that spot a repeated pattern and predict its continuation.

One connection that's easy to miss and matters shortly: **$d_{head}$ is the $d_k$ in the attention formula.** The $\sqrt{d_k}$ scale is set by the per-head width — 128, not 4,096. When we measure that scale below, that's the number in play.

There's also a trade-off lurking. More heads means more distinct patterns but narrower ones — 64 heads gives each only 64 dimensions to represent a query in. Models land at 32–64 heads because the extremes are both bad. And since each head carries its own keys and values, the head count is exactly what drives the memory cost that post 2 is about.

### Where attention sits in the model {#where-attention-sits-in-the-model}

Attention is a *component*, not the whole model. A decoder-only transformer stacks N identical blocks, and each block does two things: attention, then a position-wise feed-forward network, each wrapped in a residual connection with a normalization layer.

![Where attention sits in a decoder block](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/block-anatomy-light.png){: .light width="700" }
![Where attention sits in a decoder block](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/block-anatomy-dark.png){: .dark width="700" }

That diagram has five kinds of box, and the names are all jargon. Here's each one in plain English.

#### Token embeddings — turning words into numbers

The model can't work with text, only with numbers. So the first step is a lookup table: every token in the vocabulary owns a list of numbers — say 4,096 of them — and that list *is* the model's representation of that token. This list is called a **vector**, and its length (4,096) is called $d_{model}$.

At the start, that vector encodes only "which token is this." Every block after it edits the vector so it comes to mean "which token is this, *in this particular context*." That's the whole job of the stack: the word *bank* enters with one generic vector, and after 32 blocks of reading its neighbors, it leaves with a vector that has been nudged toward *riverbank* or toward *savings account*.

#### RMSNorm — keeping the numbers in a sane range

As a vector passes through dozens of layers, each doing arithmetic on it, the numbers tend to drift — growing until they overflow, or shrinking until they vanish. **Normalization** rescales a vector back to a standard size before handing it to the next layer, so every layer receives inputs in the range it expects.

The mixing-desk analogy is close enough: you're setting every track to a consistent level so nothing clips and nothing is inaudible. Note it rescales the *whole vector at once* — the relative proportions between the 4,096 numbers survive, only the overall magnitude is standardized.

The original transformer used **LayerNorm**: subtract the vector's mean, then divide by its standard deviation. **RMSNorm** does less — it skips the mean subtraction entirely and just divides by the root-mean-square of the numbers:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{n}\sum x_i^2}} \cdot g
$$

where $g$ is a learned per-dimension scale. **Its contribution is a discovery, not an invention**: the mean-centering half of LayerNorm turned out not to be doing meaningful work. Dropping it costs no quality and saves a pass over the data, so essentially every model since Llama uses it. This is a common pattern in this field — a component gets simpler once someone checks which half of it mattered.

#### The FFN — where the model actually knows things

**FFN** stands for **feed-forward network**, and it's the plainest thing in the architecture: take a vector, multiply it by a matrix to make it *wider*, apply a nonlinear function, multiply by another matrix to bring it back to the original width.

$$
\text{FFN}(x) = W_{\text{down}}\big(\,\text{nonlinearity}(W_{\text{up}}\,x)\,\big)
$$

Typically it expands 4× — from 4,096 up to 16,384 and back. Why expand and then contract? The wide middle layer gives the model room to test many specialized questions at once. Loosely: each of those 16,384 intermediate dimensions can learn to respond to some particular pattern ("is this a plural noun?", "is this talking about France?"), the nonlinearity keeps the ones that fire and suppresses the ones that don't, and the down-projection recombines the survivors back into an updated vector.

Two things make the FFN important out of proportion to how boring it looks:

- **It is where knowledge lives.** Interpretability work has traced specific facts to specific FFN weights, and a productive way to read the FFN is as a key–value memory: the up-projection matches patterns, the down-projection writes the associated content back. When we say a model "knows" that Paris is in France, that association is mostly stored here.
- **It is most of the model.** As the parameter table below shows, the FFN is 67–82% of every block.

The word **position-wise** attached to it just means each token's vector goes through the FFN completely on its own — token 5 has no idea token 6 exists. That's the exact complement to attention, which is the only place tokens see each other.

#### SwiGLU — a better nonlinearity for the FFN

The original FFN used **ReLU** as its nonlinearity: keep positive values, set negatives to zero. Crude but effective, and just two matrices.

**SwiGLU** replaces it with a **gate**. Instead of one up-projection, run two in parallel. One produces the content. The other is squashed by a smooth S-shaped function (SiLU, also called Swish) into something that acts like a set of dimmer switches, and the two are multiplied together element by element:

$$
\text{SwiGLU}(x) = W_{\text{down}}\big(\,\text{SiLU}(W_{\text{gate}}\,x) \;\odot\; W_{\text{up}}\,x\,\big)
$$

In words: **the network learns, per feature and per input, how much of each signal to let through** — rather than applying one fixed rule (ReLU's "negatives become zero") to everything. It's the difference between a hard on/off switch and a dimmer the model controls.

The cost is a third matrix. To keep the parameter count fair, models shrink the expansion from $4d$ to about $\tfrac{8}{3}d$ — which is exactly why Llama-3-8B's $d_{ff}$ is 14,336 rather than a rounder 16,384. The benefit is consistently lower loss at matched parameters, which is why it's now standard.

Worth knowing for its honesty: the paper that introduced it tested a family of gated variants, found they simply worked better, and offered no clean theory. It closes by attributing their success "to divine benevolence." Much of what's in a modern transformer is there because it measured better, not because someone derived it.

#### Putting the block together

Now the diagram reads straightforwardly. Each block does two passes over the vector:

1. **Normalize → attention → add back.** The token gathers context from other tokens.
2. **Normalize → FFN → add back.** The token processes what it gathered, on its own.

Stack that N times (32 for an 8B model, 80 for a 70B), finish with one more normalization, and multiply by an output matrix — the **LM head** — that converts the final vector into one score per vocabulary word. Softmax those scores and you have a probability distribution over the next token.

Two structural details are worth naming, because they're what "modern transformer" means in practice.

**The residual is a real bypass.** Each sublayer's input branches off, skips both the norm and the sublayer, and is added back at the $\oplus$. So a sublayer never computes its output — it computes a *correction* to what came in: $x \leftarrow x + \text{Attention}(\text{Norm}(x))$. If the sublayer learns nothing useful, the block degrades to the identity rather than to noise. That unbroken path from embeddings to output is also the path the gradient travels back down, undiminished, which is what lets you stack 80 of these.

**"Pre-norm" describes where the norm sits relative to that bypass.** The original 2017 transformer normalized *after* adding the residual, putting a norm directly on the trunk. Every modern decoder moved the norm *inside* the branch instead, as drawn. The payoff is that the residual highway stays unnormalized end to end — which is why deep models train stably today without the learning-rate warmup gymnastics the original recipe needed.

Which gets more of the model? Counting parameters in a single block:

```text
  block shape                  attention params  FFN params  FFN share
  ----------------------------------------------------------------------
  GPT-2 style (MHA, ReLU FFN)             10.2M       20.5M        67%
  Llama-3-8B (GQA, SwiGLU)                41.9M      176.2M        81%
  Llama-3-70B (GQA, SwiGLU)              151.0M      704.6M        82%
```

Attention gets the name and the diagrams, but it's the **minority of the weights**. The classic two-thirds figure comes from the original shapes — attention is $4d^2$ (four $d \times d$ projections), the FFN is $2 \cdot d \cdot 4d = 8d^2$. Modern decoders push it further from both ends: GQA shrinks the K and V projections while SwiGLU adds a third FFN matrix.

A useful summary: **routing lives in attention, knowledge lives in the FFN.** That framing comes back repeatedly — it's why LoRA targeting only attention underperforms (post 5), and why Mixture-of-Experts replaces the *FFN* rather than the attention (post 9).

The whole block on one page:

| Piece | Its job, in one line | What it replaced, and why |
| --- | --- | --- |
| **Embeddings** | Turn each token into a vector of numbers | — |
| **RMSNorm** | Rescale the vector so the next layer sees a sane range | LayerNorm; mean-centering turned out not to matter, so it was dropped for speed |
| **Attention** | Let each token gather information from other tokens | The only component that mixes across positions |
| **Residual (⊕)** | Add the sublayer's output back to its input | Lets each sublayer learn a small correction, and gives the gradient a clear path down |
| **SwiGLU FFN** | Process each token on its own; store most of the knowledge | ReLU FFN; a learned gate beats a fixed threshold, at matched parameters |
| **LM head** | Turn the final vector into a score per vocabulary word | — |

If you take one thing from this section: **attention is the only place tokens talk to each other.** Everything else in the stack is a per-token transformation. That single fact explains why the KV cache works (post 2), why the FFN is the natural thing to shard across experts (post 9), and why position has to be injected deliberately — which is the last thing this post measures.

With that map in place, the rest of this post takes the mechanism apart and measures it.

### Table of Contents

1. [First, what attention is](#first-what-attention-is)
2. [Where attention sits in the model](#where-attention-sits-in-the-model)
3. [Attention is a soft dictionary lookup](#attention-is-a-soft-dictionary-lookup)
4. [The whole mechanism, in five lines](#the-whole-mechanism-in-five-lines)
5. [Receipt 1: our version vs the fused kernel](#receipt-1-our-version-vs-the-fused-kernel)
6. [Why divide by sqrt(d_k)?](#why-divide-by-sqrt-dk)
7. [Causal masking: what makes it autoregressive](#causal-masking-what-makes-it-autoregressive)
8. [RoPE: absolute rotation, relative score](#rope-absolute-rotation-relative-score)
9. [A detour: how a silent MPS bug deleted RoPE](#a-detour-how-a-silent-mps-bug-deleted-rope)
10. [Sidebar: the probe](#sidebar-the-probe)
11. [What's next](#whats-next)

---

### Attention is a soft dictionary lookup {#attention-is-a-soft-dictionary-lookup}

The mental model that stuck for me is a **Python dictionary with fuzzy matching**.

A normal dictionary lookup is exact. You hand it a key, it finds the one entry whose key matches, and returns that entry's value:

```python
scores = {"cat": 0, "mat": 1, "sat": 0}   # exact match: one winner
```

Attention softens every step of that. Testing keys for *equality* becomes taking a dot product, which measures how aligned a query and key are. Returning *one* value becomes returning a weighted average of all of them.

So a "lookup" returns 70% of the value at position 3, 20% at position 1, and a sprinkle of everything else. That softness is the whole trick: a hard lookup has no gradient — nudge the query slightly and either nothing changes or the winner flips discontinuously. A soft one is differentiable everywhere, so the model can **learn** what to look for.

Two consequences fall out immediately, and both matter later in the series:

- **The dot product is the only place tokens interact.** Everything else in a transformer block — the FFN, the norms — processes each position independently. All the mixing happens in $QK^\top$.
- **Nothing in the equation knows about order.** Shuffle the tokens and you get the same set of outputs, shuffled. Position has to be injected separately, which is what RoPE is for, below.

### The whole mechanism, in five lines {#the-whole-mechanism-in-five-lines}

Here is the entire thing. Shapes are `(..., seq, head_dim)`, where the leading dimensions carry batch and heads — multi-head attention is not a different algorithm, just this function applied to several projections in parallel.

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

I return the `weights` alongside the output, because the weights are exactly what the optimized kernel throws away — and they are the thing worth looking at.

### Receipt 1: our version vs the fused kernel {#receipt-1-our-version-vs-the-fused-kernel}

Before trusting anything else, check the implementation against PyTorch's fused `F.scaled_dot_product_attention`:

```python
ours, _ = scaled_dot_product_attention(q, k, v, causal=causal)
theirs = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
print((ours - theirs).abs().max().item())
```

```text
  max |ours - torch|  (causal=False) 5.364e-07
  max |ours - torch|  (causal=True)  4.768e-07
```

Agreement to `5e-07` in float32 — that is floating-point reassociation noise, not an algorithmic difference. This is worth internalizing early, because it is the same fact that makes Flash Attention work: **the fast kernel is an optimization, not an approximation.** It reorders memory traffic and adds up the same numbers in a different order. We'll come back to that in post 3.

### Why divide by sqrt(d_k)? {#why-divide-by-sqrt-dk}

Now the part I wanted to actually see.

The argument is short. If $q$ and $k$ have independent components with mean 0 and variance 1, then

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

is a sum of $d_k$ independent mean-zero terms, so $\operatorname{Var}(q \cdot k) = d_k$ and its standard deviation is $\sqrt{d_k}$. Logits with a standard deviation of 32 going into a softmax are not "large numbers" — they are a **temperature setting**. Softmax over logits that spread that wide puts essentially all the mass on the single largest one.

So let's measure it. Sample random queries and keys at several `d_k`, and report two diagnostics: the average largest attention weight (1.0 = fully one-hot), and the entropy of the weight distribution in nats (0.0 = one-hot, $\ln 8 = 2.079$ = uniform over our 8 keys).

```python
for d_k in (4, 16, 64, 256, 1024):
    q = torch.randn(trials, 1, d_k, device=device)
    k = torch.randn(trials, seq, d_k, device=device)
    logits = (q @ k.transpose(-2, -1)).squeeze(1)

    for name, scale in (("unscaled", 1.0), ("scaled", 1.0 / math.sqrt(d_k))):
        w = torch.softmax(logits * scale, dim=-1)
        entropy = -(w * torch.log(w.clamp_min(1e-12))).sum(-1).mean()
```

```text
  d_k   logit std  max w (raw)  H (raw)  max w (/sqrt)  H (/sqrt)
  -----------------------------------------------------------------
  4        1.9642       0.5222   1.2986         0.3401     1.7548
  16       3.9493       0.7428   0.6825         0.3523     1.7350
  64       7.9399       0.8738   0.3223         0.3506     1.7407
  256     15.9159       0.9326   0.1691         0.3533     1.7318
  1024    32.4533       0.9735   0.0674         0.3701     1.7103

  uniform-over-8 entropy would be ln(8) = 2.0794
```

![Softmax saturation without the sqrt(d_k) scale](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/softmax-saturation-light.png){: .light width="1000" }
![Softmax saturation without the sqrt(d_k) scale](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/softmax-saturation-dark.png){: .dark width="1000" }

Read the `logit std` column first: `1.96, 3.95, 7.94, 15.92, 32.45`. Those are $\sqrt{4}, \sqrt{16}, \sqrt{64}, \sqrt{256}, \sqrt{1024}$. The theory is not approximately right, it is exactly right.

Then the consequence. Unscaled, entropy falls from 1.30 nats to **0.067** — at `d_k=1024` the average largest weight is 0.97, so attention has stopped being a weighted average and become a hard `argmax`. Scaled, entropy sits at **~1.74 across a 256× range of `d_k`**. That flat blue line is the entire justification for the constant.

Why does saturation hurt? Two reasons, and the second is the one that actually kills training:

1. **It stops being attention.** A near-one-hot distribution ignores all but one token, so the mechanism can't blend information at all.
2. **The gradient vanishes.** The Jacobian of softmax is $\operatorname{diag}(p) - pp^\top$. As $p \to$ one-hot, every entry of that matrix goes to zero. No gradient flows back to $Q$ and $K$, and the model cannot learn to attend differently. It gets stuck.

That second point reframes the whole thing. $1/\sqrt{d_k}$ is not "for numerical stability" in the overflow sense — softmax handles large logits fine by subtracting the max. It is there to **keep the softmax in a regime where it still has a gradient**, independent of how wide you make the heads.

### Causal masking: what makes it autoregressive {#causal-masking-what-makes-it-autoregressive}

Attention as written lets every token see every other token. For a language model that is cheating: predicting token 5 while looking at token 6 is not prediction. The fix is one line — set the scores above the diagonal to $-\infty$ before the softmax, so $e^{-\infty} = 0$ and those positions receive exactly zero weight.

Printing the weight matrix makes the structure obvious:

```text
        key0    key1    key2    key3    key4    key5
  q0    1.000    0.000    0.000    0.000    0.000    0.000
  q1    0.180    0.820    0.000    0.000    0.000    0.000
  q2    0.366    0.477    0.157    0.000    0.000    0.000
  q3    0.717    0.176    0.072    0.035    0.000    0.000
  q4    0.238    0.015    0.342    0.345    0.060    0.000
  q5    0.126    0.348    0.240    0.160    0.040    0.086

  mass on future positions           0.0000
  each row sums to 1                 yes
```

![Causal attention weight matrix](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/causal-mask-light.png){: .light width="820" }
![Causal attention weight matrix](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/causal-mask-dark.png){: .dark width="820" }

Note row `q0`: a weight of exactly 1.000 on itself. The first token has nothing to attend to but itself, so softmax over a single unmasked score returns 1. And every row still sums to 1 — masking happens *before* the softmax, so the surviving weights renormalize among themselves. Masking after the softmax would break that and is a classic bug.

This one line is why the same architecture serves both BERT (no mask, every token sees everything, good for understanding) and GPT (masked, good for generation).

### RoPE: absolute rotation, relative score {#rope-absolute-rotation-relative-score}

Back to the gap noted at the start: attention is permutation-equivariant. It has no idea what order the tokens came in. Something has to inject position.

The original transformer added a position vector to each embedding. That works, but it encodes *absolute* position, and what language actually cares about is mostly *relative* — "the adjective two tokens back" is a useful pattern at position 5 and at position 5000.

**Rotary Position Embedding** gets relative position out of absolute rotation, using one fact from high-school geometry: rotating two vectors and taking their dot product gives you something that depends only on the *difference* of the rotation angles.

$$
\langle R_m q,\; R_n k \rangle = \langle R_{m-n}\, q,\; k \rangle
$$

Split the head dimension into 2D subspaces, and rotate the $i$-th subspace by an angle of $\text{position} \times \theta_i$, where the frequencies $\theta_i$ decrease geometrically:

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

Low $i$ rotates fast (resolving nearby tokens), high $i$ rotates slowly (carrying long-range position). Every context-extension method you've heard of — position interpolation, NTK-aware scaling, YaRN — is some way of **rescaling that frequency vector** so the rotations a model learned at 4k tokens still mean something at 128k.

Now the receipt. Take one fixed query vector and one fixed key vector, place them at wildly different absolute positions but always 3 apart, and compare the scores:

```text
  query pos m  key pos n  offset m-n  q_m . k_n
  -----------------------------------------------
  0                    3          -3    -1.1905
  5                    8          -3    -1.1905
  105                108          -3    -1.1905
  4093              4096          -3    -1.1905

  spread across absolute positions   1.073e-06
```

Position 5 and position 4093 give the same score to seven digits. The model does not need to relearn "3 tokens back" for every position in the context window — it gets that invariance from the geometry, with **zero learned parameters**.

And the other half of the claim: the score does still respond to relative distance. Pin the query at 0, sweep the key's position, and the score moves around freely.

![RoPE scores are invariant to absolute position, sensitive to relative offset](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/rope-relative-light.png){: .light width="1000" }
![RoPE scores are invariant to absolute position, sensitive to relative offset](/assets/picture/2026-08-01-llm-architectures-attention-and-rope/rope-relative-dark.png){: .dark width="1000" }

Two panels, same y-axis. Left: slide both vectors along 128 positions holding the gap at +3 — a perfectly flat line. Right: pin the query and sweep the offset — structure everywhere. Flat where you want invariance, expressive where you want sensitivity. That is the whole design in one picture.

One more practical consequence worth flagging now, because it comes back when we discuss DeepSeek's MLA in post 12: **RoPE is applied to $q$ and $k$ only, never to $v$.** Values are content, not addresses. And because the rotation happens after the $Q$/$K$ projections, RoPE does not commute with tricks that try to absorb those projections into neighboring matrices — which is exactly the complication MLA has to work around.

### A detour: how a silent MPS bug deleted RoPE {#a-detour-how-a-silent-mps-bug-deleted-rope}

While building this demo, the RoPE section printed a score that was *identical for every offset*. Not wrong-looking — suspiciously perfect. The culprit turned out to be one line, and it is worth sharing because it fails silently on Apple Silicon.

The angle table needs float64: at position 4096 the fastest frequency has accumulated ~4096 radians, and float32 has about 7 decimal digits to spend on an argument that large. So I moved positions to the CPU and cast in one call — the idiomatic thing to write:

```python
pos = positions.to("cpu", torch.float64)     # looks fine. is not.
```

On an **int64 MPS** tensor, torch 2.13 reinterprets the bits rather than converting them:

```python
p = torch.tensor([105], device='mps')
p.to('cpu', torch.float64)    # tensor([5.1877e-322])   <- the bit pattern of 105, read as a float
p.cpu().to(torch.float64)     # tensor([105.])          <- correct
```

`5.19e-322` is a denormal indistinguishable from zero. So all the angles became 0, `cos = 1`, `sin = 0`, and `x * 1 + rotate_half(x) * 0` returned `x` unchanged. **RoPE had silently degraded into the identity function.** No exception, no warning — just a model with no positional information whatsoever, which would have trained to a mediocre loss and left me blaming the hyperparameters.

The fix is to separate the device move from the dtype cast. The lesson generalizes past this one bug: when a numerical result looks *too* clean, verify it on a second device before believing it. A CPU cross-check (`LLMR_DEVICE=cpu uv run demo01`) takes two seconds and would have caught this immediately.

### Sidebar: the probe {#sidebar-the-probe}

A question I've seen separate people who have read the paper from people who have read a summary of it:

> **"Why is there a $1/\sqrt{d_k}$ in the attention equation?"**

**A weak answer:** "For numerical stability — it keeps the dot products from getting too large."

This isn't wrong, but it doesn't survive a follow-up. Softmax is already robust to large logits: every implementation subtracts the row max first, so `[1000, 1001, 1002]` and `[-2, -1, 0]` produce identical output. Overflow is not the problem being solved.

**A stronger answer:** "The dot product of two $d_k$-dimensional unit-variance vectors has variance $d_k$, so without the scale the logits grow like $\sqrt{d_k}$ and the softmax sharpens as you widen the heads. Once it saturates toward one-hot, the softmax Jacobian $\operatorname{diag}(p) - pp^\top$ goes to zero and no gradient reaches $Q$ and $K$ — the model can't learn what to attend to. The scale keeps the temperature roughly constant so that head width is a free architectural choice instead of something that silently changes training dynamics."

The difference is that the second answer says what breaks, and at which step.

### What's next {#whats-next}

Post 2 takes the same measure-it-yourself approach to the **KV cache** — the thing that actually decides what you can serve. We'll time generation with and without the cache, work out why decode is memory-bandwidth-bound while prefill is compute-bound, and see why that single distinction explains batching, quantization, and speculative decoding all at once.

### References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — the scale factor is justified in a footnote in §3.2.1.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021).
- Chen et al., [Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/abs/2306.15595) (2023).
- Peng et al., [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) (2023).
- Code for this post: [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher), `uv run demo01`.
