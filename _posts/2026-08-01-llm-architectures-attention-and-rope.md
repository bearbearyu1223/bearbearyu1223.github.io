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

### Table of Contents

1. [Attention is a soft dictionary lookup](#attention-is-a-soft-dictionary-lookup)
2. [The whole mechanism, in five lines](#the-whole-mechanism-in-five-lines)
3. [Receipt 1: our version vs the fused kernel](#receipt-1-our-version-vs-the-fused-kernel)
4. [Why divide by sqrt(d_k)?](#why-divide-by-sqrt-dk)
5. [Causal masking: what makes it autoregressive](#causal-masking-what-makes-it-autoregressive)
6. [RoPE: absolute rotation, relative score](#rope-absolute-rotation-relative-score)
7. [A detour: how a silent MPS bug deleted RoPE](#a-detour-how-a-silent-mps-bug-deleted-rope)
8. [Sidebar: the probe](#sidebar-the-probe)
9. [What's next](#whats-next)

---

### Attention is a soft dictionary lookup {#attention-is-a-soft-dictionary-lookup}

The mental model that stuck for me is a **Python dictionary with fuzzy matching**.

A normal dictionary lookup is exact. You hand it a key, it finds the one entry whose key matches, and returns that entry's value:

```python
scores = {"cat": 0, "mat": 1, "sat": 0}   # exact match: one winner
```

Attention softens every step of that. Each token emits a **query** — "what am I looking for?". Every token emits a **key** — "what am I?" — and a **value** — "what do I contribute if you pick me?". Instead of testing keys for equality, you take a dot product, which measures how *aligned* the query and key are. Instead of returning one value, you return a weighted average of all of them, with the weights from a softmax over those alignment scores.

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

So a "lookup" returns 70% of the value at position 3, 20% at position 1, and a sprinkle of everything else. That softness is the whole trick: it is differentiable, so the model can *learn* what to look for.

Two consequences fall out immediately, and both matter later in the series:

- **The dot product is the only place tokens interact.** Everything else in a transformer block — the FFN, the norms — processes each position independently. All the mixing happens in $QK^\top$.
- **Nothing in the equation knows about order.** Shuffle the tokens and you get the same set of outputs, shuffled. Position has to be injected separately, which is what §6 is about.

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
