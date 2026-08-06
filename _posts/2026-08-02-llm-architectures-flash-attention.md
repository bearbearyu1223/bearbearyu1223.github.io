---
title: "LLM Architecture Refresh [3]: Flash Attention Is Exact, and Here's the Proof"
date: 2026-08-02 01:00:00 -0700
categories: [LLM Architecture Refresh, Inference]
tags: [flash-attention, online-softmax, attention, memory-bandwidth, tiling, pytorch]
description: >-
  Implementing online softmax and tiled attention from scratch to show that
  Flash Attention returns the same answer as the textbook version — and that
  the win is memory traffic, not arithmetic.
math: true
---

## The optimization that doesn't change the answer

Post 2 ended on a principle: don't move bytes you don't have to. That was about weights and the KV cache. This post applies the same idea one level down, to attention itself.

The first thing to say about Flash Attention — and the thing that trips people up — is that **it is exact**. It is not a sparse approximation, not a low-rank factorization, not a kernel trick. It computes precisely the same function as the attention you wrote in post 1. It just computes it in an order that touches memory far less.

Which means it's a claim you can check. Let's check it.

```bash
uv run demo03
```

### Table of Contents

1. [The problem is memory traffic, not FLOPs](#the-problem-is-memory-traffic-not-flops)
2. [Online softmax: the one idea](#online-softmax-the-one-idea)
3. [Receipt 1: partial softmaxes compose](#receipt-1-partial-softmaxes-compose)
4. [Tiled attention in 40 lines](#tiled-attention-in-40-lines)
5. [Receipt 2: exact, and what "exact" is worth](#receipt-2-exact-and-what-exact-is-worth)
6. [Receipt 3: the memory that never gets allocated](#receipt-3-the-memory-that-never-gets-allocated)
7. [Causal masking gets a bonus](#causal-masking-gets-a-bonus)
8. [Where the speed actually comes from](#where-the-speed-actually-comes-from)
9. [Sidebar: the probe](#sidebar-the-probe)

---

### The problem is memory traffic, not FLOPs {#the-problem-is-memory-traffic-not-flops}

Standard attention does this:

1. Compute $S = QK^\top$ — an $n \times n$ matrix. **Write it to HBM.**
2. Read it back. Compute $P = \text{softmax}(S)$. **Write it to HBM.**
3. Read it back. Compute $O = PV$. Write the output.

Three round-trips through main memory for a matrix that grows quadratically with sequence length. At $n = 8192$ with 8 heads in fp32, that intermediate is **2 GiB** — per layer, per forward pass.

Now the hardware. On an A100, HBM bandwidth is roughly 2 TB/s, while on-chip SRAM runs at about 19 TB/s — an order of magnitude faster, but there's only ~20 MB of it. So attention at long context isn't waiting on arithmetic. It's waiting on the trip to HBM and back, exactly like decode in post 2.

The fix is the standard one for memory-bound problems: **fuse the steps so intermediates never leave fast memory**. Compute a tile of $S$, softmax it, multiply by $V$, accumulate, discard — all while the tile sits in SRAM.

There's an obvious objection, and it's the interesting part. Softmax has a denominator that sums over the *entire* row. How can you normalize a tile before you've seen all the tiles?

### Online softmax: the one idea {#online-softmax-the-one-idea}

Recall the numerically stable softmax subtracts the row max first:

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \qquad m = \max_j x_j
$$

You need $m$ before you can exponentiate anything, and $m$ depends on the whole row. That's the blocker.

The trick is to keep *running* statistics and retroactively correct them. Carry a running max $m$ and a running sum $\ell$. When a new block arrives with a larger max, every term you've already accumulated was rebased against the old max — so rescale it:

$$
m_{\text{new}} = \max(m_{\text{old}},\; \max(\text{block})), \qquad
\ell_{\text{new}} = \ell_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum e^{\text{block} - m_{\text{new}}}
$$

That correction factor $e^{m_{\text{old}} - m_{\text{new}}}$ is the whole idea. In code:

```python
m = torch.full((*lead, 1), float("-inf"))
l = torch.zeros((*lead, 1))

for start in range(0, n, block_size):
    block = x[..., start : start + block_size]
    m_new = torch.maximum(m, block.max(dim=-1, keepdim=True).values)
    l = l * torch.exp(m - m_new) + torch.exp(block - m_new).sum(dim=-1, keepdim=True)
    m = m_new
```

Note `l * torch.exp(m - m_new)`. When the max doesn't change, that factor is $e^0 = 1$ and costs nothing. When it does, it corrects the entire accumulated history in one multiply.

### Receipt 1: partial softmaxes compose {#receipt-1-partial-softmaxes-compose}

Does it actually agree with `torch.softmax`? Testing on logits scaled by 8, so each row spans a range of about 80 — enough that a careless implementation returns `inf` or `nan`:

```text
  block size  max |online - torch|  rows sum to
  -----------------------------------------------
  64                     5.960e-08       1.0000
  128                    1.192e-07       1.0000
  512                    1.788e-07       1.0000
  2048                   1.192e-07       1.0000
```

Agreement at `1e-7` in fp32, and the rows still sum to 1. **Block size changes the memory schedule, not the answer.** That's the property everything else rests on.

### Tiled attention in 40 lines {#tiled-attention-in-40-lines}

Now build attention on it. Outer loop over query blocks; inner loop streams key/value blocks past them, updating the running statistics *and* the accumulated output:

```python
for i in range(0, seq_q, block_q):
    qi = q[:, :, i : i + block_q]
    m = torch.full((batch, heads, rows, 1), float("-inf"))
    l = torch.zeros((batch, heads, rows, 1))
    acc = torch.zeros((batch, heads, rows, dim))

    for j in range(0, seq_k, block_k):
        kj, vj = k[:, :, j : j + block_k], v[:, :, j : j + block_k]
        scores = (qi @ kj.transpose(-2, -1)) * scale

        m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
        correction = torch.exp(m - m_new)
        p = torch.exp(scores - m_new)

        l = l * correction + p.sum(dim=-1, keepdim=True)
        acc = acc * correction + p @ vj   # rescale history, then add this tile
        m = m_new

    out[:, :, i : i + block_q] = acc / l
```

The line that carries the algorithm is `acc = acc * correction + p @ vj`. The accumulated output — not just the normalizer — has to be rebased when the max moves. Miss that and you get a plausible-looking result that is quietly wrong.

Also notice the division by `l` happens **once at the end**, outside the inner loop. You accumulate an unnormalized weighted sum and normalize last, which is what makes the whole thing associative.

### Receipt 2: exact, and what "exact" is worth {#receipt-2-exact-and-what-exact-is-worth}

Against `F.scaled_dot_product_attention`, across tile shapes:

```text
  causal=False                       causal=True
  tile (q x k)  max abs difference   tile (q x k)  max abs difference
  ----------------------------------  ----------------------------------
  64 x 64                4.768e-07   64 x 64                1.431e-06
  128 x 128              5.066e-07   128 x 128              1.431e-06
  256 x 512              4.917e-07   256 x 512              1.431e-06
```

That's floating-point reassociation noise — the same order as post 1's check. The tile shape doesn't move it, because the tile shape isn't part of the math.

Now the comparison that gives "exact" its meaning. Sliding-window attention is a *genuine* approximation — each query attends only to the last $w$ keys:

```text
  method                 max abs difference vs exact
  ----------------------------------------------------
  Flash / tiled (exact)                    1.431e-06
  sliding window, w=256                       0.6047
  sliding window, w=64                        1.5660
```

**Five orders of magnitude apart.** This is the distinction to hold onto: sparse attention, linear attention, and low-rank attention all change the function being computed, and you have to evaluate whether the change costs you anything. Flash Attention changes only the order of memory accesses. There is nothing to evaluate — if it produced different results, it would be a bug.

### Receipt 3: the memory that never gets allocated {#receipt-3-the-memory-that-never-gets-allocated}

```text
  seq   score matrix (n^2)  naive measured  one tile  tiled residual
  --------------------------------------------------------------------
  512              8.0 MiB         9.0 MiB   0.5 MiB         0.0 MiB
  1024            32.0 MiB        33.0 MiB   0.5 MiB         0.0 MiB
  2048           128.0 MiB       130.0 MiB   0.5 MiB         0.0 MiB
  4096           512.0 MiB       524.0 MiB   0.5 MiB         0.0 MiB

  8x the sequence, score matrix grows 64x
  tile size, independent of sequence 0.5 MiB
```

The measured naive column tracks the analytic $n^2$ column to within a couple of MiB. The tiled residual is 0 because each tile is freed as the loop advances — that's the claim, not a measurement artifact; the meaningful number is the constant 0.5 MiB in column four.

![Attention memory scaling](/assets/picture/2026-08-02-llm-architectures-flash-attention/memory-scaling-light.png){: .light width="1000" }
![Attention memory scaling](/assets/picture/2026-08-02-llm-architectures-flash-attention/memory-scaling-dark.png){: .dark width="1000" }

Extrapolating past what my laptop can hold:

```text
  seq     score matrix, 8 heads fp32
  ------------------------------------
  8192                       2.0 GiB
  16384                      8.0 GiB
  32768                     32.0 GiB
  131072                   512.0 GiB
```

Half a terabyte for one attention layer's intermediate at 128k context. **This is why long context was infeasible before 2022** — not because the FLOPs were unaffordable, but because you could not allocate the intermediate. $O(n^2) \to O(n)$ memory is the entire unlock.

### Causal masking gets a bonus {#causal-masking-gets-a-bonus}

Once you're looping over tiles, causal masking stops being a `masked_fill` and becomes a scheduling decision. If a key block starts after the query block ends, every score in that tile would be $-\infty$. So don't compute it:

```python
if causal and j > i + rows - 1:
    counters["blocks_skipped"] += 1
    continue
```

```text
  block size  tiles computed  tiles skipped  fraction skipped
  -------------------------------------------------------------
  128                    136            120               47%
  256                     36             28               44%
  512                     10              6               38%
```

![Causal tiling schematic](/assets/picture/2026-08-02-llm-architectures-flash-attention/tiling-light.png){: .light width="820" }
![Causal tiling schematic](/assets/picture/2026-08-02-llm-architectures-flash-attention/tiling-dark.png){: .dark width="820" }

Just under half the work disappears. The naive path computes that entire upper triangle, writes it to HBM, reads it back, and softmaxes it into zeros. Only the diagonal tiles need an actual mask — everything below is fully visible, everything above is skipped outright. Smaller blocks skip a larger fraction, because the diagonal band they must compute is thinner.

### Where the speed actually comes from {#where-the-speed-actually-comes-from}

An honest caveat: **the Python tiled loop above is slower than naive attention.** Every tile still round-trips through HBM, plus Python overhead per iteration. The algorithm buys memory; fusing it into a single kernel is what buys time.

So for timing, compare naive attention against `F.scaled_dot_product_attention`, which dispatches to a fused kernel doing exactly this:

```text
  seq   naive (ms)  fused SDPA (ms)  speedup
  --------------------------------------------
  512       1.2210           0.3307    3.69x
  1024      3.5331           0.7961    4.44x
  2048     13.4070           2.0592    6.51x
  4096     55.1674           7.0320    7.85x
```

![Naive vs fused attention timing](/assets/picture/2026-08-02-llm-architectures-flash-attention/timing-light.png){: .light width="1000" }
![Naive vs fused attention timing](/assets/picture/2026-08-02-llm-architectures-flash-attention/timing-dark.png){: .dark width="1000" }

The speedup **grows with sequence length** — 3.7× at 512, 7.9× at 4096 — because the naive path's memory traffic grows quadratically while the fused path's grows linearly. Extrapolate and it keeps widening.

One more piece: the backward pass. Training normally stores the attention probabilities for the backward pass, which is the $O(n^2)$ tensor you just avoided allocating. Flash Attention instead **recomputes** the tiles during the backward pass from $Q$, $K$, $V$ and the saved statistics. Recomputation sounds expensive, but it's arithmetic — and arithmetic is the resource you have in surplus. It's cheaper to redo the FLOPs than to have stored and re-read the result.

A note on versions, since interviews like the specifics. **FA2** improved work partitioning across warps and cut non-matmul FLOPs — tensor cores are roughly 16× faster at matmul than at anything else, so the non-matmul work dominates disproportionately. **FA3** targets Hopper: asynchronous copies via TMA, warp specialization, and FP8.

### Sidebar: the probe {#sidebar-the-probe}

> **"Does Flash Attention change your model's output?"**

**A weak answer:** "It's an approximation that trades a little accuracy for a big speedup on long sequences."

This is wrong, and it's a common wrong answer — the name sounds like the efficient-attention family (Linformer, Performer, Longformer), which *are* approximations.

**A stronger answer:** "No. It's exact — an IO-aware reordering of the same computation. It tiles Q, K and V so the score tiles stay in SRAM, and uses online softmax with a running max and sum so partial results compose. Outputs match a naive implementation to floating-point reassociation noise, around 1e-6 in fp32; I've measured it. Memory goes from $O(n^2)$ to $O(n)$, which is what made long context feasible, and speed comes from eliminating HBM round-trips. That's the opposite of sparse or linear attention, which genuinely change the function — sliding-window attention differs from exact attention by about 0.6 on the same test, five orders of magnitude more than Flash does."

The tell is whether someone distinguishes it from the approximate-attention family, and whether they know the mechanism is online softmax rather than "it's optimized CUDA."

### What's next {#whats-next}

Post 4 turns to **quantization**, where the tradeoff is real: fewer bytes per weight genuinely does lose information, and the interesting question is *what breaks first*. We'll implement INT8 and NF4 by hand, watch per-tensor scaling get destroyed by a single outlier channel, and see why perplexity is a misleading way to decide whether a quantized model is safe to ship.

### References

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022).
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023).
- Shah et al., [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) (2024).
- Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (2018) — where the running-statistics trick comes from.
- Rabe & Staats, [Self-attention Does Not Need $O(n^2)$ Memory](https://arxiv.org/abs/2112.05682) (2021).
- Code for this post: [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher), `uv run demo03`.
