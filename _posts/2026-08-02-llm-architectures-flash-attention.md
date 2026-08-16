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
published: false
---

## The optimization that doesn't change the answer

[Post 2](/posts/llm-architectures-kv-cache/) ended on a principle: don't move bytes you don't have to. That was about weights and the KV cache. This post applies the same idea one level down, to attention itself.

The first thing to say about Flash Attention — Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022) — and the thing that trips people up, is that **it is exact**. The word is in the title for a reason. It is not a sparse approximation, not a low-rank factorization, not a kernel trick. It computes precisely the same function as the attention in [post 1](/posts/llm-architectures-attention-and-rope/). It just computes it in an order that touches memory far less.

Which means it's a claim you can check — and as in the earlier posts, every claim here gets a **receipt** you can run yourself, from the companion repo [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher):

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo03
```

Every number and figure below came out of that command on my M-series Mac. The code for this post is in [`demos/d03_flash_attention.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d03_flash_attention.py).

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

### 1. The problem is memory traffic, not FLOPs {#the-problem-is-memory-traffic-not-flops}

Two names for memory first, because the whole argument is about the distance between them. **HBM** is *high-bandwidth memory* — the GPU's main memory, the tens of gigabytes a spec sheet quotes when it says "80 GB". **SRAM** is the small pool of memory sitting on the chip itself, next to the arithmetic units. HBM is where your tensors live; SRAM is where they have to be for the chip to touch them, and everything moves between the two.

Standard attention does this:

1. Compute $S = QK^\top$ — an $n \times n$ matrix. **Write it to HBM.**
2. Read it back. Compute $P = \text{softmax}(S)$. **Write it to HBM.**
3. Read it back. Compute $O = PV$. Write the output.

Three round-trips through main memory for a matrix that grows quadratically with sequence length. At $n = 8192$ with 8 heads in fp32, that intermediate is **2 GiB** — per layer, per forward pass.

Now the hardware, and the two numbers that matter:

| | Bandwidth | How much of it |
| --- | --- | --- |
| **HBM** — main memory | ~2 TB/s | 40–80 GB |
| **SRAM** — on-chip | ~19 TB/s | ~20 MB |

Roughly an order of magnitude apart in speed, and four orders apart in size. That combination is the entire problem: SRAM is fast enough to keep the arithmetic units fed, and far too small to hold an $n \times n$ score matrix. So attention at long context isn't waiting on arithmetic. It's waiting on the trip to HBM and back, exactly like decode in [post 2](/posts/llm-architectures-kv-cache/).

The fix is the standard one for memory-bound problems: **fuse the steps so intermediates never leave fast memory**. Compute a tile of $S$, softmax it, multiply by $V$, accumulate, discard — all while the tile sits in SRAM.

There's an obvious objection, and it's the interesting part. Softmax has a denominator that sums over the *entire* row. How can you normalize a tile before you've seen all the tiles?

### 2. Online softmax: the one idea {#online-softmax-the-one-idea}

Recall the numerically stable softmax subtracts the row max first:

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \qquad m = \max_j x_j
$$

You need $m$ before you can exponentiate anything, and $m$ depends on the whole row. That's the blocker.

The trick predates Flash Attention by four years: Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (2018), showed you can compute the normalizer in a single streaming pass instead of two. Keep *running* statistics and retroactively correct them. Carry a running max $m$ and a running sum $\ell$. When a new block arrives with a larger max, every term you've already accumulated was rebased against the old max — so rescale it:

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

### 3. Receipt 1: partial softmaxes compose {#receipt-1-partial-softmaxes-compose}

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

### 4. Tiled attention in 40 lines {#tiled-attention-in-40-lines}

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

### 5. Receipt 2: exact, and what "exact" is worth {#receipt-2-exact-and-what-exact-is-worth}

Against `F.scaled_dot_product_attention`, across tile shapes:

```text
  causal=False: tiled vs F.scaled_dot_product_attention

  tile (q x k)  max abs difference
  ----------------------------------
  64 x 64                4.768e-07
  128 x 128              5.066e-07
  256 x 512              4.917e-07

  causal=True: tiled vs F.scaled_dot_product_attention

  tile (q x k)  max abs difference
  ----------------------------------
  64 x 64                1.431e-06
  128 x 128              1.431e-06
  256 x 512              1.431e-06
```

That's floating-point reassociation noise — the same order as [post 1](/posts/llm-architectures-attention-and-rope/)'s check. The tile shape doesn't move it, because the tile shape isn't part of the math.

Now the comparison that gives "exact" its meaning. Sliding-window attention is a *genuine* approximation — each query attends only to the last $w$ keys:

```text
  method                 max abs difference vs exact
  ----------------------------------------------------
  Flash / tiled (exact)                    1.431e-06
  sliding window, w=256                       0.6047
  sliding window, w=64                        1.5660
```

**Five orders of magnitude apart.** This is the distinction to hold onto: sparse attention, linear attention, and low-rank attention all change the function being computed, and you have to evaluate whether the change costs you anything. Flash Attention changes only the order of memory accesses. There is nothing to evaluate — if it produced different results, it would be a bug.

### 6. Receipt 3: the memory that never gets allocated {#receipt-3-the-memory-that-never-gets-allocated}

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

![Attention memory scaling](/assets/picture/2026-08-02-llm-architectures-flash-attention/memory-scaling-light.png){: .light width="1000" height="684" }
![Attention memory scaling](/assets/picture/2026-08-02-llm-architectures-flash-attention/memory-scaling-dark.png){: .dark width="1000" height="684" }

Extrapolating past what my laptop can hold:

```text
  seq     score matrix, 8 heads fp32
  ------------------------------------
  8192                       2.0 GiB
  16384                      8.0 GiB
  32768                     32.0 GiB
  131072                   512.0 GiB
```

Half a terabyte for one attention layer's intermediate at 128k context. **This is why long context was infeasible before 2022** — not because the FLOPs were unaffordable, but because you could not allocate the intermediate. $O(n^2) \to O(n)$ memory is the entire unlock, and Rabe & Staats, [Self-attention Does Not Need $O(n^2)$ Memory](https://arxiv.org/abs/2112.05682) (2021), had made the same point a year earlier from the memory side alone, without the IO-aware kernel that turned it into a speedup.

### 7. Causal masking gets a bonus {#causal-masking-gets-a-bonus}

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

![Causal tiling schematic](/assets/picture/2026-08-02-llm-architectures-flash-attention/tiling-light.png){: .light width="820" height="781" }
![Causal tiling schematic](/assets/picture/2026-08-02-llm-architectures-flash-attention/tiling-dark.png){: .dark width="820" height="781" }

Just under half the work disappears. The naive path computes that entire upper triangle, writes it to HBM, reads it back, and softmaxes it into zeros. Only the diagonal tiles need an actual mask — everything below is fully visible, everything above is skipped outright. Smaller blocks skip a larger fraction, because the diagonal band they must compute is thinner.

### 8. Where the speed actually comes from {#where-the-speed-actually-comes-from}

An honest caveat: **the Python tiled loop above is slower than naive attention.** Every tile still round-trips through HBM, plus Python overhead per iteration. The algorithm buys memory; fusing it into a single kernel is what buys time.

So for timing, compare naive attention against `F.scaled_dot_product_attention`, which dispatches to a fused kernel doing exactly this:

```text
  seq   naive (ms)  fused SDPA (ms)  speedup
  --------------------------------------------
  512       1.1330           0.3411    3.32x
  1024      3.5072           0.7662    4.58x
  2048     12.7797           2.0381    6.27x
  4096     54.8068           6.9735    7.86x
```

![Naive vs fused attention timing](/assets/picture/2026-08-02-llm-architectures-flash-attention/timing-light.png){: .light width="1000" height="684" }
![Naive vs fused attention timing](/assets/picture/2026-08-02-llm-architectures-flash-attention/timing-dark.png){: .dark width="1000" height="684" }

The speedup **grows with sequence length** — about 3× at 512, about 8× at 4096 — because the naive path's memory traffic grows quadratically while the fused path's grows linearly. Extrapolate and it keeps widening. (Wall-clock again, so the exact multiples move a little between runs; the widening is the part that holds.)

One more piece: the backward pass. Training normally stores the attention probabilities for the backward pass, which is the $O(n^2)$ tensor you just avoided allocating. Flash Attention instead **recomputes** the tiles during the backward pass from $Q$, $K$, $V$ and the saved statistics. Recomputation sounds expensive, but it's arithmetic — and arithmetic is the resource you have in surplus. It's cheaper to redo the FLOPs than to have stored and re-read the result.

A note on versions, since interviews like the specifics. Two more pieces of vocabulary make it readable: a **warp** is the group of 32 GPU threads that execute in lockstep, the unit work gets divided into; **tensor cores** are the dedicated circuits that do matrix multiplies, separate from the general-purpose arithmetic units beside them.

**[FlashAttention-2](https://arxiv.org/abs/2307.08691)** (2023) improved how work is partitioned across warps and cut the number of *non-matmul* FLOPs. That sounds like a strange thing to optimize until you see the gap it exploits: on an A100 the tensor cores do 312 TFLOP/s of FP16 matmul against 19.5 TFLOP/s of non-matmul FP32, so as the paper puts it, "each non-matmul FLOP is 16× more expensive than a matmul FLOP." Rescalings and sums are exactly the non-matmul work online softmax adds, which is why trimming them mattered.

**[FlashAttention-3](https://arxiv.org/abs/2407.08608)** (2024) targets Hopper hardware: asynchronous memory copies through the TMA (Tensor Memory Accelerator, a unit that moves tiles between HBM and SRAM without occupying the compute threads), warp specialization so different warps handle copying and computing, and FP8.

### 9. Sidebar: the probe {#sidebar-the-probe}

> **"Does Flash Attention change your model's output?"**

**A weak answer:** "It's an approximation that trades a little accuracy for a big speedup on long sequences."

This is wrong, and it's a common wrong answer — the name sounds like the efficient-attention family (Linformer, Performer, Longformer), which *are* approximations.

**A stronger answer:** "No. It's exact — an IO-aware reordering of the same computation. It tiles Q, K and V so the score tiles stay in SRAM, and uses online softmax with a running max and sum so partial results compose. Outputs match a naive implementation to floating-point reassociation noise, around 1e-6 in fp32; I've measured it. Memory goes from $O(n^2)$ to $O(n)$, which is what made long context feasible, and speed comes from eliminating HBM round-trips. That's the opposite of sparse or linear attention, which genuinely change the function — sliding-window attention differs from exact attention by about 0.6 on the same test, five orders of magnitude more than Flash does."

The tell is whether someone distinguishes it from the approximate-attention family, and whether they know the mechanism is online softmax rather than "it's optimized CUDA."

### What's next {#whats-next}

Next in the series — planned as post 4 — is **quantization**, where the tradeoff is real: fewer bytes per weight genuinely does lose information, and the interesting question is *what breaks first*. The plan is to implement INT8 and NF4 by hand, watch per-tensor scaling get destroyed by a single outlier channel, and work out why perplexity is a misleading way to decide whether a quantized model is safe to ship.

### References

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022) — the paper this post reimplements. "Exact" is in its title.
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023) — §8's version notes. The 16× figure is its §3.1: 312 TFLOP/s of matmul against 19.5 TFLOP/s of non-matmul on an A100.
- Shah et al., [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) (2024) — the Hopper-specific work in §8.
- Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (2018) — where §2's running-statistics trick comes from, four years before Flash Attention put it to this use.
- Rabe & Staats, [Self-attention Does Not Need $O(n^2)$ Memory](https://arxiv.org/abs/2112.05682) (2021) — §6's point, made a year earlier without the fused kernel.
- Code for this post: [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher), `uv run demo03`.
