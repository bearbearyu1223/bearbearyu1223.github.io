---
title: "LLM Architecture Refresh [3]: Flash Attention Is Exact, and Here's the Proof"
date: 2026-08-16 01:00:00 -0700
categories: [LLM Architecture Refresh, Inference]
tags: [flash-attention, online-softmax, attention, memory-bandwidth, tiling, pytorch]
description: >-
  Implementing online softmax and tiled attention from scratch to show that
  Flash Attention returns the same answer as the textbook version — and that
  the win is memory traffic, not arithmetic.
math: true
pin: true
published: false
---

## The optimization that doesn't change the answer

[Post 2](/posts/llm-architectures-kv-cache/) ended on a principle: don't move bytes you don't have to. That was about weights and the KV cache. This post applies the same idea one level down, to attention itself.

The first thing to say about Flash Attention — Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022) — and the thing that trips people up, is that **it is exact**. The word is in the title for a reason. It is not a sparse approximation, not a low-rank factorization, not a kernel trick. It computes precisely the same function as the attention in [post 1](/posts/llm-architectures-attention-and-rope/). It just computes it in an order that touches memory far less.

### The short version {#the-short-version}

**First, what attention actually does**, since everything below is about one specific cost buried inside it. To work out what a token means in context, a model scores that token's **query** vector against the **key** vector of every token it can see. Every pair — which is where the $n \times n$ comes from. Those scores go through a **softmax**, which turns each row into weights that sum to 1, and the weights are then used to average the tokens' **value** vectors. Score, normalize, average. ([Post 1](/posts/llm-architectures-attention-and-rope/) builds all three from scratch; here only their shape matters.)

The middle step is the problem. It needs a matrix with a row and a column for every token in the sequence: 2 GiB at 8k tokens, for a single layer. That matrix exists only long enough to be normalized and thrown away. **Flash Attention is the trick of never building it**, while returning the same answer anyway. These are the things this post establishes:

- **It is exact, and that is checkable.** Tiled attention and PyTorch's own reference agree to **1.4e-6**, which is floating-point noise from summing the same numbers in a different order — not a different answer. Sliding-window attention, which genuinely is an approximation, differs by **0.60** on the same inputs. Five orders of magnitude separate a reordering from a shortcut.
- **One idea makes it work: online softmax.** A softmax needs the largest value in a row before it can start, and a tile has seen only part of the row. So carry a running max and a running sum, and rescale everything accumulated so far whenever a later block raises the max. On this run, 22 of 31 blocks raised nothing and cost nothing.
- **Memory drops from quadratic to constant.** The score matrix is 512 MiB at 4,096 tokens and **512 GiB at 128k**, per layer. The tiled path's peak is one $128 \times 128$ tile — **0.5 MiB, identical at every sequence length**. That is what made long context possible; the arithmetic was never what stood in the way.
- **It does not save a single multiply.** Counted rather than timed, with causal masking off, the naive and tiled FLOP totals match **to the digit** at every length — 34.4 G apiece at 4,096 tokens. What vanishes is the score matrix's 2,048 MiB of round-trips to main memory. Traffic is the mechanism; arithmetic is untouched.
- **Causal masking then hands back half the work, free.** Once the loop walks tiles, a tile whose keys all lie in the future gets skipped rather than computed and masked — **47%** of them at 2,048 tokens with 128-wide blocks. That fraction is exactly $\frac{B-1}{2B}$ for $B$ blocks a side, climbing toward one half and never reaching it.
- **Tiling buys memory; fusing buys time.** A tiled loop written in Python is *slower* than naive attention, because every tile still round-trips to memory. The speedup needs the loop compiled into a single kernel — and against one, naive attention runs **almost 4× slower at 512 tokens and nearly 8× at 4,096**, a gap that widens as the sequence grows.
- **Training gets the same trick backwards.** The backward pass normally wants the attention probabilities it just declined to store, so instead it recomputes them from $Q$, $K$, $V$ and the saved running statistics. Redoing arithmetic to avoid storing and re-reading a result is a good trade whenever memory is the binding constraint — at long context, it is.

Every one of those has a **receipt** behind it — a small program that prints the number, so you can check it rather than take my word. The code lives in the companion repo [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher):

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo03
```

Every number and figure below came out of that command on my M-series Mac. The code for this post is in [`demos/d03_flash_attention.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d03_flash_attention.py).

### Table of Contents

Skip to [the short version](#the-short-version) for the findings without the derivations.

1. [The problem is memory traffic, not FLOPs](#the-problem-is-memory-traffic-not-flops)
2. [Online softmax: the one idea](#online-softmax-the-one-idea)
3. [Receipt 1: partial softmaxes compose](#receipt-1-partial-softmaxes-compose)
4. [Tiled attention in 40 lines](#tiled-attention-in-40-lines)
5. [Receipt 2: exact, and what "exact" is worth](#receipt-2-exact-and-what-exact-is-worth)
6. [Receipt 3: the memory that never gets allocated](#receipt-3-the-memory-that-never-gets-allocated)
7. [Causal masking gets a bonus](#causal-masking-gets-a-bonus)
8. [Where the speed actually comes from](#where-the-speed-actually-comes-from)
9. [Sidebar: the probe](#sidebar-the-probe)

Plus an [appendix of all notation](#appendix-all-notation) at the end, if a symbol ever goes by without introduction.

---

### 1. The problem is memory traffic, not FLOPs {#the-problem-is-memory-traffic-not-flops}

Two names for memory first, because the whole argument is about the distance between them. **HBM** is *high-bandwidth memory* — the GPU's main memory, the tens of gigabytes a spec sheet quotes when it says "80 GB". **SRAM** is the small, fast memory sitting on the chip itself, next to the arithmetic units. HBM is where your tensors live; SRAM is where they have to be for the chip to touch them, and everything moves between the two.

Standard attention does this:

1. Compute $S = QK^\top$ — an $n \times n$ matrix. **Write it to HBM.**
2. Read it back. Compute $P = \text{softmax}(S)$. **Write it to HBM.**
3. Read it back. Compute $O = PV$. Write the output.

Three round-trips through main memory for a matrix that grows quadratically with sequence length. At $n = 8192$ with 8 heads in **fp32** (32-bit floating point, four bytes a number), that intermediate is **2 GiB**, per layer, per forward pass.

Now the hardware, and the two numbers that matter. These are the A100 figures from the FlashAttention paper's own §2.1, which is where the design pressure came from:

| | Bandwidth | How much of it |
| --- | --- | --- |
| **HBM** — main memory | 1.5–2.0 TB/s | 40–80 GB |
| **SRAM** — on-chip | ~19 TB/s | ~20 MB |

That SRAM figure is worth unpacking, because "20 MB" sounds like a strange amount of memory for a chip to have. It isn't one pool: it's 192 KB on each of the A100's 108 **streaming multiprocessors**, the independent processor blocks a GPU is built out of, each with its own private scratchpad and its own share of the work. And $108 \times 192\ \text{KB} \approx 20\ \text{MB}$ only if you add every SM's scratchpad together. No single tile ever gets to use 20 MB — it gets 192 KB.

Roughly an order of magnitude apart in speed, and four orders apart in size. That combination is the entire problem: SRAM is fast enough to keep the arithmetic units fed, and far too small to hold an $n \times n$ score matrix. So attention at long context isn't waiting on arithmetic. It's waiting on the trip to HBM and back, exactly like decode in [post 2](/posts/llm-architectures-kv-cache/).

The fix is the standard one for memory-bound problems: **fuse the steps so intermediates never leave fast memory**. Compute a **tile** of $S$ (a small rectangular block of it, not the whole matrix), softmax that, multiply by $V$, accumulate, discard, all while the tile sits in SRAM. Fusing means putting all of that in one **kernel**: a single program the GPU runs start to finish, so nothing in the middle has to be written out to HBM and read back.

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

That is easier to see than to read. The numbers going into a softmax are called **logits**: raw scores, before normalizing turns them into weights that sum to 1. Streaming 2,048 of them past the loop 64 at a time, here is every block's own max against the running max, and what each one costs:

![Online softmax: the running max and the rescale it forces](/assets/picture/2026-08-02-llm-architectures-flash-attention/online-softmax-light.png){: .light width="1000" height="742" }
![Online softmax: the running max and the rescale it forces](/assets/picture/2026-08-02-llm-architectures-flash-attention/online-softmax-dark.png){: .dark width="1000" height="742" }

The top panel is the reason the bottom one is mostly flat. A block forces a rescale only by beating *every* block before it, and a running maximum gets harder to beat the longer it runs — so on this run 22 of the 31 blocks cost exactly nothing, and the ones that do cost something bunch up at the start. The deepest correction, $\times 0.023$, is a single multiply applied to one accumulator, not a second pass over the data.

### 3. Receipt 1: partial softmaxes compose {#receipt-1-partial-softmaxes-compose}

Does it actually agree with `torch.softmax`? Testing on logits scaled by 8, so each row spans a range of about 57:

```text
  logit row range (max - min)        56.9

  block size  max |online - torch|  rows sum to
  -----------------------------------------------
  64                     5.960e-08       1.0000
  128                    1.192e-07       1.0000
  512                    1.788e-07       1.0000
  2048                   1.192e-07       1.0000
```

Agreement at `1e-7` in fp32, and the rows still sum to 1. **Block size changes the memory schedule, not the answer.** That's the property everything else rests on.

#### And the running max — what is it for?

Worth being precise here, because the usual story ("without it the exponentials overflow") is not quite true and it's easy to repeat. Compare against the naive `exp(x)/sum(exp(x))` at both **precisions** — the number formats the arithmetic runs in, `fp32` being 32 bits per number and `fp16` sixteen:

```python
for label, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
    xd = x.to(dtype)
    e = torch.exp(xd)                               # the naive way: no max subtracted
    naive = e / e.sum(-1, keepdim=True)
    overflow_at = math.log(torch.finfo(dtype).max)  # past here, exp stops fitting
    bad = int(torch.isinf(e).sum() + torch.isnan(naive).sum())
```

The `exp overflows past` column is nothing but `log` of the largest number each format can hold, so it is a fact about the format and not about these logits: fp32 tops out near $3.4 \times 10^{38}$, fp16 at 65,504.

```text
  dtype  exp overflows past  actual row max  overflowed?  bad values  naive error
  ---------------------------------------------------------------------------------
  fp32                 88.7            29.6           no           0    5.960e-08
  fp16                 11.1            29.6          yes       1,340          nan
```

In fp32, `exp` overflows only past $x \approx 88.7$, and a row max near 30 isn't close — the unstable version **survives these logits perfectly well**, matching to `6e-8`. In fp16 the threshold drops to $x \approx 11.1$, the same logits overflow 1,340 times, and every one becomes a `nan`.

So the max subtraction isn't there to rescue a computation that would otherwise blow up in the abstract. It's there because inference runs in the precision where it *does* blow up. That matters for Flash Attention specifically: the running max is what lets the algorithm carry the same guarantee **per tile**, at whatever precision the kernel is compiled for, without ever seeing the whole row.

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

That's floating-point **reassociation** noise: adding the same numbers in a different order gives a very slightly different answer, because each intermediate rounding lands differently. Tiling adds them in a different order, so a difference at the last bit or two is expected — the same order as [post 1](/posts/llm-architectures-attention-and-rope/)'s check. The tile *shape* doesn't move it, because the tile shape isn't part of the math.

Now the comparison that gives "exact" its meaning. Sliding-window attention is a *genuine* approximation — each query attends only to the last $w$ keys:

```python
def sliding_window_attention(q, k, v, window: int):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    pos = torch.arange(scores.shape[-1], device=q.device)
    too_old = (pos[None, :] < pos[:, None] - window + 1) | (pos[None, :] > pos[:, None])
    scores = scores.masked_fill(too_old, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
```

That `too_old` mask is the entire difference. It throws away scores the exact version keeps, so what comes out the other end is a **different function** — and the table below is measuring how different:

```text
  method                 max abs difference vs exact
  ----------------------------------------------------
  Flash / tiled (exact)                    1.431e-06
  sliding window, w=256                       0.6047
  sliding window, w=64                        1.5660
```

**Five orders of magnitude apart.** This is the distinction to hold onto: sparse attention, linear attention, and low-rank attention all change the function being computed, and you have to evaluate whether the change costs you anything. Flash Attention changes only the order of memory accesses. There is nothing to evaluate — if it produced different results, it would be a bug.

### 6. Receipt 3: the memory that never gets allocated {#receipt-3-the-memory-that-never-gets-allocated}

Measuring this takes a little care, because an allocator reuses memory the instant nothing points at it any more — run naive attention and drop the result, and the $n \times n$ matrix can vanish from the peak reading before you ever see it. So the naive path's score matrix is deliberately held alive across the measurement:

```python
reset_peak_memory(device)
base = peak_memory_bytes(device) or 0
_, weights = naive_attention(q, k, v, causal=True)   # keep the n x n matrix referenced
sync(device)
naive_bytes = (peak_memory_bytes(device) or 0) - base
del weights
```

The tiled path is measured by the same four lines with `flash_attention` in the middle and nothing held. Two of the four columns below come out of that; the other two are arithmetic, not measurement. **Score matrix** is $H \cdot n^2 \cdot 4$ bytes — 8 heads, four bytes per fp32 number — and **one tile** is the same expression with $n^2$ replaced by the tile's $128 \times 128$:

$$
8 \times 4096^2 \times 4\ \text{B} = 512\ \text{MiB}, \qquad 8 \times 128^2 \times 4\ \text{B} = 0.5\ \text{MiB}
$$

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

So columns two and four are analytic, columns three and five are measured, and the point of putting them side by side is that they agree. The measured naive column tracks the analytic $n^2$ column to within a couple of MiB, which is the check that the formula describes what the machine actually does. The tiled residual is 0 because each tile is freed as the loop advances — that's the claim, not a measurement artifact; the meaningful number is the constant 0.5 MiB in column four.

And that constant is where [§1](#the-problem-is-memory-traffic-not-flops)'s 192 KB comes back. The 0.5 MiB is a $128 \times 128$ tile across all 8 heads at once; per head it's 64 KiB, and a head is what one streaming multiprocessor works on. So the tile was sized to fit an SM's private scratchpad with room to spare — that's the constraint the block size is chosen against, and why it's a constant rather than a function of $n$.

![Attention memory scaling](/assets/picture/2026-08-02-llm-architectures-flash-attention/memory-scaling-light.png){: .light width="1000" height="684" }
![Attention memory scaling](/assets/picture/2026-08-02-llm-architectures-flash-attention/memory-scaling-dark.png){: .dark width="1000" height="684" }

Extrapolating past what my laptop can hold — the same $H \cdot n^2 \cdot 4$ bytes, now with nothing measured, because none of these would fit:

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

Just under half the work disappears. The naive path computes that entire upper triangle, writes it to HBM, reads it back, and softmaxes it into zeros. Only the diagonal tiles need an actual mask — everything below is fully visible, everything above is skipped outright.

Nothing in that table is a measurement. It is a tile count, so it comes out identical on any machine, and it has a closed form: with $B = n/\text{block}$ blocks along each side, the loop computes the lower triangle *including* the diagonal, $B(B+1)/2$ tiles out of $B^2$, and skips

$$
\frac{B^2 - B(B+1)/2}{B^2} = \frac{B-1}{2B}
$$

At 2,048 tokens that is $15/32 = 46.9\%$ for `block=128`, $7/16 = 43.8\%$ for 256, and $3/8 = 37.5\%$ for 512 — the three rows above. That is also why smaller blocks skip more: the diagonal they are forced to compute is a thinner slice of the matrix, one row of tiles in sixteen at `block=128` against one in four at `block=512`. The ceiling is 50%, approached and never reached, because the diagonal always has to be computed.

### 8. Where the speed actually comes from {#where-the-speed-actually-comes-from}

#### First: it is not from doing less arithmetic

This post has been asserting that the win is memory traffic rather than FLOPs. That's a comparison, so it needs both quantities in one table.

The tiled column is **tallied as the loop runs** rather than derived afterwards — one line inside the inner loop, incremented on every tile actually computed, so tiles the causal skip drops never get counted:

```python
counters["matmul_flops"] += 2 * (2 * batch * heads * rows * cols * dim)
```

Two **matmuls** (matrix multiplies) per tile, $QK^\top$ and $PV$; each costs $2 \cdot \text{rows} \cdot \text{cols} \cdot d$, the factor of two being one multiply and one add per element accumulated. The naive column is that same expression with the tile replaced by the whole matrix, and the traffic column counts the score matrix crossing HBM on the passes §1 listed — write $S$, read $S$, write $P$, read $P$, four crossings of $H \cdot n^2 \cdot 4$ bytes. At $n = 4096$:

$$
\underbrace{2 \times (2 \cdot 8 \cdot 4096^2 \cdot 64)}_{34.4\ \text{GFLOP}}, \qquad \underbrace{4 \times 8 \times 4096^2 \times 4\ \text{B}}_{2048\ \text{MiB}}
$$

Both columns are counted, not timed; they reproduce exactly on any machine. With causal masking **off**, where both paths must touch every position:

```text
  seq   naive FLOPs  tiled FLOPs  ratio  naive score traffic  tiled
  -------------------------------------------------------------------
  512         0.5 G        0.5 G  1.00x               32 MiB  0 MiB
  1024        2.1 G        2.1 G  1.00x              128 MiB  0 MiB
  2048        8.6 G        8.6 G  1.00x              512 MiB  0 MiB
  4096       34.4 G       34.4 G  1.00x             2048 MiB  0 MiB
```

**The FLOP columns are equal to the digit.** The tiled path does not save a single multiply. What it removes is the entire third column — 2 GiB of score traffic at 4k context, gone, because the tile never leaves the accumulator.

That is the trade in one line, and it explains the shape of everything else in this post: FLOPs and score traffic both grow as $n^2$, but after tiling only one of them is still being paid.

![Arithmetic against score traffic, naive vs tiled](/assets/picture/2026-08-02-llm-architectures-flash-attention/flops-vs-traffic-light.png){: .light width="1000" height="438" }
![Arithmetic against score traffic, naive vs tiled](/assets/picture/2026-08-02-llm-architectures-flash-attention/flops-vs-traffic-dark.png){: .dark width="1000" height="438" }

Two panels because the two quantities have nothing to do with each other dimensionally, and the contrast between them *is* the result: same bars on the left, one bar missing entirely on the right.

Turn causal masking on and the tiled path does strictly less of *both*, because of the skipped blocks from [§7](#causal-masking-gets-a-bonus):

```text
  seq   naive FLOPs  tiled FLOPs  ratio  naive score traffic  tiled
  -------------------------------------------------------------------
  512         0.5 G        0.3 G  0.62x               32 MiB  0 MiB
  1024        2.1 G        1.2 G  0.56x              128 MiB  0 MiB
  2048        8.6 G        4.6 G  0.53x              512 MiB  0 MiB
  4096       34.4 G       17.7 G  0.52x             2048 MiB  0 MiB
```

That ratio column isn't measured either — it's [§7](#causal-masking-gets-a-bonus)'s tile count seen from the other side, $\frac{B+1}{2B}$ of the naive arithmetic. The 0.53 row is exactly §7's 47%, both at 2,048 tokens with `block=128`; by 4,096 tokens $B$ has doubled and it has tightened to 0.52, still creeping toward one half.

Keep the two savings straight, because only one of them is the point: the arithmetic saving is a **bonus that tiling makes available**, while the traffic elimination is the **mechanism**. Flash Attention would still be worth it at ratio 1.00.

#### Then: why the loop above is still slow

An honest caveat: **the Python tiled loop is slower than naive attention.** Every tile still round-trips through HBM, plus Python overhead per iteration. The algorithm buys memory; fusing it into a single kernel is what buys time.

So for timing, compare naive attention against `F.scaled_dot_product_attention`, which dispatches to exactly the fused kernel [§1](#the-problem-is-memory-traffic-not-flops) described. Two warmup passes, then the median of five — the first call on this machine pays for shader compilation (on an NVIDIA box it would be CUDA context creation), either of which can run 100× the steady-state cost and would swamp a plain average:

```python
naive_ms = benchmark_ms(lambda: naive_attention(q, k, v, causal=True),
                        device=device, warmup=2, repeats=5)
fused_ms = benchmark_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
                        device=device, warmup=2, repeats=5)
```

Unlike every other table in this post, this one is **wall-clock on my laptop** — it will not reproduce digit for digit on yours, and the shape is what to read:

```text
  seq   naive (ms)  fused SDPA (ms)  speedup
  --------------------------------------------
  512       1.2697           0.3402    3.73x
  1024      3.9000           0.7573    5.15x
  2048     13.3890           2.0275    6.60x
  4096     55.4718           7.0307    7.89x
```

![Naive vs fused attention timing](/assets/picture/2026-08-02-llm-architectures-flash-attention/timing-light.png){: .light width="1000" height="684" }
![Naive vs fused attention timing](/assets/picture/2026-08-02-llm-architectures-flash-attention/timing-dark.png){: .dark width="1000" height="684" }

The speedup **grows with sequence length** — not quite 4× at 512, nearly 8× at 4,096 — because the naive path's memory traffic grows quadratically while the fused path's grows linearly. Extrapolate and it keeps widening. (Wall-clock again, so the exact multiples move a little between runs; the widening is the part that holds.)

One more piece: the backward pass. Training normally stores the attention probabilities for the backward pass, which is the $O(n^2)$ tensor you just avoided allocating. Flash Attention instead **recomputes** the tiles during the backward pass from $Q$, $K$, $V$ and the saved statistics. Recomputation sounds expensive, but it's arithmetic — and arithmetic is the resource you have in surplus. It's cheaper to redo the FLOPs than to have stored and re-read the result.

A note on versions, since interviews like the specifics. Two more pieces of vocabulary make it readable: a **warp** is the group of 32 GPU threads that execute in lockstep, the unit work gets divided into; **tensor cores** are the dedicated circuits that do matrix multiplies, separate from the general-purpose arithmetic units beside them.

**[FlashAttention-2](https://arxiv.org/abs/2307.08691)** (2023) improved how work is partitioned across warps and cut the number of *non-matmul* FLOPs. That sounds like a strange thing to optimize until you see the gap it exploits: on an A100 the tensor cores do 312 TFLOP/s of FP16 matmul against 19.5 TFLOP/s of non-matmul FP32, so as the paper puts it, "each non-matmul FLOP is 16× more expensive than a matmul FLOP." Rescalings and sums are exactly the non-matmul work online softmax adds, which is why trimming them mattered.

**[FlashAttention-3](https://arxiv.org/abs/2407.08608)** (2024) targets Hopper hardware: asynchronous memory copies through the TMA (Tensor Memory Accelerator, a unit that moves tiles between HBM and SRAM without occupying the compute threads), warp specialization so different warps handle copying and computing, and FP8.

### 9. Sidebar: the probe {#sidebar-the-probe}

> **"Does Flash Attention change your model's output?"**

**A weak answer:** "It's an approximation that trades a little accuracy for a big speedup on long sequences."

This is wrong, and it's a common wrong answer — the name sounds like the efficient-attention family (Linformer, Performer, Longformer), which *are* approximations.

**A stronger answer:** "No. It's exact — an IO-aware reordering of the same computation. It tiles Q, K and V so the score tiles stay in SRAM, and uses online softmax with a running max and sum so partial results compose. Outputs match a naive implementation to floating-point reassociation noise, around 1e-6 in fp32; I've measured it. It doesn't even do less arithmetic — non-causal, the FLOP counts are identical to the digit. What it removes is the quadratic score traffic to HBM, which is why memory goes from $O(n^2)$ to $O(n)$ and why long context became feasible. That's the opposite of sparse or linear attention, which genuinely change the function — sliding-window attention differs from exact attention by about 0.6 on the same test, five orders of magnitude more than Flash does."

The tell is whether someone distinguishes it from the approximate-attention family, and whether they know the mechanism is online softmax rather than "it's optimized CUDA."

### What's next {#whats-next}

Next in the series — planned as post 4 — is **quantization**, where the tradeoff is real: fewer bytes per weight genuinely does lose information, and the interesting question is *what breaks first*. The plan is to implement INT8 and NF4 by hand, watch per-tensor scaling get destroyed by a single outlier channel, and work out why perplexity is a misleading way to decide whether a quantized model is safe to ship.

### Appendix: all notation {#appendix-all-notation}

Every symbol this post uses, in one place. [Post 1's appendix](/posts/llm-architectures-attention-and-rope/#appendix-all-notation) covers the ones inherited from attention itself in more depth, and [post 2's](/posts/llm-architectures-kv-cache/#appendix-all-notation) the ones about serving.

| Symbol | Means | In this post's runs |
| --- | --- | --- |
| $n$ | tokens in the sequence — the quantity everything here scales in | 512–4,096 measured, 128k extrapolated |
| $d$ | numbers in one query, key or value vector (the head dimension) | 64 |
| $H$ | attention heads computed side by side | 8 |
| $Q$, $K$, $V$ | the **query**, **key** and **value** matrices, one row per token | $n \times d$ each |
| $S = QK^\top$ | every query scored against every key — the score matrix | $n \times n$ |
| $P = \text{softmax}(S)$ | those scores turned into weights that sum to 1 along each row | $n \times n$ |
| $O = PV$ | the output: each token's values averaged by its weights | $n \times d$ |
| $x_i$, $x_j$ | single entries of one row on its way through a softmax | — |
| $m$ | the running **max**, carried across blocks; $m_{\text{old}}$ and $m_{\text{new}}$ are it before and after a block | one number per query row |
| $\ell$ | the running **sum** — the softmax denominator — rebased whenever $m$ moves | one number per query row |
| $e^{m_{\text{old}} - m_{\text{new}}}$ | the correction factor that rebases everything already accumulated | exactly 1 for 22 of 31 blocks |
| block, `block_q`, `block_k` | the tile's edges: how many queries and how many keys one tile covers | $128 \times 128$ unless stated |
| rows, cols | the tile actually in hand, which is smaller at the end of a sequence | $\le 128$ |
| $B$ | blocks along one side of the score matrix, $B = n/\text{block}$ | 4–32 |
| $w$ | window width in sliding-window attention — the approximation used for contrast | 64, 256 |
| $O(n^2)$, $O(n)$ | how memory grows with sequence length: with its square, or in step with it | — |
| FLOP, GFLOP | one floating-point add or multiply; $10^9$ of them | 0.5–34.4 G per pass |
| fp32, fp16 | 32- and 16-bit floating point — four and two bytes a number | fp32 unless stated |
| KiB, MiB, GiB | 1024, $1024^2$, $1024^3$ bytes — what an allocator reports | every computed figure |

Three things worth keeping straight, because the post leans on all of them:

- **$O$ is overloaded, and unavoidably so.** $O = PV$ is attention's output matrix; $O(n^2)$ is growth notation. They are unrelated, both are standard, and both appear here. Context separates them — an $O$ on its own is the output, an $O$ with a parenthesised expression is a rate.
- **KB and MB against KiB and MiB.** The A100 figures in [§1](#the-problem-is-memory-traffic-not-flops) (192 KB per SM, ~20 MB total, 40–80 GB of HBM) are quoted exactly as the vendor publishes them. Every number this post *computes or measures* is binary: KiB, MiB, GiB, powers of 1024, because that is what an allocator reports. [§6](#receipt-3-the-memory-that-never-gets-allocated) puts a 64 KiB tile next to a 192 KB scratchpad; that comparison survives either reading of "KB", since the tile fits with about threefold room to spare on both.
- **$n$ here is one length, not two.** Post 2 split a sequence into prompt $p$ and generated tokens $n$, because generation makes that split matter. This post is about a single forward pass over $n$ tokens, queries and keys the same length, which is why the score matrix is square.

### References

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022) — the paper this post reimplements. "Exact" is in its title.
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023) — §8's version notes. The 16× figure is its §3.1: 312 TFLOP/s of matmul against 19.5 TFLOP/s of non-matmul on an A100.
- Shah et al., [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) (2024) — the Hopper-specific work in §8.
- Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (2018) — where §2's running-statistics trick comes from, four years before Flash Attention put it to this use.
- Rabe & Staats, [Self-attention Does Not Need $O(n^2)$ Memory](https://arxiv.org/abs/2112.05682) (2021) — §6's point, made a year earlier without the fused kernel.
- Code for this post: [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher), `uv run demo03`.
