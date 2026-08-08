---
title: "LLM Architecture Refresh [2]: The KV Cache, and Why Decode Is Memory-Bound"
date: 2026-08-02 00:00:00 -0700
categories: [LLM Architecture Refresh, Inference]
tags: [kv-cache, inference, gqa, mqa, batching, memory-bandwidth, pytorch]
description: >-
  Measuring the KV cache — why generation is quadratic without it, why the
  cache outgrows the model weights at long context, and why decode is
  memory-bandwidth-bound while prefill is compute-bound.
math: true
---

## The number that decides what you can serve

[Post 1](/posts/llm-architectures-attention-and-rope/) was about how attention works. This one is about the single fact that determines what you can actually deploy: **generating a token is a fundamentally different workload from reading one**, and almost every inference optimization you've heard of follows from that one asymmetry.

The setup is the same as before: every claim gets a **receipt** — a small program that prints the number the claim asserts. The code lives in a companion repo, [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher), and runs unchanged on Apple Silicon or a Linux + NVIDIA box:

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo02
```

Every number and figure below came out of that command on my M-series Mac. The Python shown alongside each result is the part that matters, trimmed of setup — the runnable version is in [`demos/d02_kv_cache.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d02_kv_cache.py).

This post needs a real model rather than loose tensors, so the repo gained one: `toy_model.py`, a Llama-shaped decoder — pre-norm, RMSNorm, RoPE, SwiGLU, no biases, configurable grouped-query attention. It's small (8–60M parameters) but not *wrong*, and the later posts on quantization and MoE will reuse it. The weights are random, because everything here measures time and memory, never output quality.

### Table of Contents

1. [Why a cache exists at all](#why-a-cache-exists-at-all)
2. [The cache changes nothing about the output](#the-cache-is-exact)
3. [Without a cache, generation is quadratic](#generation-is-quadratic)
4. [How big does the cache actually get?](#how-big-does-the-cache-actually-get)
5. [Shrinking the cache: GQA, MQA, and what you give up](#gqa-mqa-and-what-you-give-up)
6. [Prefill vs decode: the whole ballgame](#prefill-vs-decode-the-whole-ballgame)
7. [The batch sweep, and where it stops working](#the-batch-sweep)
8. [What follows from all this](#what-follows-from-all-this)
9. [Sidebar: the probe](#sidebar-the-probe)

---

### 1. Why a cache exists at all {#why-a-cache-exists-at-all}

Start from what [post 1](/posts/llm-architectures-attention-and-rope/) established, because the cache falls straight out of it.

To compute attention for one token, you need three things: that token's **query**, and the **key** and **value** of every token it is allowed to look at. Causal masking means "allowed to look at" is "itself and everything before it". So for token 5:

$$
\text{output}_5 \;=\; \sum_{j \le 5} w_{5j} \cdot V_j
\qquad \text{where} \qquad
w_{5j} \;\propto\; Q_5 \cdot K_j
$$

Now generate text. Each step appends one token and asks for the next, so the model runs again with a sequence one token longer:

```text
step 1:  [The cat sat]           -> needs  Q3  and  K1 K2 K3,  V1 V2 V3   -> "on"
step 2:  [The cat sat on]        -> needs  Q4  and  K1..K4,   V1..V4     -> "the"
step 3:  [The cat sat on the]    -> needs  Q5  and  K1..K5,   V1..V5     -> "mat"
```

Read down the K and V columns. Step 2 needs $K_1, K_2, K_3$ — **the same $K_1, K_2, K_3$ step 1 already computed.** Step 3 needs them again. Every step recomputes almost everything the previous step just finished computing.

#### Why they're safe to reuse

That only works if those keys and values are genuinely unchanged, and two facts from post 1 guarantee it:

1. **$K_j$ and $V_j$ depend only on token $j$ and its position.** They're $W_k$ and $W_v$ applied to one token's vector, with RoPE applied for position $j$. Appending a token later changes neither input.
2. **Causal masking means no token's representation can depend on anything after it.** Token 3 could not see token 4 even in principle, so token 4's arrival cannot disturb it.

That's an argument, so let's check it. Run the model on a 4-token prefix, then on those same tokens plus two more, and compare what each run computed for the first four positions:

```python
short = KVCache(cfg, 1, device, dtype)
model(tokens[:, :4], start=0, cache=short)   # the first four tokens

long = KVCache(cfg, 1, device, dtype)
model(tokens[:, :6], start=0, cache=long)    # the same four, plus two more

(short.k[:, :, :, :4] - long.k[:, :, :, :4]).abs().max()
```

```text
  max difference in K                0.0000
  max difference in V                0.0000
  bitwise identical                  yes
```

Not merely close — **bitwise identical**. So recomputing them is pure waste, and storing them is safe.

#### Why K and V but not Q

The name is "KV cache", not "QKV cache", and the reason is in the table above. Look at when each tensor is *used*:

```text
  tensor  cached?                                                   why
  -----------------------------------------------------------------------
  K           yes                   every later query scores against it
  V           yes                         every later query averages it
  Q            no  position i's query is used at step i and never again
```

$K_3$ and $V_3$ get read at step 3, and again at step 4, and at every step after. $Q_3$ is used once — to produce token 4 — and is then dead. Storing it would cost memory and save nothing.

This is also the answer to a question post 1 raised and left hanging: why does grouped-query attention shrink $K$ and $V$ but leave $Q$ at full width? Because $K$ and $V$ are what you have to keep. $Q$ is recomputed from scratch every step regardless.

#### The cache itself

So: compute $K$ and $V$ for exactly one new token per step, append them, and attend over everything stored.

```python
class KVCache:
    def __init__(self, cfg, batch, device, dtype):
        shape = (cfg.n_layers, batch, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    def append(self, layer, k, v, start):
        end = start + k.shape[-2]
        self.k[layer, :, :, start:end] = k
        self.v[layer, :, :, start:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]
```

Two details in that code are worth pausing on.

**It's pre-allocated to `max_seq_len`**, not grown per step. Growing would reallocate and copy the whole cache every token. The price is that every sequence reserves its worst case up front whether it needs it or not — which is the fragmentation problem [PagedAttention](https://arxiv.org/abs/2309.06180) was built to solve.

**RoPE is applied before K goes into the cache.** Cached keys carry their rotation permanently, and you never re-rotate one as the sequence grows. That's the point: position 3 stays position 3. Getting this wrong — re-rotating cached keys, or rotating a new token as though it were at position 0 — is the most common way to break a cache implementation.

### 2. The cache changes nothing about the output {#the-cache-is-exact}

Before optimizing anything, verify that it changes nothing. The model can generate
either way, so run both on the same prompt and compare the tokens that come out:

```python
with_cache = model.generate(prompt, max_new_tokens=24, use_cache=True)
without    = model.generate(prompt, max_new_tokens=24, use_cache=False)

torch.equal(with_cache, without)
```

```text
  generated shape                    (2, 40)
  token ids identical                yes
  first 8 new tokens (cached)        [228, 432, 131, 158, 24, 111, 281, 506]
  first 8 new tokens (uncached)      [228, 432, 131, 158, 24, 111, 281, 506]
```

Identical token ids. This is the same category of claim as [post 1](/posts/llm-architectures-attention-and-rope/)'s check against the fused kernel, and it's worth stating plainly because it's the thing people get uneasy about: **the KV cache is memoization, not approximation.** If your cached and uncached outputs diverge, you have a bug — most often a position-offset error where the new token is rotated as though it were at position 0.

### 3. Without a cache, generation is quadratic {#generation-is-quadratic}

Now the cost. Generate from a 64-token prompt, with and without the cache:

```text
  prompt: 64 tokens; model: 7.8M params

  tokens generated  cached (ms)  uncached (ms)  speedup
  -------------------------------------------------------
  64                    54.7214        97.1847    1.78x
  128                   93.6767       224.2902    2.39x
  256                  153.4499       618.5234    4.03x
  512                  293.5081      1920.9685    6.54x

  8x more tokens costs (cached)      5.4x
  8x more tokens costs (uncached)    19.8x
  tokens processed at n=512 (cached)   576
  tokens processed at n=512 (uncached) 163584
  wasted work multiplier             284x
```

![Cached vs uncached generation time](/assets/picture/2026-08-02-llm-architectures-kv-cache/cache-vs-nocache-light.png){: .light width="1000" height="682" }
![Cached vs uncached generation time](/assets/picture/2026-08-02-llm-architectures-kv-cache/cache-vs-nocache-dark.png){: .dark width="1000" height="682" }

The bottom rows are the clearest statement of it. To produce 512 tokens, the cached path embeds 576 tokens total. The uncached path embeds **163,584** — a 284× multiplier of pure repeated work, because step $i$ re-processes all $64 + i$ preceding tokens:

$$
\sum_{i=0}^{n-1} (p + i) \;=\; np + \frac{n(n-1)}{2}
$$

Quadratic in the number of generated tokens. The measured growth reflects it: 8× more tokens costs 5.4× with a cache and **19.8×** without.

One honest note about this measurement. I used a *short* prompt on purpose. With a long prompt the $np$ term dominates at these values of $n$ and the curve looks straight — you'd still see a big absolute saving, but not the shape. A short prompt isolates the $n^2/2$ term. Real serving has both: long prompts *and* long generations.

### 4. How big does the cache actually get? {#how-big-does-the-cache-actually-get}

Here's where it stops being a nice optimization and starts being the constraint. The size is exactly:

$$
\text{KV bytes} = 2 \times L \times H_{kv} \times d_{head} \times S \times B \times \text{bytes}
$$

The 2 is for K and V. Note what's in there: sequence length $S$ **and** batch size $B$, both linear. The weights are a fixed cost; this is not.

Llama-3-8B shapes, fp16, batch 1 — the weights are 15.0 GiB. The three rows are the
three ways a model can allocate key/value heads: one per query head (**MHA**, multi-head
attention), eight shared across 32 (**GQA**, grouped-query attention — what Llama-3-8B
actually does), or a single one for all of them (**MQA**, multi-query attention).
[§5](#gqa-mqa-and-what-you-give-up) covers what each costs you.

```text
  variant           kv heads   4k ctx   8k ctx   32k ctx  128k ctx
  ------------------------------------------------------------------
  as MHA                  32  2.00 GiB  4.00 GiB  16.00 GiB  64.00 GiB
  Llama-3-8B (GQA)         8  0.50 GiB  1.00 GiB   4.00 GiB  16.00 GiB
  as MQA                   1  0.06 GiB  0.12 GiB   0.50 GiB   2.00 GiB
```

![KV cache size vs context length](/assets/picture/2026-08-02-llm-architectures-kv-cache/cache-size-light.png){: .light width="1000" height="678" }
![KV cache size vs context length](/assets/picture/2026-08-02-llm-architectures-kv-cache/cache-size-dark.png){: .dark width="1000" height="678" }

At 128k context, an 8B model's KV cache is **16 GiB against 15 GiB of weights**. The cache is bigger than the model. And that's one user:

```text
  model        batch   weights  KV cache @128k  cache/weights
  -------------------------------------------------------------
  Llama-3-8B       1   15.0 GiB         16.0 GiB          1.07x
  Llama-3-8B       8   15.0 GiB        128.0 GiB          8.56x
  Llama-3-8B      32   15.0 GiB        512.0 GiB         34.23x
  Llama-3-70B      1  131.5 GiB         40.0 GiB          0.30x
  Llama-3-70B      8  131.5 GiB        320.0 GiB          2.43x
  Llama-3-70B     32  131.5 GiB       1280.0 GiB          9.73x
```

Thirty-two concurrent users at full context on an 8B model needs half a terabyte of KV cache. This is why "how many users can I serve?" is a KV-cache question, not a model-size question — and why the answer changes completely with context length.

It's also why the 70B row is interesting: at batch 1 its cache is only 0.30× its weights, so a big model at short context is weight-dominated, while a small model at long context is cache-dominated. Two very different engineering problems wearing the same "LLM inference" label.

### 5. Shrinking the cache: GQA, MQA, and what you give up {#gqa-mqa-and-what-you-give-up}

Notice that $H_{kv}$ — the number of *key/value* heads — is in the formula, but the number of *query* heads isn't. That's the lever.

- **MHA**: every query head gets its own K/V head. Maximum expressiveness, maximum cache.
- **MQA**: all query heads share a single K/V head. 32× smaller cache here, but a real quality cost — every head is forced to look up against the same keys.
- **GQA**: query heads are split into groups, one K/V head each. Llama-3-8B uses 8 KV heads for 32 query heads — **4× less cache** than MHA at a quality cost small enough that it's now the default.

In code the sharing is just a broadcast before the attention call:

```python
if self.n_rep > 1:
    k = k.repeat_interleave(self.n_rep, dim=1)
    v = v.repeat_interleave(self.n_rep, dim=1)
```

The expansion happens *after* the cache read, which is the entire point: you store 8 heads and compute against 32. Post 12 covers DeepSeek's MLA, which attacks the same problem differently — compressing K and V into a shared low-rank latent and caching that instead.

### 6. Prefill vs decode: the whole ballgame {#prefill-vs-decode-the-whole-ballgame}

Generation has two phases with completely different performance characteristics.

**Prefill** processes the entire prompt in one parallel pass. **Decode** produces one token per pass. In code the only difference is how many tokens go in — the model, the weights and the cache are the same:

```python
# prefill: the whole 512-token prompt at once, into an empty cache
model(prompt, start=0, cache=cache)          # prompt is (1, 512)

# decode: one token, against a cache that already holds 512
model(one_token, start=512, cache=cache)     # one_token is (1, 1)
```

Timing both on the same 62.6M-parameter model:

```text
  phase    tokens/pass  ms/pass  ms/token    tokens/s
  -----------------------------------------------------
  prefill          512  33.0275    0.0645  15502.2522
  decode             1   5.2635    5.2635    189.9862

  per-token cost, decode / prefill   81.6x
```

**A token costs 82× more to generate than to read.** Both passes stream the same weights through the chip's arithmetic units (ALUs — the circuits that actually do the multiplying). Prefill amortizes that read across 512 tokens; decode pays it in full for one.

The clean way to see why is arithmetic intensity — FLOPs performed per byte of weights moved:

```text
  phase    tokens  FLOPs (2*N*P)  weight bytes  FLOP/byte
  ---------------------------------------------------------
  prefill     512         64.1 G      0.23 GiB   256.0000
  decode        1        0.125 G      0.23 GiB     0.5000
```

Modern accelerators need roughly 100–300 FLOP/byte to keep their tensor cores busy. Prefill sits at 256 — right in the productive zone, **compute-bound**. Decode sits at **0.5**, two to three orders of magnitude short of what the hardware needs to stay busy. The hardware spends essentially all of decode waiting on memory. It is **memory-bandwidth-bound**.

This single fact explains a startling amount:

- **Batching works** — the weight read is already paid for, so more sequences are nearly free (until they aren't; see below).
- **Quantization speeds up decode** even when the arithmetic isn't faster, because there are simply fewer bytes to move.
- **Speculative decoding wins** because verifying $k$ draft tokens in one pass costs about the same as generating one — you were bandwidth-bound, not compute-bound.
- **Bigger GPUs don't help decode much** unless their *bandwidth* went up.

### 7. The batch sweep, and where it stops working {#the-batch-sweep}

If decode is really memory-bound, batching should buy throughput almost for free. The sweep fills a cache for `batch` sequences, then times a single decode step across all of them at once:

```python
for batch in (1, 2, 4, 8, 16, 32):
    cache = KVCache(cfg, batch, device, dtype)
    model(warm, start=0, cache=cache)        # warm is (batch, prefix)

    step = torch.randint(0, cfg.vocab_size, (batch, 1))   # one token per sequence
    ms = benchmark_ms(lambda: model(step, start=prefix, cache=cache))
```

With a short (32-token) prefix:

```text
  batch  ms/step  latency vs b=1  tokens/s  throughput vs b=1
  -------------------------------------------------------------
  1       4.0446           1.00x       247               1.0x
  2       4.2466           1.05x       471               1.9x
  4       4.6923           1.16x       852               3.4x
  8       5.5535           1.37x      1441               5.8x
  16      6.3332           1.57x      2526              10.2x
  32      7.9067           1.95x      4047              16.4x
```

32× the work for **1.95×** the time — 16.4× the throughput. The GPU was idling on memory, so the extra sequences rode along inside a weight read that was happening anyway. That is what memory-bound looks like, and it's why every serving stack batches aggressively.

Now the same sweep with a **512-token** prefix:

```text
  batch  ms/step  latency vs b=1  tokens/s  throughput vs b=1
  -------------------------------------------------------------
  1       5.3005           1.00x       189               1.0x
  2       6.6377           1.25x       301               1.6x
  4       9.0194           1.70x       443               2.4x
  8      14.0736           2.66x       568               3.0x
  16     22.3713           4.22x       715               3.8x
  32     40.2305           7.59x       795               4.2x
```

The free lunch is gone: 16.4× throughput becomes **4.2×**.

![Batch sweep at two prefix lengths](/assets/picture/2026-08-02-llm-architectures-kv-cache/batch-sweep-light.png){: .light width="1000" height="540" }
![Batch sweep at two prefix lengths](/assets/picture/2026-08-02-llm-architectures-kv-cache/batch-sweep-dark.png){: .dark width="1000" height="540" }

The reason is the formula from §4. **Weights are shared across the batch; the KV cache is not.** Every sequence brings its own cache, so KV traffic scales with batch while weight traffic stays flat. At batch 32 with a 512-token prefix, this model reads 0.250 GiB of KV cache against 0.23 GiB of weights — the cache has become the *majority* of memory traffic, and batching can't amortize it.

I find this the most useful thing in the post, because the textbook version ("decode is memory-bound, so batch it") is only half the story. Batching amortizes one term. The other term grows with exactly the thing you were batching. That crossover is why production serving is a scheduling problem — continuous batching, prefix sharing for common system prompts, paged caches to stop reserving worst-case memory, and KV-cache quantization to shrink the term that doesn't amortize.

*(These are laptop numbers on Apple Silicon with a small model, so the absolute values reflect a modest bandwidth budget and some kernel-launch overhead. The shape of both curves, and the crossover between them, is what transfers — that's arithmetic, not hardware.)*

### 8. What follows from all this {#what-follows-from-all-this}

A short mental checklist I now use when reasoning about an inference setup:

| Symptom | What's actually binding | Lever |
| --- | --- | --- |
| Time-to-first-token is slow | Prefill — compute-bound | Better kernels, more FLOPs, chunked prefill |
| Tokens-per-second is slow | Decode — bandwidth-bound | Quantization, speculative decoding, faster memory |
| Can't fit more users | KV cache | GQA/MLA, cache quantization, paging, shorter context |
| Throughput plateaus as batch grows | KV traffic overtook weight traffic | Prefix sharing, cache quantization, smaller batch × longer context tradeoff |

### 9. Sidebar: the probe {#sidebar-the-probe}

> **"Why does generating 500 tokens take so much longer than reading a 5,000-token prompt?"**

**A weak answer:** "Generation is sequential — you can't compute token 2 until token 1 exists, whereas the prompt is processed in parallel."

That's true, and it's the answer most people give. But it describes the *dependency structure* without saying why sequential is expensive. If the work per token were the same, 500 sequential steps would cost a tenth of 5,000 parallel ones.

**A stronger answer:** "Both phases stream the whole weight matrix through the ALUs. Prefill amortizes that read over thousands of tokens and lands around 250 FLOPs per byte, which is roughly where the hardware saturates — it's compute-bound. Decode does the same read for a single token, landing near 0.5 FLOPs per byte, so it's memory-bandwidth-bound and the ALUs sit idle. In my measurement a decoded token cost 82× a prefilled one. That's also why batching, quantization, and speculative decoding all help decode specifically: they either amortize the read or shrink it. The caveat is that batching only amortizes the *weight* read — KV-cache traffic scales with batch, so past a point throughput plateaus."

The difference is that the second answer names the bottleneck resource and predicts which optimizations work.

### What's next {#whats-next}

[Post 3](/posts/llm-architectures-flash-attention/) is **Flash Attention** — the other half of the memory-traffic story. We've been treating attention itself as cheap, but at long context the $n \times n$ score matrix is the problem, and the fix is the same insight as this post applied one level down: don't move bytes you don't have to. We'll implement online softmax from scratch and confirm it's exact to floating-point noise.

### References

- Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (2019) — MQA.
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (2023).
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023) — vLLM.
- Pope et al., [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (2022) — the roofline analysis in depth.
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022).
- Code for this post: [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher), `uv run demo02`.
