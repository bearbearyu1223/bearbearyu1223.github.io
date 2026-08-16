---
title: "LLM Architecture Refresh [2]: The KV Cache, and Why Decode Is Memory-Bound"
date: 2026-08-16 00:00:00 -0700
categories: [LLM Architecture Refresh, Inference]
tags: [kv-cache, inference, gqa, mqa, batching, memory-bandwidth, pytorch]
description: >-
  Measuring the KV cache — why generation is quadratic without it, why the
  cache outgrows the model weights at long context, and why decode is
  memory-bandwidth-bound while prefill is compute-bound.
math: true
pin: true
published: false
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
  step       sequence so far  needs     and     and  predicts
  -------------------------------------------------------------
  1            [The cat sat]     Q3  K1..K3  V1..V3      "on"
  2         [The cat sat on]     Q4  K1..K4  V1..V4     "the"
  3     [The cat sat on the]     Q5  K1..K5  V1..V5     "mat"
```

Read down the K and V columns. Step 2 needs $K_1, K_2, K_3$ — **the same $K_1, K_2, K_3$ step 1 already computed.** Step 3 needs them again. Every step recomputes almost everything the previous step just finished computing.

Counting those reads makes the case, and three steps is enough to see it:

```text
  Q vectors read, over 3 steps       3
  K vectors read, over 3 steps       12
  of which are recomputations        7
```

Three queries, each used once. Twelve key reads for five distinct keys, seven of them redoing work that was already done. The Q column is a diagonal, the K column a triangle. [The figure below](#why-k-and-v-but-not-q) draws exactly that, and the quadratic in [§3](#generation-is-quadratic) is what the triangle costs once the sequence is long.

#### Why they're safe to reuse

That only works if those keys and values are genuinely unchanged, and two facts from post 1 guarantee it:

1. **$K_j$ and $V_j$ depend only on token $j$ and its position.** They're $W_k$ and $W_v$ applied to one token's vector, with RoPE applied for position $j$. Appending a token later changes neither input.
2. **Causal masking means no token's representation can depend on anything after it.** Token 3 could not see token 4 even in principle, so token 4's arrival cannot disturb it.

That's an argument, so let's check it. Run the model on a 4-token prefix, then on those same tokens plus two more, and compare what each run stored for the first four positions:

```python
short = KVCache(cfg, 1, device, dtype)
model(tokens[:, :4], start=0, cache=short)   # the first four tokens

long = KVCache(cfg, 1, device, dtype)
model(tokens[:, :6], start=0, cache=long)    # the same four, plus two more
```

A cache holds one tensor per layer, so `cache.k` has five axes. Slicing the fourth to `:4` keeps every layer, sequence, head and feature, and takes only the token positions the two runs have in common:

```text
                         shape                                       meaning
  ----------------------------------------------------------------------------
  cache.k    (2, 1, 4, 64, 32)  layers, batch, kv_heads, positions, head_dim
  the slice   (2, 1, 4, 4, 32)          same, but only the first 4 positions

  numbers being compared             1,024
```

Now compare them. Subtract one from the other, take absolute values, and keep the largest — if nothing moved, that largest value is zero:

```python
(short.k[:, :, :, :4] - long.k[:, :, :, :4]).abs().max()
```

```text
  max |K_short - K_long|             0.0000
  max |V_short - V_long|             0.0000
```

`0.0000` is a rounded display though, so it's worth the stronger check — exact equality, which rules out a tiny non-zero difference hiding behind the rounding:

```text
  torch.equal(K_short, K_long)       yes
```

Not merely close — **identical, across all 1,024 numbers**. So recomputing them is pure waste, and storing them is safe.

#### Why K and V but not Q

The name is "KV cache", not "QKV cache". The reason is easiest to see as a picture. Put generation steps down the side and token positions across the top, then shade in which tensors each step actually needs:

![Why the cache holds K and V but not Q](/assets/picture/2026-08-02-llm-architectures-kv-cache/why-cache-light.png){: .light width="1000" height="618" }
![Why the cache holds K and V but not Q](/assets/picture/2026-08-02-llm-architectures-kv-cache/why-cache-dark.png){: .dark width="1000" height="618" }

**The shapes are the whole argument.**

$Q$ fills a **diagonal**. Step 3 computes $Q_3$, uses it to produce token 4, and is then done with it — no later step ever asks for $Q_3$ again. A diagonal has nothing to reuse, so there is nothing a cache could save you.

$K$ and $V$ fill a **triangle**. Step 3 needs $K_1, K_2, K_3$; step 4 needs those *plus* $K_4$; step 5 needs all five. Every column extends downward forever. Across five steps, five keys get computed once each but **read fifteen times** between them — and by the arithmetic in the next section, that gap widens quadratically.

So the rule is not "keys and values are special". It's simply: **cache what gets read again.** In the diagram, that's everything below the diagonal.

The same thing said as a table:

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
  64                    54.1744        95.6451    1.77x
  128                  108.0281       225.1538    2.08x
  256                  160.2089       614.7870    3.84x
  512                  312.1989      1927.7645    6.17x

  8x more tokens costs (cached)      5.8x
  8x more tokens costs (uncached)    20.2x
  tokens processed at n=512 (cached) 576
  tokens processed at n=512 (uncached) 163584
  wasted work multiplier             284x
```

![Cached vs uncached generation time](/assets/picture/2026-08-02-llm-architectures-kv-cache/cache-vs-nocache-light.png){: .light width="1000" height="682" }
![Cached vs uncached generation time](/assets/picture/2026-08-02-llm-architectures-kv-cache/cache-vs-nocache-dark.png){: .dark width="1000" height="682" }

The bottom rows are the clearest statement of it. To produce 512 tokens, the cached path embeds 576 tokens total. The uncached path embeds **163,584** — a 284× multiplier of pure repeated work, because step $i$ re-processes all $64 + i$ preceding tokens:

$$
\sum_{i=0}^{n-1} (p + i) \;=\; np + \frac{n(n-1)}{2}
$$

Quadratic in the number of generated tokens, and the measured growth reflects it: 8× more tokens costs roughly 5–6× with a cache and about **20×** without. Those are wall-clock numbers so they wobble a little between runs; the contrast between "grows a bit faster than linear" and "grows like the square" is the part that holds.

One honest note about this measurement. I used a *short* prompt on purpose. With a long prompt the $np$ term dominates at these values of $n$ and the curve looks straight — you'd still see a big absolute saving, but not the shape. A short prompt isolates the $n^2/2$ term. Real serving has both: long prompts *and* long generations.

### 4. How big does the cache actually get? {#how-big-does-the-cache-actually-get}

First, what is physically in there. **The cache stores computed numbers, not weights** — and that distinction matters, because the two are different kinds of thing:

| | Model weights | KV cache |
| --- | --- | --- |
| What it is | the learned parameters | K and V vectors computed from your tokens |
| Where it comes from | training | running the model on this conversation |
| Size | fixed | grows with every token |
| Shared between users? | yes, one copy serves everyone | **no, every sequence has its own** |

So they aren't competing for the same reason. The weights are a one-time cost you pay to load the model. The cache is a running cost you pay *per conversation, per token*.

#### What one token costs

Concretely: for every token, at every layer, each key/value head stores one key vector and one value vector. Nothing else — no queries, no attention weights, no FFN activations. For Llama-3-8B in fp16:

```text
  what                      count   running total
  -------------------------------------------------
  one key vector      128 numbers             128
  + one value vector  128 numbers             256
  x 8 KV heads                              2,048
  x 32 layers                      65,536 numbers
  x 2 bytes (fp16)                        128 KiB
```

**One token of context costs 128 KiB.** That's the number worth remembering — everything else is multiplication.

*(One label to be precise about: the layer shapes above are Llama-3-8B's, but its own context window is 8k. The 128k figures throughout this post are Llama-3.1-8B, which has the identical layer shapes and differs only in how far the context extends. The per-token cost is the same either way; only how many tokens you can accumulate changes.)*

And you pay it for *every* token in the conversation, not just the new one: the prompt you sent, the reply so far, all of it, for as long as the conversation lives.

```text
  context         x KiB/token  cache (batch 1)
  ----------------------------------------------
  8,192 tokens        128 KiB         1.00 GiB
  32,768 tokens       128 KiB         4.00 GiB
  131,072 tokens      128 KiB        16.00 GiB
```

That walk up from one token is the formula. Here it is in one line, with its symbols first, in the order they appear:

| Symbol | What it is | Llama-3-8B |
| --- | --- | --- |
| $L$ | how many blocks are stacked | 32 |
| $H_{kv}$ | key/value heads per block | 8 |
| $d_{head}$ | numbers in one key or value vector | 128 |
| $S$ | tokens in the sequence so far | grows every step |
| $B$ | sequences being served at once | your batch size |

$$
\text{KV bytes} = \underbrace{2}_{K \text{ and } V} \times L \times H_{kv} \times d_{head} \times S \times B \times \text{bytes}
$$

The leading 2 is there because every position stores a key *and* a value. Everything else is a count of how many of those you end up holding.

Note which two are in there: sequence length $S$ **and** batch size $B$, both linear. The weights have neither — that difference is what the rest of this section is about.

Llama-3-8B shapes, fp16, batch 1 — the weights are 15.0 GiB. The three rows are the
three ways a model can allocate key/value heads: one per query head (**MHA**, multi-head
attention), eight shared across 32 (**GQA**, grouped-query attention — what Llama-3-8B
actually does), or a single one for all of them (**MQA**, multi-query attention).
[§5](#gqa-mqa-and-what-you-give-up) covers what each costs you.

```text
  variant           kv heads    4k ctx    8k ctx    32k ctx   128k ctx
  ----------------------------------------------------------------------
  as MHA                  32  2.00 GiB  4.00 GiB  16.00 GiB  64.00 GiB
  Llama-3-8B (GQA)         8  0.50 GiB  1.00 GiB   4.00 GiB  16.00 GiB
  as MQA                   1  0.06 GiB  0.12 GiB   0.50 GiB   2.00 GiB
```

![KV cache size vs context length](/assets/picture/2026-08-02-llm-architectures-kv-cache/cache-size-light.png){: .light width="1000" height="678" }
![KV cache size vs context length](/assets/picture/2026-08-02-llm-architectures-kv-cache/cache-size-dark.png){: .dark width="1000" height="678" }

#### So caching costs memory?

Yes — and it's worth being explicit about that, because everything so far has made the cache sound like free money. It isn't. **It's a time-memory trade.**

Without a cache you still compute exactly the same keys and values every step. You just throw them away immediately, so they exist only *while that layer is running* — one layer's worth at a time, then freed. The cache keeps all of them, for all 32 layers, alive for the whole conversation:

```text
  approach             K/V memory held                           for how long
  -----------------------------------------------------------------------------
  with a cache               16.00 GiB  all 32 layers, the whole conversation
  without a cache             0.50 GiB  one layer, freed as the pass moves on
    + its activations         1.00 GiB                         also transient

  memory held, cached vs not         10.7x more
```

So you hold roughly **an order of magnitude more memory** than you otherwise would. What does that buy?

A request costs roughly in proportion to how many tokens the model has to push through itself. With a cache that's the prompt and the reply, once. Without one, it's the entire prefix again on every single step:

```text
  prompt  reply  cached    uncached  compute saved
  --------------------------------------------------
  512       256     768     163,712           213x
  2,048     512   2,560   1,179,392           461x
  8,192   1,024   9,216   8,912,384           967x
  32,768  2,048  34,816  69,204,992          1988x
```

**Roughly 10× the memory, for 200–2000× the compute.** And the two sides scale differently: the memory cost grows *linearly* with context, while the compute saving grows *quadratically*. The longer the conversation, the better the bargain looks.

That's why no serving stack ships without a KV cache. It isn't a tuning option you enable for extra throughput — a 2,048-token prompt with a 512-token reply would cost 461× more to serve without it, which is the difference between a viable product and an impossible one.

That's the honest framing for the rest of this section: the numbers below aren't the cost of a mistake, they're the price of a deliberate bargain. What makes them interesting is how quickly the price grows.

So at a 128k context, one conversation's cache is **16 GiB** — against 15 GiB for the entire model. One user's scratch space outweighs the thing that took a fortune to train.

And that's **one** user. Serving more doesn't mean loading the model again — one copy of the weights answers everybody. But each concurrent conversation brings its own cache. So read the next table down the columns: the weights stay at 15.0 GiB no matter how many people you serve, while the cache multiplies by however many of them there are.

```text
  model        batch    weights  KV cache @128k  cache/weights
  --------------------------------------------------------------
  Llama-3-8B       1   15.0 GiB        16.0 GiB          1.07x
  Llama-3-8B       8   15.0 GiB       128.0 GiB          8.56x
  Llama-3-8B      32   15.0 GiB       512.0 GiB         34.23x
  Llama-3-70B      1  131.5 GiB        40.0 GiB          0.30x
  Llama-3-70B      8  131.5 GiB       320.0 GiB          2.43x
  Llama-3-70B     32  131.5 GiB      1280.0 GiB          9.73x
```

That last row is worth doing by hand, because it's the one that decides hardware budgets:

$$
\underbrace{128\ \text{KiB}}_{\text{per token}} \times \underbrace{131{,}072}_{\text{tokens}} = \underbrace{16\ \text{GiB}}_{\text{one conversation}}
\qquad
16\ \text{GiB} \times 32\ \text{users} = 512\ \text{GiB}
$$

**Thirty-two concurrent conversations at 128k context need half a terabyte of KV cache** — about six 80 GiB accelerators' worth, for a model whose weights fit comfortably on one. This is why "how many users can I serve?" is a KV-cache question, not a model-size question, and why the answer changes completely with context length.

It's also why the 70B row is interesting: at batch 1 its cache is only 0.30× its weights, so a big model at short context is weight-dominated, while a small model at long context is cache-dominated. Two very different engineering problems wearing the same "LLM inference" label.

### 5. Shrinking the cache: GQA, MQA, and what you give up {#gqa-mqa-and-what-you-give-up}

Notice that $H_{kv}$ — the number of *key/value* heads — is in the formula, but the number of *query* heads isn't. That's the lever.

- **MHA**: every query head gets its own K/V head. Maximum expressiveness, maximum cache.
- **MQA**: all query heads share a single K/V head. 32× smaller cache here, but a real quality cost — every head is forced to look up against the same keys. Shazeer proposed it in [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150) (2019) for exactly this reason: decoding was already memory-bound.
- **GQA**: query heads are split into groups, one K/V head each. Llama-3-8B uses 8 KV heads for 32 query heads — **4× less cache** than MHA at a quality cost small enough that it's now the default. Ainslie et al., [GQA](https://arxiv.org/abs/2305.13245) (2023), also gave a recipe for *uptraining* an existing multi-head checkpoint into a grouped-query one for about 5% of the original pre-training compute — you didn't have to retrain to adopt it, which is part of why it spread so fast.

In code the sharing is just a broadcast before the attention call:

```python
if self.n_rep > 1:
    k = k.repeat_interleave(self.n_rep, dim=1)
    v = v.repeat_interleave(self.n_rep, dim=1)
```

The expansion happens *after* the cache read, which is the entire point: you store 8 heads and compute against 32.

#### What that actually buys, measured

That's two claims — the cache shrinks, the computation doesn't — so here they are side by side. Same model built three times, 12 query heads throughout, only `n_kv_heads` changing:

```text
  variant  kv heads  q per kv  params  cache @512       vs MHA  ms/decode step
  ------------------------------------------------------------------------------
  MHA            12         1   68.9M    24.0 MiB            —          4.3953
  GQA             4         3   62.6M     8.0 MiB   3x smaller          5.3112
  MQA             1        12   60.3M     2.0 MiB  12x smaller          5.4696
```

The cache column tracks `n_kv_heads` exactly — 24 → 8 → 2 MiB, no approximation in it. The time column barely moves. That asymmetry is the whole reason GQA is worth doing: **it's a storage decision the compute side hardly notices.** (The toy shares 3 query heads per KV head where Llama-3-8B shares 4; the ratio is whatever `n_kv_heads` says it is.)

One thing in that table is worth not glossing over, because it points at something real. MQA has *fewer* parameters than MHA and still takes slightly *longer* per step. The cause is the `repeat_interleave` above: it allocates the widened tensor rather than viewing it, and the fewer K/V heads you store, the more there is to expand. Production kernels take the shared K/V directly and never build that tensor — PyTorch exposes it as `scaled_dot_product_attention`'s `enable_gqa` flag. So the small penalty here is my implementation's, not GQA's, and it's a good reminder that "the cache got smaller" and "the kernel got faster" are separate claims that have to be measured separately.

Post 12 covers DeepSeek's MLA, which attacks the same problem differently — compressing K and V into a shared low-rank latent and caching that instead.

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
  prefill          512  32.8005    0.0641  15609.4982
  decode             1   5.5220    5.5220    181.0925

  per-token cost, decode / prefill   86.2x
```

**A token costs roughly two orders of magnitude more to generate than to read.** Both passes stream the same weights through the chip's arithmetic units (ALUs — the circuits that actually do the multiplying). Prefill amortizes that read across 512 tokens; decode pays it in full for one.

The exact multiple is the least trustworthy number in this post. Across repeated runs on this laptop it lands anywhere between about 80× and 100×, depending on what else the machine is busy with. The order of magnitude is what holds, and it is the only part the argument needs.

The clean way to see why is arithmetic intensity — FLOPs performed per byte of weights moved:

```text
  phase    tokens  FLOPs (2*N*P)  weight bytes  FLOP/byte
  ---------------------------------------------------------
  prefill     512         64.1 G      0.23 GiB   256.0000
  decode        1        0.125 G      0.23 GiB     0.5000
```

#### Where "100–300 FLOP/byte" comes from

That threshold gets quoted a lot, usually with no source attached, so it's worth not leaving it as folklore. It isn't a rule of thumb — it's the **ridge point** of the roofline model: peak arithmetic throughput divided by memory bandwidth. Below it, a kernel cannot saturate the arithmetic units however well it's written, because the bytes can't arrive fast enough to feed them.

Divide one datasheet number by the other and you get it directly:

```text
  accelerator     dense fp16  HBM bandwidth    ridge point
  ----------------------------------------------------------
  A100 40GB SXM  312 TFLOP/s     1,555 GB/s  201 FLOP/byte
  A100 80GB SXM  312 TFLOP/s     2,039 GB/s  153 FLOP/byte
  H100 SXM       989 TFLOP/s     3,350 GB/s  295 FLOP/byte
  H200 SXM       989 TFLOP/s     4,800 GB/s  206 FLOP/byte

  ridge point, across these four     153-295 FLOP/byte
```

These are vendor peak figures, not anything I measured — the one table here that comes off a spec sheet rather than out of my laptop, and labelled as such. But the spread is the interesting part: **153 to 295**, across two architectures and a 3× jump in raw FLOP/s. Bandwidth and arithmetic have grown closely enough in step that the ridge point barely moved, which is why a rule of thumb this crude has survived the hardware it was coined for. Note also that H100 → H200 is a pure bandwidth upgrade at identical FLOP/s, and it *lowers* the ridge — more workloads land on the compute-bound side of it.

Now place the two phases against the narrowest of those ridges:

```text
  phase    FLOP/byte  vs ridge (153)                      verdict
  -----------------------------------------------------------------
  prefill      256.0           1.67x  at or above — compute-bound
  decode         0.5         0.0033x  far below — bandwidth-bound

  decode is short of the ridge by    306x
```

Prefill sits at 256 — past the ridge on the two A100s, near it on the Hopper parts, comfortably in the productive zone either way. Decode sits at **0.5**, short of the easiest ridge by **306×**. The hardware spends essentially all of decode waiting on memory. It is **memory-bandwidth-bound**.

And that's a statement about the workload, not the implementation. There is no kernel that fixes a 306× shortfall — you have to change the arithmetic-per-byte ratio itself, which is exactly what every optimization below does. Pope et al., [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (2022), work the same analysis through at production scale if you want it in more depth than one table.

This single fact explains a startling amount:

- **Batching works** — the weight read is already paid for, so more sequences are nearly free (until they aren't; see below).
- **Quantization speeds up decode** even when the arithmetic isn't faster, because there are simply fewer bytes to move.
- **[Speculative decoding](https://arxiv.org/abs/2211.17192) wins** because verifying $k$ draft tokens in one pass costs about the same as generating one — you were bandwidth-bound, not compute-bound.
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
  1       4.1567           1.00x       241               1.0x
  2       5.4191           1.30x       369               1.5x
  4       5.0820           1.22x       787               3.3x
  8       5.8200           1.40x      1375               5.7x
  16      6.8876           1.66x      2323               9.7x
  32      9.1050           2.19x      3515              14.6x
```

32× the work for **2.19×** the time — 14.6× the throughput. The GPU was idling on memory, so the extra sequences rode along inside a weight read that was happening anyway. That is what memory-bound looks like, and it's why every serving stack batches aggressively.

Now the same sweep with a **512-token** prefix:

```text
  batch  ms/step  latency vs b=1  tokens/s  throughput vs b=1
  -------------------------------------------------------------
  1       5.6198           1.00x       178               1.0x
  2       7.3265           1.30x       273               1.5x
  4       9.6397           1.72x       415               2.3x
  8      14.8361           2.64x       539               3.0x
  16     23.1560           4.12x       691               3.9x
  32     40.5192           7.21x       790               4.4x
```

The free lunch is gone: 14.6× throughput becomes **4.4×**.

![Batch sweep at two prefix lengths](/assets/picture/2026-08-02-llm-architectures-kv-cache/batch-sweep-light.png){: .light width="1000" height="540" }
![Batch sweep at two prefix lengths](/assets/picture/2026-08-02-llm-architectures-kv-cache/batch-sweep-dark.png){: .dark width="1000" height="540" }

The reason is the formula from §4. **Weights are shared across the batch; the KV cache is not.** Every sequence brings its own cache, so KV traffic scales with batch while weight traffic stays flat. Which is a claim about two numbers, so here are the two numbers, per decode step, at the 512-token prefix:

```text
  batch  weight bytes   KV bytes      total  KV share  bound by
  ---------------------------------------------------------------
  1         0.233 GiB  0.008 GiB  0.241 GiB        3%   weights
  2         0.233 GiB  0.016 GiB  0.249 GiB        6%   weights
  4         0.233 GiB  0.031 GiB  0.265 GiB       12%   weights
  8         0.233 GiB  0.062 GiB  0.296 GiB       21%   weights
  16        0.233 GiB  0.125 GiB  0.358 GiB       35%   weights
  32        0.233 GiB  0.250 GiB  0.483 GiB       52%  KV cache

  KV overtakes weights at batch (32-tok prefix) 478
  KV overtakes weights at batch (512-tok prefix) 30
```

Read the weight column: it never changes. Read the KV column: it doubles every row. Somewhere between batch 16 and 32 they cross, and the KV share goes from a rounding error at 3% to the **majority** of memory traffic at 52%.

Line that up against the throughput column above and the two tell one story. Throughput is still climbing steeply while KV share is under 20%; it flattens as the share passes half. Batching amortizes one term and multiplies the other, and the shape of the curve is just which term is winning.

The last two rows are the same crossover stated as a batch size, and the gap between them is the point: with a 32-token prefix you'd need batch **478** before the cache matters, with a 512-token prefix you need batch **30**. Sixteen times the context, sixteen times sooner. The context length you support and the batch size you can profitably run are the same decision.

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

**A stronger answer:** "Both phases stream the whole weight matrix through the ALUs. Prefill amortizes that read over thousands of tokens and lands around 250 FLOPs per byte, which is roughly where the hardware saturates — it's compute-bound. Decode does the same read for a single token, landing near 0.5 FLOPs per byte, so it's memory-bandwidth-bound and the ALUs sit idle. In my measurements a decoded token cost somewhere around 80–100× a prefilled one. That's also why batching, quantization, and speculative decoding all help decode specifically: they either amortize the read or shrink it. The caveat is that batching only amortizes the *weight* read — KV-cache traffic scales with batch, so past a point throughput plateaus."

The difference is that the second answer names the bottleneck resource and predicts which optimizations work.

### What's next {#whats-next}

[Post 3](/posts/llm-architectures-flash-attention/) is **Flash Attention** — the other half of the memory-traffic story. We've been treating attention itself as cheap, but at long context the $n \times n$ score matrix is the problem, and the fix is the same insight as this post applied one level down: don't move bytes you don't have to. We'll implement online softmax from scratch and confirm it's exact to floating-point noise.

### References

- Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (2019) — MQA, behind §5. Its abstract names the problem this whole post is about: incremental inference is slow "due to the memory-bandwidth cost of repeatedly loading the large 'keys' and 'values' tensors."
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (2023) — §5's middle option, and the uptraining recipe that made it adoptable.
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023) — vLLM, and the fix for the pre-allocation problem in §1.
- Pope et al., [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (2022) — the arithmetic-intensity analysis of §6, worked through at production scale.
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022) — why §6's bandwidth bound is what makes drafting pay.
- Code for this post: [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher), `uv run demo02`.
