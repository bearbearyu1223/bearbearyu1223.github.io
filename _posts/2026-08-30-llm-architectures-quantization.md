---
title: "LLM Architecture Refresh [4]: Quantization, and Why Perplexity Won't Tell You It Broke"
date: 2026-08-30 02:00:00 -0700
categories: [LLM Architecture Refresh, Inference]
tags: [quantization, int8, nf4, qlora, outlier-features, perplexity, pytorch]
description: >-
  Implementing INT8 and NF4 by hand on a real trained model to find out what
  quantization actually damages — and why the metric everyone checks is the one
  least able to see it.
math: true
pin: true
published: false
---

## The first tradeoff that is actually a tradeoff

[Post 1](/posts/llm-architectures-attention-and-rope/) took a transformer block apart and put a cost on every piece, including the one that matters most here: the feed-forward network holds more of a model's parameters than attention does. Those parameters are what this post is about to start rounding off.

[Post 2](/posts/llm-architectures-kv-cache/) and [post 3](/posts/llm-architectures-flash-attention/) then each removed a cost. They are worth separating, because they spend different currencies.

The **KV cache** removes work that would otherwise be *repeated*. Every generation step needs a key and a value for each earlier token, and without a cache it recomputes all of them — a **284×** multiplier of pure repeated work on post 2's run. Storing them makes that linear instead of quadratic, at a cost of **10.7× more memory held**. The cache spends memory to save compute.

**Flash Attention** removes work that would otherwise be *written down and fetched back*: the $n \times n$ table of every token scored against every other, 2 GiB for one layer at 8k tokens, built in memory only to be normalized and thrown away. Computing it one tile at a time takes that memory from quadratic to constant and moves **3.9× fewer bytes** between the GPU's main memory and the small fast memory beside the arithmetic units. What it does not do is less arithmetic — [the FLOP totals are identical to the digit](/posts/llm-architectures-flash-attention/#where-the-speed-actually-comes-from). It is faster only because attention at long context waits on memory rather than on multiplies.

Different trades, same guarantee: **neither one changes a single number in the model.** Both posts showed it rather than claimed it. Generating with the cache and without it produces the same text, token for token. Tiled attention and PyTorch's own kernel [agree to about **1.4e-6**](/posts/llm-architectures-flash-attention/#exact-and-what-exact-is-worth) — the size of disagreement you get from adding the same numbers in a different order, not from computing a different answer.

**Quantization is where that ends.** Storing a weight in eight bits instead of sixteen throws information away. There is no clever reordering that gets it back. So the question stops being *is it exact* and becomes two harder ones: what exactly gets damaged, and how would you know if it were?

The second question is the interesting one, because the metric almost everyone reaches for cannot answer it. **Perplexity** is one number for how surprised a model is by text it did not train on, lower being better. It is also an *average*, and §7 builds it up from scratch to show why that single fact is what blinds it.

### The short version {#the-short-version}

**First, what quantization is**, because the word is heavier than the idea. A trained model is a large pile of numbers, and by default each is stored in 16 bits. Quantization stores each one in fewer — 8 bits, or 4 — by picking a small set of allowed values and rounding every weight to the nearest one. That is the mechanism: a **grid** of permitted values, and rounding.

Why bother: [post 2](/posts/llm-architectures-kv-cache/) established that generating a token is bound by *memory bandwidth*, not arithmetic. The chip spends its time hauling weights in from memory and waiting. Halve the bytes and you halve the haul, so the model both fits in less memory and runs faster — for exactly the same reason, which is that there is less of it to move.

What it costs is precision, and these are the things this post establishes about that cost:

- **Rounding error is half a grid step, and the step is set by the largest value sharing the grid.** On a real weight matrix, `absmax/127/2` predicts `2.414e-03` and the measured worst error is `2.414e-03`. Nothing about quantization damage is mysterious once you see it as one division — every result below is a consequence of who is forced to share a divisor with whom.
- **The outliers are in the activations, not the weights.** Two words there. A **channel** is one of the model's internal feature dimensions — one row of a weight matrix, or one column of the numbers flowing between layers. Those flowing numbers are the **activations**: the weights are fixed once training ends, the activations are computed fresh for every input. Weight channels sit within 4–7× of their median; the activations flowing into the MLP run **10–85×**. That gap is why essentially every shipped quantization scheme leaves activations alone and touches only weights.
- **So where you put the scale matters more than how many bits you keep.** The same 8-bit format, with one divisor per tensor rather than one per row, costs **3.8% relative error on weights but 58% on activations**. Same bits, same spacing, two orders of magnitude apart in damage.
- **The textbook story about outliers is only half true at this scale.** The received account says a few fixed dimensions are extreme everywhere. Measured here: 2 of the top 10 channels persist across five unrelated inputs — including Python source and Spanish — but **0 of 10 are shared between adjacent layers**. Stable enough to matter, not stable enough to hard-code.
- **NF4's codebook is derived, not designed.** Its 16 levels are the equal-probability quantiles of a normal distribution, and deriving them reproduces the constants bitsandbytes ships to **6e-08**. Against uniform 4-bit at the same width it cuts error by about a fifth. It does not beat 8-bit and never could — 16 levels against 256.
- **"4-bit" does not mean four times smaller.** NF4 carries a 32-bit scale per 64 weights, so it is really 4.5 bits, and the embedding table nobody quantizes is 28% of this model. Actual shrink: **2.09×**, not 4×.
- **Perplexity moves +0.21% while the worst token moves 68× the average.** INT8 weight-only looks free by any perplexity gate you would set. In the same run, one token's predicted distribution shifted by 0.176 nats against a mean of 0.0026. A mean over thousands of tokens is exactly the statistic that cannot see a tail.
- **One line decides more than the format does.** Many models save space by making the table that turns tokens into vectors on the way in and the layer that turns vectors back into token scores on the way out *the same numbers*, stored once. They are called **tied**. In this model they are, so quantizing "every linear layer" silently quantizes the embedding table too. That one decision costs **11 points of perplexity**, more than the gap between 8-bit and 4-bit.

Every one of those has a **receipt** behind it — a program that prints the number, so you can check it rather than take my word:

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo04
```

Every number and figure below came out of that command on my M-series Mac. The code is in [`demos/d04_quantization.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d04_quantization.py), with the formats themselves in [`quantizers.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/quantizers.py).

One break from the earlier posts, and it matters. Posts 2 and 3 ran on `toy_model.py`, a Llama-shaped decoder with **random** weights, because they measured time and memory and random weights are fine for that. They are useless here. Outlier channels and perplexity are both products of *training*: a random tensor measures 1.34× max-to-median across channels where a trained one measures 85×, so the central phenomenon of this post simply does not exist in an untrained model. Post 4 therefore runs on **Qwen2.5-0.5B** — small enough for a laptop, real enough to break properly.

### Table of Contents

Skip to [the short version](#the-short-version) for the findings without the derivations.

1. [What rounding to a grid costs](#what-rounding-to-a-grid-costs)
2. [Where the outliers actually live](#where-the-outliers-actually-live)
3. [Are they the same channels every time?](#are-they-the-same-channels-every-time)
4. [Same bits, different scale placement](#same-bits-different-scale-placement)
5. [NF4: a grid shaped like the data](#nf4-a-grid-shaped-like-the-data)
6. [What it actually saves](#what-it-actually-saves)
7. [Perplexity says fine; the tail says otherwise](#perplexity-says-fine)
8. [What follows from all this](#what-follows-from-all-this)
9. [Sidebar: the probe](#sidebar-the-probe)

Plus an [appendix of all notation](#appendix-all-notation) at the end, if a symbol ever goes by without introduction.

---

### 1. What rounding to a grid costs {#what-rounding-to-a-grid-costs}

Take one real weight matrix and quantize it to **INT8**: 8 bits per weight, which allows 256 distinct values.

The matrix is layer 11's `down_proj`, of shape $(896, 4864)$. Every transformer block ends with a small two-layer network applied to each token on its own — [post 1](/posts/llm-architectures-attention-and-rope/) calls it the **FFN**, and the code calls it the **MLP**. It widens each token's vector, applies a nonlinearity, then narrows it back; `down_proj` is that narrowing step, and its partner at the end of the attention half of the block is `o_proj`; those two names carry most of this post.

The simplest scheme is *absmax symmetric*. **Absmax** is just the largest magnitude in the tensor, ignoring sign. Find it, spread the available levels evenly from $-\text{absmax}$ to $+\text{absmax}$, and round:

```python
def int8_per_tensor(w):
    scale = w.abs().max().clamp(min=1e-12) / 127
    return torch.round(w / scale).clamp(-127, 127) * scale
```

That is all quantization is: one division, one rounding, one multiplication back. The **scale**, meaning the divisor, is the only design decision in the whole thing, and the rest of this post is about where to put it.

[`what_rounding_costs`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d04_quantization.py) applies that to the matrix and prints what it cost:

```text
  absmax |w|                         0.6133
  std of w                           0.0183
  absmax / std                       33.6

  step = absmax / 127                4.829e-03
  worst rounding error = step/2      2.414e-03
  measured max |q(w) - w|            2.414e-03
```

The last two lines are the check, and they agree exactly. **The error is half a grid step.** Nothing more mysterious than that is going on, and the grid step is $\text{absmax}/127$ — set entirely by the single largest weight in the tensor.

Which puts the whole problem in one line: *every weight in that matrix is paying for the biggest one*. Note the third row — the largest weight is 33.6 standard deviations out. The grid has to stretch that far, while almost all the weights huddle near zero, so most of the 256 levels sit in a region that is essentially empty.

That observation has two possible fixes, and this post measures both. Either **narrow what shares a grid** (§4), or **stop spacing the levels evenly** (§5).

### 2. Where the outliers actually live {#where-the-outliers-actually-live}

Before fixing anything, find out where the trouble is. §1 showed that the damage is set by the ratio between the biggest value on a grid and the typical one, the **dynamic range** of whatever shares that grid. So the question is which tensors have a wide one.

The measurement: for each channel, take its largest magnitude and express it as a multiple of the median channel's. A channel, from [the short version](#the-short-version), is one row of a weight matrix or one column of an activation tensor.

[`where_the_outliers_are`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d04_quantization.py) measures both. Weights first:

```text
  tensor                max / median  channels > 5x  channels
  -------------------------------------------------------------
  L0 down_proj.weight           4.2x              0       896
  L5 down_proj.weight           7.6x              3       896
  L11 down_proj.weight          7.0x              4       896
  L17 down_proj.weight          4.6x              0       896
  L23 down_proj.weight          7.3x              1       896
```

Tame. No channel is more than about 7× its median neighbour, and at most 4 of 896 exceed 5×. Now the activations. The same function captures the values flowing *into* those same layers during a real forward pass, using a **forward hook**, which is PyTorch's way of asking to be handed a layer's inputs as they go past, without changing what the model computes:

```text
  tensor         max / median  channels > 5x  channels
  ------------------------------------------------------
  L0 o_proj              7.9x              5       896
  L0 down_proj          11.7x            155      4864
  L5 o_proj              3.8x              0       896
  L5 down_proj          85.5x            224      4864
  L11 o_proj             5.4x              2       896
  L11 down_proj         10.3x             28      4864
  L17 o_proj             5.5x              2       896
  L17 down_proj         20.9x            158      4864
  L23 o_proj             8.2x             22       896
  L23 down_proj         19.1x             97      4864
```

The `down_proj` inputs run to **85×** their median channel, an order of magnitude worse than any weight tensor. And it is specifically `down_proj`, the narrowing step at the end of the MLP, which sees the widened vector after the nonlinearity. `o_proj`, at the end of the attention half, stays in the same mild range the weights do.

![Per-channel spread: weights against activations](/assets/picture/2026-08-02-llm-architectures-quantization/outlier-channels-light.png){: .light width="1000" height="654" }
![Per-channel spread: weights against activations](/assets/picture/2026-08-02-llm-architectures-quantization/outlier-channels-dark.png){: .dark width="1000" height="654" }

Both axes are logarithmic, and the two curves are not the same shape. The weights descend gently from a low peak. The activations start two orders of magnitude up and fall off a cliff — a small number of channels carrying magnitudes nothing else comes close to. These are the **outlier features** described in Dettmers et al., [LLM.int8()](https://arxiv.org/abs/2208.07339) (2022).

**Weight-only quantization is what essentially everyone ships** (GPTQ, AWQ, bitsandbytes' NF4), and the reason is the two tables above: weights are well behaved and activations are not. Quantizing activations means confronting an 85× dynamic range on every forward pass; quantizing weights means confronting a 7× one, once, offline.

### 3. Are they the same channels every time? {#are-they-the-same-channels-every-time}

Here the received account and this model part company.

The standard story is that outlier features are *systematic*: a few specific dimensions are extreme regardless of input, which is what would let you handle them specially. That is a checkable claim, so check it. [`outlier_persistence`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d04_quantization.py) takes one layer, runs five unrelated inputs through it — English prose, popular science, economics, Python source, Spanish — and compares which channels land in the top ten:

```text
  input pair                  shared of top-10
  ----------------------------------------------
  English prose vs science                3/10
  science vs economics                    2/10
  economics vs Python source              2/10
  Python source vs Spanish                2/10

  shared by all five inputs          2/10
  distinct channels across inputs    41

  shared by all 6 sampled layers     0/10
```

So: **partly**. Two channels are in the top ten for all five inputs — including Python source and Spanish, which is not something you get by coincidence, and is real evidence of a persistent core. But 41 distinct channels appear across just five inputs, so most of the outlier set moves with what you feed it. And across layers there is **no overlap at all**: each layer has its own outlier channels, and knowing layer 5's tells you nothing about layer 11's.

Be careful about what this does and doesn't say. The strong systematic version is documented at 6.7B parameters and above; this is a 0.5B model, and whether the effect sharpens with scale is **not measured here** — I have no 7B result to offer, and the literature says it would look different. What the measurement does establish is that at this scale you cannot pick the outlier channels once and hard-code them.

Which is fine, because the fix doesn't require knowing which channels they are. It only requires never letting a whole tensor share one divisor.

### 4. Same bits, different scale placement {#same-bits-different-scale-placement}

Keep the format identical, 8 bits and evenly spaced, and change only the *scope* of the scale. Per-tensor uses one divisor for everything. **Per-channel** gives each output row its own:

```python
def int8_per_channel(w):
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127
    return torch.round(w / scale).clamp(-127, 127) * scale
```

One line different, `amax(dim=-1, keepdim=True)` instead of `max()`, and it confines each outlier to the row it lives in. The overhead is one float per row against `in_features` weights per row, a fraction of a percent.

Error is reported as **relative RMSE**. Root-mean-square is the ordinary way to average a set of numbers when you want their size and not their sign: square each one, take the mean, take the square root. Do that to the errors, divide by the same quantity computed on the original weights, and the result reads as a percentage of typical magnitude rather than an absolute number whose scale you'd have to remember. [`scale_placement`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d04_quantization.py) reports it for both scopes:

```text
  tensor                      per-tensor  per-row  ratio
  --------------------------------------------------------
  L0 down_proj.weight              3.81%    1.14%   3.3x
  L11 down_proj.weight             7.63%    1.41%   5.4x
  L23 down_proj.weight             6.69%    1.14%   5.9x
  L5 down_proj (activations)      58.10%    6.10%   9.5x
```

For weights, splitting the scale is worth 3–6× less error, which is useful without being dramatic.

For the activation tensor it is the difference between a number you can use and one you can't. **58% relative error** means the quantized tensor barely resembles the original — and the cause is the row from §2: that tensor has a channel 85× its median, and per-tensor scaling makes every other channel share a grid built to survive it.

So "is 8 bits enough?" is not a question about 8 bits. The bit width was identical in both columns; what changed was how much dynamic range was forced through a single divisor.

### 5. NF4: a grid shaped like the data {#nf4-a-grid-shaped-like-the-data}

The other fix attacks the grid instead of the scale. Evenly spaced levels are optimal only if the values are evenly spread, and §1 already showed they aren't: weights are roughly normal, clustered hard around zero, with an absmax 33 standard deviations out.

**NF4**, NormalFloat-4 from Dettmers et al.'s [QLoRA](https://arxiv.org/abs/2305.14314) (2023), puts its 16 levels at the *equal-probability quantiles* of a standard normal instead, so each level claims about the same share of the weights. The set of levels a format rounds to is its **codebook**, and NF4's is derived rather than chosen. [`nf4_codebook`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/quantizers.py) is the derivation, and `icdf` is what does the work: it is the inverse of the normal distribution's cumulative function, so handing it a probability returns the value with that much of the distribution below it. Ask for sixteen evenly spaced probabilities and you get back sixteen levels that each cover an equal share:

```python
def nf4_codebook():
    dist = Normal(torch.tensor(0.0), torch.tensor(1.0))
    offset = 0.9677083
    positive = dist.icdf(torch.linspace(offset, 0.5, 9)[:-1])
    negative = -dist.icdf(torch.linspace(offset, 0.5, 8)[:-1])
    levels = torch.cat([negative, torch.zeros(1), positive]).sort().values
    return levels / levels.abs().max()
```

[`nf4_derivation`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d04_quantization.py) runs that and compares the result against the table bitsandbytes — the library most 4-bit models are loaded with — actually ships:

```text
  levels derived                     16
  max |derived - bitsandbytes|       5.96e-08

  narrowest step (near zero)         0.0796
  widest step (near +/-1)            0.3038
  ratio                              3.82x
```

They agree to **6e-08**, which is float noise. The table in the library is not a magic set of tuned numbers; it is what falls out of asking where to put 16 levels if the data is normal.

The last three lines are the point. NF4's levels are **3.82× closer together near zero** than at the extremes, which is exactly where the weights are.

![Sixteen levels, evenly spaced or at normal quantiles](/assets/picture/2026-08-02-llm-architectures-quantization/quant-grids-light.png){: .light width="1000" height="683" }
![Sixteen levels, evenly spaced or at normal quantiles](/assets/picture/2026-08-02-llm-architectures-quantization/quant-grids-dark.png){: .dark width="1000" height="683" }

That figure is drawn on **block-normalized** weights, which is not a cosmetic choice. NF4 is applied blockwise: every 64 weights are divided by their own absmax before meeting the codebook. On a raw axis the 33× ratio from §1 pushes every level of both grids out into empty tails and the comparison shows nothing. Divided by their block's absmax, the weights form the bell the codebook was designed for, and you can see the dashed NF4 levels crowding the middle where the mass is while the solid uniform levels ignore it. (The spikes at $\pm 1$ are an artifact of the normalization: each block contributes exactly one weight at its own absmax.)

Now measure all of it on the same matrix, at real stored cost. The same function quantizes the matrix five ways and reports error against storage:

```text
  format                  rel RMSE  bits/weight incl. scales
  ------------------------------------------------------------
  INT8 per-tensor            7.63%                     8.000
  INT8 per-channel           1.41%                     8.007
  INT4 uniform, block=64    12.21%                     4.500
  NF4 block=64               9.64%                     4.500
  NF4 block=256             11.15%                     4.125
```

The comparison that tests NF4's actual claim is the **middle pair**: uniform INT4 against NF4, same width, same block size, same number of scales. The only variable left is where the levels sit, and moving them to the quantiles cuts error from 12.21% to 9.64% — about a fifth.

It does **not** beat 8-bit, and it was never going to: 16 levels against 256. What it does is land within a few points of per-tensor INT8 at *half the storage*. And per-channel INT8, at 1.41%, beats both by a wide margin — which is worth holding onto, because "4-bit" gets discussed as though it were strictly better than 8-bit rather than a different point on a curve.

Note the bits column too. NF4 at `block=64` carries a 32-bit scale per 64 weights: $4 + 32/64 = 4.5$ bits. A "4-bit" model is a 4.5-bit model. Widening to `block=256` brings that to 4.125 and pays for it in error, because more weights then share one absmax — the same trade as §4, at a finer grain.

### 6. What it actually saves {#what-it-actually-saves}

Quantization is sold in multiples: "4-bit means 4× smaller". It does not. [`memory_cost`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d04_quantization.py) counts what is actually there:

```text
  parameters                         494.0M
  of which embedding (tied)          136.1M  (28%)
  of which other Linear              357.8M  (72%)

  scheme            bits/weight (Linear)  model size  vs fp16
  -------------------------------------------------------------
  fp16 (baseline)                 16.000     942 MiB    1.00x
  INT8 per-channel                 8.036     603 MiB    1.56x
  NF4 block=64                     4.500     452 MiB    2.09x
```

Nominally 16 → 4 bits is a 4× cut. Measured, it is **2.09×**. Two things dilute it, and both are counted rather than estimated:

- **The scales ride along.** 4 bits become 4.5, an eighth of the saving gone before anything else.
- **28% of this model is an embedding table that nobody quantizes**, and it stays at 16 bits. Small models are hit hardest here — the embedding is a fixed cost set by vocabulary size, so it is a much larger share of a 0.5B model than of a 70B one. Expect the ratio to look better at scale, though this post doesn't measure that.

These are **analytic** figures — parameter counts times bits per weight, not a measurement of a file on disk. They'll track a real checkpoint closely but won't match it to the byte, since formats carry headers and padding of their own.

### 7. Perplexity says fine; the tail says otherwise {#perplexity-says-fine}

Now the question that matters: does the model still work?

The standard answer is **perplexity**, and it is worth building up rather than quoting, because its blind spot follows directly from how it is made.

Run some real text through the model. At each position the model has produced a probability for every token in its vocabulary, and one of those tokens is the one that actually came next. Pull out the probability it gave that correct token. A confident, well-fitted model gives it a high probability; a damaged one gives it a lower one.

Now turn that into a score. Take the logarithm of each of those probabilities and negate it, so a probability of 1 scores 0 and anything less scores positive — that is how much the model was *surprised* at that position. Average the surprise over every position. Then exponentiate the average, which converts it out of log units back into a count. That count is perplexity, and it reads as: *the model was as uncertain as if it had been choosing uniformly among this many options at each step.* A perplexity of 20 means roughly a 20-way guess, and lower is better.

That is four lines of the demo, and the third one is where the rest of this section comes from:

```python
def perplexity(lg):
    logp = torch.log_softmax(lg[:, :-1], dim=-1)
    nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return nll.mean().exp().item()
```

`nll.mean()` is an average over every token in the passage. **Every conclusion in this section falls out of that one call**: an average is a single number standing in for thousands, and it has no way to report that a few of them are far from the rest.

Evaluating perplexity needs text, so the demo ships a fixed 550-token passage covering six unrelated topics. That choice is load-bearing, and my first attempt got it wrong: it repeated one paragraph a dozen times, which is *far* easier to predict, and it reported NF4 costing 2.5% where honest non-repeating prose reports 15.5%. **Repetitive eval text flatters a quantized model.**

So alongside perplexity, measure something an average cannot see. Perplexity looked at one number per position, the probability of the token that actually came next. But the model produced a probability for *every* token in its vocabulary at that position, and quantizing shifts the whole shape of that distribution. **KL divergence** puts a single number on how far the shape moved: zero means the two distributions are identical, larger means the model now expects something different. It is reported per token, in **nats** — the unit you get when the logarithms are natural ones, as they were above. Here 0.18 nats is a small but real change of mind and 10 nats is a different model.

[`quality`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d04_quantization.py) runs each scheme against the unquantized reference and reports both:

```text
  scheme                perplexity  vs fp32   mean KL    max KL  max/mean
  -------------------------------------------------------------------------
  fp32 reference            20.869        —         —         —         —
  INT8 per-channel          20.913   +0.21%  2.60e-03  1.76e-01       68x
  NF4 block=64              24.104  +15.50%  1.57e-01  1.09e+01       69x
  NF4 block=64, + head      26.486  +26.92%  2.44e-01  1.09e+01       45x
```

Read the INT8 row twice. **Perplexity moved 0.21%.** That would clear any ship/no-ship threshold anyone sets — you would call it lossless and move on. In the very same run, one token's predicted distribution moved **68× the average**.

Both numbers are correct. Perplexity is a *mean* over hundreds of tokens, and a mean is precisely the statistic that cannot see a tail. The same function sorts every token's KL and reports where the damage sits. A **quantile** is the value below which that fraction of tokens falls, so quantile 0.99 is the level only the worst 1% exceed:

```text
  INT8 KL at quantile 0.5            1.730e-03
  INT8 KL at quantile 0.9            4.341e-03
  INT8 KL at quantile 0.99           1.194e-02
  INT8 KL at quantile 0.999          9.441e-02
  INT8 KL at quantile 1.0            1.756e-01
```

Flat across the bulk, then a hundred-fold climb in the last tenth of a percent.

![KL divergence by percentile, with the means perplexity reports](/assets/picture/2026-08-02-llm-architectures-quantization/kl-tail-light.png){: .light width="1000" height="641" }
![KL divergence by percentile, with the means perplexity reports](/assets/picture/2026-08-02-llm-architectures-quantization/kl-tail-dark.png){: .dark width="1000" height="641" }

Each dotted line is the mean — the kind of number a perplexity comparison reports. Each curve is where the tokens actually are. The means sit *above* the median of their own curves, dragged up by a handful of tokens at the right edge, and every curve hooks sharply upward in its last few percent. **The tokens that changed most are exactly the ones a mean is worst at reporting.**

Whether that matters depends on what the changed tokens *are*. Perplexity treats every position as interchangeable; a product does not. A model that is imperceptibly worse on ordinary prose and meaningfully worse on the rare tokens carrying names, digits, code syntax, or a refusal has a problem that perplexity is not built to report, and this post doesn't identify which tokens moved — only that some did, by a lot. That is the honest boundary of the measurement, and it is also the argument for evaluating on what you actually care about.

#### The line that costs more than the format

The bottom row is a different lesson. In Qwen2.5-0.5B the embeddings are **tied**: `lm_head.weight` and `embed_tokens.weight` are the same tensor under two names, sharing storage.

A **tensor** is one array of numbers, and two names can point at the same one. `data_ptr()` returns the address where a tensor's numbers actually live, so comparing the two addresses settles whether they are one tensor or two copies. [`memory_cost`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d04_quantization.py) runs that check:

```text
  lm_head.weight is embed_tokens.weight  yes
```

So the natural-looking loop, "quantize every `nn.Linear`" — every one of the model's matrix-multiply layers — also quantizes the embedding table, which every token in the sequence reads on the way in. Perplexity goes from +15.50% to **+26.92%**: that single decision costs 11 points, comparable to the entire gap between 8-bit and 4-bit. Real tools exclude the head and embeddings by default, and this is why.

### 8. What follows from all this {#what-follows-from-all-this}

| If you're deciding | The thing that decides it |
| --- | --- |
| 8-bit or 4-bit? | Not error alone — per-channel INT8 was **1.41%** against NF4's 9.64%. 4-bit is a memory decision you pay for in quality, not a free upgrade. |
| Weights only, or activations too? | Weights only, unless you have a specific plan for an 85× dynamic range that changes every forward pass. |
| Per-tensor or per-channel? | Per-channel, always. It costs one float per row and was worth 3–6× on weights, 9.5× on activations. |
| Is my quantized model safe to ship? | Not answerable by perplexity. It moved 0.21% while the worst token moved 68× the mean. Look at the tail, and at the tokens you actually care about. |
| Why is my 4-bit model only half the size? | Scales (4 → 4.5 bits) and the embedding table you correctly left alone (28% here). |
| Why did quality collapse when I quantized everything? | Check whether your embeddings are tied. `lm_head` is often the same tensor as `embed_tokens`. |

Underneath all of it, **quantization damage is a dynamic-range problem rather than a bit-width problem.** Every result here reduces to who was forced to share a divisor with whom — per-tensor against per-channel, block 64 against block 256, activations against weights. Bits set the ceiling on how good it can get; scale placement decides how much of that ceiling you reach.

### 9. Sidebar: the probe {#sidebar-the-probe}

> **"You quantized the model to 4-bit and perplexity barely moved. Is it safe to ship?"**

**A weak answer:** "Yes — perplexity is within a percent of the original, so it's effectively lossless."

That is the answer the metric invites, and it mistakes an average for a guarantee.

**A stronger answer:** "Perplexity being flat is necessary, not sufficient. It's a mean over every token, so it can't see a tail. When I measured per-token KL divergence against the unquantized model, INT8 weight-only moved perplexity 0.2% while the worst token's distribution moved about 68× the mean — flat across the bulk and then a hundred-fold climb in the last tenth of a percent. So the real question is *which* tokens moved: if the damage lands on rare tokens carrying names, numbers, or code syntax, perplexity will never show it and your users will. I'd also check what got quantized — if the embeddings are tied to `lm_head`, quantizing every linear layer hits the embedding table too, which cost more here than the difference between 8-bit and 4-bit."

The tell is whether someone treats perplexity as a summary statistic with known blind spots, or as a verdict.

### What's next {#whats-next}

Post 5 is **mixture-of-experts** — the other way to make a large model cheap to run, and one that trades in a completely different currency. Quantization shrinks every weight; MoE keeps them all and simply declines to use most of them on any given token. The interesting questions there are what the router actually learns, why "active parameters" is the number that predicts speed while total parameters predicts your memory bill, and what happens to a batch when every sequence in it wants a different expert.

### Appendix: all notation {#appendix-all-notation}

Every symbol this post uses, in one place. [Post 1's appendix](/posts/llm-architectures-attention-and-rope/#appendix-all-notation) covers attention's own notation, and [post 2's](/posts/llm-architectures-kv-cache/#appendix-all-notation) the memory and serving terms.

| Symbol | Means | In this post's runs |
| --- | --- | --- |
| $w$ | a weight tensor, before quantization | e.g. $(896, 4864)$ |
| $q(w)$ | the same tensor after rounding to the format's levels | same shape, fewer distinct values |
| $s$ | the **scale**: the divisor applied before rounding | one per tensor, row, or block |
| absmax | the largest magnitude in whatever shares a scale | 0.6133 for L11 `down_proj` |
| step | spacing between adjacent levels, $\text{absmax}/127$ for INT8 | 4.829e-03 |
| $b$ | bits per stored weight, **scales included** | 8.007, 4.5, 4.125 |
| block | how many weights share one scale, in a blockwise format | 64 or 256 |
| channel | one row of a weight matrix, or one feature dimension of an activation | 896 or 4864 of them |
| max/median | a channel's peak magnitude over the median channel's — the outlier measure | 4–7× weights, 10–85× activations |
| rel RMSE | root-mean-square error over the original's root-mean-square | 1.41% to 58.10% |
| PPL | **perplexity** — how surprised the model is by held-out text, lower being better | 20.869 unquantized |
| $D_{KL}$ | **KL divergence** in nats between two predicted distributions, per token | mean 2.6e-03, max 1.76e-01 |
| INT8 / INT4 | integer formats with **evenly spaced** levels, 256 and 16 of them | — |
| NF4 | NormalFloat-4: 16 levels at the **quantiles of a normal** | derived, matches bitsandbytes to 6e-08 |
| fp32 / fp16 | 32- and 16-bit floating point — four and two bytes a number | fp32 reference, fp16 baseline |

Three things worth keeping straight:

- **Per-tensor, per-channel, and blockwise are the same idea at three grains.** Each names how many weights share one scale: all of them, one row's worth, or 64. Every quality result in this post is a consequence of that number, not of the bit width beside it.
- **"4-bit" is a level count, not a storage cost.** 16 levels is what makes it 4-bit; the fp32 scale per block is what makes it 4.5 bits on disk. Both numbers are honest and they describe different things.
- **Error and divergence answer different questions.** Relative RMSE (§4, §5) measures how far the *weights* moved and is a property of the format alone. KL divergence (§7) measures how far the *predictions* moved and is a property of the whole model. A format can look bad on the first and fine on the second, which is roughly what INT8 does.

### References

- Dettmers et al., [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) (2022) — where §2's outlier features are described, and the mixed-precision decomposition that works around them.
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (2023) — NF4, its derivation from the normal distribution, and blockwise scaling. §5.
- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (2022) — weight-only quantization that uses second-order information rather than absmax alone.
- Lin et al., [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (2023) — uses the activation outliers of §2 to decide which weights matter, rather than treating all of them alike.
- Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) (2022) — the other response to §2: migrate the activation range into the weights, where it is survivable.
- Code for this post: [`llm-architectures-refresher`](https://github.com/bearbearyu1223/llm-architectures-refresher), `uv run demo04`.
