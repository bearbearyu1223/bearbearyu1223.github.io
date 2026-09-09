---
title: "LLM Architecture Refresh [5]: Mixture-of-Experts, and Why Sparsity Doesn't Survive a Batch"
date: 2026-09-09 09:00:00 -0700
categories: [LLM Architecture Refresh, Inference]
tags: [mixture-of-experts, moe, routing, sparsity, olmoe, load-balancing, pytorch]
description: >-
  Taking a real 64-expert model apart to find out what the router learns, why
  active parameters predict speed while total parameters predict your memory
  bill, and where per-token sparsity quietly disappears.
math: true
pin: true
---

## A model that mostly declines to run

[Post 1](/posts/llm-architectures-attention-and-rope/) took a transformer block apart and put a cost on every piece. One finding from it matters more here than anything else: the **feed-forward network** holds more of a model's parameters than attention does. (The feed-forward network, or FFN, is the per-token stack of two or three matrix multiplies that sits after attention in every block. Attention is where tokens look at each other; the FFN is where each token is transformed on its own. This post also calls it the **MLP**, which is the same thing under an older name.)

The three posts since then all tried to make a model cheaper without changing what it computes, or while admitting exactly what they changed.

[Post 2](/posts/llm-architectures-kv-cache/) removed work that was being *repeated*, caching keys and values instead of recomputing them at every generation step: a **284×** cut in repeated compute, paid for with **10.7× more memory held**. [Post 3](/posts/llm-architectures-flash-attention/) removed work that was being *written down and fetched back*, computing attention one tile at a time to move **3.9× fewer bytes**, with the arithmetic identical to the digit. [Post 4](/posts/llm-architectures-quantization/) stopped being exact on purpose, storing each weight in fewer bits for a **2.09×** shrink, and spent most of its length on the fact that the usual quality check cannot see what that breaks.

Every one of those works on the model you already have. **Mixture-of-experts is a different kind of move: it changes what gets built.**

The idea, in one sentence: instead of one FFN per block that every token goes through, build sixty-four of them and send each token to eight. Nothing is compressed and nothing is approximated. The model simply has far more capacity than it uses on any given token, and a small learned component decides which slice to use.

The arithmetic is genuinely strange the first time you meet it. The model in this post has **6.9 billion** parameters. Running it on a token multiplies through about **1.2 billion** of them. The other 5.7 billion sit in memory, doing nothing, waiting for a token that wants them.

### Setup

Everything below is measured, and you can re-run all of it:

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo05
```

The code is in [`demos/d05_moe.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d05_moe.py). Each receipt below names the function that prints it.

The model is **[OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924)**, and the choice was not casual. Like [post 4](/posts/llm-architectures-quantization/), this post needs a *trained* checkpoint, because a router with random weights routes nothing in particular — the specialization in §4 is a product of training and does not exist before it. OLMoE is the smallest fully open MoE that fits on a 24 GB laptop, and its layers are the plain form of the mechanism: 64 experts, top-8, and neither of the two common variations. It has no **shared expert** (an extra FFN that every token goes through in addition to its chosen few), and it was not **upcycled** (built by copying a finished dense model's FFN into many experts and continuing training). It was trained sparse from scratch, which matters for §4: upcycled models are reported to specialize noticeably less. Most open MoE models are tens of billions of parameters in total even when only two or three billion are active, and it is the total that has to fit in memory.

One break from post 4: this demo runs in **bfloat16**, not fp32. 6.92B parameters is 25.8 GiB in fp32 and does not fit in 24 GB; in bf16 it is 12.9 GiB and does. Every claim here is a count or a ratio, both of which survive that change.

### Table of Contents

Skip to [the short version](#the-short-version) for the findings without the derivations.

1. [Where the parameters actually are](#where-the-parameters-actually-are)
2. [The router, which is smaller than you would guess](#the-router)
3. [One token, routed](#one-token-routed)
4. [What the router actually learns](#what-the-router-learns)
5. [Two bills: memory and time](#two-bills)
6. [Sparsity does not survive a batch](#sparsity-and-batching)
7. [Nothing balances the load for free](#load-balance)
8. [What follows from all this](#what-follows)
9. [Sidebar: the probe](#sidebar-the-probe)

Plus an [appendix of all notation](#appendix-all-notation) at the end, if a symbol ever goes by without introduction.

### The short version {#the-short-version}

A **mixture-of-experts** layer replaces one FFN with many copies of it, called **experts**, plus a small **router** that picks a few experts per token. "Sparse" here means only the picked ones run. That is the entire idea; everything else is consequence.

- **The experts are essentially the model.** 93.1% of OLMoE's parameters are experts, against 3.9% for attention. A token activates 17.0% of the total ([§1](#where-the-parameters-actually-are)).
- **The router costs almost nothing.** One 64×2048 matrix per layer, 2.10M parameters, **0.030%** of the model, deciding how the other 93% get spent ([§2](#the-router)).
- **Routing is a softmax, a cut, and a weighted sum** (softmax turns raw scores into probabilities that add to 1), and OLMoE does not renormalize after the cut. The eight kept weights on the token walked through in [§3](#one-token-routed) sum to **0.4281**, not 1, so the router's confidence becomes a scale on the layer's output.
- **The router does specialize, and it is measurable rather than folklore.** Two halves of the *same* passage route differently by 0.216; prose against code differs by 0.731, **2.56×** the noise floor, and the gap widens with depth ([§4](#what-the-router-learns)).
- **Active parameters predict time; total parameters predict memory.** Forcing all 64 experts on costs **2.26×** the elapsed time and exactly zero extra bytes of weights ([§5](#two-bills)).
- **Per-token sparsity is not batch sparsity.** One token needs 8 experts of 64. Two hundred and fifty-six tokens together need **60.9** ([§6](#sparsity-and-batching)). This is the single most useful fact in the post, and it is why an MoE saves arithmetic without saving memory.
- **Nothing keeps the experts equally busy on its own.** The busiest expert in layer 0 takes **5.71×** an even share while four experts go completely unused ([§7](#load-balance)).

---

### 1. Where the parameters actually are {#where-the-parameters-actually-are}

Start with the question [post 4](/posts/llm-architectures-quantization/) opened on, because for an MoE it has a much more interesting answer: which parts of the model does this technique even touch?

For quantization the answer was "nearly all of it, a bit at a time." For mixture-of-experts the answer is that it touches exactly one component — the FFN — and it touches it by making sixty-four copies.

![A dense block against an MoE block: one MLP becomes 64 experts with a router in front, and 8 are lit for this token](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/moe-block-light.png){: .light width="1000" height="526" }
![A dense block against an MoE block: one MLP becomes 64 experts with a router in front, and 8 are lit for this token](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/moe-block-dark.png){: .dark width="1000" height="526" }

Attention is untouched. The eight filled squares are the experts this post's own demo routed the word `' harbour'` to in layer 0, so the picture and [§3](#one-token-routed)'s table are the same measurement.

Counting the parameters by role (`where_the_parameters_are`):

```text
  role          parameters   share
  ----------------------------------
  experts           6.442B   93.1%
  attention         0.269B    3.9%
  embed + head      0.206B    3.0%
  router             2.10M  0.030%
  norms              0.07M  0.001%
  TOTAL             6.919B  100.0%

  experts per layer                  64
  experts used per token             8
  parameters in one expert           6.29M
  total parameters                   6.919B
  active parameters per token        1.177B
  active share                       17.0%
```

![Where OLMoE-1B-7B's 6.9B parameters live: a single bar, 93.1% of it experts](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/weight-census-light.png){: .light width="1000" height="317" }
![Where OLMoE-1B-7B's 6.9B parameters live: a single bar, 93.1% of it experts](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/weight-census-dark.png){: .dark width="1000" height="317" }

Those numbers are checkable from the model's published configuration without downloading a single weight, which is a good habit for any "model X has shape Y" claim. OLMoE has 16 layers, a model width of 2048, an expert width of 1024, and 64 experts per layer. Each expert is a [SwiGLU](/posts/llm-architectures-attention-and-rope/) FFN — the three-matrix form of feed-forward network post 1 pulled apart, where two matrices project the token up into a wider space and a third brings it back down — so it holds $3 \times 2048 \times 1024 = 6{,}291{,}456$ parameters, which is the **6.29M** printed above. Multiply by 64 experts and 16 layers and you get 6.442B, and adding attention, embeddings and norms lands on 6.919B against the 6.92B the checkpoint reports.

The **active** count is the same arithmetic with 8 experts in place of 64: 8 experts × 6.29M × 16 layers is 0.805B, plus attention, norms and the embedding lookup, giving 1.177B. That ratio, **17.0%**, is the number the design exists to produce.

> **"1B active, 7B total"** is what the model's name means, and it is two different measurements of the same object. Neither is a lie and neither is sufficient on its own. Which one you should care about depends entirely on whether you are buying memory or buying time, which is [§5](#two-bills).
{: .prompt-info }

### 2. The router, which is smaller than you would guess {#the-router}

Something has to choose the eight. That something is the **router** (also called the gate): a single matrix per layer that takes a token's vector and produces one score per expert.

```text
  router matrices                    16 (one per layer)
  shape of each                      (64, 2048)
  router parameters                  2.10M
  as a share of the model            0.030%
  router bytes (bf16)                4.0 MiB
  everything else (bf16)             12.88 GiB
```

From `the_router_is_tiny`. Sixteen matrices of 64×2048, 2.10M parameters, **4.0 MiB** in bf16 against **12.88 GiB** for everything else. Three hundredths of one percent of the model decides how the rest of it is spent, on every token, at every layer.

That asymmetry is where the risk in this architecture lives. A router that chooses badly does not merely lose a little accuracy; it wastes the capacity the other 93% of the parameters represent. [§7](#load-balance) is about what happens when it chooses lopsidedly, and why training has to actively push against that.

### 3. One token, routed {#one-token-routed}

Here is the mechanism in full. Before the formula, every symbol in it:

| Symbol | Means | Shape here |
| --- | --- | --- |
| $x$ | one token's vector arriving at the MoE layer | $(2048,)$ |
| $d_{model}$ | the model's width, the length of $x$ | 2048 |
| $E$ | how many experts the layer has | 64 |
| $k$ | how many experts each token is routed to | 8 |
| $W_r$ | the router matrix | $(64, 2048)$ |
| $z$ | **router logits** — one raw, unbounded score per expert | $(64,)$ |
| $p$ | those scores turned into probabilities by softmax | $(64,)$ |
| $\mathcal{K}$ | the set of the $k$ highest-scoring experts | 8 indices |
| $\text{FFN}_e$ | expert $e$, an ordinary feed-forward network | — |
| $y$ | what the layer outputs for this token | $(2048,)$ |

A **logit** is a raw score before it has been turned into a probability; it can be any real number, positive or negative. **Softmax** is the function that turns a list of logits into positive numbers that sum to 1, by exponentiating each one and dividing by the total, so the largest logit gets the largest share.

$$z = W_r\,x \qquad p = \mathrm{softmax}(z) \qquad \mathcal{K} = \operatorname*{top-k}_{e}\ p_e \qquad y = \sum_{e \in \mathcal{K}} p_e \cdot \text{FFN}_e(x)$$

Read left to right: score every expert, turn the scores into probabilities, keep the eight highest, run only those eight, and add their outputs together weighted by how strongly the router wanted each one.

Taking the word `' harbour'` at layer 0 (`one_token_routed`):

```text
  token                              1 = ' harbour'
  layer                              0
  router output                      64 logits -> softmax -> top-8

  rank  expert    logit  softmax weight
  ---------------------------------------
  1          5  +1.8984          0.1161
  2         14  +1.6016          0.0862
  3         41  +1.3516          0.0672
  4         18  +1.2344          0.0597
  5          6  +0.7148          0.0355
  6         17  +0.3340          0.0243
  7         28  +0.1201          0.0196
  8          9  +0.1143          0.0195

  sum of the 8 kept weights          0.4281
  sum of all 64 weights              1.0000
  discarded by the top-k cut         0.5719
  renormalized? (norm_topk_prob)     no
```

Add the eight printed weights and you get 0.4281, which is the printed total. That is deliberate: of the 1488 (layer, position) pairs in this passage, 676 reconcile exactly at four decimal places, and the demo pins one of them so a reader who checks the arithmetic finds it correct.

The last line is the part most explanations skip. The softmax runs over all 64 experts and its outputs sum to 1, but only 8 survive the cut, and **OLMoE does not rescale them afterwards**. The eight weights sum to 0.4281, so 0.5719 of the probability mass is discarded and the layer's output is scaled down by roughly that much.

This is a real design choice with a name in the config, `norm_topk_prob`, and models differ on it. Setting it true would divide the eight weights by their sum so they add to 1, making every token's expert mixture equally strong. Leaving it false, as here, lets the router express *confidence*: a token whose top eight experts are all strongly wanted gets a larger contribution from this layer than a token the router is ambivalent about.

Routing happens independently at every layer, which is easy to miss (`routing_depth_profile`):

```text
  layer  top-1 expert  its weight  kept total  share
  ----------------------------------------------------
  0                 5      0.1161      0.4281   0.27
  1                53      0.0583      0.2540   0.23
  2                36      0.0866      0.3215   0.27
  3                 0      0.0697      0.3567   0.20
  4                35      0.0761      0.3316   0.23
  5                23      0.0639      0.3003   0.21
  6                17      0.0647      0.3298   0.20
  7                43      0.0443      0.3162   0.14
  8                 6      0.0636      0.3282   0.19
  9                41      0.0540      0.3338   0.16
  10               23      0.1208      0.4334   0.28
  11               55      0.0755      0.3685   0.20
  12               48      0.0764      0.4592   0.17
  13                0      0.0733      0.4784   0.15
  14               46      0.0982      0.4830   0.20
  15               23      0.1158      0.4851   0.24
```

One word visits sixteen separate committees of eight on its way through the model. There is no such thing as "the expert for `' harbour'`" — there are 16 independent choices, and the number of distinct paths a single token could take is astronomically large. Note also that `kept total` climbs with depth, from 0.25 in layer 1 to 0.49 in layer 15: the router gets more decisive about the later layers.

### 4. What the router actually learns {#what-the-router-learns}

Now the question everyone asks, and the one where it is easiest to fool yourself.

The folk story is that experts specialize — one handles code, one handles French, one handles punctuation. It is an appealing story, and it is the reason the word "expert" was chosen. So: is it true?

The honest way to ask is to route three passages of clearly different character (English prose, Python, and a paragraph of group theory), then measure how differently they use the experts. The measure is **total variation distance**, which for two distributions over the same 64 experts is half the sum of the absolute differences between them. It runs from 0, meaning the two used the experts identically, to 1, meaning they shared no expert at all.

Here is where it would be easy to go wrong. Any two finite samples differ, even when drawn from the same source. A hundred tokens of prose will not use the experts in exactly the same proportions as another hundred tokens of the same prose, so a nonzero distance between prose and code proves nothing on its own. **The number needs a noise floor.**

So the demo splits each passage in half and measures the distance between the two halves of the *same* text. That is what sampling noise looks like. Every other row is read against it (`what_the_router_learns`):

```text
  comparison           layer 0  layer 8  layer 15   mean
  --------------------------------------------------------
  same passage, split    0.240    0.210     0.173  0.216
  prose vs code          0.465    0.736     0.855  0.731
  prose vs math          0.362    0.508     0.596  0.535
  code vs math           0.371    0.491     0.505  0.397

  noise floor, mean over layers      0.216
  cross-domain, mean over layers     0.554
  cross-domain over noise floor      2.56x
```

The specialization is real. Two halves of one passage differ by 0.216; two different kinds of text differ by 0.554, **2.56×** as much.

The more interesting structure is in the columns. Averaging the three cross-domain rows at each depth and setting them against the floor:

```text
  noise floor, first layer           0.240
  cross-domain, first layer          0.400
  noise floor, last layer            0.173
  cross-domain, last layer           0.652
```

At layer 0 the cross-domain distance is 0.400 against a noise floor of 0.240, a ratio of only 1.7, so early routing is barely about content at all. By layer 15 it is 0.652 against a floor of 0.173, a ratio of 3.8. **The noise floor falls with depth while the cross-domain distance rises**, which is two separate signs of the same thing: deep routing is consistent within a kind of text and sharply different between kinds. Early layers route on something much closer to surface form.

![The same 64 experts used differently by different text, with two halves of one passage on top as the noise floor](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/specialization-light.png){: .light width="1000" height="349" }
![The same 64 experts used differently by different text, with two halves of one passage on top as the noise floor](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/specialization-dark.png){: .dark width="1000" height="349" }

The top two rows are the same prose passage split in half; the bottom two are code and mathematics. The comparison the eye should make is row 1 against row 2 (noise) versus row 1 against row 3 (signal), which is why the noise pair is in the figure rather than described in a caption.

Individual experts do lean hard (`what_the_router_learns`):

```text
  most code-leaning expert (layer 15) 17
    share of code routing slots      9.73%
    share of prose routing slots     0.13%
    an even share would be           1.56%
  experts unused by prose but used by code 13
```

Expert 17 takes 9.73% of code's routing slots against 0.13% of prose's, where an even share would be 1.56%. Thirteen experts are used by code and never by prose at all.

> **Two cautions, because this is where the story runs ahead of the evidence.**
>
> First, "the router distinguishes code from prose" is a much weaker claim than "expert 17 is the code expert." Code and English prose differ in their *tokens* before they differ in anything conceptual, and a router that keyed purely on surface form would produce a table much like the one above.
>
> Second, this is contested in the literature, and the objection is sharper than "the effect is small." Wang, Hayou and Nalisnick's [*The Myth of Expert Specialization in MoEs*](https://arxiv.org/abs/2604.09780) (2026) points out that a router is a **linear map**, so two inputs get similar expert usage exactly when their hidden states are similar. Specialization on that account is a property of the representation space the model has learned, not of the routing architecture, and a table like the one above measures the former while appearing to describe the latter. Their paper names OLMoE's own specialization figure as an example of the genre it is questioning.
>
> They also report something that cuts directly against the depth trend above: in the models they study, *deeper* layers show near-identical expert activation across semantically unrelated inputs. This post measures the opposite direction on a different model, and both can be true of their respective checkpoints. What neither establishes is *why* a token goes where it goes.
{: .prompt-warning }

### 5. Two bills: memory and time {#two-bills}

"1B active, 7B total" describes two costs that behave completely differently, and confusing them is the most common practical mistake with these models.

**Memory is billed on total parameters.** Every expert has to be resident, because the router decides at run time and any token might want any of them. There is no subset you could have left on disk. That is **12.89 GiB** in bf16 for a model whose name starts with "1B".

**Time is billed on active parameters**, and the cleanest way to show it is to change nothing but $k$. (The output below says **FLOPs**, floating-point operations: a count of the individual multiplies and adds the model performs, independent of how fast any particular chip gets through them.) Same weights, same memory, same everything — route to all 64 experts instead of 8 and measure (`two_bills`):

```text
  experts per token  forward (ms)  vs top-8
  -------------------------------------------
  8                         407.4     1.00x
  64                        920.1     2.26x

  expert FLOPs ratio (64/8)          8x
  measured wall-clock ratio          2.26x
    below 8x because attention, norms and the LM head are unchanged
```

Eight times the expert arithmetic costs 2.26× the wall clock, and zero extra bytes of weights.

Both halves of that deserve a note. The **2.26× rather than 8×** is because only the expert multiplies grew; attention, the norms and the LM head are unchanged, and on a 94-token forward pass those are a large share of the total. The gap between 8× and 2.26× is a useful reminder that "active parameters" predicts the *trend* of speed, not a clean multiplier.

And the top-64 row is a measurement of **cost only**. Because `norm_topk_prob` is false, routing to all 64 experts changes what the model computes; it is a timing experiment, not a quality one.

> This is the practical trap. A 7B-total, 1B-active model is *not* a drop-in replacement for a 1B dense model — it needs seven times the memory. It is also not equivalent to a 7B dense model, since it does a fraction of the arithmetic. It buys the quality that comes with more parameters at close to the speed that comes with fewer, and it pays for that in RAM.
{: .prompt-tip }

### 6. Sparsity does not survive a batch {#sparsity-and-batching}

Here is the result that reframes everything above, and the one I found least obvious.

Every claim so far has been about *one token*. One token uses 8 of 64 experts. But nothing is served one token at a time — you process a prompt of hundreds of tokens at once, and you batch requests from many users together. So the question that decides real cost is: how many *distinct* experts does a group of tokens need between them?

Each token picks its own 8. If two tokens pick differently, the hardware has to touch the union of their choices. Counting that union over every window of a given size in a 356-token passage (`batch_collapse`):

```text
  tokens together  experts needed  of 64  expert-slots used
  -----------------------------------------------------------
  1                           8.0    12%                  8
  2                          13.1    20%                 16
  4                          21.1    33%                 32
  8                          31.4    49%                 64
  16                         40.8    64%                128
  32                         48.5    76%                256
  64                         53.9    84%                512
  128                        58.0    91%               1024
  256                        60.9    95%               2048
```

![Distinct experts required against tokens processed together: 8 for one token, 61 for 256](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/batch-collapse-light.png){: .light width="1000" height="550" }
![Distinct experts required against tokens processed together: 8 for one token, 61 for 256](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/batch-collapse-dark.png){: .dark width="1000" height="550" }

Eight tokens already need half the experts. Two hundred and fifty-six need **60.9 of 64**, which is 95%.

So the sparsity that makes an MoE cheap is a property of a *token*, and it evaporates the moment you process tokens together. At any realistic batch size, essentially every expert is needed by somebody, which is exactly why all of them must be resident. The dotted line on the chart is the worst case where no two tokens ever agree; the measured curve sits below it, which is [§4](#what-the-router-learns)'s specialization showing up again as tokens genuinely sharing experts. It is not far enough below to change the conclusion.

This resolves the apparent paradox in the name. **An MoE saves arithmetic, not memory.** Per token, 17% of the parameters multiply. Per batch, essentially 100% of them have to be in RAM and get read. Post 2's lesson is worth recalling here: generating a token is limited by memory bandwidth rather than arithmetic, which is precisely the resource an MoE does *not* economize on.

### 7. Nothing balances the load for free {#load-balance}

One more consequence of letting a learned component do the choosing: nothing makes the experts equally popular.

There is a degenerate outcome sitting in this architecture. If the router slightly prefers a few experts early in training, those experts get more gradient, improve faster, get preferred more strongly, and the rest starve. You would end up paying for 64 experts and training perhaps 10.

Measuring the spread over 2848 routing slots per layer (`load_balance`). The last column is the **coefficient of variation**, the standard deviation of expert usage divided by its mean: a scale-free measure of unevenness where 0 is perfectly even and around 1.0 means the spread between experts is about as large as the average usage itself.

```text
  layer  busiest  quietest  busiest vs even  coeff of var
  ---------------------------------------------------------
  0        8.92%     0.04%            5.71x          0.88
  8        7.30%     0.00%            4.67x          0.89
  15       8.32%     0.00%            5.33x          1.07

  even share would be                1.56%
  router_aux_loss_coef               0.0100
  experts never used (last layer)    4
```

An even share would be 1.56%. The busiest expert in layer 0 takes 8.92%, which is 5.71× that, while the quietest takes 0.04% and four experts in the last layer are never used by this passage at all. (That last number is passage-dependent: a broader sample would use more of them. It is a statement about this text, not a claim that four experts are dead weight.)

This is why MoE training carries an **auxiliary loss** — an extra penalty term added to the training objective, on top of the usual next-token prediction loss, that grows when routing is lopsided. OLMoE's `router_aux_loss_coef` of 0.01 is how strongly that penalty counts. Note what the numbers above mean: even *with* that penalty applied throughout training, the busiest expert still runs about five times as often as an even share. The auxiliary loss keeps the distribution from collapsing; it does not make it flat, and it is not trying to.

### 8. What follows from all this {#what-follows}

Pulling the six measurements into the shape they make together:

**A mixture-of-experts model is a memory-for-arithmetic trade, and it runs in the opposite direction to post 4's.** Quantization keeps every weight and makes each one smaller. MoE keeps every weight at full size and declines to multiply by most of them. One shrinks the bytes, the other shrinks the FLOPs, and they are fully compatible — which is why nearly every large open model shipping today is both quantized and sparse.

**The scaling argument underneath it is simple.** Adding experts adds capacity at almost no cost in per-token compute, since $k$ stays fixed while $E$ grows. What it does cost is memory, linearly and unforgivingly. That is the trade the whole family of models is built on.

**And the number to hold onto is [§6](#sparsity-and-batching)'s.** Per-token sparsity is real and it is what makes these models fast. It is not batch sparsity, it never was, and any capacity plan that assumes otherwise will be wrong by roughly the ratio of total to active parameters.

### 9. Sidebar: the probe {#sidebar-the-probe}

One question to close on — the kind this material gets asked, and what separates an answer that sounds right from one that is.

> **"You're serving a 7B-total, 1B-active MoE. How much GPU memory do you need, and how fast will it be compared to a 1B dense model?"**

**The tempting answer:** *"It's 1B active, so roughly 1B-dense speed and something like 1B-dense memory. That's the point of MoE."*

Half of that is right, which is what makes it dangerous.

**A better answer comes in three moves.**

**1. Separate the two bills immediately.** Memory is billed on **total** parameters and compute on **active** ones. You need all 6.9B resident — **12.9 GiB** in bf16 — because the router chooses at run time and any token can want any expert. Speed is the part that tracks the 1B figure.

**2. Then refuse the "1B-dense speed" claim as stated.** Active parameters predict the trend, not a clean multiplier. Measured on the same weights with only $k$ changed, 8× the expert arithmetic produced **2.26×** the wall clock, because attention and the LM head do not scale with $k$. Which side of that you land on depends on sequence length and batch size.

**3. And say why the memory does not improve with batching.** One token needs 8 of 64 experts; 256 tokens together need **60.9**. Sparsity is per token, so at any serving batch size essentially every expert is live. If the interviewer's real question is "can I fit this on a smaller card," the answer is no, and the reason is that the union of what a batch needs is nearly everything.

What the question is really testing is whether "1B active, 7B total" is being read as two numbers describing two different resources, or as one number with a marketing adjective attached.

### What's next {#whats-next}

Post 6 is **LoRA**, and it turns the attention of this series from inference to training for the first time. Everything in posts 2 through 5 changed how a finished model runs. LoRA changes how one gets adapted: instead of updating all 6.9B parameters to teach a model a new task, it freezes them and trains a pair of much smaller matrices alongside. The questions there are what rank actually buys you, why the update can be low-rank at all when the weights it modifies are not, and what it costs to serve fifty fine-tunes of the same base model at once — which, after [§6](#sparsity-and-batching), should sound like a familiar kind of question.

### Appendix: all notation {#appendix-all-notation}

Every symbol this post uses, in one place. [Post 1's appendix](/posts/llm-architectures-attention-and-rope/#appendix-all-notation) covers attention's own notation, [post 2's](/posts/llm-architectures-kv-cache/#appendix-all-notation) the memory and serving terms, and [post 4's](/posts/llm-architectures-quantization/#appendix-all-notation) the quantization formats.

| Symbol | Means | In this post's runs |
| --- | --- | --- |
| $E$ | **experts** — how many copies of the FFN a layer holds | 64 |
| $k$ | how many experts each token is routed to, the "top-k" | 8 |
| $d_{model}$ | the model's width, the length of one token's vector | 2048 |
| $W_r$ | the **router** matrix, one per layer | $(64, 2048)$ |
| $z$ | **router logits** — one raw score per expert, any real number | $(64,)$ |
| $p$ | the logits after softmax; over all $E$ they sum to 1 | $(64,)$ |
| $\mathcal{K}$ | the set of experts that survive the top-k cut | 8 of 64 |
| $\text{FFN}_e$ | expert $e$ — an ordinary SwiGLU feed-forward network | 6.29M parameters |
| total parameters | every weight in the model, all of which must be resident | 6.919B |
| active parameters | the weights that multiply for one token | 1.177B (17.0%) |
| `norm_topk_prob` | whether the kept weights are rescaled to sum to 1 | false in OLMoE |
| TV distance | **total variation** — half the summed absolute difference between two distributions; 0 is identical, 1 is disjoint | floor 0.216, cross-domain 0.554 |
| coeff of var | standard deviation over mean, a scale-free measure of unevenness | 0.88 to 1.07 |
| auxiliary loss | an extra training penalty that grows when routing is lopsided | coefficient 0.01 |

Three things worth keeping straight:

- **"Active" and "total" are not two estimates of one quantity.** They are exact counts of two different things, and they bill to two different budgets. Total sets your memory; active sets your arithmetic.
- **Sparse does not mean small.** A sparse model is one that declines to use most of itself per token. It is still all there, and it is still all in RAM.
- **The router is not a classifier over topics.** It is a learned scoring matrix whose behaviour correlates with input distribution ([§4](#what-the-router-learns)). Reading it as "expert 17 handles code" is a story laid over a measurement, and one the literature actively disputes.

### References

- Muennighoff et al., [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060) (2024) — the model measured throughout this post, and the source of its router-saturation, expert-co-activation and domain-specialization analyses. Backs §1, §4 and §7.
- Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (2017) — where top-k gating and the load-balancing auxiliary loss of §7 were introduced.
- Fedus et al., [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) (2021) — top-1 routing, expert capacity, and the batching and token-dropping concerns that §6 measures the root of.
- Lepikhin et al., [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) (2020) — expert parallelism, and why the union-of-experts problem in §6 becomes a communication cost at scale.
- Wang, Hayou and Nalisnick, [The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily Domain Expertise](https://arxiv.org/abs/2604.09780) (2026) — the dissenting reading of §4. Argues that because routers are linear maps, hidden-state similarity is necessary and sufficient to explain expert-usage similarity, so specialization is emergent from the representation space. Their models are Qwen- and DeepSeek-family rather than OLMoE. Their §on load balancing also bears on §7: they prove the balancing loss suppresses shared hidden-state directions.
