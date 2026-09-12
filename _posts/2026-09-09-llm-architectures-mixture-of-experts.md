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

[Post 1](/posts/llm-architectures-attention-and-rope/) took a transformer block apart and put a cost on every piece. One finding from it matters more here than anything else: the **feed-forward network** holds more of a model's parameters than attention does. (The feed-forward network, or FFN, is the per-token stack of two or three matrix multiplies that sits after attention in every block. Attention is where tokens look at each other; the FFN is where each token is transformed on its own. You will also see it called the **MLP**, which is the same thing under a different name; OLMoE's own code calls it `mlp`.)

The three posts since then all tried to make a model cheaper without changing what it computes, or while admitting exactly what they changed.

[Post 2](/posts/llm-architectures-kv-cache/) removed work that was being *repeated*, caching keys and values instead of recomputing them at every generation step: a **284×** cut in repeated compute, paid for with **10.7× more memory held**. [Post 3](/posts/llm-architectures-flash-attention/) removed work that was being *written down and fetched back*, computing attention one tile at a time to move **3.9× fewer bytes**, landing on the same answer to about **1e-6** — the size of disagreement you get from summing in a different order. [Post 4](/posts/llm-architectures-quantization/) stopped being exact on purpose, storing each weight in fewer bits for a **2.09×** shrink, and spent most of its length on the fact that the usual quality check cannot see what that breaks.

Every one of those works on the model you already have. **Mixture-of-experts is a different kind of move: it changes what gets built.**

Think of it as a firm with a hundred specialists on staff. A question about a contract does not go to all hundred people; a partner reads it, decides it is a tax matter, and hands it to the three who do tax. Only those three bill any hours against that question. The other ninety-seven stay on the payroll, and the firm goes on renting desks for all hundred.

So the firm pays two different bills, and only one of them got smaller. **The hours billed depend on how many specialists you consulted. The rent does not.** That is the trade a mixture-of-experts model makes, and the two bills have names: the hours are **compute**, and the desks are **memory**. Running a token multiplies through a handful of experts, while every expert has to be sitting in memory anyway, in case the next token is the one that wants it. [§5](#two-bills) puts measured numbers on both.

The analogy is worth holding loosely, though, and [§4](#what-the-router-learns) is where it starts to strain. Real experts do not divide themselves into tidy human categories like "tax" and "litigation", and calling them experts was a naming decision, not a finding.

The idea, in one sentence: instead of one FFN per block that every token goes through, build many narrower ones and send each token to a few of them. The model measured here builds sixty-four and uses eight, though those two numbers are its own choice rather than part of the definition — [§1](#why-sixty-four) is about where they come from. Nothing is compressed and nothing is approximated. The model simply has far more capacity than it uses on any given token, and a small learned component decides which slice to use.

The arithmetic is genuinely strange the first time you meet it. The model in this post has **6.9 billion** parameters. Running it on a token multiplies through about **1.2 billion** of them. The other 5.7 billion sit in memory, doing nothing, waiting for a token that wants them.

If you would rather see the shape of the thing before reading about it, the whole architecture is on one page here. Every number on it is measured later in the post. The grid of numbered boxes in the middle panel is the layer's **64 experts, one box each** — not a matrix; each box is an entire feed-forward network. The eight filled boxes are the ones a single real token was routed to.

![Mixture-of-experts end to end: the whole model, one MoE layer in detail, what is inside one expert, the key numbers, and what happens to a single token](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/moe-architecture-light.png){: .light width="1100" height="1239" }
![Mixture-of-experts end to end: the whole model, one MoE layer in detail, what is inside one expert, the key numbers, and what happens to a single token](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/moe-architecture-dark.png){: .dark width="1100" height="1239" }

### Setup

Everything below is measured, and you can re-run all of it:

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo05
```

The code is in [`demos/d05_moe.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d05_moe.py). Each receipt below names the function that prints it.

The model is **[OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924)**, and the choice was not casual. Like [post 4](/posts/llm-architectures-quantization/), this post needs a *trained* checkpoint, because a router with random weights routes nothing in particular — the specialization in §4 is a product of training and does not exist before it. OLMoE is the smallest fully open MoE that fits on a 24 GB laptop, and its layers are the plain form of the mechanism: 64 experts, top-8, and neither of the two common variations. It has no **shared expert** (an extra FFN that every token goes through in addition to its chosen few), and it was not **upcycled** (built by copying a finished dense model's FFN into many experts and continuing training). It was trained sparse from scratch, which turns out to matter for [§4](#what-the-router-learns), for reasons [§9](#how-its-trained) gives. Most open MoE models are tens of billions of parameters in total even when only two or three billion are active, and it is the total that has to fit in memory.

One break from post 4: this demo runs in **bfloat16**, not fp32. 6.92B parameters is 25.8 GiB in fp32 and does not fit in 24 GB; in bf16 it is 12.9 GiB and does. Every claim here is a count or a ratio, both of which survive that change.

### Table of Contents

Skip to [the short version](#the-short-version) for the findings without the derivations.

1. [Where the parameters actually are](#where-the-parameters-actually-are)
2. [The router, which is smaller than you would guess](#the-router)
3. [One token, routed](#one-token-routed)
4. [What the router actually learns](#what-the-router-learns)
5. [Two bills: memory and time](#two-bills)
6. [Sparsity does not survive a batch](#sparsity-and-batching)
7. [When the model spans many GPUs](#across-gpus)
8. [Nothing balances the load for free](#load-balance)
9. [How an MoE is trained, and where "mid-training" fits](#how-its-trained)
10. [What follows from all this](#what-follows)
11. [Sidebar: the probe](#sidebar-the-probe)

Plus an [appendix of all notation](#appendix-all-notation) at the end, if a symbol ever goes by without introduction.

### The short version {#the-short-version}

A **mixture-of-experts** layer replaces one FFN with many smaller ones, called **experts**, plus a small **router** that picks a few experts per token. "Sparse" here means only the picked ones run. That is the entire idea; everything else is consequence.

- **The experts are essentially the model.** 93.1% of OLMoE's parameters are experts, against 3.9% for attention. A token activates 17.0% of the total ([§1](#where-the-parameters-actually-are)).
- **The router costs almost nothing.** One matrix per layer, 64 rows each 2,048 wide, 2.10M parameters, **0.030%** of the model, deciding how the other 93% get spent ([§2](#the-router)).
- **Routing is a softmax, a cut, and a weighted sum** (softmax turns raw scores into probabilities that add to 1), and OLMoE does not renormalize after the cut. The eight kept weights on the token walked through in [§3](#one-token-routed) sum to **0.4281**, not 1, so the router's confidence becomes a scale on the layer's output.
- **The router does specialize, and it is measurable rather than folklore.** Two halves of the *same* passage route differently by 0.216; the three pairs of different text average 0.554, **2.56×** that noise floor, and the gap widens with depth ([§4](#what-the-router-learns)).
- **Active parameters predict time; total parameters predict memory.** Forcing all 64 experts on costs **2.16×** the elapsed time and exactly zero extra bytes of weights ([§5](#two-bills)).
- **Per-token sparsity is not batch sparsity.** One token needs 8 experts of 64. Two hundred and fifty-six tokens together need **60.9** ([§6](#sparsity-and-batching)). That is why an MoE saves arithmetic without saving memory.
- **Splitting the experts across GPUs makes the router a network problem.** With 64 experts on 8 GPUs, one token's eight experts land on **5.54** different GPUs on average ([§7](#across-gpus)).
- **Nothing keeps the experts equally busy on its own.** The busiest expert in layer 0 takes **5.71×** an even share, and four experts in the last layer go unused by the test passage ([§8](#load-balance)).
- **MoE is an architecture, not a training stage.** The router and the experts learn together during pre-training, and the same model then carries on through mid-training and post-training unchanged ([§9](#how-its-trained)).

---

### 1. Where the parameters actually are {#where-the-parameters-actually-are}

Start with the question [post 4](/posts/llm-architectures-quantization/) opened on, because for an MoE it has a much more interesting answer: which parts of the model does this technique even touch?

For quantization the answer was "nearly all of it, a bit at a time." For mixture-of-experts the answer is that it touches exactly one component, the FFN, and replaces it with sixty-four smaller ones.

![A dense block against an MoE block: one MLP becomes 64 experts with a router in front, and 8 are lit for this token](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/moe-block-light.png){: .light width="1000" height="539" }
![A dense block against an MoE block: one MLP becomes 64 experts with a router in front, and 8 are lit for this token](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/moe-block-dark.png){: .dark width="1000" height="539" }

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

Those numbers are checkable from the model's published configuration without downloading a single weight, which is a good habit for any "model X has shape Y" claim. OLMoE has 16 layers, a model width of 2048, an expert width of 1024, and 64 experts per layer. Each expert is a [SwiGLU](/posts/llm-architectures-attention-and-rope/) FFN — the three-matrix form of feed-forward network post 1 pulled apart, where two matrices project the token into the expert's own width (1,024 here, which is *half* the model width, for reasons [below](#why-sixty-four)) and a third brings it back to 2,048 — so it holds $3 \times 2048 \times 1024 = 6{,}291{,}456$ parameters, which is the **6.29M** printed above. Multiply by 64 experts and 16 layers and you get 6.442B, and adding attention, embeddings and norms lands on 6.919B against the 6.92B the checkpoint reports.

The **active** count is the same arithmetic with 8 experts in place of 64: 8 experts × 6.29M × 16 layers is 0.805B, plus attention, the norms and the LM head, giving 1.177B. That ratio, **17.0%**, is the number the design exists to produce.

> **"1B active, 7B total"** is what the model's name means, and it is two different measurements of the same object. Neither is a lie and neither is sufficient on its own. Which one you should care about depends entirely on whether you are buying memory or buying time, which is [§5](#two-bills).
{: .prompt-info }

#### Why sixty-four? {#why-sixty-four}

Nothing so far explains where 64 and 8 came from, and they are worth pulling apart, because they are neither arbitrary nor universal. They are OLMoE's choices. Mixtral used 8 experts and picked 2; Qwen3-30B-A3B uses 128 and picks 8; DeepSeek-V3 uses 256 routed experts and picks 8. The mechanism is the same in all of them; the two numbers are a design decision made per model.

The useful way to read them is that they are not two independent knobs. What a token costs is $k$ multiplied by the width of one expert, and that product is the real budget (`why_this_many_experts`):

```text
  model width (hidden_size)          2048
  one expert's width                 1024
    as a multiple of the model width 0.50x
  width actually used per token (8 x 1024) 8192
    as a multiple of the model width 4.00x
  width held in total (64 x 1024)    65536
    as a multiple of the model width 32.00x
  capacity over compute              8x
```

An ordinary dense FFN is conventionally about **4× the model width**, and the dense sibling from the same lab, [OLMo-2-1B](https://huggingface.co/allenai/OLMo-2-0425-1B), has exactly that: hidden 2,048, FFN width 8,192. OLMoE's eight chosen experts come to `8 × 1024 = 8192`, which is **the same number**. Per token it does precisely as much feed-forward arithmetic as the dense model of its shape. What it adds is the other 56 experts, which is why it holds **8×** the FFN capacity for the same per-token cost.

So $k$ is set by the compute you are willing to spend, and the expert count is set by the capacity you want. That leaves one genuine question: given a fixed budget on both, do you want a few wide experts or many narrow ones?

```text
  experts  each of width  used per token  possible combinations
  ---------------------------------------------------------------
  8                 8192               1                      8
  16                4096               2                    120
  32                2048               4                 35,960
  64                1024               8          4,426,165,368
```

Every row in that table stores the same number of parameters and runs the same arithmetic per token. What changes is how many distinct combinations of experts a token can be assigned to: **8** at the coarse end, **4.4 billion** at the fine end. A model with 8 experts picking 1 has eight possible behaviours at that layer. OLMoE has more than four billion.

That is the argument for **fine-grained experts**, and it is the reason each of OLMoE's experts is *half* the model width rather than four times it. It is not free — more, smaller experts mean more routing decisions, more scattered memory access, and a token with more experts to reach touches more GPUs, which is [§7](#across-gpus)'s problem. Where to sit on that curve is what differs between Mixtral, OLMoE, Qwen3 and DeepSeek-V3.

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

From `the_router_is_tiny`. Sixteen matrices, one per layer. Each has **64 rows, one per expert, and each row is 2,048 numbers wide** — the same width as a token's vector, because scoring an expert is a dot product against the token. That is 2.10M parameters in total, **4.0 MiB** in bf16 against **12.88 GiB** for everything else. Three hundredths of one percent of the model decides how the rest of it is spent, on every token, at every layer.

That asymmetry is where the risk in this architecture lives. A router that chooses badly does not merely lose a little accuracy; it wastes the capacity the other 93% of the parameters represent. [§8](#load-balance) is about what happens when it chooses lopsidedly, and why training has to actively push against that.

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

The last line is the part most explanations skip. The softmax runs over all 64 experts and its outputs sum to 1, but only 8 survive the cut, and **OLMoE does not rescale them afterwards**. The eight weights sum to 0.4281, so 0.5719 of the probability mass is discarded, and the layer's output is exactly 0.4281 times what it would be if the eight were rescaled to sum to 1.

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

One word visits sixteen separate committees of eight on its way through the model. There is no such thing as "the expert for `' harbour'`" — there are 16 independent choices, and the number of distinct paths a single token could take is astronomically large. Note also that `kept total` climbs with depth, from 0.25 in layer 1 to 0.49 in layer 15: the router is more decisive in the later layers.

The same thing in shapes, with the real numbers underneath:

![The router as a pipeline of tensor shapes, and all 64 routing probabilities for one real token with the 8 that survive the cut](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/router-flow-light.png){: .light width="1000" height="640" }
![The router as a pipeline of tensor shapes, and all 64 routing probabilities for one real token with the 8 that survive the cut](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/router-flow-dark.png){: .dark width="1000" height="640" }

The lower panel is the reason 0.4281 is not a surprise once you see it. Eight bars are tall, and the remaining fifty-six are individually small but collectively larger than the eight — which is what a softmax over sixty-four options looks like when the model has a mild preference rather than a strong one.

#### Not every router works this way {#router-variants}

OLMoE's router is the plain form, which is why it is the one to learn first. Nearly every part of it is a decision that other models make differently, and the differences are visible in their published configs rather than being a matter of interpretation. DeepSeek-V3 makes a different choice on all four:

| Decision | OLMoE | DeepSeek-V3 |
| --- | --- | --- |
| turning scores into weights | softmax over all 64 | sigmoid per expert: each score squashed to between 0 and 1 on its own, so they need not sum to 1 |
| after the top-k cut | left as-is (`norm_topk_prob: false`) | rescaled to sum to 1 (`true`) |
| keeping experts busy | an extra training penalty for lopsided routing, the **auxiliary loss** of [§8](#load-balance) | no auxiliary loss; a per-expert bias nudged during training |
| where a token may go | any of the 64 | at most 4 of 8 expert groups |

The third row is a direct alternative to what [§8](#load-balance) describes rather than a tweak of it. Instead of penalizing imbalance in the loss, DeepSeek-V3 keeps a bias per expert, adds it to the score *when choosing* but not when weighting, and nudges it up or down between steps depending on whether that expert was overloaded. The stated motivation in the [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) is that an auxiliary loss pulls against the language-modelling objective, and a bias that only affects selection does not.

The fourth row is [§7](#across-gpus) turned into a design constraint. If a token can only reach experts in 4 of 8 groups, the traffic between GPUs that §7 measures is bounded by construction rather than by luck.

Two more variations you will meet in the literature: **noisy routing**, which adds random noise to the scores during training so the router explores experts it would otherwise never try ([Shazeer et al.](https://arxiv.org/abs/1701.06538)), and **expert capacity**, a hard cap on how many tokens one expert may accept per batch, with the overflow either dropped or passed through unchanged ([Fedus et al.](https://arxiv.org/abs/2101.03961)). Capacity limits exist because of exactly what [§8](#load-balance) measures: if the load is uneven and your hardware allocated equal space per expert, something has to give.

### 4. What the router actually learns {#what-the-router-learns}

Now the question everyone asks, and the one where it is easiest to fool yourself.

The folk story is that experts specialize — one handles code, one handles French, one handles punctuation. It is an appealing story. Is it true?

The honest way to ask is to route three passages of clearly different character (English prose, Python, and a paragraph of group theory), then measure how differently they use the experts. Each token makes 8 expert picks per layer, and this post calls each pick a **routing slot**: a 94-token passage fills 752 slots at every layer, and a passage's *expert usage* is the share of those slots each expert received. The measure is **total variation distance**, which for two distributions over the same 64 experts is half the sum of the absolute differences between them. It runs from 0, meaning the two used the experts identically, to 1, meaning they shared no expert at all.

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

Expert choice clearly depends on the kind of text. Two halves of one passage differ by 0.216; two different kinds of text differ by 0.554, **2.56×** as much.

The more interesting structure is in the columns. Averaging the three cross-domain rows at each depth and setting them against the floor:

```text
  noise floor, first layer           0.240
  cross-domain, first layer          0.400
  noise floor, last layer            0.173
  cross-domain, last layer           0.652
```

At layer 0 the cross-domain distance is 0.400 against a noise floor of 0.240, a ratio of only 1.7, so early routing is barely about content at all. By layer 15 it is 0.652 against a floor of 0.173, a ratio of 3.8. **The noise floor falls with depth while the cross-domain distance rises**, which is two separate signs of the same thing: deep routing is consistent within a kind of text and sharply different between kinds. Early layers route on something much closer to surface form.

![The same 64 experts used differently by different text, with two halves of one passage on top as the noise floor](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/specialization-light.png){: .light width="1000" height="382" }
![The same 64 experts used differently by different text, with two halves of one passage on top as the noise floor](/assets/picture/2026-09-09-llm-architectures-mixture-of-experts/specialization-dark.png){: .dark width="1000" height="382" }

The top two rows are the same prose passage split in half; the bottom two are code and mathematics. The comparison to make by eye is row 1 against row 2 (noise) versus row 1 against row 3 (signal).

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

"1B active, 7B total" describes two costs that behave completely differently, and confusing them is an easy practical mistake to make with these models.

**Memory is billed on total parameters.** Every expert has to be resident, because the router decides at run time and any token might want any of them. There is no subset you could have left on disk. That is **12.89 GiB** in bf16 for a model whose name starts with "1B".

**Time is billed on active parameters**, and the cleanest way to show it is to change nothing but $k$. (The output below says **FLOPs**, floating-point operations: a count of the individual multiplies and adds the model performs, independent of how fast any particular chip gets through them.) Same weights, same memory, same everything — route to all 64 experts instead of 8 and measure (`two_bills`):

```text
  experts per token  forward (ms)  vs top-8
  -------------------------------------------
  8                         430.8     1.00x
  64                        931.6     2.16x

  expert FLOPs ratio (64/8)          8x
  measured wall-clock ratio          2.16x
    below 8x because attention, norms and the LM head are unchanged
```

Eight times the expert arithmetic costs 2.16× the wall clock, and zero extra bytes of weights.

Both halves of that deserve a note. The **2.16× rather than 8×** is because only the expert multiplies grew; attention, the norms and the LM head are unchanged, and on a 94-token forward pass those are a large share of the total. The gap between 8× and 2.16× is a useful reminder that "active parameters" predicts the *trend* of speed, not a clean multiplier.

Unlike every other number in this post, this one is a wall-clock measurement and it moves a little from run to run; the counts and ratios elsewhere do not. And the top-64 row is a measurement of **cost only**. Because `norm_topk_prob` is false, routing to all 64 experts changes what the model computes; it is a timing experiment, not a quality one.

> This is the practical trap. A 7B-total, 1B-active model is *not* a drop-in replacement for a 1B dense model — it needs about 5.9 times the memory (6.92B parameters resident against 1.18B active). It is also not equivalent to a 7B dense model, since it does a fraction of the arithmetic. It buys the quality that comes with more parameters at close to the speed that comes with fewer, and it pays for that in RAM.
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

So the sparsity that makes an MoE cheap is a property of a *token*, and it evaporates the moment you process tokens together. At any realistic batch size, essentially every expert is needed by somebody, which is exactly why all of them must be resident. The dotted line on the chart is the worst case, where no two tokens ever share an expert; it reaches all 64 by eight tokens. The measured curve sits below it because tokens do share experts, which is [§4](#what-the-router-learns)'s specialization showing up again, but it does not sit far enough below to change the conclusion.

This resolves the apparent paradox in the name. **An MoE saves arithmetic, not memory.** Per token, 17% of the parameters multiply. Per batch, essentially 100% of them have to be in RAM and get read. That connects to [post 2](/posts/llm-architectures-kv-cache/)'s finding that generating a token is limited by memory bandwidth rather than arithmetic. For a single token an MoE reads only its eight experts, which is why it generates quickly; for a batch it reads nearly all of them, so at serving batch sizes the bandwidth saving mostly disappears.

### 7. When the model spans many GPUs {#across-gpus}

Everything so far assumed the model fits on one machine. OLMoE does, at 12.89 GiB. The models this architecture actually exists for do not: DeepSeek-V3 is 671B parameters, over a terabyte of weights in bf16, so it has to be split across many GPUs.

There are two ways to split a transformer, and MoE makes the second one natural.

**Tensor parallelism** cuts every matrix into pieces and gives each GPU a slice, so all GPUs work on every token. **Expert parallelism** cuts along the experts instead: GPU 0 gets experts 0–7, GPU 1 gets experts 8–15, and so on. Nobody has to slice a matrix, because the experts were already separate objects.

The catch follows directly from [§6](#sparsity-and-batching). Routing is per token, and a token's eight experts are wherever the router says they are. If those eight live on five different GPUs, that token has to be sent to five GPUs and its results gathered back — for every layer, for every token. That exchange is the **all-to-all**, and it is one of the central engineering problems in serving MoE models.

How much traffic that is depends entirely on how scattered the routing is, which is measurable from the routing indices alone (`experts_across_gpus`):

```text
  GPUs  experts each  GPUs per token  of all  tokens needing all
  ----------------------------------------------------------------
  2               32            1.99    100%                 99%
  4               16            3.67     92%                 68%
  8                8            5.54     69%                  0%
  16               4            6.84     43%                  0%
```

Split across four GPUs, the average token needs **3.67 of them**, and **68%** of tokens need all four. At two GPUs, 99% of tokens need both — the router almost never keeps a token's work on one side.

The absolute number climbs while the share falls, and both halves matter. Going from 8 GPUs to 16 raises the GPUs a token must reach from 5.54 to 6.84, so each token's work is spread thinner and communicated wider. It cannot exceed 8, because a token only picks 8 experts, which is why the "of all" column drops. What you are watching is [§6](#sparsity-and-batching)'s finding in a different costume: a token's choices are scattered, so nothing about them stays local.

> **What this section does and does not measure.** The routing is real, measured on the actual model. The GPU assignment is arithmetic on top of it — experts dealt out in contiguous blocks — not a benchmark on a multi-GPU host, which is not something a laptop can honestly produce. What it gives you is the quantity that *determines* the communication cost, rather than the cost itself, which depends on the serving stack and on **interconnect bandwidth** — how fast the links between GPUs can carry data, as distinct from how fast the GPUs compute.
{: .prompt-info }

For scale, here is how OLMoE sits next to two models built the same way. These are published configuration values rather than measurements of mine:

| | OLMoE-1B-7B | Qwen3-30B-A3B | DeepSeek-V3 |
| --- | --- | --- | --- |
| total parameters | 6.9B | 30.5B | 671B |
| active per token | 1.2B | ~3B | 37B |
| layers | 16 | 48 | 61 |
| routed experts per layer | 64 | 128 | 256 |
| experts per token | 8 | 8 | 8 |
| shared expert | none | none | 1, always on |
| fits on one 24 GB card | yes, in bf16 | no | not remotely |

Two details in that last column are worth noticing, because simplified diagrams of DeepSeek-V3 routinely get them wrong. It has a **shared expert** that every token passes through in addition to its top-8, so "8 of 256" understates what runs. And its first three layers are ordinary dense FFNs, not MoE layers, so it has 58 MoE layers rather than 61. Its experts are also much narrower than its dense layers — an intermediate width of 2,048 against the 18,432 of the dense FFN — which is what lets it hold 256 of them per layer.

### 8. Nothing balances the load for free {#load-balance}

One more consequence of letting a learned component do the choosing: nothing makes the experts equally popular.

There is a degenerate outcome sitting in this architecture. If the router slightly prefers a few experts early in training, those experts get more of the **gradient** — the signal that says which way to nudge each weight to reduce the error — so they improve faster, get preferred more strongly, and the rest starve. You would end up paying for 64 experts and effectively training a handful.

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

An even share would be 1.56%. The busiest expert in layer 0 takes 8.92%, which is 5.71× that, while the quietest takes 0.04% and four experts in the last layer are never used by this passage at all. (That count is passage-dependent: a broader sample would use more of them. It is a statement about this text, not a claim that four experts are dead weight.)

This is why MoE training carries an **auxiliary loss**: an extra penalty added to the training objective that grows when routing is lopsided. The objective being optimized is a sum of two terms.

| Symbol | Means |
| --- | --- |
| $\mathcal{L}_{\text{LM}}$ | the ordinary next-token prediction loss — how surprised the model was by the real next token |
| $\mathcal{L}_{\text{balance}}$ | the load-balancing penalty, large when a few experts take most of the traffic and small when the load is spread |
| $\alpha$ | how much the penalty counts against the prediction loss |

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha \cdot \mathcal{L}_{\text{balance}}$$

OLMoE sets $\alpha$ to **0.01**, which is the `router_aux_loss_coef` printed above. That number is small on purpose: the balancing term is there to stop a collapse, not to drive the model, and setting it too high would trade away prediction quality to make a histogram look tidy. And even *with* that penalty applied throughout training, the busiest expert still runs about five times as often as an even share. The auxiliary loss keeps the distribution from collapsing; it does not make it flat, and it is not trying to.

### 9. How an MoE is trained, and where "mid-training" fits {#how-its-trained}

A question worth separating out, because two different things get talked about in the same breath: mixture-of-experts is an **architecture**, while pre-training, mid-training and post-training are **stages in a model's life**. They are different dimensions, not alternatives. You can ask "is this model MoE?" and "what stage is it in?" independently.

**The experts and the router learn together, during pre-training.** They are not bolted on afterwards. At the start of training every expert is random and the router is random, so the first routing decisions are meaningless. Then ordinary next-token prediction runs: the model reads text, predicts what comes next, and is scored on how wrong it was. **Backpropagation** then works that error backwards through the network to get a gradient for every weight involved, and the weights move. The router, the chosen experts and the rest of the network all update. The skipped experts do not, because they contributed nothing to the output.

There is a subtlety here that [§3](#one-token-routed) set up. Picking the top 8 is a discrete choice, and discrete choices have no useful gradient — you cannot differentiate "expert 5 was ranked higher than expert 6". So how does the router learn anything at all? Through the weights. Each chosen expert's output is scaled by its router probability $p_e$ before being added in, so $p_e$ sits in the middle of an ordinary differentiable path. If an expert's contribution improved the prediction, the gradient pushes its $p_e$ up, which means pushing that expert's router score up for tokens like this one. **The router learns because the routing weights multiply, not because the selection itself is differentiable.**

Specialization is emergent from that process. Nobody assigns expert 17 to code. It becomes whatever it becomes, because it happened to be picked slightly more often for certain inputs early on and improved at them, which made it get picked more. That is the same runaway dynamic [§8](#load-balance)'s auxiliary loss exists to keep from going too far.

Not every MoE is trained this way. **Sparse upcycling** takes a finished dense model, copies its FFN into many identical experts, and continues training — much cheaper than starting over, since it inherits everything the dense model learned. The tradeoff shows up in [§4](#what-the-router-learns)'s measurement: upcycled models are reported to specialize noticeably less ([Muennighoff et al.](https://arxiv.org/abs/2409.02060) compare OLMoE against the upcycled Mixtral), which makes sense if every expert starts as a copy of the same function. OLMoE was trained sparse from scratch, which is part of why its routing is worth measuring.

#### The three stages

**Pre-training** is the enormous run: trillions of tokens of web pages, books, code and papers, with the single objective of predicting the next token. This is where a model learns language, facts and patterns, and for an MoE it is where the router and experts sort themselves out.

**Post-training** is the other end, and it is about behaviour rather than knowledge. A raw pre-trained model is not an assistant; it is a very well-read text continuer. Post-training teaches it to follow instructions, hold a conversation, use tools, and decline things it should decline, through **supervised fine-tuning** (showing it worked examples of good answers and training it to reproduce them) and preference-based methods (showing it pairs of answers with a judgement of which is better). Posts 8 and 9 of this series are about two of those methods.

**Mid-training** is the term with the least agreement behind it, which is worth saying plainly rather than pretending it is a settled piece of vocabulary. Broadly it means continued training after the main pre-training run but before the behaviour-focused stage, using deliberately chosen data rather than a broad web scrape. You may also see it called continued pre-training, annealing, or a second curriculum stage.

The clearest concrete example comes from the same lab that built the model in this post. [OLMo 2](https://arxiv.org/abs/2501.00656) pre-trains in two stages: roughly 3.9 trillion tokens of general mixture, then a second stage of about 5–10% of the compute budget on a curated mix heavy in high-quality web text, academic content, instruction data and synthetic mathematics, aimed at capabilities the first stage left weak. (OLMo 2 is a dense model, not an MoE — it is cited here for the training stage, not the architecture.)

**Why not just put that data in the pre-training mix?** Because when data arrives matters, not only whether it arrived. Spread a small amount of excellent mathematics across four trillion tokens of general text and its influence is diluted by everything around it. Concentrated near the end, when the model already has general competence, the same tokens have more room to shape it, which is the reasoning labs give for scheduling it that way. The analogy that survives scrutiny is a curriculum: broad schooling first, then specialization, then learning how to do the job with other people.

For an MoE, all three stages run on the same architecture. The router is learned in pre-training and keeps routing through mid-training, post-training and inference. So when someone says "we trained an MoE with a long mid-training stage", those are two separate facts: how the network is built, and how the training run was scheduled.

### 10. What follows from all this {#what-follows}

Pulling the measurements into the shape they make together:

**A mixture-of-experts model is a memory-for-arithmetic trade, and it runs in the opposite direction to post 4's.** Quantization keeps every weight and makes each one smaller. MoE keeps every weight at full size and declines to multiply by most of them. One shrinks the bytes, the other shrinks the FLOPs, and they are fully compatible. [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) is an MoE that ships its experts in 4-bit, for instance.

**The scaling argument underneath it is simple.** Adding experts adds capacity at almost no cost in per-token compute, since $k$ stays fixed while $E$ grows. What it does cost is memory, linearly and unforgivingly.

**Serving one is a networking problem before it is a compute problem.** The experts have to be split across GPUs because they are most of the model, and routing then scatters each token's work across most of those GPUs ([§7](#across-gpus)). That is why interconnect bandwidth matters as much as FLOPs when serving these models.

**And the number to hold onto is [§6](#sparsity-and-batching)'s.** Per-token sparsity is real and it is what makes these models fast. It is not batch sparsity, it never was, and any capacity plan that assumes otherwise will be wrong by roughly the ratio of total to active parameters.

### 11. Sidebar: the probe {#sidebar-the-probe}

One question to close on — the kind this material gets asked, and what separates an answer that sounds right from one that is.

> **"You're serving a 7B-total, 1B-active MoE. How much GPU memory do you need, and how fast will it be compared to a 1B dense model?"**

**The tempting answer:** *"It's 1B active, so roughly 1B-dense speed and something like 1B-dense memory. That's the point of MoE."*

Half of that is right, which is what makes it dangerous.

**A better answer comes in four moves.**

**1. Separate the two bills immediately.** Memory is billed on **total** parameters and compute on **active** ones. You need all 6.9B resident — **12.9 GiB** in bf16 — because the router chooses at run time and any token can want any expert. Speed is the part that tracks the 1B figure.

**2. Then refuse the "1B-dense speed" claim as stated.** Active parameters predict the trend, not a clean multiplier. Measured on the same weights with only $k$ changed, 8× the expert arithmetic produced **2.16×** the wall clock, because attention and the LM head do not scale with $k$. How close you get to 1B-dense speed depends on sequence length and batch size.

**3. And say why the memory does not improve with batching.** One token needs 8 of 64 experts; 256 tokens together need **60.9**. Sparsity is per token, so at any serving batch size essentially every expert is live. If the interviewer's real question is "can I fit this on a smaller card," the answer is no, and the reason is that the union of what a batch needs is nearly everything.

**4. And if they follow up with "so shard it across more GPUs".** That helps with capacity and hurts with communication. Splitting 64 experts over 8 GPUs leaves the average token needing **5.54** of them at every layer, so the all-to-all exchange grows as you spread out. Expert parallelism converts a memory problem into a bandwidth problem; it does not make the problem go away.

What the question is really testing is whether "1B active, 7B total" is being read as two numbers describing two different resources, or as one number with a marketing adjective attached.

### What's next {#whats-next}

Post 6 is **LoRA**, and it moves this series from inference to training for the first time. Posts 2 through 4 changed how a finished model runs, and this one changed what gets built. LoRA changes how a model gets adapted: instead of updating all 6.9B parameters to teach a model a new task, it freezes them and trains a pair of much smaller matrices alongside. The questions there are what rank actually buys you, why the update can be low-rank at all when the weights it modifies are not, and what it costs to serve fifty fine-tunes of the same base model at once — which, after [§6](#sparsity-and-batching), should sound like a familiar kind of question.

### Appendix: all notation {#appendix-all-notation}

Every symbol this post uses, in one place. [Post 1's appendix](/posts/llm-architectures-attention-and-rope/#appendix-all-notation) covers attention's own notation, [post 2's](/posts/llm-architectures-kv-cache/#appendix-all-notation) the memory and serving terms, and [post 4's](/posts/llm-architectures-quantization/#appendix-all-notation) the quantization formats.

| Symbol | Means | In this post's runs |
| --- | --- | --- |
| $E$ | **experts** — how many separate, smaller FFNs a layer holds | 64 |
| $k$ | how many experts each token is routed to, the "top-k" | 8 |
| $d_{model}$ | the model's width, the length of one token's vector | 2048 |
| $W_r$ | the **router** matrix, one per layer | 64 rows, each 2,048 wide |
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
| expert parallelism | splitting the *experts* across GPUs, rather than slicing every matrix | 64 experts over 2–16 GPUs |
| all-to-all | the exchange that sends each token to whichever GPUs hold its experts, and gathers the results back | 5.54 GPUs per token at 8 |
| shared expert | an extra expert every token uses, on top of its top-k | none in OLMoE; 1 in DeepSeek-V3 |
| noisy routing | random noise added to router scores during training, so it explores | not used in OLMoE |
| expert capacity | a cap on how many tokens one expert accepts per batch; the overflow is dropped or passed through | a consequence of §8's imbalance |
| sparse upcycling | building an MoE by copying a trained dense model's FFN into many experts | not how OLMoE was built |
| mid-training | a later, deliberately curated training stage between pre- and post-training | ~5–10% of OLMo 2's budget |

Three things worth keeping straight:

- **"Active" and "total" are not two estimates of one quantity.** They are exact counts of two different things, and they bill to two different budgets. Total sets your memory; active sets your arithmetic.
- **Sparse does not mean small.** A sparse model is one that declines to use most of itself per token. It is still all there, and it is still all in RAM.
- **The router is not a classifier over topics.** It is a learned scoring matrix whose behaviour correlates with input distribution ([§4](#what-the-router-learns)). Reading it as "expert 17 handles code" is a story laid over a measurement, and one the literature actively disputes.

### References

- Muennighoff et al., [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060) (2024) — the model measured throughout this post, and the source of its router-saturation, expert-co-activation and domain-specialization analyses. Backs §1, §4 and §8.
- Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (2017) — where top-k gating and the load-balancing auxiliary loss of §8 were introduced.
- Fedus et al., [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) (2021) — top-1 routing, expert capacity, and the batching and token-dropping concerns that §6 measures the root of.
- Lepikhin et al., [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) (2020) — expert parallelism, and why the union-of-experts problem in §6 becomes a communication cost at scale.
- OLMo Team, [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) (2025) — §9's concrete example of a mid-training stage: a curated second-stage mix over roughly 5–10% of the compute budget, after ~3.9T tokens of general pre-training. A dense model, cited for the training schedule rather than the architecture.
- Komatsuzaki et al., [Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/abs/2212.05055) (2022) — the alternative to training sparse from scratch, and the reason §4's result depends on which of the two produced the model.
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (2024) — the 671B/37B model in §7's comparison, its shared expert, and the serving architecture that §7's all-to-all describes.
- Wang, Hayou and Nalisnick, [The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily Domain Expertise](https://arxiv.org/abs/2604.09780) (2026) — the dissenting reading of §4. Argues that because routers are linear maps, hidden-state similarity is necessary and sufficient to explain expert-usage similarity, so specialization is emergent from the representation space. Their models are Qwen- and DeepSeek-family rather than OLMoE. Their result on load balancing also bears on §8: they prove the balancing loss suppresses shared hidden-state directions.
