---
title: "LLM Architecture Refresh [6]: LoRA, and What Rank Actually Buys"
date: 2026-09-26 09:00:00 -0700
categories: [LLM Architecture Refresh, Training]
tags: [lora, fine-tuning, low-rank, peft, adapters, qwen, pytorch]
description: >-
  Fine-tuning a real model twice, once fully and once through a rank-8 adapter,
  to find out what rank buys, why a full fine-tune's update turns out not to be
  low-rank at all, and what it costs to serve fifty adaptations of one model.
math: true
pin: true
published: false  # draft: publishing is a separate decision
---

## The first post that changes the weights

Five posts in, everything so far has been about a model that already exists.

[Post 1](/posts/llm-architectures-attention-and-rope/) took a transformer block apart and put a cost on every piece. [Post 2](/posts/llm-architectures-kv-cache/) stopped recomputing keys and values, removing a **284×** multiplier of repeated work, and paid for it in memory. [Post 3](/posts/llm-architectures-flash-attention/) computed attention one tile at a time and moved **3.9× fewer bytes** to get the same answer, agreeing with the ordinary kernel to about **1.4e-6**. [Post 4](/posts/llm-architectures-quantization/) stored each weight in fewer bits, shrinking a model **2.09×**, from 942 MiB to 452 MiB, and gave up exactness on purpose. [Post 5](/posts/llm-architectures-mixture-of-experts/) built a different model instead, one that holds 64 experts per layer and runs 8, so that **17.0%** of its parameters do the work for any one token.

Not one of them changed a single weight. They changed how the weights are stored, or in what order they're used, or which of them run. The weights themselves came out of training and stayed exactly as they were.

**This post changes them.** Fine-tuning takes a trained model and continues training it on data you care about, so that it ends up with different numbers in it than the ones it shipped with. That is the one operation the series hasn't touched, and it's where a surprising amount of practical LLM work actually happens.

The obvious way to do it is to keep training every parameter, which is called a **full fine-tune**. It works, and its bill is worse than it looks. The model isn't the expensive part.

### The thing that makes fine-tuning expensive

Training needs three numbers for every parameter, not one.

The **gradient** is the first: for each weight, a number saying which direction to move it and how strongly, computed from how wrong the model was on the current batch. It's the same size as the model.

The other two belong to the optimizer. Nobody trains a large model by stepping straight down the gradient, because a raw gradient is noisy: one batch says move left, the next says move right. **AdamW**, the optimizer almost everything uses, smooths that out by keeping a running average of the gradient (its "first moment") and a running average of the gradient *squared* (its "second moment"), and using the two together to decide each weight's step. That's two more numbers per parameter, and both are the same size as the model again.

So a full fine-tune holds four copies of the model before it stores a single activation from the forward pass. [§1](#where-a-fine-tunes-cost-is) measures it: **7.36 GiB** for a model whose weights are 1.84 GiB.

**LoRA attacks exactly that.** The name is short for *Low-Rank Adaptation*, from [Hu et al.](https://arxiv.org/abs/2106.09685) in 2021. It freezes every weight the model shipped with, so none of them need a gradient or a moment, and trains a small pair of new matrices beside each one. The trained pair is called an **adapter**. What you keep at the end is the adapter, and in this post's configuration that's **16.8 MiB** next to a 1.84 GiB model.

Think of the frozen model as a printed map, and the adapter as a transparent overlay laid over it. The map is what it is, and no one is reprinting it. The overlay adds the route, and it's cheap to store, cheap to send to someone, and easy to swap for a different one. When you're certain you want one particular route forever, you can print the overlay into the map, after which there's no overlay left to carry and looking something up costs exactly what it always did.

The catch, and the subject of most of this post, is the sense in which the overlay is thin. It isn't thin because the changes it makes are small. It's thin because of a constraint on what kind of marks it's allowed to draw at all, and the practical question is whether the change you need happens to be drawable under that constraint.

That constraint is **rank**, and it's the one word you need before the rest of this makes sense.

### What rank means, in one paragraph

A matrix is a grid of numbers, and its **rank** is how many genuinely independent directions of information it holds. A grid of 896 rows and 896 columns *could* hold 896 independent directions, and that's the most it could have. It might hold far fewer: if every row of a matrix is just some multiple of one single row, then however large the grid is, it carries one direction's worth of information, and its rank is 1. **Low rank means a big grid whose contents are simpler than its size suggests.**

The useful thing is that a low-rank grid can be *built* out of two thin ones. Multiply a grid of 896 rows and 8 columns by a grid of 8 rows and 896 columns and you get a full 896 × 896 grid back, but one that can never have rank above 8, because the 8 in the middle is a bottleneck everything has to squeeze through. That product is what LoRA trains. It's the same shape as the weight it modifies, and [§2](#the-mechanism) shows it's **56× smaller** to store.

So the adapter is not an approximation of the model. It's a *complete* change to the weights, drawn under a restriction on how complicated that change may be.

![One adapted matrix: the frozen weight on top, the two trainable matrices below it squeezing through a rank-8 bottleneck, the alpha/r scaling, and the sum of the two paths](/assets/picture/2026-09-26-llm-architectures-lora/lora-mechanism-light.png){: .light width="1000" height="521" }
![One adapted matrix: the frozen weight on top, the two trainable matrices below it squeezing through a rank-8 bottleneck, the alpha/r scaling, and the sum of the two paths](/assets/picture/2026-09-26-llm-architectures-lora/lora-mechanism-dark.png){: .dark width="1000" height="521" }

The rest of the post asks what that restriction costs. It turns out to depend almost entirely on what you're trying to teach the model, and [§5](#what-rank-buys) measures two tasks that answer the question in opposite directions.

### Setup

Everything below is measured, and you can re-run all of it:

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo06
```

The code is in [`demos/d06_lora.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py), and LoRA itself is in [`lora.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/lora.py), written from the paper rather than imported, because this series builds the thing it's explaining. Every number this post quotes is printed by that program, and each block of its output, which I call a **receipt**, names the function that printed it.

Two warnings about this one. It's the first demo that *trains*, so it takes about **25 minutes** where the others took a minute, and most of that is the rank sweep in §5.

And its numbers come from a training run rather than from arithmetic on a fixed checkpoint, so they don't reproduce the way the earlier posts' numbers do. I ran the demo twice and diffed the output to find out how far that goes. Everything counted or derived from shapes is identical, as are the losses and scores in §4 and §5, down to the last digit printed. What moved between the two runs were the wall-clock timings in §4 and §7, and the last digit of some of §6's spectrum percentages, which shifted by about a tenth of a point. Floating-point addition isn't associative, and a GPU is free to order a reduction differently from one run to the next. So read §6's tables as good to about a part in a thousand rather than exact, and expect a different machine to move them further.

The model is **Qwen2.5-0.5B**, the same one [post 4](/posts/llm-architectures-quantization/) quantized: [the weights](https://huggingface.co/Qwen/Qwen2.5-0.5B), [the report](https://arxiv.org/abs/2412.15115). Using post 4's model twice is deliberate. Post 4 measured what happens when you make each of its weights smaller; this post measures what happens when you change them, and the two results describe the same 494 million numbers.

Everything runs in **fp32**, four bytes per number. Post 5 ran in bf16 to fit a 7B model on a laptop, but this post compares training runs against each other, and mixed precision would add a second reason for two runs to differ. [Post 7](#whats-next) takes precision up properly, since that's exactly what QLoRA is about.

#### Two tasks, chosen to disagree {#two-tasks}

The post needs something to fine-tune *on*, and the choice turns out to be the most important decision in it, because §5's answer changes completely depending on which task you ask about. The demo generates both, deterministically, so there's no dataset to download:

**`format`** teaches the model to rewrite a written date as ISO 8601: `Date: April 22, 1944` becomes `1944-04-22`. Three input styles, 2,000 training examples. This is a **rule**. Learning it means learning to map twelve month names to numbers, zero-pad, and reorder, and nothing about the answer depends on which particular date it is. It's scored on 40 held-out dates, and the receipt in §5 confirms all 40 are absent from training, so the score measures whether the rule generalized rather than whether the examples were memorized.

**`recall`** teaches the model 64 arbitrary facts: `Q: Where is the amber anchor archive?` is answered `Sofia`. The name-to-city map is random, generated from a fixed seed. This is **memorization**, the opposite kind of learning. There's no rule to find, and knowing 63 of the pairs tells you nothing about the 64th. It's scored on 40 of the same facts it was trained on, which is the only fair test: the question isn't whether it generalizes, because nothing here can, but whether the adapter had room to store what it was given.

Neither task is interesting in itself. They're a pair of opposite corners, and the point is that LoRA behaves differently in each.

### Table of Contents

Skip to [the short version](#the-short-version) for the findings without the derivations.

1. [Where a fine-tune's cost is](#where-a-fine-tunes-cost-is)
2. [The mechanism: W plus a rank-8 update](#the-mechanism)
3. [Where the adapters go](#where-the-adapters-go)
4. [What training actually costs](#what-training-costs)
5. [What rank buys](#what-rank-buys)
6. [Why the update can be low-rank when the weights are not](#why-low-rank-works)
7. [Merging, and the cost of not merging](#merging)
8. [Fifty fine-tunes of one base model](#fifty-finetunes)
9. [What follows from all this](#what-follows)
10. [Sidebar: the probe](#sidebar-the-probe)

Plus two appendices: [counting the adapter parameters](#appendix-counting-adapters), which derives §3's census matrix by matrix, and [all notation](#appendix-all-notation).

---

### The short version {#the-short-version}

A **LoRA adapter** freezes a trained weight matrix $W$ and trains a small pair of matrices beside it, adding $(\alpha/r)BA$ to whatever $W$ computes. The pair can only ever produce a change of **rank $r$**, which is what makes it small, and $B$ starts at zero, so the adapted model starts out identical to the base model.

- **A fine-tune's cost is four copies of the model, not one.** Weights, gradient, and AdamW's two moments: 7.36 GiB in fp32 for a model whose weights are 1.84 GiB ([§1](#where-a-fine-tunes-cost-is)).
- **The adapter is 0.89% of the model.** 4,399,104 parameters across 168 matrices, **16.8 MiB** against the model's 1.84 GiB ([§3](#where-the-adapters-go)).
- **But training memory falls 3.9×, not 112×.** The gradient and the two moments go away; the frozen weights don't, and they're the largest single piece ([§4](#what-training-costs)).
- **What rank buys depends entirely on the task.** A date-reformatting rule is learned perfectly at **rank 1**, and rank 32 adds nothing. Memorizing 64 arbitrary facts takes **rank 4** to get them all. Untrained, the model scores 0 of 40 on both ([§5](#what-rank-buys)).
- **A full fine-tune's update is not low-rank.** Measured on the change a real full fine-tune made, 90% of its energy needs 25 to 214 directions, not 8. What *is* true is that it's far more concentrated than the weight it modifies, and that a rank-8 change exists that solves the task anyway ([§6](#why-low-rank-works)).
- **Merging is exact enough to be free.** Folding all 168 adapters into the weights moves the largest logit by 2.9e-06 of its size and doesn't change the predicted token ([§7](#merging)).
- **Fifty fine-tunes cost 92.0 GiB; fifty adapters over one base cost 2.66 GiB**, which is **34.6× less**, and only while they stay unmerged ([§8](#fifty-finetunes)).

---

### 1. Where a fine-tune's cost is {#where-a-fine-tunes-cost-is}

Start where post 5 started, with a census, because the answer decides what there is to save.

Qwen2.5-0.5B's parameters, grouped by what they do ([`the_full_bill`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  role        parameters   share
  --------------------------------
  embedding  136,134,656   27.6%
  attention   44,067,840    8.9%
  MLP        313,786,368   63.5%
  norms           43,904    0.0%
  TOTAL      494,032,768  100.0%
```

The shape is post 1's finding again: the MLP, which is the same component post 5 called the FFN, holds far more than attention does. What's different at this scale is the embedding row. **27.6% of this model is the table that turns tokens into vectors**, because a 151,936-row vocabulary is expensive in a model this small, and Qwen2.5-0.5B *ties* that table to its output layer, using one set of numbers for both. Post 4 found the sharp edge on that: quantizing "every layer" hits the tied table too, and [it cost 11 points of perplexity](/posts/llm-architectures-quantization/#perplexity-says-fine). It matters here for a different reason, which [§3](#where-the-adapters-go) gets to.

Now the training bill. Every parameter needs a gradient and AdamW's two moments, and in fp32 each of those is 4 bytes:

```text
  step                                   result
  -----------------------------------------------------------
  parameters in the model           494,032,768  parameters
  the weights themselves          1,976,131,072       bytes
  a gradient for each weight      1,976,131,072       bytes
  AdamW's first moment            1,976,131,072       bytes
  AdamW's second moment           1,976,131,072       bytes
  total held while training       7,904,524,288       bytes
  / 1,073,741,824 bytes in a GiB           7.36         GiB
```

Reading down it: 494,032,768 parameters at 4 bytes each is 1,976,131,072 bytes, which is **1.84 GiB**, and that same figure appears four times because the gradient, the first moment and the second moment each hold one number per parameter exactly as the weights do. Four copies is **7.36 GiB**, and that's before the backward pass stores a single activation.

**The model is a quarter of its own training bill.** That's the number LoRA is aimed at, and it's the interesting target because you can't avoid holding the weights, since the forward pass needs them. You *can* avoid the other three copies, if you can arrange for almost no parameters to be trainable.

### 2. The mechanism: W plus a rank-8 update {#the-mechanism}

Here's the whole of LoRA. Before the formula, every symbol in it:

| Symbol | Means | Shape here |
| --- | --- | --- |
| $x$ | the vector coming into one matrix | $(896,)$ |
| $W$ | the trained weight, **frozen**: never updated | $(896, 896)$ |
| $A$ | the first trainable matrix, initialized randomly | $(8, 896)$ |
| $B$ | the second trainable matrix, initialized to **zero** | $(896, 8)$ |
| $r$ | the **rank**: the width of the bottleneck between them | 8 |
| $\alpha$ | a fixed number that scales the update | 16 |
| $y$ | what comes out, the same shape as without the adapter | $(896,)$ |

$$y = W x + \frac{\alpha}{r}\,B\,A\,x$$

Read it right to left. The token's vector $x$ goes two ways. Down the frozen path it meets $W$ and produces exactly what the base model would have produced. Down the trainable path it meets $A$ first, which squeezes 896 numbers down to 8, then $B$, which expands those 8 back out to 896. The two results are added, and the sum is what the next layer sees.

**The bottleneck is the idea.** Everything the adapter can ever do to this matrix has to pass through those 8 numbers in the middle. That's what makes $BA$ a rank-8 matrix no matter how its entries are trained, and it's what makes the adapter small.

Three details decide whether an implementation is right, and the demo checks each ([`the_mechanism`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  matrix                 shape  parameters
  ------------------------------------------
  W, frozen         (896, 896)     802,816
  A, trained          (8, 896)       7,168
  B, trained          (896, 8)       7,168
  A and B together                  14,336
  W / (A and B)                      56.0x

  check                                     result
  --------------------------------------------------
  largest change before training           0.0e+00
  module output vs the formula by hand     0.0e+00
  shape of (alpha/r) B A                (896, 896)
  singular values above 1e-6              8 of 896
  scaling alpha/r                              2.0
```

**First, the size.** $W$ holds $896 \times 896 = 802{,}816$ numbers. $A$ holds $8 \times 896 = 7{,}168$ and $B$ holds $896 \times 8 = 7{,}168$, so the pair is 14,336, which is **56.0×** smaller. In general an adapter on a matrix of $d_{in}$ inputs and $d_{out}$ outputs holds $r \times (d_{in} + d_{out})$ numbers against the matrix's $d_{in} \times d_{out}$, and that ratio is what §3 counts for the whole model.

**Second, $B$ starts at zero,** so $BA$ starts at zero and the adapted model starts out computing bit-for-bit what the base model computed. The receipt's `0.0e+00` is that, measured rather than asserted. This is not a nicety: if the adapter perturbed the model before training began, every fine-tune would start by damaging it. $A$ is random rather than zero because if both were zero the gradient through the pair would be zero too, and neither would ever move.

**Third, the rank really is capped at 8.** The `singular values above 1e-6` line is the check. A **singular value** is a measure of how much a matrix stretches along one of its independent directions, and the number of non-zero ones is exactly the matrix's rank. The full 896 × 896 update has **8** of them and 888 that are zero, which is the bottleneck showing up as a measurement.

The last line is the scaling. $\alpha/r$ multiplies the update, and with $\alpha = 16$ and $r = 8$ it's 2.0. The paper's reason for expressing it as a ratio rather than a single number is practical: [Hu et al.](https://arxiv.org/abs/2106.09685) note in §4.1 that "tuning $\alpha$ is roughly the same as tuning the learning rate", and dividing by $r$ keeps the size of the update roughly fixed when you change the rank. Every run in this post uses $\alpha = 2r$, so the scaling is 2.0 at every rank and §5's sweep changes rank and nothing else.

### 3. Where the adapters go {#where-the-adapters-go}

An adapter is attached to a matrix, so "using LoRA" means choosing *which* matrices. This post adapts all seven of the projections in each block: attention's four ($W_q$, $W_k$, $W_v$, $W_o$) and the MLP's three (`gate_proj`, `up_proj`, `down_proj`), in all 24 layers ([`where_the_adapters_go`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  target     in x out  count    each  parameters
  ------------------------------------------------
  down_proj  4864x896     24  46,080   1,105,920
  gate_proj  896x4864     24  46,080   1,105,920
  k_proj      896x128     24   8,192     196,608
  o_proj      896x896     24  14,336     344,064
  q_proj      896x896     24  14,336     344,064
  up_proj    896x4864     24  46,080   1,105,920
  v_proj      896x128     24   8,192     196,608
  TOTAL                  168           4,399,104
```

Each row is one kind of matrix and how many of them there are, one per layer. The `each` column is $r \times (d_{in} + d_{out})$ from §2: for `q_proj` that's $8 \times (896 + 896) = 14{,}336$, and for `down_proj`, which takes 4,864 numbers in and puts 896 out, it's $8 \times (4864 + 896) = 46{,}080$. Multiply by 24 layers and add up the seven kinds to get **4,399,104**.

The MLP rows are the big ones, because the MLP's matrices are the wide ones, and they're three of the seven. And `k_proj` and `v_proj` are much narrower than `q_proj`, at 896 × 128 rather than 896 × 896, because Qwen2.5-0.5B uses **grouped-query attention**: it has 14 query heads but only 2 key/value heads, so the key and value projections produce a seventh as many numbers. Post 2 is where that shows up as a memory saving; here it shows up as a smaller adapter.

What that comes to against the model:

```text
  step                                   result
  -----------------------------------------------------------
  base model                        494,032,768  parameters
  matrices the adapters sit beside  357,826,560  parameters
  adapter parameters                  4,399,104  parameters
  adapters / base model                   0.89%
  x 4 bytes each (fp32)              17,596,416       bytes
  / 1,048,576 bytes in a MiB               16.8         MiB
```

**0.89% of the model is trainable.** The adapters sit beside 357,826,560 parameters, which is 72% of the model, so they reach most of it. What's left over is 136,206,208 parameters: the embedding table (136,134,656), the norms (43,904), and the small bias vectors on attention's query, key and value projections (27,648).

That exclusion is a real choice rather than an oversight, and it's the same 27.6% [§1](#where-a-fine-tunes-cost-is) flagged. Adapting the tied embedding table would touch both how tokens come in and how scores go out, which is the part of the model post 4 found most fragile. Nothing stops you doing it, and for a task that needs new vocabulary you might have to. For these two tasks the model already has every token it needs, and the adapters leave the table alone.

### 4. What training actually costs {#what-training-costs}

Now train the thing twice, on the same task, with the same data in the same order, and change only what's trainable. The task is `recall`, 200 steps, batch of 8 ([`what_training_costs`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
                                full fine-tune   LoRA r=8
  ---------------------------------------------------------
  trainable parameters             494,032,768  4,399,104
  share of the model                   100.00%      0.89%
  weights held (MiB)                   1,884.6    1,884.6
  gradient + two moments (MiB)         5,653.8       50.3
  total held (MiB)                     7,538.3    1,934.9
  loss at step 0                        6.9305     6.9305
  loss, mean of last 10 steps           0.0002     0.0014
  exact match                         40 of 40   40 of 40
  seconds per step                        0.65       0.21
```

Both runs learn the task: 40 of 40 facts recalled exactly. The full fine-tune ends at a slightly lower loss, and both are far below where they started.

The memory rows are the ones to read carefully, because they're where the usual summary of LoRA goes wrong. **Both columns hold the weights.** A frozen parameter is still a parameter the forward pass has to multiply by, so it lives in memory either way, and the 1,884.6 MiB row is identical in both columns by construction. What LoRA removes is the *per-trainable-parameter* cost: the gradient and AdamW's two moments, 12 bytes each in fp32:

```text
  step                                           result
  -------------------------------------------------------------------
  trainable, full fine-tune                 494,032,768  parameters
  trainable, LoRA r=8                         4,399,104  parameters
  ratio                                            112x       fewer

  x 12 bytes each (gradient + 2 moments)  5,928,393,216       bytes
  the same for the adapters                  52,789,248       bytes
  ratio                                            112x        less

  + the weights, held either way          1,976,131,072       bytes
  total, full fine-tune                   7,904,524,288       bytes
  total, LoRA r=8                         2,028,920,320       bytes
  ratio of totals                                  3.9x        less
```

**112× fewer trainable parameters, but only 3.9× less memory.** That gap is the practical content of this section. If you read "LoRA trains 0.89% of the parameters" as "LoRA needs 0.89% of the memory", you'll size a machine wrong by a factor of thirty.

And 3.9× is close to the best this comparison can ever give. [§1](#where-a-fine-tunes-cost-is) counted four copies of the model, all the same size. LoRA removes three of them almost entirely and cannot touch the fourth, because the forward pass needs the weights. **So 4× is the ceiling**, whatever the rank, whatever the model. Measuring 3.9× means the adapters are already down in the noise; there is nothing left to win on this axis, and the next real saving has to come from making the weights themselves smaller, which is [post 7](#whats-next).

The two runs also used different learning rates: `2e-4` for the adapters against `1e-5` for the full fine-tune. That isn't a thumb on the scale, it's a consequence. A full fine-tune at `2e-4` moves every weight in the model by a lot, and what it mostly does is destroy what the model already knew. The adapter can take a step twenty times larger safely, because the thing it's stepping on is a freshly initialized matrix sitting beside a frozen model rather than the model itself. Being able to move fast without breaking anything is part of what freezing buys.

One number in that table is not the saving it appears to be. The seconds-per-step row improves, but by nothing like 112×, and the reason is that **the frozen parameters still take part in the backward pass.** To reach $A$ and $B$ down in layer 0, the gradient has to be propagated back through every layer above them, which means multiplying through weights that will never be updated. What LoRA skips is *storing* a gradient for them and applying an optimizer step to them, not computing through them. Speed was never the main claim.

### 5. What rank buys {#what-rank-buys}

Rank is the one dial LoRA really has, so the question is what turning it up gets you.

The demo runs the same fine-tune at six ranks, on both tasks, with everything else identical: same data, same order, same 200 steps, and $\alpha = 2r$ throughout so the $\alpha/r$ scaling stays at 2.0 and rank is genuinely the only thing that changes ([`what_rank_buys`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  task: format, scored on 40 dates it never saw
  rank  parameters   share  final loss  exact match
  ---------------------------------------------------
  none           0  0.000%    2.06e+00      0 of 40
  1        549,888  0.111%    3.16e-05     40 of 40
  2      1,099,776  0.223%    1.52e-05     40 of 40
  4      2,199,552  0.445%    6.77e-06     40 of 40
  8      4,399,104  0.890%    4.77e-06     40 of 40
  16     8,798,208  1.781%    8.24e-06     40 of 40
  32    17,596,416  3.562%    5.32e-06     40 of 40

  task: recall, scored on 40 of the facts it was trained on
  rank  parameters   share  final loss  exact match
  ---------------------------------------------------
  none           0  0.000%    7.25e+00      0 of 40
  1        549,888  0.111%    1.42e-01     37 of 40
  2      1,099,776  0.223%    6.36e-02     37 of 40
  4      2,199,552  0.445%    1.12e-02     40 of 40
  8      4,399,104  0.890%    1.11e-03     40 of 40
  16     8,798,208  1.781%    4.06e-04     40 of 40
  32    17,596,416  3.562%    2.79e-03     40 of 40

  'final loss' is the mean of the last 10 steps, in scientific
  notation because on the format task every rank rounds to zero at
  four decimal places. A single step's loss depends on which
  examples its batch drew, so the mean is what to read.
  'share' is of the base model's 494,032,768 parameters, and every
  run sees the same examples in the same order, so rank is the
  only difference between rows. The 'none' row is the model before
  any training, scored the same way: the control that says whether
  a rank bought anything at all.
```

The `none` row is the control, and it's what makes the rest of the table readable: before training, the model gets **0 of 40** on both tasks. Everything below that row was bought.

![What rank buys on two tasks: a rule learned perfectly at rank 1, memorization needing rank 4 and improving to rank 16](/assets/picture/2026-09-26-llm-architectures-lora/rank-sweep-light.png){: .light width="1000" height="419" }
![What rank buys on two tasks: a rule learned perfectly at rank 1, memorization needing rank 4 and improving to rank 16](/assets/picture/2026-09-26-llm-architectures-lora/rank-sweep-dark.png){: .dark width="1000" height="419" }

**The two tasks answer the question differently, and that's the finding.**

On `format`, rank 1 is already the ceiling. 549,888 trainable parameters, 0.111% of the model, and all 40 unseen dates come out right. Ranks 2 through 32 do exactly as well, and the loss sits in the same neighbourhood at every one of them. Thirty-two times the adapter buys nothing at all, because at rank 1 there was nothing left to buy.

On `recall`, rank is the binding constraint. Rank 1 gets 37 of 40 and its loss stalls at `1.42e-01`, more than a hundred times rank 8's. Rank 2 cuts that by more than half and still gets 37. **Rank 4 is the first to get all 40**, and the loss goes on improving to rank 16. The adapter is buying somewhere to put the facts, and until it has enough room, three of them have nowhere to go.

The pattern behind it is what transfers to a task this post didn't run. A **rule** has to be *found*, and once found it applies to every example, so what the adapter stores is one procedure. The capacity that takes is small and largely independent of how many examples you have. **Arbitrary facts** have to be *stored*, one at a time, and 64 of them need more room than 1 does. Rank is capacity, and only the second kind of task is shopping for it.

This is also why the original paper's headline about rank isn't the contradiction it might look like. [Hu et al.](https://arxiv.org/abs/2106.09685) report in §7.2 that "a rank as small as one suffices for adapting both $W_q$ and $W_v$ on these datasets", and their datasets are WikiSQL and MultiNLI, which are rule-shaped in exactly this sense. Their result and this post's `format` result are the same result. The `recall` column is what that headline doesn't cover.

One honest wrinkle in the table: on `recall` the loss falls all the way to rank 16 and then gets *worse* at rank 32. That's a single run at a fixed step count, not a law, and the likeliest reading is that a larger adapter has more parameters to move and 200 steps is no longer enough to settle them. It's a reminder that a rank sweep measures rank *and* everything held fixed alongside it. [Kalajdzievski](https://arxiv.org/abs/2312.03732) argues the $\alpha/r$ scaling used here systematically under-serves high ranks and proposes $\alpha/\sqrt{r}$ instead, which is one reason not to read the right-hand end of a sweep like this too confidently.

### 6. Why the update can be low-rank when the weights are not {#why-low-rank-works}

Now the question the whole method rests on. A rank-8 change to a 896 × 896 matrix is an extremely restricted thing to be allowed to do. Why should it be enough?

The usual answer is that the change fine-tuning needs to make is *itself* low-rank, so nothing is lost by only being able to draw low-rank changes. [Hu et al.](https://arxiv.org/abs/2106.09685) put it as a hypothesis in their §1: "the change in weights during model adaptation also has a low intrinsic rank." A hypothesis is exactly the kind of thing this series should measure rather than repeat.

So measure it. Fine-tune the model **fully**, with no low-rank constraint anywhere, and look at what it did: subtract the weights before from the weights after to get $\Delta W$, the change, one matrix at a time. Then take the singular values of that change.

Two measurements on the same spectrum, each answering a different question. The first asks how many directions it takes to account for 90% of a matrix ([`why_the_update_is_low_rank`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  Directions needed to carry 90% of a matrix's energy, layer 12:

  matrix     rank  W itself  change, format  change, recall
  -----------------------------------------------------------
  q_proj      896       299              68              38
  k_proj      128        70              39              25
  v_proj      128       106              51              38
  o_proj      896       324              54              42
  gate_proj   896       629             214             127
  up_proj     896       665             192             106
  down_proj   896       597             203             115
```

The second asks the question §5 actually cares about: of everything in the matrix, how much would the best possible rank-8 version keep?

```text
  Share of each matrix held in its largest 8 directions, which is
  what the best possible rank-8 approximation of it would keep:

  matrix     W itself  change, format  change, recall
  -----------------------------------------------------
  q_proj        10.1%           63.7%           68.3%
  k_proj        22.5%           68.4%           74.1%
  v_proj        11.1%           59.6%           67.5%
  o_proj         7.6%           72.4%           71.2%
  gate_proj      5.3%           50.2%           54.6%
  up_proj        2.7%           54.4%           58.2%
  down_proj      5.4%           49.5%           54.5%
```

![Cumulative energy against directions used: the fine-tune's change concentrates in far fewer directions than the weight it modifies](/assets/picture/2026-09-26-llm-architectures-lora/update-spectrum-light.png){: .light width="1000" height="554" }
![Cumulative energy against directions used: the fine-tune's change concentrates in far fewer directions than the weight it modifies](/assets/picture/2026-09-26-llm-architectures-lora/update-spectrum-dark.png){: .dark width="1000" height="554" }

Take `q_proj` in layer 12, the matrix [§2](#the-mechanism) walked through. It has 896 directions available to it.

**The trained weight uses nearly all of them.** It takes 299 directions to account for 90% of $W$, and its largest 8 hold just **10.1%** of it. That's what a dense, thoroughly trained matrix looks like, and it's the reason you can't simply compress $W$ itself to rank 8 and expect a working model.

**The change a full fine-tune made to that same matrix is a different shape entirely.** 90% of it fits in **68** directions for the `format` fine-tune and **38** for `recall`, four to eight times fewer than the weight it modifies. Its largest 8 directions hold **63.7%** and **68.3%** of it, against the weight's 10.1%.

**So the hypothesis is half right, and the half that's wrong is the interesting half.**

The change really is *far* more concentrated than the weight it modifies. That much is not subtle, and it's the part of the low-rank story that holds up.

But it is not rank 8, and it isn't close. If LoRA's job were to reproduce what a full fine-tune does, rank 8 would capture around two thirds of it and miss the rest.

The resolution is that reproducing the full fine-tune was never the job. A full fine-tune finds *one* way to solve the task, out of many, and it has no reason to look for a tidy one: every direction is free to it, so it uses them. LoRA is asked a different question — is there a rank-8 change that solves the task? — and [§5](#what-rank-buys) answers it by finding one. **"The update is low-rank" is the wrong claim. "A low-rank update suffices" is the right one**, and only the second is what the measurements support.

That distinction stops being academic as soon as a task gets harder. [Biderman et al.](https://arxiv.org/abs/2405.09673) measured the same gap at real scale and found it matters: fine-tuning on programming and mathematics, "full finetuning learns perturbations with a rank that is 10-100X greater than typical LoRA configurations", and in those settings LoRA "substantially underperforms full finetuning". The compensation they report is on the other side of the ledger, and it follows from the same constraint: LoRA "better maintains the base model's performance on tasks outside the target domain", because a change restricted to a few directions is a change that disturbs less of what was already there. This post's two tasks are both easy enough that the gap never bites. That is a fact about the tasks.

Two smaller results in those tables.

**The MLP's updates are markedly higher-rank than attention's.** In the `format` fine-tune, attention's four matrices need 39 to 68 directions to carry 90% of their change; the MLP's three need 192 to 214. The top-8 share says it from the other side: roughly 60–72% for attention against roughly 50–54% for the MLP. So "how low-rank is the update" isn't one number for a model, it's a number per kind of matrix, and the wide ones are the stubborn ones. That bears directly on where adapters should go, and it lines up with [QLoRA's](https://arxiv.org/abs/2305.14314) finding that "LoRA on all linear transformer block layers are required to match full finetuning performance" rather than attention alone.

**And depth changes the answer, for the weight as much as for the change.** Layer 0's `q_proj` needs only 55 directions for 90% of itself, against 321 at layer 8 and 239 at the last layer: the first layer's weight is a simpler object than the ones above it. Its change is the lowest-rank of the four too, at 28 directions for `format` and 18 for `recall`. Nothing in that column suggests one rank is right for every layer, which is what per-layer rank allocation methods exist to exploit.

A last note on what this measurement is *not*. [Shuttleworth et al.](https://arxiv.org/abs/2410.21228) run a spectral analysis that sounds similar and asks something different: they look at the singular vectors of the *resulting weight matrix*, not of the change, and find that LoRA introduces "new, high-ranking singular vectors, which we call intruder dimensions" that a full fine-tune doesn't, then show by intervening on them that those dimensions cause forgetting. That's a claim about how the two methods differ in where they leave the model, and this section's measurement, which is about the shape of the change a full fine-tune makes, neither supports nor contradicts it. They're worth reading together, and easy to conflate.

### 7. Merging, and the cost of not merging {#merging}

An adapter's update is the same shape as the matrix it modifies, which has a consequence: you can add it in and throw the adapter away.

$$W' = W + \frac{\alpha}{r}BA$$

$W'$ is an ordinary 896 × 896 matrix. Nothing about it says it was ever adapted. Load a model whose weights are all $W'$ and it has no adapters, no extra matrices, and no extra multiplies, so it runs at exactly the speed the base model ran at. This is what [Hu et al.](https://arxiv.org/abs/2106.09685) mean in §4.1 when they say LoRA introduces no additional inference latency "by construction", and it's the property that distinguishes LoRA from the adapter methods that came before it, which inserted new layers that had to be run.

The demo merges all 168 adapters and checks that the model still computes the same thing ([`merging`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  check                                     result
  --------------------------------------------------
  adapters folded into their base weight       168
  largest logit difference                8.34e-05
  largest logit, for scale                 28.8243
  difference as a share of that            2.9e-06
  same next token                              yes
  weight change after merge then unmerge  7.45e-09
```

All 168 adapters fold in, and the model's prediction survives it. The largest logit moves by a few parts in a million of its own size, and the token the model predicts doesn't change.

That residue is floating-point noise, of exactly the kind [post 3](/posts/llm-architectures-flash-attention/#exact-and-what-exact-is-worth) measured when tiled attention agreed with the fused kernel to about 1.4e-6. It has the same cause, too: adding $BA$ into $W$ and then multiplying is the same arithmetic as multiplying separately and adding afterwards, but not in the same order, and floating-point addition doesn't promise those agree to the last bit.

The reverse operation is how a serving system switches tasks: subtract the update and the base model is back. It isn't exactly the inverse, though. Merging adds a number and unmerging subtracts the same number, and in floating point that round trip doesn't always land on the bit you started from. The receipt's last line measures how far off it lands. That's harmless once, and it's the reason a system that swaps adapters thousands of times keeps a clean copy of $W$ rather than unmerging repeatedly.

```text
  forward pass             ms  vs merged
  ----------------------------------------
  adapter kept separate  42.1      1.00x
  adapter merged into W  41.9      1.00x
```

Those two are wall clock, so they move a little from run to run, and the gap is small on a laptop-sized model at this sequence length. The direction is what matters and it isn't going to change: an unmerged adapter does two extra matrix multiplies in each of 168 places, and a merged one does none. On this run the difference is a few percent. It grows with how much of the model is adapted and shrinks as the rest of the forward pass gets heavier.

So merging is free at inference and costs a one-off pass over the weights. **The only reason not to merge is that you have more than one adapter**, which is [§8](#fifty-finetunes).

### 8. Fifty fine-tunes of one base model {#fifty-finetunes}

Here's the situation LoRA is actually deployed for. You have one model and fifty customers, each wanting it adapted to their own data. A full fine-tune gives you fifty models.

```text
  step                                    result
  -------------------------------------------------------------
  one full copy of the model       1,976,131,072        bytes
  x 50 fine-tunes, kept whole     98,806,553,600        bytes
  / 1,073,741,824 bytes in a GiB            92.0          GiB

  one adapter, r=8                    17,596,416        bytes
  / 1,048,576 bytes in a MiB                16.8          MiB
  x 50 adapters                      879,820,800        bytes
  + one shared base model, total   2,855,951,872        bytes
  / 1,073,741,824 bytes in a GiB            2.66          GiB

  whole copies / shared base               34.6x  more memory
```

**92.0 GiB against 2.66 GiB, a factor of 34.6.** Fifty whole models don't fit on one card; one model and fifty 16.8 MiB adapters barely notice each other. That ratio is the entire commercial argument for LoRA, and it gets better with the number of adapters, since each new customer costs 16.8 MiB instead of 1.84 GiB.

The catch is the one §7 set up. **You can only merge one adapter at a time.** Merging customer A's update into $W$ produces a model that is customer A's, and a request from customer B now has to unmerge it and merge a different one, which is a pass over 358 million parameters between requests. That's fine if requests arrive in long runs per customer, and useless if they interleave.

So a real multi-tenant server keeps every adapter unmerged and pays the extra multiplies on every forward pass, which is the cost §7 measured. It also means the batch can no longer be one matrix multiply: tokens belonging to different customers need different $A$ and $B$, so the adapted part of the layer becomes a *grouped* multiply, one group per adapter in the batch. [S-LoRA](https://arxiv.org/abs/2311.03285) is built around exactly this, keeping adapters in host memory, fetching the ones a batch needs, and using custom kernels to batch the heterogeneous part.

Post 5 ended on a version of this same shape, so put the two side by side. There, per-token sparsity was real and batch sparsity wasn't, because a batch needs the union of what its tokens want. Here, the single-adapter saving is real and the multi-adapter saving is smaller than it looks, because a batch needs the union of what its *requests* want. Batching is where clean per-item stories go to be complicated.

### 9. What follows from all this {#what-follows}

**LoRA does not make the model cheaper to hold; it makes the *update* cheap.** The frozen weights are still resident during training, so the total memory falls by 3.9× while the trainable parameter count falls by 112×. Post 5's lesson in a new place: the number in the headline and the number on the invoice measure different resources.

**What rank you need is a property of the task, not of the method.** A rule that applies to every example was learned perfectly at rank 1. Sixty-four arbitrary facts needed more. Neither number transfers to your problem, and the useful move is to sweep, which is cheap: the whole sweep in §5 is a fraction of one full fine-tune.

**"The update is low-rank" is not what the measurement says.** What a full fine-tune actually does to a matrix is high-rank. What §5 shows is that a low-rank update can reach the same behaviour, which is a claim about the existence of a good solution, not about the shape of the one gradient descent finds unaided. [Biderman et al.](https://arxiv.org/abs/2405.09673) found the gap matters most exactly where these two tasks were easiest: on domains far from what the base model already knows.

**And the serving story is the reason LoRA won.** One base model plus a directory of 16.8 MiB adapters is a different operational shape from fifty copies of a model, and it's why every serving stack now has a multi-adapter path. The cost is that the adapters can't be merged when a batch mixes them, which is the one place LoRA's "no additional inference latency" has an asterisk.

### 10. Sidebar: the probe {#sidebar-the-probe}

One question to close on, the kind this material gets asked, and what separates an answer that sounds right from one that is.

> **"We need a fine-tune per customer, fifty of them. LoRA trains under 1% of the parameters, so training is about a hundred times cheaper and we can serve all fifty on one GPU. Right?"**

**The tempting answer:** *"Yes. 0.9% trainable is ~100× less memory, and the adapters are megabytes, so fifty of them is nothing."*

Both halves are half right, and they fail in different directions.

**1. Separate trainable parameters from memory.** The parameter count falls **112×**; the memory falls **3.9×**, from 7,538.3 MiB to 1,934.9 MiB. What disappeared is the gradient and AdamW's two moments, 12 bytes per trainable parameter. What did not is the weights, because a frozen parameter still has to be there for the forward pass. If the model didn't fit before, LoRA may not be the thing that makes it fit.

**2. And separate memory from time.** The frozen parameters still participate in the backward pass: gradients have to flow *through* them to reach the adapters, even though none is stored for them. A step got faster here, but by a few times rather than a hundred, and the saving comes mostly from not applying an optimizer update to 494 million parameters.

**3. Refuse "rank 8" as a default.** This post's two tasks disagree completely about what rank buys. A systematic rule was learned perfectly at rank 1. Sixty-four arbitrary facts needed more, because the capacity to store them is the thing being bought. Sweeping rank costs a fraction of one full fine-tune, so measure it rather than inherit it.

**4. Then say what serving fifty actually costs.** Fifty whole fine-tunes are 92.0 GiB of weights; fifty adapters over one shared base are 2.66 GiB, **34.6× less**. That is real, and it's the reason multi-adapter serving exists. The asterisk is that this only holds while the adapters stay *unmerged*, and an unmerged adapter costs extra work on every forward pass. A batch that mixes customers can't merge any of them, which is the case production systems are built around ([S-LoRA](https://arxiv.org/abs/2311.03285) keeps adapters in host memory and batches them with custom kernels).

What the question is really testing is whether "1% of the parameters" is being read as one number that describes training cost, memory, latency and serving all at once, or as what it is: a statement about how many numbers receive a gradient.

### What's next {#whats-next}

Post 7 is **QLoRA**, which is this post and [post 4](/posts/llm-architectures-quantization/) at the same time: quantize the frozen base model to 4-bit, then train a LoRA adapter on top of it. Post 4 already derived the format it uses, NF4, to within 6e-08 of the shipped table, so the piece that's new is the interaction. The questions there are what happens to gradients that have to flow back through quantized weights, why the adapter stays in higher precision when the thing it's adapting doesn't, and whether the two approximations compound or stay out of each other's way. [§1](#where-a-fine-tunes-cost-is)'s four copies is the number to keep in mind: QLoRA attacks the one this post left alone.

### Appendix: counting the adapter parameters {#appendix-counting-adapters}

This appendix derives [§3's census](#where-the-adapters-go) matrix by matrix, so every number in it can be checked by hand. One rule does all the work: an adapter on a matrix with $d_{in}$ inputs and $d_{out}$ outputs holds

$$r \times d_{in} \;+\; d_{out} \times r \;=\; r\,(d_{in} + d_{out})$$

parameters, because $A$ has $r$ rows of $d_{in}$ numbers and $B$ has $d_{out}$ rows of $r$ numbers.

Qwen2.5-0.5B has 24 identical layers, each holding seven adapted matrices. Their shapes come from the model's configuration: a width of 896, an MLP inner width of 4,864, and 14 query heads against 2 key/value heads of 64 numbers each, so the key and value projections produce $2 \times 64 = 128$ numbers rather than $14 \times 64 = 896$.

| matrix | $d_{in}$ | $d_{out}$ | $d_{in} + d_{out}$ | at $r = 8$ |
| --- | --- | --- | --- | --- |
| `q_proj` | 896 | 896 | 1,792 | 14,336 |
| `k_proj` | 896 | 128 | 1,024 | 8,192 |
| `v_proj` | 896 | 128 | 1,024 | 8,192 |
| `o_proj` | 896 | 896 | 1,792 | 14,336 |
| `gate_proj` | 896 | 4,864 | 5,760 | 46,080 |
| `up_proj` | 896 | 4,864 | 5,760 | 46,080 |
| `down_proj` | 4,864 | 896 | 5,760 | 46,080 |
| **one layer** | | | **22,912** | **183,296** |

One layer's seven matrices sum to $22{,}912$ in the $d_{in} + d_{out}$ column, and 24 layers make $24 \times 22{,}912 = 549{,}888$. That number is the adapter's size **per unit of rank**, which is why every row of §5's sweep is a multiple of it: rank 1 is 549,888 parameters, rank 8 is $8 \times 549{,}888 = 4{,}399{,}104$, and rank 32 is $32 \times 549{,}888 = 17{,}596{,}416$. An adapter's size is exactly linear in its rank.

In bytes, at 4 bytes per fp32 number:

$$4{,}399{,}104 \times 4 = 17{,}596{,}416 \text{ bytes} = 16.78 \text{ MiB}$$

And what the adapters leave alone, which is the model's 494,032,768 parameters minus the 357,826,560 they sit beside:

| not adapted | parameters |
| --- | --- |
| embedding table, tied to the output head | 136,134,656 |
| RMSNorm scales, 2 per layer plus a final one | 43,904 |
| bias vectors on `q_proj`, `k_proj`, `v_proj` | 27,648 |
| **total** | **136,206,208** |

$357{,}826{,}560 + 136{,}206{,}208 = 494{,}032{,}768$, which is the model, so nothing is unaccounted for.

### Appendix: all notation {#appendix-all-notation}

The terms and symbols this post leans on; anything not listed is defined where it first appears. [Post 1's appendix](/posts/llm-architectures-attention-and-rope/#appendix-all-notation) covers attention's notation, [post 2's](/posts/llm-architectures-kv-cache/#appendix-all-notation) the memory and serving terms, [post 4's](/posts/llm-architectures-quantization/#appendix-all-notation) the quantization formats, and [post 5's](/posts/llm-architectures-mixture-of-experts/#appendix-all-notation) the mixture-of-experts terms.

| Symbol | Means | In this post's runs |
| --- | --- | --- |
| $W$ | a weight matrix the model shipped with, frozen during LoRA training | e.g. $(896, 896)$ |
| $A$, $B$ | the adapter's two trainable matrices, $A$ narrowing and $B$ widening | $(8, 896)$ and $(896, 8)$ |
| $r$ | the **rank**: the width of the bottleneck between $A$ and $B$ | 1 to 32 in §5, 8 elsewhere |
| $\alpha$ | the constant scaling the update, always $2r$ here so $\alpha/r$ is fixed | 16 at $r = 8$ |
| $\Delta W$ | the change to a weight matrix: $(\alpha/r)BA$ for LoRA, $W_{after} - W_{before}$ for a full fine-tune | $(896, 896)$ |
| rank | how many independent directions a matrix holds, at most its smaller side | 8 for an adapter, up to 896 for $W$ |
| singular value | how far a matrix stretches along one of its independent directions | 8 non-zero for a rank-8 update |
| energy | the square of a singular value; the squares sum to the matrix's squared size | §6's measure |
| effective rank | how many directions it takes to carry 90% of a matrix's energy | §6's column |
| full fine-tune | training every parameter, the thing LoRA is an alternative to | 494,032,768 trainable |
| adapter | the trained $A$ and $B$ for every adapted matrix, saved as one file | 4,399,104 parameters, 16.8 MiB |
| merging | folding $(\alpha/r)BA$ into $W$, after which no adapter is left to run | §7 |
| gradient | one number per trainable parameter, saying how to change it | same size as the trainable set |
| AdamW moments | the optimizer's two running averages, one number each per trainable parameter | 2 more copies |
| optimizer state | gradient plus both moments: 12 bytes per trainable parameter in fp32 | 5,653.8 MiB full, 50.3 MiB LoRA |
| GQA | grouped-query attention: fewer key/value heads than query heads | 14 query, 2 key/value |
| exact match | the share of held-out prompts whose completion is character-for-character right | §4 and §5's score |

### References

- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021) — the method this post implements. §4.1 is the source of the $\alpha/r$ scaling and of the merging argument ("this guarantees that we do not introduce any additional latency during inference compared to a fine-tuned model by construction"), which §7 measures. Their main GPT-3 experiments adapt only $W_q$ and $W_v$; §7.1 reports that adapting both beats adapting either alone. §7.2 is the finding §5 revisits: "a rank as small as one suffices for adapting both $W_q$ and $W_v$ on these datasets", on GPT-3 with WikiSQL and MultiNLI.
- Aghajanyan, Zettlemoyer and Gupta, [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255) (2020) — the prior result LoRA's low-rank hypothesis is built on, cited in its §1.
- Biderman et al., [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (2024) — the counterweight to §5's easy case, and the direct corroboration of §6. On programming and mathematics, in both instruction tuning and continued pre-training on 20B tokens, LoRA "substantially underperforms full finetuning" while better preserving performance outside the target domain. Their measurement of the update is the one §6 reproduces at laptop scale: "full finetuning learns perturbations with a rank that is 10-100X greater than typical LoRA configurations."
- Shuttleworth et al., [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228) (2024) — a different spectral measurement from §6's, and worth not conflating with it. They analyze the singular vectors of the *resulting* weight matrix rather than of the change, and find LoRA introduces "new, high-ranking singular vectors, which we call intruder dimensions" that full fine-tuning does not, then show by intervening on them that they cause forgetting.
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (2023) — post 7's subject, and the source of §3's target list: they report that "LoRA on all linear transformer block layers are required to match full finetuning performance", having found query and value alone insufficient on LLaMA 7B with Alpaca.
- Sheng et al., [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285) (2023) — §8's problem solved at production scale. Adapters are kept unmerged and stored in host memory, fetched per query, with custom kernels for batching heterogeneous adapters; they report up to 4× the throughput of the baselines and "several orders of magnitude" more served adapters.
- Kalajdzievski, [A Rank Stabilization Scaling Factor for LoRA](https://arxiv.org/abs/2312.03732) (2023) — argues the $\alpha/r$ scaling this post uses under-scales high ranks, and proposes $\alpha/\sqrt{r}$ instead. Relevant to how §5's sweep should be read.
- Qwen Team, [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) (2024) — the model fine-tuned throughout, the same checkpoint post 4 quantized.
