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

### The short version {#the-short-version}

Fine-tuning means continuing to train a model that has already been trained, on data you care about. Done the obvious way it's expensive, and not mainly because of the model's size: training keeps three more numbers for every weight. For this post's 494-million-parameter model, a full fine-tune in fp32 holds **7.36 GiB** for weights that take 1.84 GiB ([§1](#where-a-fine-tunes-cost-is)).

LoRA avoids most of that by never changing the original weights. It freezes them, and beside 168 of the model's weight matrices it trains a small pair of new matrices whose product is added to the frozen one. The pair is held to a low **rank**, which in plain terms means it can only change a matrix along a few independent directions, and that limit is what makes it small. The rest of the post measures what the limit saves and what it costs.

Three findings are about cost:

- **The adapter is 0.89% of the model**: 4,399,104 parameters, **16.8 MiB** ([§3](#where-the-adapters-go)).
- **But training memory falls 3.9×, not 112×.** The gradients and optimizer state go away; the frozen weights don't, and in fp32 they cap the saving at 4× ([§4](#what-training-costs)).
- **And a step gets about three times faster, not a hundred.** The backward pass still carries the gradient through every frozen weight, and most of the time saved is the optimizer's ([§4](#what-training-costs)).

Three are about what rank buys:

- **It depends on the task.** A date-reformatting rule is learned perfectly at **rank 1**, and rank 32 adds nothing. On 64 memorized facts, every rank gets 37 to 40 right across three seeds, and rank shows up in the loss instead, which falls about 390× from rank 1 to rank 16 ([§5](#what-rank-buys)).
- **Where the adapters go matters too.** On attention alone at rank 8 they fell well short; on the MLP alone they did as well as on all seven kinds of matrix ([§5](#where-adapters-pay)).
- **A full fine-tune's change is not low-rank.** Across all 168 matrices, 90% of it needs 10 to 280 directions, never 8 or fewer. It is far more concentrated than the weights it changes, though, and a rank-8 change that solves the task exists anyway ([§6](#why-low-rank-works)).

And three are about living with it:

- **On these tasks LoRA didn't forget less.** At the learning rate each needs, a rank-8 adapter and a full fine-tune disturbed unrelated text about equally. Forgetting grew with rank, and a full fine-tune at LoRA's learning rate wrecked it ([§6](#what-it-forgot)).
- **Merging is exact enough to be free.** Folding the adapter into the weights moves the largest logit by 2.6e-06 of its size and doesn't change the predicted token. Kept separate, it costs 1.16× to 1.21× a merged forward pass ([§7](#merging)).
- **Fifty fine-tunes cost 92.0 GiB; fifty adapters over one base cost 2.66 GiB**, **34.6× less**. One batch can serve several adapters at once, for about a fifth more compute per forward pass ([§8](#fifty-finetunes)).

### The thing that makes fine-tuning expensive

Training keeps three more numbers for every parameter, on top of the weight itself.

The **gradient** is the first: for each weight, a number saying which way a small change would make the model's error worse, and by how much, computed from how wrong the model was on the current batch. Training steps each weight the opposite way. It's one number per weight, so it takes as much memory as the weights do.

The other two belong to the optimizer, the rule that turns gradients into changes to the weights. **AdamW**, the optimizer almost everything uses, keeps two running averages for every weight, and they do different jobs. The running average of the gradient (its "first moment") smooths the *direction*: a raw gradient is noisy, one batch says move left and the next says move right, and averaging cancels the flip-flopping. The running average of the gradient *squared* (its "second moment") sets each weight's *step size*: AdamW divides each step by its square root, so a weight whose gradients run large takes proportionally smaller steps, and one whose gradients run small takes larger ones. That's two more numbers per parameter.

So a full fine-tune holds four model-sized blocks of memory, the weights, the gradient and the two moments, before it stores a single activation from the forward pass. They're only the same size when every number is stored in the same precision, and this post keeps them all in fp32, 4 bytes each, so the rest of it calls them **four copies** of the model for short. [§1](#where-a-fine-tunes-cost-is) measures it: **7.36 GiB** for a model whose weights are 1.84 GiB. The usual large-scale setup, mixed precision, stores the weights and gradients in 2-byte bf16 but keeps the moments, and usually a master copy of the weights, in 4-byte fp32. That comes to about 16 bytes per parameter, eight times the bf16 weights rather than four.

**LoRA attacks exactly that.** The name stands for *Low-Rank Adaptation*, introduced by [Hu et al.](https://arxiv.org/abs/2106.09685) in 2021.

The idea is simple. Freeze every weight the model shipped with, so none of them need a gradient or either of AdamW's moments. Then, beside some of those weight matrices, add a small pair of new matrices and train only those. In this post that's the seven projection matrices in each of the model's 24 layers, 168 in all, and [§3](#where-the-adapters-go) says why those and not the rest. Together, the trained pairs are called an **adapter**. At the end, the adapter is all you keep: in this post's setup, **16.8 MiB** next to a 1.84 GiB model.

Think of the frozen model as a printed map, and the adapter as a transparent overlay on top of it. The overlay draws your route. It's cheap to store, cheap to share, and easy to swap for another. If you decide you want one route permanently, you can print the overlay onto the map. Then there's nothing extra to carry, and using the map costs exactly what it did before.

The catch, and the subject of most of this post, is what the overlay is allowed to draw. Its changes aren't necessarily small; they can be large. What's limited is their shape: the overlay can only draw certain kinds of marks. The practical question is whether the change you need is one of them.

That limit is called **rank**, and it's the one idea you need for the rest of this post.

### What rank means

A matrix is a grid of numbers, so start with a small one:

$$
\begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{pmatrix}
$$

It has nine numbers, but it isn't really holding nine pieces of information. Every row is just the first row times something: ×1, ×2, ×3. Once you know the row 1 2 3 and the multipliers 1, 2, 3, you can rebuild the whole grid. That's six numbers instead of nine.

That's what **rank** measures: how many independent rows a grid really has, once you set aside the ones you can build by adding up multiples of the others. This grid has rank 1.

Notice how we rebuilt it: a column (the multipliers) times a row. A thin piece times a thin piece gives a full grid. The general rule is that if you multiply a tall, thin grid by a short, wide one, the result is full-sized, but its rank can't exceed the width the two pieces share, because everything has to squeeze through that narrow middle. The figure's middle row shows it with two columns times two rows. The result has three rows, but the third is the first two added together, so its rank is 2.

![What rank means: a 3 by 3 grid rebuilt as a column times a row, a rank-2 grid whose third row is the sum of the first two, and LoRA's 896 by 8 times 8 by 896 product](/assets/picture/2026-09-26-llm-architectures-lora/rank-explained-light.png){: .light width="1000" height="807" }
![What rank means: a 3 by 3 grid rebuilt as a column times a row, a rank-2 grid whose third row is the sum of the first two, and LoRA's 896 by 8 times 8 by 896 product](/assets/picture/2026-09-26-llm-architectures-lora/rank-explained-dark.png){: .dark width="1000" height="807" }

Now scale up. Take `q_proj`, one of the weight matrices in this post's model: 896 × 896, about 800,000 numbers. LoRA adds two thin matrices beside it, one 896 × 8 and one 8 × 896, and trains only those. Their product is a full 896 × 896 change to the weight, but it can have rank at most 8, and the two thin pieces take 56× less storage than the full grid ([§2](#the-mechanism) counts it). That smaller count is what saves you money, and it does so in two places. During training, only the thin pieces are trained, so they're the only numbers that need a gradient and two moments: the training bill from above shrinks with them. After training, they're all you keep: the adapter you save and share is those two thin pieces, not a new 896 × 896 grid. The full grid is never even built unless you choose to merge it ([§7](#merging)): the adapter multiplies the input by the short, wide piece first, down to 8 numbers, and then by the tall, thin one, back up to 896.

So the adapter isn't a blurry or partial copy of the model. It's a change to every entry of the full weight matrix, restricted to rank 8. That also makes it an approximation of something else: the change a full fine-tune would have made, which is free to use any rank. Whether the restriction matters depends on whether the change you need is simple enough to fit, and [§6](#why-low-rank-works) measures how far a real full fine-tune's change is from fitting.

The rest of the post asks what that restriction costs. It turns out to depend almost entirely on what you're trying to teach the model, and [§5](#what-rank-buys) measures two tasks that answer the question in opposite directions.

### Setup

Everything below is measured, and you can re-run all of it:

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo06
```

The code is in [`demos/d06_lora.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py), and LoRA itself is in [`lora.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/lora.py), written from the paper rather than imported, because this series builds the thing it's explaining. Every number this post quotes is printed by that program, and each block of its output, which I call a **receipt**, names the function that printed it.

Two warnings about this one. It's the first demo that *trains*, 55 training runs in all, so it takes about **80 minutes** where the others took a minute, and most of that is the rank sweep in §5.

And its numbers come from training runs rather than from arithmetic on a fixed checkpoint, so they don't reproduce the way the earlier posts' numbers do. I ran the whole demo twice and diffed the output to find out how far that goes. Everything counted or derived from shapes is identical, and so is every result of every LoRA run, loss, score and perplexity, down to the last digit printed, because each run starts its random matrices from a fixed seed. Three things moved. The wall-clock timings in §4, §7 and §8, which always do. The full fine-tune at a deliberately too-high learning rate in §4, which is chaotic, and which the post discusses as such. And §6's figures for the `format` full fine-tune, where a few cells moved by one direction or a few tenths of a percentage point: floating-point addition isn't associative, and a GPU is free to order a sum differently from one run to the next. So read §6's `format` column as good to within a direction and a few tenths of a point, and expect a different machine to move the other numbers too.

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

Now the training bill. Every parameter needs a gradient and AdamW's two moments, and in fp32 each of those is 4 bytes ([`the_full_bill`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py) again):

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

![One adapted matrix: the frozen weight W on top; below, A narrows 896 numbers down to 8 and B widens them back to 896, scaled by alpha over r, and the two paths are added](/assets/picture/2026-09-26-llm-architectures-lora/lora-mechanism-light.png){: .light width="1000" height="624" }
![One adapted matrix: the frozen weight W on top; below, A narrows 896 numbers down to 8 and B widens them back to 896, scaled by alpha over r, and the two paths are added](/assets/picture/2026-09-26-llm-architectures-lora/lora-mechanism-dark.png){: .dark width="1000" height="624" }

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

Adapting all seven is the choice the [QLoRA paper](https://arxiv.org/abs/2305.14314) recommends, and [§5](#where-adapters-pay) tests it on this model. The MLP rows are the big ones, because the MLP's matrices are the wide ones, and they're three of the seven. And `k_proj` and `v_proj` are much narrower than `q_proj`, at 896 × 128 rather than 896 × 896, because Qwen2.5-0.5B uses **grouped-query attention**: it has 14 query heads but only 2 key/value heads, so the key and value projections produce a seventh as many numbers. Post 2 is where that shows up as a memory saving; here it shows up as a smaller adapter.

What that comes to against the model ([`where_the_adapters_go`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py) again):

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
  loss, mean of last 10 steps           0.0002     0.0013
  exact match                         40 of 40   40 of 40
  seconds per step                        0.67       0.21
```

Both runs learn the task: 40 of 40 facts recalled exactly. The full fine-tune ends at a slightly lower loss, and both are far below where they started.

The memory rows are the ones to read carefully, because they're where the usual summary of LoRA goes wrong. **Both columns hold the weights.** A frozen parameter is still a parameter the forward pass has to multiply by, so it lives in memory either way, and the 1,884.6 MiB row is identical in both columns by construction. What LoRA removes is the *per-trainable-parameter* cost: the gradient and AdamW's two moments, 12 bytes each in fp32 ([`what_training_costs`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py) again):

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

And 3.9× is close to the best this comparison can ever give. [§1](#where-a-fine-tunes-cost-is) counted four copies of the model, all the same size. LoRA removes three of them almost entirely and cannot touch the fourth, because the forward pass needs the weights. **So in fp32, 4× is the ceiling**, whatever the rank, whatever the model. In mixed precision the ceiling is nearer 8×, since a full fine-tune holds about 16 bytes per parameter and a frozen bf16 weight holds 2. Measuring 3.9× means the adapters are already down in the noise; there is nothing left to win on this axis, and the next real saving has to come from making the weights themselves smaller, which is [post 7](#whats-next).

![Memory held while training, drawn to scale: the full fine-tune's four equal copies of the model against LoRA's one copy plus an adapter sliver too thin to see](/assets/picture/2026-09-26-llm-architectures-lora/training-memory-light.png){: .light width="1000" height="325" }
![Memory held while training, drawn to scale: the full fine-tune's four equal copies of the model against LoRA's one copy plus an adapter sliver too thin to see](/assets/picture/2026-09-26-llm-architectures-lora/training-memory-dark.png){: .dark width="1000" height="325" }

The two runs also used different learning rates: `2e-4` for the adapters against `1e-5` for the full fine-tune. That isn't a thumb on the scale, it's a consequence, and the demo checks it by running the full fine-tune a second time at the adapters' rate ([`the_learning_rate`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  full fine-tune, recall          lr 1e-05  lr 0.0002
  -----------------------------------------------------
  loss, mean of last 10 steps       0.0002     0.1775
  exact match                     40 of 40   38 of 40
  perplexity on post 4's passage     22.08     65,115
    vs untrained                     1.06x     3,120x
```

The last two rows need a measure this post hasn't used yet. **Perplexity** is the one [post 4](/posts/llm-architectures-quantization/#perplexity-says-fine) judged quantization by: have the model predict every token of a fixed passage of ordinary prose, and ask how many tokens it is, on average, torn between. Lower is better. The passage is post 4's own, 550 tokens about harbours, photosynthesis and interest rates, so the untrained model's 20.87 is post 4's fp32 reference, and nothing in it looks like a date or a made-up archive. What happens to it is a measure of what a fine-tune did to everything it wasn't trying to change.

At `1e-5` the full fine-tune learns all 40 facts, and the passage barely notices: 22.08. At `2e-4` it learns fewer of them, and the passage goes to 65,115: the model no longer predicts ordinary English. That run is also the least reproducible thing in this post. Three runs of the same program sent the passage to about 65 thousand, 3 million and 87 million, because a model knocked that far off course amplifies every rounding difference along the way. What never changed is that it ends up thousands of times worse than where it started, and learns the task less well while doing it. The adapter can take the twenty-times-larger step safely because what it's stepping on is a freshly initialized matrix beside a frozen model, not the model itself. Being able to move fast without breaking anything is part of what freezing buys.

One number in the first table isn't the saving it appears to be. Seconds per step improve about three times, nowhere near 112×. Splitting one step into its three phases shows where the time goes ([`step_breakdown`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  phase             full fine-tune  LoRA r=8  full / LoRA
  ---------------------------------------------------------
  forward pass                  84        93         0.9x
  backward pass                186       113         1.6x
  optimizer update             406        16        25.9x
  one step                     676       221         3.1x
```

**The forward pass costs the same**, a little more for LoRA, which is the same model plus 168 small extra multiplies. **The backward pass is 1.6× cheaper, not 112×.** At every layer, backpropagation does two jobs: it passes the gradient down to the layer below, and it works out the gradient of that layer's own weights. A frozen weight skips the second job but not the first, because $A$ and $B$ down in layer 0 can only be reached by passing the gradient back through every layer above them. **The optimizer update is where the big ratio lives.** AdamW reads and rewrites four numbers for each of 494 million parameters on every step, and for 4.4 million it barely registers. Speed was never LoRA's main claim, and most of the speed-up it does deliver comes from the optimizer.

### 5. What rank buys {#what-rank-buys}

Rank is the one dial LoRA really has, so the question is what turning it up gets you.

The demo runs the same fine-tune at six ranks, on both [tasks](#two-tasks), with everything else identical: same data, same order, same 200 steps, and $\alpha = 2r$ throughout so the $\alpha/r$ scaling stays at 2.0 and rank is genuinely the only thing that changes. It trains every rank three times, because one thing does differ between otherwise identical runs: $A$ starts out random, and a different random start can land somewhere different. Each of the three runs starts $A$ from a different fixed seed ([`what_rank_buys`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  task: format, scored on 40 dates it never saw
  rank  parameters   share  final loss  exact match  passage
  ------------------------------------------------------------
  none           0  0.000%    2.06e+00      0  0  0    20.87
  1        549,888  0.111%    2.58e-05     40 40 40    20.93
  2      1,099,776  0.223%    1.55e-05     40 40 40    21.00
  4      2,199,552  0.445%    6.43e-06     40 40 40    20.98
  8      4,399,104  0.890%    1.23e-05     40 40 40    21.21
  16     8,798,208  1.781%    9.96e-07     40 40 40    21.76
  32    17,596,416  3.562%    3.08e-06     40 40 40    22.90

  task: recall, scored on 40 of the facts it was trained on
  rank  parameters   share  final loss  exact match  passage
  ------------------------------------------------------------
  none           0  0.000%    7.25e+00      0  0  0    20.87
  1        549,888  0.111%    1.56e-01     40 38 38    21.60
  2      1,099,776  0.223%    3.52e-02     37 40 40    21.62
  4      2,199,552  0.445%    3.86e-03     40 40 39    22.37
  8      4,399,104  0.890%    1.34e-03     40 40 39    22.51
  16     8,798,208  1.781%    3.99e-04     40 40 40    23.46
  32    17,596,416  3.562%    1.20e-02     40 40 40    25.39

  Every rank is trained 3 times, with A initialized from
  seeds 0, 1, 2; the data and its order are the same
  each time. 'exact match' lists all 3 runs, each out of 40.
  'final loss' (mean of the last 10 steps) and 'passage' are the
  median of the three. 'share' is of the base model's 494,032,768
  parameters. The 'none' row is the model before any training,
  scored the same way: the control that says whether a rank bought
  anything at all. 'passage' is perplexity on post 4's held-out
  prose, which section 6 reads; lower is better.
```

The `none` row is the control, and it's what makes the rest of the table readable: before training, the model gets **0 of 40** on both tasks. Everything below that row was bought.

![What rank buys on two tasks: a rule learned perfectly at rank 1, and memorization whose loss keeps falling as rank grows](/assets/picture/2026-09-26-llm-architectures-lora/rank-sweep-light.png){: .light width="1000" height="419" }
![What rank buys on two tasks: a rule learned perfectly at rank 1, and memorization whose loss keeps falling as rank grows](/assets/picture/2026-09-26-llm-architectures-lora/rank-sweep-dark.png){: .dark width="1000" height="419" }

**The two tasks answer the question differently, and that's the finding.**

On `format`, rank 1 is already the ceiling. 549,888 trainable parameters, 0.111% of the model, and all 40 unseen dates come out right, in all three runs. Ranks 2 through 32 do exactly as well. Their losses wander between about 1e-06 and 3e-05 with no trend, which is what a loss looks like once the model is already certain of every answer. Thirty-two times the adapter buys nothing at all, because at rank 1 there was nothing left to buy.

On `recall`, the exact-match column is the wrong one to read, and the three seeds are what show it. Every rank gets between 37 and 40 of the facts, and the misses land anywhere: rank 1 gets all 40 on one seed, while ranks 4 and 8 each miss one on another. Which fact slips is decided by where $A$ happened to start, not by rank. A single run can't tell you that: with one run per rank, rank 4 could easily look like the threshold where recall becomes perfect, when the difference is noise.

**The loss is where rank shows up.** On `recall` the median final loss falls from `1.56e-01` at rank 1 to `3.99e-04` at rank 16, about 390 times lower. Rank 1 gets nearly every fact over the line but holds them loosely, and each step up holds them more firmly. The same climb on `format` buys nothing. The adapter is buying somewhere to put the facts.

The pattern behind it is what transfers to a task this post didn't run. A **rule** has to be *found*, and once found it applies to every example, so what the adapter stores is one procedure. The capacity that takes is small and largely independent of how many examples you have. **Arbitrary facts** have to be *stored*, one at a time, and 64 of them need more room than 1 does. Rank is capacity, and only the second kind of task is shopping for it.

This is also why the original paper's headline about rank isn't the contradiction it might look like. [Hu et al.](https://arxiv.org/abs/2106.09685) report in §7.2 that "a rank as small as one suffices for adapting both $W_q$ and $W_v$ on these datasets", and their datasets are WikiSQL and MultiNLI, which are rule-shaped in exactly this sense. Their result and this post's `format` result are the same result. The `recall` column is what that headline doesn't cover.

The last column, `passage`, is new, and [§6](#what-it-forgot) comes back to it. It scores each fine-tuned model on text that has nothing to do with either task (its perplexity, from [§4](#what-training-costs): lower is better), and it climbs with rank: from 20.93 at rank 1 to 22.90 at rank 32 on `format`, and from 21.60 to 25.39 on `recall`, against 20.87 untrained. On `format` the bigger adapter learned its task no better and disturbed more of everything else.

#### Rank 32, trained longer {#rank-32-longer}

On `recall` the loss rises at rank 32, from `3.99e-04` at rank 16 to `1.20e-02`, and the spread bars in the figure show it isn't one unlucky seed: every rank-32 run ended above every rank-16 run. A bigger adapter has more parameters to settle, so the obvious suspect is that 200 steps isn't enough. The demo trains the top three ranks three times as long ([`training_longer`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  task: recall, the same runs at 200 and at 600 steps
  rank  loss, 200 steps  loss, 600 steps  exact match
  -----------------------------------------------------
  8            1.29e-03         8.91e-05     40 of 40
  16           7.35e-04         6.64e-05     40 of 40
  32           5.02e-02         8.28e-05     40 of 40

  The 200-step column is the sweep's seed-0 run. Same seed, same
  data, same order; the longer runs keep going 400 more steps.
```

At 600 steps rank 32 reaches `8.28e-05`, level with ranks 8 and 16. The rise was the step count, not the rank. A rank sweep measures rank *and* everything held fixed alongside it, and a fixed step budget favours the adapters with less to settle. [Kalajdzievski](https://arxiv.org/abs/2312.03732) argues the $\alpha/r$ scaling used here makes that worse, because dividing by $r$ shrinks a high-rank adapter's updates enough to slow its learning, and proposes $\alpha/\sqrt{r}$ instead.

#### Where the adapters pay {#where-adapters-pay}

[§3](#where-the-adapters-go) put adapters on all seven projections because the [QLoRA paper](https://arxiv.org/abs/2305.14314) found attention alone wasn't enough. That choice can be tested here rather than inherited. The demo trains the `recall` adapter on attention's four matrices only, on the MLP's three only, and on all seven ([`where_adapters_pay`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  task: recall, 200 steps, adapters on different matrices
  adapters on     rank  parameters  final loss  exact match
  -----------------------------------------------------------
  attention only     8   1,081,344    1.26e-01     39 of 40
  attention only    32   4,325,376    3.59e-03     40 of 40
  MLP only           8   3,317,760    1.53e-03     40 of 40
  all seven          8   4,399,104    1.29e-03     40 of 40

  Attention only at rank 32 holds about as many parameters as all
  seven at rank 8, so that pair differs in placement, not in size.
  One run each, seed 0; 'all seven' is the sweep's seed-0 run.
```

Attention alone at rank 8 is the smallest adapter here, 1,081,344 parameters, and the only clearly worse one: it misses a fact and ends at `1.26e-01`, 35 to 100 times the loss of the other three. The MLP alone, at 3,317,760 parameters, does as well as all seven do. Attention alone at rank 32 is the same size as the all-seven adapter and ends at `3.59e-03` against `1.29e-03`, but that gap is inside the spread the three seeds produced for the all-seven adapter at rank 8, so one run each can't say it's real.

So on this task and this model, the measurement supports something narrower than QLoRA's advice. Attention at rank 8 is too little. Most of what the adapter needs, it can get from the MLP. Whether attention *at equal size* is worse needs more seeds than this post ran. [§6](#why-low-rank-works) finds a reason to expect the MLP to matter: what a full fine-tune changes there is much higher-rank than what it changes in attention.

### 6. Why the update can be low-rank when the weights are not {#why-low-rank-works}

Now the question the whole method rests on. A rank-8 change to a 896 × 896 matrix is an extremely restricted thing to be allowed to do. Why should it be enough?

The usual answer is that the change fine-tuning needs to make is *itself* low-rank, so nothing is lost by only being able to draw low-rank changes. [Hu et al.](https://arxiv.org/abs/2106.09685) put it as a hypothesis in their §1: "the change in weights during model adaptation also has a low intrinsic rank." A hypothesis is exactly the kind of thing this series should measure rather than repeat.

So measure it. Fine-tune the model **fully**, with no low-rank constraint anywhere, and look at what it did: subtract the weights before from the weights after to get $\Delta W$, the change, one matrix at a time. Then take the singular values of that change.

Reading singular values needs one more idea on top of [§2](#the-mechanism)'s. Square every entry of a matrix and add them up, and you get a single number for how big the matrix is overall, which this post calls its **energy**. The useful fact is that the energy splits exactly across the matrix's independent directions: square each singular value, and those squares add up to the same total. So each direction carries a measurable share of the matrix, its singular value squared over the sum of all of them. A matrix whose energy sits almost entirely in a few directions behaves like a low-rank matrix even when its other singular values aren't exactly zero, because those directions carry almost nothing.

![Energy worked through on two 4 by 4 matrices: one spreads its energy over all four directions and needs all of them for 90%, the other puts 49 of its 60 in one direction and needs two](/assets/picture/2026-09-26-llm-architectures-lora/energy-explained-light.png){: .light width="1000" height="719" }
![Energy worked through on two 4 by 4 matrices: one spreads its energy over all four directions and needs all of them for 90%, the other puts 49 of its 60 in one direction and needs two](/assets/picture/2026-09-26-llm-architectures-lora/energy-explained-dark.png){: .dark width="1000" height="719" }

The figure works this through on two small matrices with four directions each. Square the entries of either one and add them up, then do the same with its singular values, and you get the same total both ways: 68 for A, 60 for B. From there it makes the two measurements this section is about to make on the real model. The first is how many directions it takes to reach 90% of the energy. The second is how much the largest direction holds on its own, which is also how much the best possible rank-1 version of the matrix would keep. A spreads its energy fairly evenly and needs all four directions. B puts 49 of its 60 in one direction and passes 90% with two. You can't tell which is which from the entries, either: A is mostly zeros, and B has none.

Here are the same two measurements on the real model, where a matrix has up to 896 directions instead of 4. The first asks how many directions it takes to account for 90% of a matrix ([`why_the_update_is_low_rank`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  Directions needed to carry 90% of a matrix's energy, layer 12:

  matrix     rank  W itself  change, format  change, recall
  -----------------------------------------------------------
  q_proj      896       299              68              38
  k_proj      128        70              39              25
  v_proj      128       106              51              38
  o_proj      896       324              54              42
  gate_proj   896       629             215             127
  up_proj     896       665             192             106
  down_proj   896       597             203             115
```

The second is the figure's "largest direction alone" widened to the largest 8, because that's the question §5 actually cares about: of everything in the matrix, how much would the best possible rank-8 version keep ([`why_the_update_is_low_rank`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py) again)?

```text
  Share of each matrix held in its largest 8 directions, which is
  what the best possible rank-8 approximation of it would keep:

  matrix     W itself  change, format  change, recall
  -----------------------------------------------------
  q_proj        10.1%           63.7%           68.3%
  k_proj        22.5%           68.4%           74.1%
  v_proj        11.1%           59.7%           67.5%
  o_proj         7.6%           72.4%           71.2%
  gate_proj      5.3%           50.2%           54.6%
  up_proj        2.7%           54.3%           58.2%
  down_proj      5.4%           49.5%           54.5%
```

![Cumulative energy against directions used: the fine-tune's change concentrates in far fewer directions than the weight it modifies](/assets/picture/2026-09-26-llm-architectures-lora/update-spectrum-light.png){: .light width="1000" height="554" }
![Cumulative energy against directions used: the fine-tune's change concentrates in far fewer directions than the weight it modifies](/assets/picture/2026-09-26-llm-architectures-lora/update-spectrum-dark.png){: .dark width="1000" height="554" }

Take `q_proj` in layer 12, the matrix [§2](#the-mechanism) walked through. It has 896 directions available to it.

**The trained weight uses nearly all of them.** It takes 299 directions to account for 90% of $W$, and its largest 8 hold just **10.1%** of it. That's what a dense, thoroughly trained matrix looks like, and it's the reason you can't simply compress $W$ itself to rank 8 and expect a working model.

**The change a full fine-tune made to that same matrix is a different shape entirely.** 90% of it fits in **68** directions for the `format` fine-tune and **38** for `recall`, four to eight times fewer than the weight it modifies. Its largest 8 directions hold **63.7%** and **68.3%** of it, against the weight's 10.1%.

Layer 12 could be a lucky pick, so the demo measures every one of the 168 matrices the adapters sit beside, 24 layers of seven kinds (also [`why_the_update_is_low_rank`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  Every layer: directions for 90%, median over the 24 layers
  (smallest-largest):

  matrix          W itself  change, format  change, recall
  ----------------------------------------------------------
  q_proj      288 (55-331)     59 (28-118)      36 (18-51)
  k_proj        70 (21-81)      33 (17-44)      23 (10-29)
  v_proj      101 (71-109)      43 (26-54)      34 (22-43)
  o_proj     311 (186-389)     57 (14-116)      42 (25-57)
  gate_proj  640 (623-678)    203 (76-270)    122 (55-146)
  up_proj    666 (643-699)    187 (61-250)     93 (48-128)
  down_proj  632 (594-680)    197 (19-280)    106 (41-140)

  All 168 adapted matrices at once:

                  need > 8 for 90%   range  top-8 median
  --------------------------------------------------------
  W itself              168 of 168  21-699          7.8%
  change, format        168 of 168  14-280         62.7%
  change, recall        168 of 168  10-146         61.2%
```

Each cell is the median over the 24 layers, with the smallest and largest in brackets. Layer 12 is typical: its `q_proj` change needed 68 and 38 directions, and the medians across all 24 layers are 59 and 36. The bottom table is the one that settles it. **Every one of the 168 changes, on both tasks, needs more than 8 directions to carry 90% of its energy.** The lowest-rank change anywhere in the model still needs 10.

**So the hypothesis is half right, and the half that's wrong is the interesting half.**

The change really is *far* more concentrated than the weight it modifies. Across all 168 matrices the largest 8 directions hold a median of 62.7% of the `format` change and 61.2% of the `recall` change, against 7.8% of the weight. That much is not subtle, and it's the part of the low-rank story that holds up.

But it is not rank 8, and it isn't close. If LoRA's job were to reproduce what a full fine-tune does, rank 8 would capture around three fifths of it and miss the rest.

The resolution is that reproducing the full fine-tune was never the job. A full fine-tune finds *one* way to solve the task, out of many, and it has no reason to look for a tidy one: every direction is free to it, so it uses them. LoRA is asked a different question — is there a rank-8 change that solves the task? — and [§5](#what-rank-buys) answers it by finding one. **"The update is low-rank" is the wrong claim. "A low-rank update suffices" is the right one**, and only the second is what the measurements support.

That distinction stops being academic as soon as a task gets harder. [Biderman et al.](https://arxiv.org/abs/2405.09673) measured the same gap at real scale and found it matters: fine-tuning on programming and mathematics, "full finetuning learns perturbations with a rank that is 10-100X greater than typical LoRA configurations", and in those settings LoRA "substantially underperforms full finetuning". The compensation they report is on the other side of the ledger, and it follows from the same constraint: LoRA "better maintains the base model's performance on tasks outside the target domain", because a change restricted to a few directions is a change that disturbs less of what was already there. This post's two tasks are both easy enough that the gap never bites. That is a fact about the tasks. The forgetting half can be measured here, and [the end of this section](#what-it-forgot) does.

Two smaller results in those tables.

**The MLP's updates are markedly higher-rank than attention's.** Across all 24 layers of the `format` fine-tune, attention's four kinds of matrix have medians of 33 to 59 directions for 90% of their change; the MLP's three have 187 to 203. The `recall` fine-tune shows the same split at lower numbers. So "how low-rank is the update" isn't one number for a model, it's a number per kind of matrix, and the wide ones are the stubborn ones. That bears directly on where adapters should go, and [§5](#where-adapters-pay) points the same way: adapters on the MLP alone did as well as adapters on all seven, while attention alone at rank 8 fell well short.

**And depth changes the answer, for the weight as much as for the change.** Here is `q_proj` at four depths, from the same [`why_the_update_is_low_rank`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py):

```text
  q_proj    W itself  change, format  change, recall
  ----------------------------------------------------
  layer 0         55              28              18
  layer 8        321              89              50
  layer 16       191              30              26
  layer 23       239              43              35
```

Layer 0's weight needs only 55 directions for 90% of itself, against 321 at layer 8 and 239 at the last layer: the first layer's weight is a simpler object than the ones above it. Its change is the lowest-rank of the four too, at 28 directions for `format` and 18 for `recall`. Nothing in that column suggests one rank is right for every layer, which is what per-layer rank allocation methods exist to exploit.

A last note on what this measurement is *not*. [Shuttleworth et al.](https://arxiv.org/abs/2410.21228) run a spectral analysis that sounds similar and asks something different: they look at the singular vectors of the *resulting weight matrix*, not of the change, and find that LoRA introduces "new, high-ranking singular vectors, which we call intruder dimensions" that a full fine-tune doesn't, then show by intervening on them that those dimensions cause forgetting. That's a claim about how the two methods differ in where they leave the model, and this section's measurement, which is about the shape of the change a full fine-tune makes, neither supports nor contradicts it. They're worth reading together, and easy to conflate.

#### What the fine-tunes forgot {#what-it-forgot}

Biderman et al.'s title makes two claims, *learns less* and *forgets less*, and so far this section has only tested the first. The second can be measured with what [§4](#what-training-costs) set up: score each fine-tune on post 4's passage, which neither task resembles, and see how far it moved from the untrained model's 20.87 ([`what_it_forgot`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  model                   exact match  passage  vs untrained
  ------------------------------------------------------------
  untrained                              20.87
  LoRA r=8, format           40 of 40    21.28         +1.9%
  full fine-tune, format     40 of 40    21.18         +1.5%
  LoRA r=8, recall           40 of 40    22.51         +7.9%
  full fine-tune, recall     40 of 40    22.08         +5.8%

  'passage' is perplexity on post 4's 550 tokens of prose, the same
  text and the same measure post 4 scored quantization with. Lower
  is better. LoRA ran at lr 0.0002 and the full fine-tunes at
  lr 1e-05, 200 steps each.
```

Every model in the table learned its task completely, 40 of 40, so a difference in the `passage` column isn't a difference in how well the task was learned.

**On these tasks, LoRA did not forget less.** On `format` the two land a tenth of a point apart, 21.28 against 21.18. On `recall` the rank-8 adapter moved the passage *more* than the full fine-tune did, +7.9% against +5.8%. Each method ran at the learning rate it needs, and that condition matters. At LoRA's rate, [§4](#what-training-costs) showed the full fine-tune taking the passage to 65,115. A full fine-tune avoids forgetting by taking small steps.

Two things keep this from contradicting Biderman et al. Their fine-tunes ran on billions of tokens of programming and mathematics, far enough to move a model a long way; these ran 200 steps on toy tasks, where neither method moves it far. And [§5](#what-rank-buys)'s `passage` column shows the variable their result turns on: forgetting climbs with rank, from 21.60 at rank 1 to 25.39 at rank 32 on `recall`. A lower-rank adapter forgets less than a higher-rank one, here as there. What a model this small, trained this briefly, can't show is how rank 8 compares with a full fine-tune pushed hard enough for the difference to matter.

### 7. Merging, and the cost of not merging {#merging}

An adapter's update is the same shape as the matrix it modifies, which has a consequence: you can add it in and throw the adapter away.

$$W' = W + \frac{\alpha}{r}BA$$

$W'$ is an ordinary 896 × 896 matrix. Nothing about it says it was ever adapted. Load a model whose weights are all $W'$ and it has no adapters, no extra matrices, and no extra multiplies, so it runs at exactly the speed the base model ran at. This is what [Hu et al.](https://arxiv.org/abs/2106.09685) mean in §4.1 when they say LoRA introduces no additional inference latency "by construction", and it's the property that distinguishes LoRA from the adapter methods that came before it, which inserted new layers that had to be run.

The demo merges all 168 adapters and checks that the model still computes the same thing ([`merging`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  check                                     result
  --------------------------------------------------
  adapters folded into their base weight       168
  largest logit difference                6.58e-05
  largest logit, for scale                 25.1507
  difference as a share of that            2.6e-06
  same next token                              yes
  weight change after merge then unmerge  0.00e+00
```

All 168 adapters fold in, and the model's prediction survives it. The largest logit moves by a few parts in a million of its own size, and the token the model predicts doesn't change.

That residue is floating-point noise, of exactly the kind [post 3](/posts/llm-architectures-flash-attention/#exact-and-what-exact-is-worth) measured when tiled attention agreed with the fused kernel to about 1.4e-6. It has the same cause, too: adding $BA$ into $W$ and then multiplying is the same arithmetic as multiplying separately and adding afterwards, but not in the same order, and floating-point addition doesn't promise those agree to the last bit.

The reverse operation is how a serving system switches tasks: subtract the update and the base model is back. It isn't exactly the inverse, though. Merging adds a number and unmerging subtracts the same number, and in floating point that round trip doesn't always land on the bit you started from. The receipt's last line measures how far off it lands, and on this run the answer is not at all: layer 12's `q_proj`, the matrix it checks, came back to the bit. That's luck of the particular numbers rather than a guarantee, since whether adding and then subtracting a value returns the original depends on how the two sizes round against each other. It's why a system that swaps adapters thousands of times keeps a clean copy of $W$ rather than unmerging repeatedly.

Merging also changes the speed, and the timings below are from the same program, in two settings. *Prefill* is reading a prompt: 8 sequences of 256 tokens in one pass. *Decode* is generating: 8 sequences each adding one new token, which is what every step of a reply costs ([`forward_timings`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py), printed by [`merging`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)).

```text
  forward pass, ms         merged  separate  separate / merged
  --------------------------------------------------------------
  prefill, 8 x 256 tokens   840.8    1018.4              1.21x
  decode, 8 x 1 token        34.5      40.1              1.16x
```

Keeping the adapters separate costs **1.21×** a merged pass on prefill and **1.16×** on decode. Those are wall clock, so the digits move between runs, but the direction won't: an unmerged adapter does two extra matrix multiplies in each of 168 places, and a merged one does none. Each extra multiply is small, 8 numbers wide in the middle, but there are 336 of them, and on a model this small they are a real share of the work.

So merging is free at inference and costs a one-off pass over the weights. **The only reason not to merge is that you have more than one adapter**, which is [§8](#fifty-finetunes).

### 8. Fifty fine-tunes of one base model {#fifty-finetunes}

Here's the situation LoRA is actually deployed for. You have one model and fifty customers, each wanting it adapted to their own data. A full fine-tune gives you fifty models. The memory arithmetic, from [`fifty_finetunes`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py):

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

The demo does it the simplest correct way. It loads the rank-8 adapters §5 trained for both tasks beside one copy of the base model, and puts a `format` prompt and a `recall` prompt in the same batch, each row routed to its own adapter. At every adapted matrix it gathers each row's $A$ and $B$ and multiplies them batch by batch ([`MultiLoRALinear`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/lora.py), [`one_batch_two_customers`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d06_lora.py)):

```text
  row's adapter    expected       alone  mixed batch       swapped
  ------------------------------------------------------------------
  format         1901-08-02  1901-08-02   1901-08-02  10000-1 A. 1
  recall              Sofia       Sofia        Sofia    2019-09-01

  'alone' is that adapter served by itself; 'mixed batch' is both
  rows in one forward pass, each routed to its own adapter;
  'swapped' is the same batch with the two routes exchanged.

  forward pass, ms         merged  separate   mixed  mixed / merged
  -------------------------------------------------------------------
  prefill, 8 x 256 tokens   840.8    1018.4  1026.8           1.22x
  decode, 8 x 1 token        34.5      40.1    42.0           1.22x
```

Both rows give the answer their own adapter gives when served alone, and both are right. The `swapped` column is the control: the same batch with the two routes exchanged hands the date to the facts adapter and the question to the dates adapter, and both answers turn to nonsense. The answers come from the routing, not from the prompts.

The bottom table is the price. A mixed batch costs **1.22×** a merged forward pass on prefill and **1.22×** on decode, slightly more than one unmerged adapter, the difference being the gather. That's the running cost behind the 34.6×: fifty customers fit in 2.66 GiB, and every forward pass they share runs about a fifth slower than one merged model would. S-LoRA's custom kernels exist to shrink that fifth.

Post 5 ended on a version of this same shape, so put the two side by side. There, per-token sparsity was real and batch sparsity wasn't, because a batch needs the union of what its tokens want. Here, the single-adapter saving is real and the multi-adapter saving is smaller than it looks, because a batch needs the union of what its *requests* want. Batching is where clean per-item stories go to be complicated.

### 9. What follows from all this {#what-follows}

**LoRA does not make the model cheaper to hold; it makes the *update* cheap.** The frozen weights are still resident during training, so the total memory falls by 3.9× while the trainable parameter count falls by 112×. Post 5's lesson in a new place: the number in the headline and the number on the invoice measure different resources.

**What rank you need is a property of the task, not of the method.** A rule that applies to every example was learned perfectly at rank 1. Sixty-four arbitrary facts kept getting firmer up to rank 16. Neither number transfers to your problem, and the useful move is to sweep, which is affordable: each run in §5 took about a third of a full fine-tune's time per step and a quarter of its memory. Run each rank more than once, too, because at this scale the seed moved exact match as much as rank did.

**"The update is low-rank" is not what the measurement says.** What a full fine-tune actually does to a matrix is high-rank. What §5 shows is that a low-rank update can reach the same behaviour, which is a claim about the existence of a good solution, not about the shape of the one gradient descent finds unaided. [Biderman et al.](https://arxiv.org/abs/2405.09673) found the gap matters most exactly where these two tasks were easiest: on domains far from what the base model already knows.

**And the serving story is the reason LoRA won.** One base model plus a directory of 16.8 MiB adapters is a different operational shape from fifty copies of a model, and it's why every serving stack now has a multi-adapter path. The cost is that the adapters can't be merged when a batch mixes them, which is the one place LoRA's "no additional inference latency" has an asterisk.

### 10. Sidebar: the probe {#sidebar-the-probe}

One question to close on, the kind this material gets asked, and what separates an answer that sounds right from one that is.

> **"We need a fine-tune per customer, fifty of them. LoRA trains under 1% of the parameters, so training is about a hundred times cheaper and we can serve all fifty on one GPU. Right?"**

**The tempting answer:** *"Yes. 0.9% trainable is ~100× less memory, and the adapters are megabytes, so fifty of them is nothing."*

Both halves are half right, and they fail in different directions.

**1. Separate trainable parameters from memory.** The parameter count falls **112×**; the memory falls **3.9×**, from 7,538.3 MiB to 1,934.9 MiB. What disappeared is the gradient and AdamW's two moments, 12 bytes per trainable parameter. What did not is the weights, because a frozen parameter still has to be there for the forward pass. If the model didn't fit before, LoRA may not be the thing that makes it fit.

**2. And separate memory from time.** A step got about three times faster here, not a hundred. The forward pass costs the same. The backward pass still has to carry the gradient *through* every frozen weight to reach the adapters, and got only 1.6× cheaper. Most of the saving is the optimizer update that no longer touches 494 million parameters.

**3. Refuse "rank 8" as a default.** This post's two tasks disagree completely about what rank buys. A systematic rule was learned perfectly at rank 1. Sixty-four arbitrary facts kept improving up to rank 16, because the capacity to store them is the thing being bought. Each sweep run costs about a third of a full fine-tune's step time and a quarter of its memory, so measure rank rather than inherit it.

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
| energy | the sum of the squares of a matrix's entries; it equals the sum of its singular values squared, so each direction carries a share of it | §6's measure |
| directions for 90% | the fewest directions whose squared singular values reach 90% of the energy | §6's first table |
| full fine-tune | training every parameter, the thing LoRA is an alternative to | 494,032,768 trainable |
| adapter | the trained $A$ and $B$ for every adapted matrix, saved as one file | 4,399,104 parameters, 16.8 MiB |
| merging | folding $(\alpha/r)BA$ into $W$, after which no adapter is left to run | §7 |
| gradient | one number per trainable parameter, saying how to change it | same size as the trainable set |
| AdamW moments | the optimizer's two running averages, one number each per trainable parameter | 2 more copies |
| optimizer state | gradient plus both moments: 12 bytes per trainable parameter in fp32 | 5,653.8 MiB full, 50.3 MiB LoRA |
| GQA | grouped-query attention: fewer key/value heads than query heads | 14 query, 2 key/value |
| exact match | the share of held-out prompts whose completion is character-for-character right | §4 and §5's score |
| seed | the fixed starting point of a random-number generator; here, what $A$'s random start is drawn from | 0, 1 and 2 in §5 |
| perplexity | how many tokens a model is, on average, torn between when predicting a passage; lower is better | 20.87 untrained, on post 4's passage |
| prefill, decode | reading a whole prompt in one pass, against generating one new token per sequence | §7 and §8's two timings |
| mixed batch | one batch whose rows belong to different adapters, so none of them can be merged | §8 |

### References

- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021) — the method this post implements. §4.1 is the source of the $\alpha/r$ scaling and of the merging argument ("this guarantees that we do not introduce any additional latency during inference compared to a fine-tuned model by construction"), which §7 measures. Their main GPT-3 experiments adapt only $W_q$ and $W_v$; §7.1 reports that adapting both beats adapting either alone. §7.2 is the finding §5 revisits: "a rank as small as one suffices for adapting both $W_q$ and $W_v$ on these datasets", on GPT-3 with WikiSQL and MultiNLI.
- Aghajanyan, Zettlemoyer and Gupta, [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255) (2020) — the prior result LoRA's low-rank hypothesis is built on, cited in its §1.
- Biderman et al., [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (2024) — the counterweight to §5's easy case, and the direct corroboration of §6. On programming and mathematics, in both instruction tuning and continued pre-training on 20B tokens, LoRA "substantially underperforms full finetuning" while better preserving performance outside the target domain. Their measurement of the update is the one §6 reproduces at laptop scale: "full finetuning learns perturbations with a rank that is 10-100X greater than typical LoRA configurations." §6's last subsection measures the forgetting half on this post's tasks, where it does not reproduce.
- Shuttleworth et al., [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228) (2024) — a different spectral measurement from §6's, and worth not conflating with it. They analyze the singular vectors of the *resulting* weight matrix rather than of the change, and find LoRA introduces "new, high-ranking singular vectors, which we call intruder dimensions" that full fine-tuning does not, then show by intervening on them that they cause forgetting.
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (2023) — post 7's subject, and the source of §3's target list: they report that "LoRA on all linear transformer block layers are required to match full finetuning performance", having found query and value alone insufficient on LLaMA 7B with Alpaca. §5 tests the choice on this post's model.
- Sheng et al., [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285) (2023) — §8's problem solved at production scale. Adapters are kept unmerged and stored in host memory, fetched per query, with custom kernels for batching heterogeneous adapters; they report up to 4× the throughput of the baselines and "several orders of magnitude" more served adapters.
- Kalajdzievski, [A Rank Stabilization Scaling Factor for LoRA](https://arxiv.org/abs/2312.03732) (2023) — argues the $\alpha/r$ scaling this post uses under-scales high ranks, and proposes $\alpha/\sqrt{r}$ instead. Relevant to §5's rank-32 row, which needed three times the steps to catch up.
- Qwen Team, [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) (2024) — the model fine-tuned throughout, the same checkpoint post 4 quantized.
