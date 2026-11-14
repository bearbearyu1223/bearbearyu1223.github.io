---
title: "LLM Architecture Refresh [8]: Learning from Rewards"
date: 2026-12-05 09:00:00 -0800
categories: [LLM Architecture Refresh, Reinforcement Learning]
tags: [reinforcement-learning, grpo, ppo, rlhf, policy-gradient, lora, qwen, pytorch]
description: >-
  Training a small language model from a score instead of an answer: it writes
  several answers to a multiplication problem, a program checks each one, and
  the model is pushed toward the answers that beat its other tries. GRPO built
  by hand on Qwen2.5-0.5B-Instruct with LoRA, two seeds, a shuffled-reward
  control, and where RLHF fits.
math: true
pin: true
published: false
---

## The first post without an answer key

Seven posts in, every model in this series has learned by copying.

[Post 1](/posts/llm-architectures-attention-and-rope/) took a transformer block apart and put a cost on every piece. [Post 2](/posts/llm-architectures-kv-cache/) cached keys and values so a model stops redoing work for every new token. [Post 3](/posts/llm-architectures-flash-attention/) computed attention in tiles to move fewer bytes. [Post 4](/posts/llm-architectures-quantization/) stored each weight in fewer bits. [Post 5](/posts/llm-architectures-mixture-of-experts/) built a model that runs only some of its parameters for each token. [Post 6](/posts/llm-architectures-lora/) fine-tuned a model by training a small adapter beside its frozen weights. [Post 7](/posts/llm-architectures-multimodal/) taught a language model to answer questions about pictures by training a small connector between two frozen models.

Whenever one of those posts trained something, it had a right answer to copy: a fact for post 6, a caption or `three` for post 7. The loss measured how far the model's next token was from that answer, and training pulled it closer.

For a lot of tasks, though, it's easier to check an answer than to write down the one right way to reach it. A program can tell whether a product came out right, whether code passes its tests, or whether a meeting ended up on the calendar, without anyone demonstrating the steps. **Reinforcement learning** (RL) trains on that check. The model tries several answers to the same question, a program scores each one, and the model is pushed toward the answers that scored better than its other tries.

This post builds that loop by hand, on a laptop, with a task small enough to watch: two-digit multiplication. The question it tests is whether a score of 1 or 0, with no answer to copy, is enough to make a small model better at problems it never trained on. And if it is, what exactly did the score teach?

### The short version {#the-short-version}

Can a model get better at a task from nothing but a right-or-wrong score?

I took **Qwen2.5-0.5B-Instruct**, a small chat model, and asked it to multiply two-digit numbers. Out of the box it gets 37.5% of 200 test problems right ([§2](#a-checkable-task)). Then I trained it with **GRPO** (Group Relative Policy Optimization), the method DeepSeek used for its math and reasoning models, written out in about twenty lines rather than imported. For each question the model writes eight answers, a program gives each a 1 if it's the right integer and a 0 otherwise, and each answer is compared with its own group's average: the ones above it are made more likely, the ones below less ([§4](#groups-and-advantages), [§5](#one-update)). Only a LoRA adapter trains, a small set of weights beside the frozen ones, as in [post 6](/posts/llm-architectures-lora/).

After 60 steps, two runs from different random starting points (seeds) got **50.5%** and **59.0%** of the same 200 unseen problems right. A control run that trained identically, except that each batch's rewards were shuffled among its answers, fell to **33.5%** ([§6](#training)). The two seeds landed far apart, so the honest summary is "it helped, by somewhere in that range", and two runs can't say more precisely.

A few other things became clearer along the way:

- **RL can only reinforce what the model already sometimes does.** The untrained model got 58.5% of the test problems right in at least one of eight tries, against 37.5% for its single best guess. That gap is the raw material ([§2](#a-checkable-task)).
- **Most groups teach nothing.** When all eight answers are right, or all are wrong, every answer scores exactly the group average and gets no push at all. Before training, that was 61.1% of groups of eight ([§4](#groups-and-advantages)).
- **The gain was mostly arithmetic, not format.** Of the problems the first seed newly got right, 27 had been a wrong number and only 5 had been something other than a lone number ([§7](#what-it-learned)).
- **The training reward was a poor guide.** Logged as ten-step averages, it was noisy, and for the first seed it hardly moved, from 0.498 to 0.537, while the same run's test score rose 13.0 points. The curve that's easiest to watch was the least useful one here ([§6](#training)).
- **RLHF is the same loop with a different judge.** In RLHF, reinforcement learning from human feedback, the judge is a model trained on which replies people preferred, and DeepSeek-R1's authors explain why they'd rather trust a program ([§8](#where-rlhf-fits)).

None of this makes a good calculator, and it isn't meant to. Multiplication is here because the reward is unambiguous and cheap, so every number below can be checked.

### Setup

Everything below is measured, and you can re-run all of it:

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo08
```

The code is in [`demos/d08_rewards.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py). There's no RL library in it: the sampling, the reward, the group comparison and the loss are all written out, and the loss itself is one function, [`grpo_objective`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py). The adapters are post 6's [`lora.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/lora.py). Every number this post quotes is printed by that program, and each block of its output, which I call a **receipt**, names the function that printed it.

The demo trains three times, and the whole run takes about half an hour on my Mac. `uv run demo08 --quick` runs every section at toy size, to check that it works before committing the time.

**The model.** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) is the instruction-tuned version of the Qwen2.5-0.5B that posts [4](/posts/llm-architectures-quantization/), [6](/posts/llm-architectures-lora/) and [7](/posts/llm-architectures-multimodal/) used ([the report](https://arxiv.org/abs/2412.15115)). Instruction-tuned means it was trained further, after pretraining, to answer questions in a chat format, which is what this task needs: a base model asked "What is 23 × 47?" is as likely to continue the text as to answer it. It's small enough to train on a laptop in fp32, and the frozen copy that training compares it against costs no extra memory ([§5](#one-update)). fp32 means four bytes per number, as in posts 6 and 7.

**The task.** Each question is "What is *a* × *b*? Answer with just the number.", with *a* and *b* between 10 and 99. The 256 training problems and 200 test problems don't overlap: if 23 × 47 is in training, neither 23 × 47 nor 47 × 23 is in the test set. The demo's opening lines, from [`main`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py):

<!-- RECEIPT: s0 -->
```text
  model                              Qwen/Qwen2.5-0.5B-Instruct
  precision                          float32
  task                               a x b, a and b in 10..99
  reward                             1 if the answer is the right integer
  format-valid                       a lone integer: 1081, 1,081 or 1081.
  train / test problems              256 / 200, disjoint
```

The **reward** is the score an answer gets: 1 if the answer is the right integer, 0 for anything else. **Format-valid** means the answer is a lone integer, so the program can read a number out of it at all. `1081`, `1,081` and `1081.` are format-valid; `= 1081` and `23 x 47 = 1081` aren't, so they score 0 even when the number in them is right. That's a deliberate choice, and [§7](#what-it-learned) shows what it did.

**Seeds.** Each training run starts from a fixed **seed**, the number that sets its random starting adapter weights and the answers it happens to sample, so rerunning the demo on the same machine prints the same results. A different machine can sum floating-point numbers in a different order, and sampling turns tiny differences into different answers, so expect different scores there. That's why [§6](#training) trains with two seeds and reports both.

### Table of Contents

Skip to [the short version](#the-short-version) for the findings without the derivations.

1. [What next-token training measures](#sequence-log-probability)
2. [A task with a checkable answer](#a-checkable-task)
3. [From rewards to a gradient](#policy-gradients)
4. [Groups instead of a value model](#groups-and-advantages)
5. [One update by hand](#one-update)
6. [Training it](#training)
7. [What it actually learned](#what-it-learned)
8. [Where RLHF fits](#where-rlhf-fits)

Then [a sidebar of questions](#sidebar), [what's next](#whats-next), and [all notation](#appendix-all-notation) at the end.

---

### 1. What next-token training measures {#sequence-log-probability}

Before training on a score, it helps to be exact about what training on an answer does, because RL ends up changing the same number.

A language model doesn't produce an answer in one go. It produces one **token** at a time, and at each step it gives every token in its vocabulary a probability of coming next. A **token's log-probability** (log-prob) is the natural logarithm of the probability the model gave that token, given everything before it. Probabilities are between 0 and 1, so log-probs are 0 or negative: a log-prob of 0 means the model was certain, and the more negative, the more surprised it was.

The probability of a whole answer is the product of its tokens' probabilities, because each token is chosen given the ones before it. Logarithms turn a product into a sum, so the **sequence log-prob** of an answer is the sum of its tokens' log-probs. That sum is the number this post is about.

To score an answer, the model is shown the question and the answer together and asked, at every answer position, how likely the real next token was. That's **teacher forcing**, the same trick [post 7](/posts/llm-architectures-multimodal/#stage-one) used: one forward pass scores the whole answer, because each position only sees the positions before it. The question is context, so its positions are **masked out**: they're read but not scored.

Here's Qwen scoring the right answer to 23 × 47, from [`sequence_logprob`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py):

<!-- RECEIPT: s1a -->
```text
  Question: What is 23 x 47? Answer with just the number.

  prompt tokens (chat template)      45, all masked out
  answer tokens scored               5: the digits, then <|im_end|>

  answer token         token  log-prob    prob
  ----------------------------------------------
  0                      '1'   -0.3165  0.7287
  1                      '0'   -0.0163  0.9838
  2                      '8'   -2.8617  0.0572
  3                      '1'   -0.0305  0.9700
  4             '<|im_end|>'   -0.0036  0.9964
```

The question becomes 45 tokens once the **chat template** wraps it in the markers the instruct model was trained with, and none of them are scored. Qwen's tokenizer writes numbers one digit per token, so `1081` is four tokens, and the fifth is `<|im_end|>`, the token that ends a chat turn. It's scored like any other, because stopping is part of the answer: a model that wrote `1081` and kept going would be wrong.

The model is confident about every token but one. It gives the third digit, `8`, a probability of only 0.0572, and that one token costs most of the answer's log-prob. Adding up the five log-probs:

<!-- RECEIPT: s1b -->
```text
  sequence log-prob (sum)            -3.2286
  probability of the whole answer    0.0396
  cross-entropy (mean of -log-prob)  0.6457

  same, for the wrong answer "1091"  -7.1006
```

The sum is −3.2286, and $$e^{-3.2286}$$ is 0.0396, the probability the model gives this exact answer, digit by digit and then stopping. The **cross-entropy** loss that supervised training minimizes is the same sum turned around: the mean of the negative log-probs over the answer tokens, here 0.6457 ([post 6](/posts/llm-architectures-lora/#loss-and-exact-match) walked through it). For comparison, the wrong answer `1091` has a sequence log-prob of −7.1006, so the model already considers it much less likely.

**Supervised fine-tuning** (SFT), what every earlier post did, pushes that sum up for an answer someone wrote. Its gradient, the signal that says which way to nudge each weight, only exists once you hand it the answer's text: it's the gradient of `1081`'s log-prob, and without `1081` there's nothing to differentiate.

Reinforcement learning keeps the same sum and changes where the answer comes from. The model writes its own answers, a program scores them, and training pushes each answer's sequence log-prob up or down depending on how it scored. Nobody has to supply `1081`, only a way to check it. The rest of the post is about how "depending on how it scored" becomes a gradient.

---

### 2. A task with a checkable answer {#a-checkable-task}

RL needs two things from a task: answers the model will sometimes get right, and a way to tell which ones those are. Two-digit multiplication has both. The check is one line of arithmetic, and a 0.5B model is right often enough to have something to build on, and wrong often enough to have room to improve.

There are two ways to get an answer out of the model. **Greedy** decoding takes the most likely token at every step, so it always gives the same answer, the model's single best guess. **Sampling** draws each token at random according to the model's probabilities, so asking eight times can give eight different answers. How spread out those draws are is set by the **temperature** $$T$$. At $$T = 1$$, used throughout this post, the model's probabilities are used exactly as they are. The demo's sampler applies nothing else, no top-k or top-p cutoff, so the probabilities GRPO computes later are exactly the ones the answers were drawn from.

How good is the model before any training? From [`the_baseline`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py), on the 200 test problems, greedy once and sampled eight times each:

<!-- RECEIPT: s2a -->
```text
  decoding          answers  right  format-valid
  ------------------------------------------------
  greedy                200  37.5%         91.0%
  sampled, T = 1.0     1600  34.3%         86.8%

  problems right in >=1 of 8 samples 117/200 (58.5%)
    of the problems greedy gets wrong 44/125 (35.2%)
```

Greedy gets 37.5% right, and 91.0% of its answers are lone integers, so the instruct model mostly does as it's told about format. Sampled answers are a little worse on both counts, 34.3% and 86.8%, which is what you'd expect from drawing less likely tokens some of the time.

**117 of the 200 problems were answered right in at least one of the eight samples**, and of the 125 problems greedy gets wrong, 44 had a right answer somewhere among the samples. The model can produce the right product for many problems it doesn't usually give. That's what RL works with: it can only raise the probability of answers the model actually writes. A problem the model never gets right in any sample gives it nothing to reinforce. DeepSeekMath's authors saw the same limit at scale: RL improved how often a majority vote over $$K$$ samples was right, but not how often at least one of $$K$$ samples was ([Shao et al., Figure 7](https://arxiv.org/abs/2402.03300)).

The sampled answers that weren't lone integers show the other way to score 0, from [`the_baseline`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py):

<!-- RECEIPT: s2b -->
```text
  Sampled answers that are not a lone integer, first three:
    "35 x 11 = 385"
    "35 x 11 = 385"
    "35 x 11 = 385"
```

All three are the same problem, because a problem's eight samples sit next to each other, and all three are right: 35 × 11 is 385. They score 0 anyway, because the answer was asked to be "just the number". The reward checks exactly what the program checks, and nothing else. Keep 35 × 11 in mind for [§7](#what-it-learned).

---

### 3. From rewards to a gradient {#policy-gradients}

SFT has an obvious gradient: raise the log-prob of the answer you were given. With a reward there's no given answer, only scores for the answers the model wrote. This section turns those scores into an update, and it needs a little vocabulary first.

In RL terms, the model is the **policy**: the rule that picks what to do. Writing an answer is the **action**. The **reward** is the score the action gets. In general an agent can collect rewards over many steps, and the total is called the **return**; here there's one reward per answer, at the end, so the return is just that reward. The symbols:

| symbol | means | here |
|---|---|---|
| $$\theta$$ | the trainable weights | the LoRA adapters' weights |
| $$q$$ | a question | "What is 23 × 47? ..." |
| $$o$$ | one answer the model wrote (its output) | `1081` and the end-of-turn token |
| $$\pi_\theta(o \mid q)$$ | the probability the policy gives answer $$o$$ to question $$q$$ | 0.0396 for `1081` |
| $$\log \pi_\theta(o \mid q)$$ | that answer's sequence log-prob, §1's sum | −3.2286 |
| $$r$$ | the reward the answer got | 1 or 0 |
| $$b$$ | a **baseline**: a reference score to compare $$r$$ against, chosen without looking at which answer was written | §4 uses the group's average |
| $$\alpha$$ | the learning rate, how big each nudge is | |
| $$\nabla_\theta$$ | the gradient with respect to the weights: which way to nudge each one to raise what follows | |

The update that comes out is due to [Williams (1992)](https://link.springer.com/article/10.1007/BF00992696), who called it **REINFORCE**, an acronym he spells out as "REward Increment = Nonnegative Factor times Offset Reinforcement times Characteristic Eligibility". In the notation used today:

$$\Delta\theta = \alpha\,(r - b)\,\nabla_\theta \log \pi_\theta(o \mid q)$$

Read it right to left. $$\nabla_\theta \log \pi_\theta(o \mid q)$$ is exactly the direction SFT would step in if someone had handed it $$o$$ as the answer: the direction that makes $$o$$ more likely. $$(r - b)$$ decides how far to step, and which way. An answer that scored above the baseline is made more likely, in proportion to how far above; an answer below it is made less likely. With $$r - b = 1$$ for every answer, this would be SFT on the model's own answers.

Williams wrote it per weight, as $$\Delta w_{ij} = \alpha_{ij}(r - b_{ij})e_{ij}$$, where $$e_{ij} = \partial \ln g_i / \partial w_{ij}$$, the derivative of the log of the probability $$g_i$$ that unit $$i$$ gave its output, is what he called the "characteristic eligibility". It's the same quantity as $$\nabla_\theta \log \pi_\theta$$, one weight at a time.

**What Williams proved, and what he didn't.** His Theorem 1 says that, on average over the answers the policy might write, this update points in the same direction as the gradient of the expected reward: $$E\{\Delta W \mid W\} = \alpha\,\nabla_W E\{r \mid W\}$$. So each update is a noisy estimate of the direction that raises the average score, and in his words, $$(r - b_{ij})\,\partial \ln g_i / \partial w_{ij}$$ "represents an unbiased estimate". The condition is that the baseline is "conditionally independent of" the action, which is why $$b$$ mustn't depend on which answer was written. Any such baseline keeps the estimate unbiased, including $$b = 0$$, and Williams says plainly that his results "offer no basis for comparing various choices of reinforcement baseline".

So why have one? Look at what $$b = 0$$ does with a reward of 1 or 0. A wrong answer has $$r - b = 0$$ and gets no update at all; only right answers move, always up. A question the model already gets right every time still pushes all its answers up, though there's nothing left to learn. With $$b$$ set to how well the model usually does on that question, a right answer to a hard question counts for a lot, a right answer to an easy one for little, and wrong answers are pushed down. The difference $$r - b$$, how much better an answer did than expected, is called its **advantage**. The update in the paper that introduced PPO ([Schulman et al.](https://arxiv.org/abs/1707.06347), Equation 1), the method [§4](#groups-and-advantages) describes, is this one with an estimated advantage $$\hat A$$ in place of $$r - b$$, and the rest of this post is about where that estimate comes from.

---

### 4. Groups instead of a value model {#groups-and-advantages}

The baseline should be "how well the model usually does on this question". Getting that number is the expensive part.

**PPO's answer: a second network.** **PPO** (Proximal Policy Optimization) is the method most language-model RL used before GRPO, including RLHF ([§8](#where-rlhf-fits)). It trains a **value model**: a second network that reads the question and the answer so far and predicts the reward to come. Its prediction is the baseline. That means training and holding a second large model next to the first, which DeepSeekMath's authors call "a substantial memory and computational burden" ([Shao et al., §4.1.1](https://arxiv.org/abs/2402.03300)).

**GRPO's answer: ask the question several times.** **GRPO** (Group Relative Policy Optimization) was introduced in the same DeepSeekMath paper. Instead of predicting how well the model usually does on a question, it measures it: sample a **group** of $$G$$ answers to the same question, score them all, and use the group's own average as the baseline. Each answer's advantage is its reward compared with the rest of its group, divided by the group's spread (§4.1.2 of the paper):

$$\hat A_i = \frac{r_i - \operatorname{mean}(r_1, \ldots, r_G)}{\operatorname{std}(r_1, \ldots, r_G) + 0.0001}$$

Here $$r_i$$ is the reward of answer $$i$$, and std is the **standard deviation**, a measure of how spread out the rewards are; the demo uses the sample standard deviation, which divides by $$G - 1$$ rather than $$G$$. The 0.0001 isn't in the paper's formula. It's there so a group whose rewards are all equal divides by something other than zero. Every token of answer $$i$$ shares the one advantage $$\hat A_i$$. In [`group_advantages`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py), with one row per group:

```python
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)
```

One thing to keep straight about history: DeepSeekMath introduced GRPO with a learned reward model doing the scoring. Scoring with a program that checks the answer came later, with [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (§2.2), and that's the version this demo builds.

**Three groups.** The demo first estimated each training question's **pass rate**, the share of its sampled answers that are right, then picked the question with the lowest, the one nearest 50%, and the highest, and sampled a group of eight answers to each. From [`groups_and_advantages`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py), the hardest:

<!-- RECEIPT: s3a -->
```text
  31 x 51 = 1581   (pass rate estimate 0.00)
  #  response  reward  advantage
  --------------------------------
  1    "1641"       0     +0.000
  2    "1501"       0     +0.000
  3    "1601"       0     +0.000
  4    "1601"       0     +0.000
  5    "1661"       0     +0.000
  6    "1591"       0     +0.000
  7    "1691"       0     +0.000
  8    "1661"       0     +0.000
  mean 0.000   std 0.000   (r - mean) / (std + 0.0001)
```

Every answer is a near miss, all eight score 0, the mean is 0 and so is the spread, and every advantage is exactly zero. The middle question:

<!-- RECEIPT: s3b -->
```text
  46 x 32 = 1472   (pass rate estimate 0.50)
  #    response  reward  advantage
  ----------------------------------
  1  "New York"       0     -0.935
  2      "1352"       0     -0.935
  3      "1472"       1     +0.935
  4      "1452"       0     -0.935
  5      "1472"       1     +0.935
  6      "1452"       0     -0.935
  7      "1472"       1     +0.935
  8      "1472"       1     +0.935
  mean 0.500   std 0.535   (r - mean) / (std + 0.0001)
```

Four right, four wrong. The mean is 0.500, the sample standard deviation is 0.535, and each answer's advantage is ±0.935: the four `1472`s are pushed up by the same amount the four wrong answers are pushed down. `New York` gets the same −0.935 as `1452`. The reward is 0 for both, and the advantage can't tell a near miss from nonsense. And the easiest:

<!-- RECEIPT: s3c -->
```text
  12 x 24 = 288   (pass rate estimate 1.00)
  #  response  reward  advantage
  --------------------------------
  1     "288"       1     +0.000
  2     "288"       1     +0.000
  3     "288"       1     +0.000
  4     "288"       1     +0.000
  5     "288"       1     +0.000
  6     "288"       1     +0.000
  7     "288"       1     +0.000
  8     "288"       1     +0.000
  mean 1.000   std 0.000   (r - mean) / (std + 0.0001)
```

All right, all zero again. A question that's always right and a question that's never right teach the model exactly the same amount, nothing. In both, every answer is exactly as good as the group's average, so there's nothing to prefer.

**How often does a group teach nothing?** For the 256 training questions, the demo sampled 16 answers each:

<!-- RECEIPT: s3d -->
```text
  Pass rates of the 256 train questions (16 samples each):
    never right                      97
    always right                     25
    in between                       134
    mean pass rate                   0.326
```

97 questions were never answered right in 16 tries and 25 always were. A group of answers to one of those questions will almost always be all-equal. Then, cutting another 16 fresh samples per question into groups of different sizes and counting the groups whose rewards are all equal, from [`groups_and_advantages`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py):

<!-- RECEIPT: s3e -->
```text
  G   groups  measured  analytic
  --------------------------------
  2     2048     0.854     0.856
  4     1024     0.733     0.738
  8      512     0.611     0.635
  16     256     0.492     0.553

  measured: 16 fresh samples per question, cut into groups of G.
  analytic: mean over questions of p^G + (1-p)^G, with each p
  estimated from 16 other samples. p^G + (1-p)^G is convex in p,
  so plugging in an estimated p overstates it (Jensen's inequality).
```

With the demo's $$G = 8$$, **61.1% of groups carry no signal**. Bigger groups help: a question with a pass rate of 20% is unlikely to give eight wrong answers in a row, and less likely still to give sixteen. But each answer costs a full generation, so a bigger group buys fewer wasted questions with more samples per question.

The **measured** column counts real groups. The **analytic** column is what you'd predict from the pass rates alone: if a question is right with probability $$p$$ and the $$G$$ answers are independent, all $$G$$ are right with probability $$p^G$$ and all wrong with probability $$(1-p)^G$$, so the group is all-equal with probability $$p^G + (1-p)^G$$, averaged over the questions. It overshoots a little, more for bigger $$G$$, and the receipt says why. Each $$p$$ is itself estimated from 16 samples, and because $$p^G + (1-p)^G$$ curves upward on both sides, an estimate that's wrong in either direction raises it on average. That's **Jensen's inequality**, and it's why the prediction is a little too high.

The field has two responses to this, both with a direct line to this table. [DAPO](https://arxiv.org/abs/2503.14476) (§3.2) uses **dynamic sampling**: it filters out groups whose answers are all right or all wrong and keeps sampling until a batch is full of groups that teach something, because "A zero advantage results in zero policy gradients". [Dr. GRPO](https://arxiv.org/abs/2503.20783) (§3.1) points at the division by the standard deviation. A group with one right answer in eight has a small spread, so dividing by it makes that one answer's advantage large. Questions that are nearly always right or nearly always wrong get weighted more than questions in the middle, a **difficulty bias**, and Dr. GRPO drops the division (§3.2). This demo keeps it, to match DeepSeekMath.

---

### 5. One update by hand {#one-update}

With advantages in hand, the update is the REINFORCE step of [§3](#policy-gradients) with two safety features from PPO bolted on. This section takes one group, the 46 × 32 group from [§4](#groups-and-advantages), and updates on it by hand.

**Reusing answers, and the ratio.** Sampling is the slow part, so PPO and GRPO allow several gradient steps, called **inner steps**, on the same batch of sampled answers. After the first step, though, the policy being trained is no longer the one that wrote the answers. The **probability ratio** tracks how far it's moved, token by token: the probability the current policy gives a token, divided by the probability the policy that sampled it gave it. A ratio of 1 means no change. On the first inner step it's exactly 1 everywhere, because the two policies are the same.

**Clipping.** PPO limits how far one batch can push a token's ratio. Its objective (Equation 7 of the [PPO paper](https://arxiv.org/abs/1707.06347)) takes the smaller of two terms: the ratio times the advantage, and the same with the ratio clipped to the range $$1 - \epsilon$$ to $$1 + \epsilon$$, "say, ϵ = 0.2". Once a token's ratio has moved past the clip in the direction the advantage wants, for example a right answer's token already more than 1.2 times as likely, the clipped term wins, it's constant, and that token stops contributing to the gradient. The paper calls the result "a lower bound (i.e., a pessimistic bound)" on the unclipped objective.

**The KL term.** **KL divergence** measures how different two probability distributions are; it's zero when they're identical and grows as they drift apart. GRPO keeps the policy close to a frozen **reference model**, here the starting model, by subtracting $$\beta$$ times an estimate of the KL divergence between them. Where that penalty goes differs by method. PPO as used for RLHF subtracts it from the reward, token by token ([§8](#where-rlhf-fits)); GRPO, in DeepSeekMath's words, "instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL ... to the loss".

The estimate is the one [Schulman's note on approximating KL divergence](http://joschu.net/blog/kl-approx.html) calls $$k_3$$. Per token, write $$d$$ for the reference model's log-prob of the token minus the policy's. Then the estimate is $$e^{d} - d - 1$$. It's zero when the two agree ($$d = 0$$), and never negative, since $$e^{d} \ge 1 + d$$ for any $$d$$; Schulman's note shows that, for tokens sampled from the policy, its average is the true KL divergence. The receipt below calls this quantity `r`; in this post $$r$$ is the reward, so I've written it as $$d$$. DeepSeekMath uses this estimator too (its Equation 4), and the demo runs DeepSeekMath's $$\beta = 0.04$$.

The demo's reference model costs no extra memory. The policy is Qwen plus LoRA adapters, and the reference is the same network with the adapters' contribution multiplied by zero, which is the starting model bit for bit ([`adapters_off`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py)). That's one reason a 0.5B model and its reference fit on a laptop.

**The whole objective.** The symbols, for answer $$i$$ of a group and its token $$t$$:

| symbol | means |
|---|---|
| $$G$$ | answers per group, 8 |
| $$o_i$$, $$\lvert o_i \rvert$$ | answer $$i$$, and its number of tokens, the end-of-turn token included |
| $$\pi_\theta$$, $$\pi_{\text{old}}$$, $$\pi_{\text{ref}}$$ | the policy being trained, the policy that sampled the answers, the frozen reference |
| $$\rho_{i,t}$$ | the probability ratio of token $$t$$: $$\pi_\theta$$'s probability for it divided by $$\pi_{\text{old}}$$'s |
| $$\hat A_i$$ | answer $$i$$'s advantage, from §4, shared by all its tokens |
| $$\epsilon$$ | the clip range, 0.2 |
| $$d_{i,t}$$ | $$\log \pi_{\text{ref}} - \log \pi_\theta$$ for token $$t$$ |
| $$\beta$$ | the weight on the KL term, 0.04 |

The demo maximizes DeepSeekMath's Equation 3, written in these symbols:

$$\mathcal{J}(\theta) = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{\lvert o_i\rvert}\sum_{t=1}^{\lvert o_i\rvert}\Big(\min\big[\rho_{i,t}\hat A_i,\ \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\,\hat A_i\big] - \beta\,\big(e^{d_{i,t}} - d_{i,t} - 1\big)\Big)$$

Working inward: each token gets its clipped term minus its KL penalty; those are averaged over the answer's tokens; the answers' averages are averaged over the group. The loss is $$-\mathcal{J}$$. In code, the core of [`grpo_objective`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py), whose inputs are each token's log-prob under the three policies:

```python
    adv = advantages.unsqueeze(1)
    ratio = torch.exp(logp - old_logp)
    clipped_ratio = ratio.clamp(1 - clip, 1 + clip)
    surrogate = torch.minimum(ratio * adv, clipped_ratio * adv)
    log_ref_ratio = ref_logp - logp
    kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1
    per_answer = masked_mean(surrogate - beta * kl, mask)
    loss = -per_answer.mean()
```

The clipped term is called the **surrogate**, because it stands in for the expected reward: its gradient is the policy-gradient step, but it isn't the reward itself.

Other papers write GRPO differently. DeepSeek-R1 computes one ratio per answer rather than per token, and DAPO and Dr. GRPO change the averaging and drop the KL term. The formula above is the one this demo runs.

**Two inner steps on one group**, from [`one_update`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py), starting from fresh adapters, whose second matrix is all zeros, so the policy is exactly the starting model:

<!-- RECEIPT: s4a -->
```text
  The group for 46 x 32 from section 3, adapters fresh, lr 3e-05.
  2 gradient steps on the same 8 answers (GRPO's inner steps).

  step  ratio min  ratio max  clipped  surrogate         KL      loss
  ---------------------------------------------------------------------
  1        1.0000     1.0000     0.0%    +0.0000  -1.44e-08   +0.0000
  2        0.0001     1.4386    28.9%    +0.0580   4.52e+02  +18.0031

  Step 1: the policy is the sampler, so every ratio is exactly 1, and
  the adapters start at zero, so the KL is 0 (to float rounding).
  Step 2 scores the same answers again. Where a token's log-prob fell
  below the reference's, the KL term exp(r) - r - 1, r = ref - logp,
  grows exponentially in r; where a ratio crossed the clip in the
  advantage's direction, that token's surrogate gradient is zero.
```

Step 1 is the clean case. The policy is the sampler, so every ratio is exactly 1, nothing is clipped, and the policy is also the reference, so the KL is zero (the −1.44e-08 is float rounding). The surrogate's value is zero too, because the advantages of a group always add up to zero, but its gradient isn't: it's §3's REINFORCE step, with the group's advantages as weights.

Step 2 reuses the same eight answers, and shows why PPO-style methods need their safety features. One step at this learning rate moved some tokens a long way. The smallest ratio is 0.0001: some token now has a ten-thousandth of the probability it had when it was sampled, and the wrong answers are the ones being pushed down. The largest is 1.4386, a right answer's token that's become more likely than the clip allows. 28.9% of answer tokens are past the clip in their advantage's direction, and contribute no gradient from the surrogate.

The KL estimate has gone from zero to 4.52e+02. Since the starting policy was also the reference, that token's $$d$$ is the log of about ten thousand, and $$e^{d}$$ grows exponentially: one token is enough to make the average enormous. The loss, +18.0031, is now almost all KL penalty, 0.04 times the KL, against a surrogate of +0.0580. And the KL estimate's guarantee no longer holds here: it's an unbiased estimate for tokens sampled from the current policy, and these were sampled from the policy of one step ago. This is why the training runs in [§6](#training) take one inner step per batch, as DeepSeekMath's own runs did ("a single update following each exploration stage", §4.2). With one inner step, every ratio is 1, so the clip never binds and the surrogate's gradient is exactly REINFORCE with group advantages, averaged over each answer's tokens.

What did the two steps do to each answer? Each answer's sequence log-prob before and after, from [`one_update`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py):

<!-- RECEIPT: s4b -->
```text
  response    reward    adv  log-prob    after  change
  ------------------------------------------------------
  "New York"       0  -0.94   -15.361  -16.150  -0.789
  "1352"           0  -0.94    -3.002   -7.167  -4.165
  "1472"           1  +0.94    -0.593   -0.049  +0.544
  "1452"           0  -0.94    -2.984   -6.749  -3.765
  "1472"           1  +0.94    -0.593   -0.049  +0.544
  "1452"           0  -0.94    -2.984   -6.749  -3.765
  "1472"           1  +0.94    -0.593   -0.049  +0.544
  "1472"           1  +0.94    -0.593   -0.049  +0.544

  right answers whose log-prob rose  4 of 4
  wrong answers whose log-prob fell  4 of 4
```

Each of the four copies of the right answer rose from −0.593 to −0.049, and every wrong answer fell. There was no value network and no labelled answer, only eight samples and their scores, and the group's own average decided which way each one moved. `1352` and `1452` fell by around four, while `New York`, already the least likely answer by far, fell by less than one.

---

### 6. Training it {#training}

Training repeats §5 many times, on fresh answers each step. The settings, from [`training`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py):

<!-- RECEIPT: s5a -->
```text
  policy                             the model + LoRA, r=8, 7 projections
  reference (for the KL)             the same model, adapters off
  per step                           8 questions x G=8 = 64 answers
  inner steps per batch              1
  KL beta                            0.04
  clip epsilon                       0.2 (inactive at 1 inner step)
  optimizer                          AdamW, lr 3e-05, no weight decay
  steps per run                      60 (480 draws from 256 questions)
  sampling                           T = 1.0, at most 24 tokens
```

Each step samples eight training questions, eight answers to each, scores them, computes advantages within each group, and takes one gradient step with **AdamW**, the optimizer posts 6 and 7 used, on the LoRA adapters. LoRA is post 6's: rank 8, on all seven projection matrices of every layer. The reference never changes. Sixty steps draw 480 questions from a pool of 256, so the model sees each training question only a few times.

**The control.** A rising score after training could come from the reward, or from something else in the loop: sampling, the KL term, the optimizer, or just nudging the weights at all. To separate them, the demo runs a **control** that does everything the same, except that before computing advantages it shuffles the batch's 64 rewards among its 64 answers ([`grpo_train`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py)). Each step keeps the same number of 1s and the same average reward; only which answer gets which reward is random. If the control improves too, the reward wasn't what taught it.

**During training.** Three runs, two real seeds and the control, each logged every ten steps:

<!-- RECEIPT: s5b -->
```text
  GRPO, seed 0

  step  mean reward  no-signal groups  KL to ref  length
  --------------------------------------------------------
  10          0.498             65.0%     0.1767    4.97
  20          0.503             71.2%     0.2318    4.81
  30          0.456             63.7%     0.2335    4.95
  40          0.464             50.0%     0.3226    4.97
  50          0.519             68.8%     0.5564    4.95
  60          0.537             73.8%     0.2401    4.90

  GRPO, seed 1

  step  mean reward  no-signal groups  KL to ref  length
  --------------------------------------------------------
  10          0.425             55.0%     0.1995    5.08
  20          0.562             56.2%     0.2615    4.90
  30          0.575             65.0%     0.2693    4.71
  40          0.528             72.5%     0.3355    5.07
  50          0.586             60.0%     0.2506    4.84
  60          0.591             66.2%     0.2588    4.94

  control: shuffled rewards, seed 0

  step  mean reward  no-signal groups  KL to ref  length
  --------------------------------------------------------
  10          0.391              3.8%     0.0493    5.72
  20          0.362              2.5%     0.0734    5.42
  30          0.277             16.2%     0.2406    5.45
  40          0.166             28.7%     0.3828    6.15
  50          0.159             27.5%     0.4323    7.05
  60          0.183             27.5%     0.4983    7.63

  Each row averages the 10 steps up to it. no-signal groups:
  all G training rewards equal (for the control, the shuffled ones).
  length: answer tokens, <|im_end|> included.
```

Read the columns one at a time.

- **mean reward** is the true reward of the sampled answers, averaged over ten steps. For the real runs it doesn't climb much once the log begins: seed 0 goes from 0.498 to 0.537 with a dip to 0.456 on the way, seed 1 from 0.425 to 0.591. Each row covers 80 questions drawn at random, some easy and some hard, so it's noisy. Both first rows are already well above the 0.326 mean pass rate the untrained model had on these same questions in [§4](#groups-and-advantages), which suggests much of the change came in the first ten steps, which the first row averages over. The control's true reward falls, to 0.183.
- **no-signal groups** is the share of groups whose training rewards were all equal. For the real runs it's between 50.0% and 73.8%: most of each step's 64 answers taught nothing, just as §4 predicted. The control's is much lower at the start, because shuffling scatters the 1s across questions and most groups end up with a mix. It has plenty of nonzero advantages, and none of them mean anything.
- **KL to ref** is the per-token KL estimate against the starting model, before each step's update. All three runs drift away from the reference, and the control ends further from it than either real run, so it isn't sitting still. It's moving, in no useful direction.
- **length** is the answer length in tokens. The real runs stay around five tokens, four digits and `<|im_end|>`; the control's answers grow from 5.72 to 7.63 tokens.

**On problems it never trained on.** The test that matters is the 200 held-out problems, answered greedily after training:

<!-- RECEIPT: s5c -->
```text
  model                              right  format-valid  change
  ----------------------------------------------------------------
  untrained                          37.5%         91.0%
  GRPO, seed 0                       50.5%         98.0%   +13.0
  GRPO, seed 1                       59.0%         99.0%   +21.5
  control: shuffled rewards, seed 0  33.5%         89.0%    -4.0
```

Both real runs improved, seed 0 by 13.0 points and seed 1 by 21.5, and both nearly stopped writing anything but lone integers. The control, with the same loop and the same number of 1s per batch, got worse: 4.0 points below the untrained model, and slightly worse at format too. So the improvement came from the reward's information about which answers were right, not from the training loop itself.

How big the improvement is, two seeds can't pin down. The runs differ only in their seed, and one ended at 50.5% while the other ended at 59.0%. With two runs there's no way to put an error bar on either number, so the claim this receipt supports is that both seeds improved a lot and the control didn't, and not that GRPO adds a particular number of points. Seed 0's training reward, which barely moved while its test score rose 13.0 points, is also a warning about watching the training curve alone, which the [sidebar](#sidebar) comes back to.

---

### 7. What it actually learned {#what-it-learned}

A test score going up could mean two quite different things here. The model could have learned to say its answer as a lone number, which the reward demands and the untrained model sometimes didn't do. Or it could have got better at the multiplication itself. Both earn the same reward.

First, a few answers. These are the first six test problems whose greedy answer went from right to wrong or wrong to right with seed 0's training, from [`what_changed`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py):

<!-- RECEIPT: s6a -->
```text
  problem  answer         untrained  trained
  --------------------------------------------
  35 x 11     385             "435"    "385"
  26 x 83    2158  "26 x 83 = 2118"   "2158"
  94 x 69    6486            "6386"   "6486"
  17 x 90    1530  "17 x 90 = 1530"   "1530"
  20 x 94    1880            "1880"   "1860"
  56 x 72    4032            "3952"   "4032"
```

They show every kind of change. 35 × 11 went from the wrong number `435` to `385`, and that's the same problem whose samples in [§2](#a-checkable-task) had the right product inside a sentence. 94 × 69 and 56 × 72 are wrong numbers made right, too. 17 × 90 was right all along, but in a sentence, so it scored 0, and now it's just `1530`. 26 × 83 was a sentence with the wrong product in it and is now the right number. And 20 × 94 went the other way: right before training, wrong after.

To count them, the demo sorts every test problem that changed:

- **arithmetic fixed**: the answer was a wrong lone integer, and became the right one.
- **format fixed**: the answer wasn't a lone integer, and became the right one. This includes cases like 26 × 83, where the product inside the sentence was wrong too.
- **broken**: the answer was right, and became wrong.

From [`what_changed`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d08_rewards.py), for all three runs:

<!-- RECEIPT: s6b -->
```text
  wrong -> right    seed 0  seed 1  shuffled, seed 0
  ----------------------------------------------------
  arithmetic fixed      27      42                 8
  format fixed           5       5                 0
  broken                 6       4                16
  net                   26      43                -8

  arithmetic fixed: a wrong lone integer became the right one.
  format fixed: not a lone integer became the right one.
  broken: right became wrong. net: problems gained.
```

Most of the gain is arithmetic. For seed 0, 27 problems went from a wrong number to the right one and 5 from a sentence to the right number, and 6 that were right broke, for a net gain of 26 problems, the +13.0 points of [§6](#training). Seed 1 fixed more arithmetic, 42, and the same 5 formats. Format couldn't have contributed much either way: 91.0% of the untrained model's greedy answers were already lone integers, so there was little format left to fix.

The control is the noise floor. It fixed 8 problems and broke 16, with a reward that carried no information. So changing a model's weights by this much moves a couple of dozen answers whether or not it learns anything, and a seed's 27 or 42 arithmetic fixes should be read against the control's 8, not against zero.

What this doesn't show is whether the model learned anything new about multiplication or got better at giving answers it could already sometimes produce. [§2](#a-checkable-task) found 44 problems that greedy got wrong but some sample got right, and RL raises the probability of answers the model samples. The demo doesn't check whether the problems it fixed are those same ones, so I can't tell the two stories apart from these receipts.

---

### 8. Where RLHF fits {#where-rlhf-fits}

Multiplication has a program for a judge. Most of what people want from a chat model doesn't: whether a reply is helpful, honest, or well put has no one-line check. **RLHF**, reinforcement learning from human feedback, runs the same loop as this post with a judge that's itself a model, trained on people's preferences. There's no demo for this section; it describes how the best-known recipe fits the pieces above.

[InstructGPT](https://arxiv.org/abs/2203.02155) (Ouyang et al.) laid out the recipe in three steps (§3.1):

1. **Supervised fine-tuning.** People write good replies to prompts, and the model is fine-tuned on them, exactly like the earlier posts.
2. **A reward model.** For a prompt, the model writes several replies, and people rank them from best to worst. A **reward model**, a language model with one number as its output, is trained to give preferred replies higher scores. InstructGPT ranked between 4 and 9 replies per prompt and turned each ranking into every pair it contains. For a pair where people preferred reply $$y_w$$ to reply $$y_l$$ for prompt $$x$$, the loss is $$-\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$$, averaged over pairs, where $$r_\phi$$ is the reward model's score and $$\sigma$$ is the sigmoid, which turns a difference in scores into a probability that $$y_w$$ is the better one (§3.5, Equation 1). InstructGPT used a 6B reward model, even for its largest policy, because "175B RM training could be unstable".
3. **RL against the reward model** with PPO. The reward model scores each reply the policy writes, in place of this post's program.

The KL term is there too, in the other place [§5](#one-update) mentioned. InstructGPT adds "a per-token KL penalty from the SFT model at each token" to the reward, so the reward PPO maximizes is the reward model's score minus $$\beta \log\big(\pi^{RL}(y \mid x) / \pi^{SFT}(y \mid x)\big)$$ (Equation 2), with $$\beta = 0.02$$. And it uses PPO's value model, initialized from the reward model.

So RLHF is §3 to §5 with a learned judge and a learned baseline. The judge is the risky part. A program that checks multiplication can't be talked into giving a wrong product a 1, but a reward model is a network trained on a finite set of comparisons, and a policy optimized hard against it can find replies it scores highly that people wouldn't, a failure called **reward hacking**. DeepSeek-R1's authors give it as their reason for scoring with rules (§2.2): "we abstain from applying neural reward models ... susceptible to reward hacking during large-scale reinforcement learning". R1's rules check whether the final answer is right and whether the reasoning sits in the expected format, which is this post's reward with a format check added.

A program can be gamed too, just more narrowly: it rewards exactly what it checks. Here that included the format, and [§7](#what-it-learned) showed the model learning it.

---

### Sidebar: questions this post can answer {#sidebar}

Some questions this material gets asked, and what this post's numbers say about them.

> **"The training reward barely moved. Did RL do anything?"**

Check the held-out score before deciding. Here the logged reward for seed 0 went from 0.498 to 0.537 (seed 1's rose more, from 0.425 to 0.591), and seed 0's greedy accuracy on 200 unseen problems went from 37.5% to 50.5%. The training reward is measured on sampled answers at $$T = 1$$, on whichever eight questions a step drew, and averaged over steps, so it's noisy and it measures something different from greedy test accuracy. The link can break the other way too: [ToolRL](https://arxiv.org/abs/2504.13958) (Qian et al.) found that "higher training rewards do not necessarily translate to better generalization". A reward curve is a training diagnostic, and the evaluation needs its own held-out set.

> **"We have the right answers. Why not just fine-tune on them?"**

For multiplication you could, and that's why it's a good task for checking the machinery: the reward can't be wrong. RL is for when you can check an answer but can't write a good one down, or when there are many right answers and you don't want to pick one. The two methods are also closer than they look. [§3](#policy-gradients) showed the policy-gradient step is SFT's step on the model's own answers, weighted by advantage, and with one inner step that's what this demo's training step is, plus the KL term.

> **"61% of our groups give zero advantage. What do we do?"**

Use bigger groups, or spend the samples where they count. §4's table shows the trade: going from 8 to 16 answers per group drops the no-signal share from 0.611 to 0.492, at twice the samples per question. DAPO's dynamic sampling throws away all-right and all-wrong groups and samples more questions instead. And a question the model never gets right can't be fixed by either: no group for it ever contains a right answer to reinforce ([§2](#a-checkable-task)).

> **"Seed 0 gained 13 points and seed 1 gained 21.5. Which goes in the report?"**

Both, and the control beside them. Two seeds that differ this much say the result is real in direction but not in size: the control, with the same loop, lost 4.0 points. Reporting only the better seed, or only the average, hides how much a single run varies. Before claiming a number, run more seeds, and compare every run against a control trained the same way.

### What's next {#whats-next}

This post trained a model on a score for a task that ends in one number. The posts after it use the same loop for something more useful: teaching a small model to call tools, where a program can check whether the call was right. Post 9 starts from demonstrations, training on examples of good tool calls the way the earlier posts trained on answers, so that the model writes calls often enough for a reward to have something to reinforce.

### Appendix: all notation {#appendix-all-notation}

The terms this post leans on; anything not listed is defined where it first appears. [Post 6's appendix](/posts/llm-architectures-lora/#appendix-all-notation) covers LoRA's notation.

| Term | Means | In this post |
| --- | --- | --- |
| token log-prob | the log of the probability the model gave a token, given everything before it | ≤ 0 |
| sequence log-prob | the sum of an answer's token log-probs: the log of the probability of the whole answer | −3.2286 for `1081` |
| cross-entropy | the mean of the negative token log-probs; the SFT loss | 0.6457 for `1081` |
| teacher forcing | scoring every position of a known answer in one pass | §1 |
| masking | reading positions as context without scoring them | the prompt's 45 tokens |
| SFT | supervised fine-tuning: raising the log-prob of answers someone wrote | posts 6 and 7 |
| greedy, sampling | always the most likely token; drawing each token at random by its probability | §2 |
| temperature $$T$$ | how spread out sampling is; at $$T = 1$$ the probabilities are used as they are | $$T = 1$$ |
| reward $$r$$ | the score an answer gets | 1 if the right lone integer, else 0 |
| format-valid | the answer is a lone integer, so it can be checked | `1081`, `1,081`, `1081.` |
| policy $$\pi_\theta$$ | the model, seen as a rule for choosing answers | Qwen2.5-0.5B-Instruct + LoRA |
| action, return | what the policy does; the total reward it collects | one answer; its one reward |
| baseline $$b$$ | a reference score that doesn't depend on which answer was written | the group's mean |
| advantage $$\hat A_i$$ | how much better an answer did than the baseline, here scaled by the group's spread | ±0.935 in §4's middle group |
| REINFORCE | Williams' update: step along $$(r - b)\,\nabla_\theta \log \pi_\theta$$ | §3 |
| value model | PPO's learned baseline: a second network predicting the reward to come | not used |
| group, $$G$$ | answers sampled for one question and compared with each other | 8 |
| pass rate | the share of a question's sampled answers that are right | §4 |
| inner steps | gradient steps on one batch of sampled answers | 1 in training, 2 in §5 |
| ratio $$\rho_{i,t}$$ | a token's probability under the current policy over its probability under the sampler | exactly 1 at the first inner step |
| clip, $$\epsilon$$ | PPO's limit on how far the ratio can usefully move | 0.2 |
| surrogate | the clipped term whose gradient is the policy-gradient step | §5 |
| KL divergence | how different two probability distributions are; 0 when equal | per token, against the reference |
| $$k_3$$, $$d_{i,t}$$ | Schulman's KL estimate $$e^{d} - d - 1$$, with $$d = \log \pi_{\text{ref}} - \log \pi_\theta$$ | §5 |
| $$\beta$$ | the weight on the KL term | 0.04 |
| reference model $$\pi_{\text{ref}}$$ | the frozen model the KL is measured against | the same network, adapters off |
| control | the same training with each batch's rewards shuffled among its answers | §6 |
| reward model | a network trained on people's rankings to score replies; RLHF's judge | §8 |
| reward hacking | a policy finding replies its judge scores highly that aren't actually good | §8 |

### References

- Williams, [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) (Machine Learning 1992) — REINFORCE, §3's update. §4 gives $$\Delta w_{ij} = \alpha_{ij}(r - b_{ij})e_{ij}$$ with $$e_{ij} = \partial \ln g_i / \partial w_{ij}$$, the "characteristic eligibility", and a baseline "conditionally independent of" the output; Theorem 1 shows the expected update follows the gradient of the expected reward, and $$(r - b_{ij})\,\partial \ln g_i / \partial w_{ij}$$ "represents an unbiased estimate". The paper makes no claim that a baseline reduces variance, and says its results "offer no basis for comparing various choices of reinforcement baseline". (Quotes checked against the author's preprint, whose page numbers differ from the journal's.)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017) — PPO. Equation 1 is the policy-gradient estimator with an advantage, cited in §3; Equation 7 is the clipped objective of §5, with "say, ϵ = 0.2" and the clip described as "a lower bound (i.e., a pessimistic bound)" (§3).
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (2022) — InstructGPT, §8's recipe: the three steps (§3.1), a 6B reward model trained on rankings of 4 to 9 replies with the pairwise loss of Equation 1, because "175B RM training could be unstable" (§3.5), and PPO with "a per-token KL penalty from the SFT model at each token" in the reward (Equation 2, $$\beta = 0.02$$ in Appendix C.4).
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) (2024) — introduces GRPO (§4.1), with a learned reward model. The value model is "a substantial memory and computational burden"; the group-normalized advantage (§4.1.2) and the objective of Equation 3 are what §4 and §5 implement; the KL is added "to the loss" rather than the reward, estimated as in Equation 4; training used $$\beta = 0.04$$, 64 answers per question, and "a single update following each exploration stage" (§4.2). Figure 7 is §2's majority-vote versus at-least-one result.
- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (Nature 2025) — GRPO with rule-based rewards for accuracy and format (§2.2), a per-answer ratio, and the reason given in §8: "we abstain from applying neural reward models ... susceptible to reward hacking during large-scale reinforcement learning".
- Schulman, [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) (2020) — the $$k_3$$ estimator of §5, $$(r - 1) - \log r$$ for a probability ratio $$r$$, unbiased and "guaranteed to be positive" (never negative).
- Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) (NeurIPS 2025) — dynamic sampling (§3.2), which keeps only groups with some right and some wrong answers because "A zero advantage results in zero policy gradients". Cited in §4.
- Liu et al., [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) (COLM 2025) — Dr. GRPO: the length bias of dividing by $$\lvert o_i \rvert$$ and the difficulty bias of dividing by the group's standard deviation (§3.1), and a version without either (§3.2). Cited in §4.
- Qian et al., [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958) (NeurIPS 2025) — GRPO for tool calls, scored against reference calls; quoted in the sidebar: "higher training rewards do not necessarily translate to better generalization".
- Qwen Team, [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) (2024) — the model family, including the instruction-tuned 0.5B used here (§1).
