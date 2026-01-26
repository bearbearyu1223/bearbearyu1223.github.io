---
layout: post
title: "Study Notes: Stanford CS336 Language Modeling from Scratch [14]"
categories: cs336
author:
- Han Yu
---

## A Beginner's Guide to Reinforcement Learning for Language Models

Recent breakthroughs in AI reasoning—like DeepSeek R1 and OpenAI's o1—have been powered by reinforcement learning (RL). But if you're new to RL, the math can feel intimidating. Terms like "policy gradients," "baselines," and "importance sampling" get thrown around, and the equations look like alphabet soup.

In this note, I am trying to break down the core concepts of RL for language models in plain English, with simple examples and step-by-step explanations of a few key formulas. 

*This guide is based on my study notes from Stanford CS336 and resources like OpenAI's Spinning Up in Deep RL and Nathan Lambert's RLHF Book.*

### Table of Contents
- [A Beginner's Guide to Reinforcement Learning for Language Models](#a-beginners-guide-to-reinforcement-learning-for-language-models)
  - [Table of Contents](#table-of-contents)
  - [The Big Picture: Training Dogs and Language Models](#the-big-picture-training-dogs-and-language-models)
  - [Part 1: Language Models as Policies](#part-1-language-models-as-policies)
    - [What is a Policy?](#what-is-a-policy)
    - [The Two Operations You Need](#the-two-operations-you-need)
  - [Part 2: Trajectories — Recording the Journey](#part-2-trajectories--recording-the-journey)
    - [What is a Trajectory?](#what-is-a-trajectory)
    - [A Concrete Example](#a-concrete-example)
  - [Part 3: Rewards and Returns — Measuring Success](#part-3-rewards-and-returns--measuring-success)
    - [The Reward Function](#the-reward-function)
    - [The Return: Adding Up Rewards](#the-return-adding-up-rewards)
    - [The Objective: Maximize Expected Return](#the-objective-maximize-expected-return)
  - [Part 4: The Policy Gradient (Vanilla REINFORCE)](#part-4-the-policy-gradient-vanilla-reinforce)
    - [The Key Equation](#the-key-equation)
    - [Symbol-by-Symbol Breakdown](#symbol-by-symbol-breakdown)
    - [The Log-Derivative Trick: Why the Math Works](#the-log-derivative-trick-why-the-math-works)
    - [Deriving the Policy Gradient Step-by-Step](#deriving-the-policy-gradient-step-by-step)
    - [Intuitive Summary](#intuitive-summary)
  - [Part 5: Baselines — Reducing the Noise](#part-5-baselines--reducing-the-noise)
    - [The Problem with Vanilla REINFORCE](#the-problem-with-vanilla-reinforce)
    - [The Solution: Subtract a Baseline](#the-solution-subtract-a-baseline)
    - [A Concrete Example](#a-concrete-example-1)
    - [Why Baselines Don't Add Bias](#why-baselines-dont-add-bias)
  - [Part 6: Off-Policy Learning — Reusing Old Data](#part-6-off-policy-learning--reusing-old-data)
    - [The Inefficiency of On-Policy Learning](#the-inefficiency-of-on-policy-learning)
    - [The Solution: Importance Sampling](#the-solution-importance-sampling)
    - [A Concrete Example](#a-concrete-example-2)
    - [The Catch: Don't Stray Too Far](#the-catch-dont-stray-too-far)
  - [Part 7: GRPO — Group Relative Policy Optimization](#part-7-grpo--group-relative-policy-optimization)
    - [The Core Idea: Compare Siblings](#the-core-idea-compare-siblings)
    - [Step 1: Sample Multiple Outputs Per Question](#step-1-sample-multiple-outputs-per-question)
    - [Step 2: Compute Group-Normalized Advantages](#step-2-compute-group-normalized-advantages)
    - [Step 3: The GRPO-Clip Objective](#step-3-the-grpo-clip-objective)
    - [Why Clipping Matters](#why-clipping-matters)
    - [The GRPO Training Loop Visualized](#the-grpo-training-loop-visualized)
    - [The Complete GRPO Algorithm](#the-complete-grpo-algorithm)
  - [Part 8: Putting It All Together](#part-8-putting-it-all-together)
    - [Summary Table](#summary-table)
    - [Key Equations at a Glance](#key-equations-at-a-glance)
    - [The PyTorch Connection](#the-pytorch-connection)

### The Big Picture: Training Dogs and Language Models

Before diving into equations, let's build intuition with an analogy.

**Imagine you're training a dog to do tricks.** You can't tell the dog exactly which muscles to move—you can only reward it when it does something good. Over time, the dog learns to repeat actions that led to treats and avoid actions that didn't.

Reinforcement learning for language models works the same way:

| Dog Training | LLM Training |
|--------------|--------------|
| Dog decides what action to take | LLM decides what token to generate |
| You give a treat (or not) | Reward function gives a score (0 or 1) |
| Dog repeats actions that got treats | LLM increases probability of tokens that led to rewards |

The key insight: **we don't tell the model what to generate (e.g., what is the groundtruth)—we just tell it whether its answer was good or bad, and it figures out the rest.**

### Part 1: Language Models as Policies

#### What is a Policy?

In RL terminology, a **policy** is just a decision-making strategy. For language models:

- **State** ($s_t$): The text generated so far (the context, or the prefix)
- **Action** ($a_t$): The next token to generate
- **Policy** ($\pi_\theta$): The probability distribution over possible next tokens

Your LLM is a policy! Given a text prefix, it outputs probabilities for each possible next token:

```
State:  "The capital of France is"
Policy: {"Paris": 0.85, "Lyon": 0.05, "the": 0.03, ...}
Action: Sample from this distribution → "Paris"
```

Mathematically, we write this as:

$$a_t \sim \pi_\theta(\cdot | s_t)$$

This reads: "action $a_t$ is sampled from the policy $\pi_\theta$ given state $s_t$."

#### The Two Operations You Need

To train a policy with RL, you only need two operations:

| Operation | What It Does | Example |
|-----------|--------------|---------|
| **Sampling** | Draw a token from the probability distribution | Pick "Paris" with 85% probability-the highest probability |
| **Scoring** | Compute the log-probability of a token | $\log \pi_\theta(\text{"Paris"} \mid s_t) = \log(0.85) \approx -0.16$ |

That's it! You don't need to know anything else about the model's internals.

### Part 2: Trajectories — Recording the Journey

#### What is a Trajectory?

A **trajectory** (also called an episode or rollout) is the complete sequence of states and actions from start to finish:

$$\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T)$$

Think of it like recording a chess game move-by-move—you capture everything that happened.

#### A Concrete Example

Let's trace a trajectory for a math problem:

**Prompt:** "What is 2+3? Think step by step."

| Timestep | State ($s_t$) | Action ($a_t$) |
|----------|---------------|----------------|
| 0 | "What is 2+3? Think step by step. <think>" | "I" |
| 1 | "... <think> I" | "need" |
| 2 | "... <think> I need" | "to" |
| 3 | "... <think> I need to" | "add" |
| ... | ... | ... |
| T | "... </think> <answer>" | "5" |
| T+1 | "... <answer> 5" | "</answer>" |

The trajectory ends when the model emits an end-of-text token (like `</answer>`) or hits a maximum length.

**Key observation:** In LLM-land, the "environment" is trivially deterministic. The next state is just the old state plus the new token:

$$s_{t+1} = s_t | a_t$$

(where $\|$ means concatenation)

### Part 3: Rewards and Returns — Measuring Success

#### The Reward Function

The **reward** $r_t = R(s_t, a_t)$ judges how good an action was. For RL on math problems, we typically use **sparse rewards**:

- **Intermediate steps:** $r_t = 0$ (no feedback until the end)
- **Final answer:** $r_T = 1$ if correct, $0$ if wrong

**Example:**

| Trajectory | Final Answer | Correct? | Reward |
|------------|--------------|----------|--------|
| "... <answer>5</answer>" | 5 | ✓ | 1 |
| "... <answer>6</answer>" | 6 | ✗ | 0 |

#### The Return: Adding Up Rewards

The **return** $R(\tau)$ is the total reward accumulated over a trajectory:

$$R(\tau) = \sum_{t=0}^{T} r_t$$

With sparse rewards, only the final step matters, so $R(\tau)$ equals the terminal reward (0 or 1).

#### The Objective: Maximize Expected Return

The goal of RL is to find policy parameters $\theta$ that maximize expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

In plain English: **"On average, how much reward does my policy get?"**

If $J(\theta) = 0.7$, that means your model solves 70% of problems correctly.

### Part 4: The Policy Gradient (Vanilla REINFORCE)

Now we get to the heart of RL: how do we actually improve the policy?

#### The Key Equation

The **Vanilla REINFORCE policy gradient** tells us how to update parameters:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$

This looks complex, but the intuition is simple: **increase the probability of actions that led to high rewards.**

#### Symbol-by-Symbol Breakdown

Let me explain every symbol in this equation:

| Symbol | Name | Meaning |
|--------|------|---------|
| $J(\theta)$ | Objective function | Expected total reward—the thing we want to maximize |
| $\theta$ | Parameters | All the weights in your language model (millions of numbers) |
| $\nabla_\theta J$ | Gradient | "Which direction should I nudge each parameter to increase J - the expected total reward?" |
| $\mathbb{E}[\cdot]$ | Expectation | Average over many samples |
| $\tau$ | Trajectory | One complete episode (e.g., prompt → response → end) |
| $\tau \sim \pi_\theta$ | Sampling | Generate trajectories by running the policy |
| $\sum_t$ | Sum over timesteps | Add up contributions from every token |
| $s_t$ | State | Text prefix at timestep $t$ |
| $a_t$ | Action | Token generated at timestep $t$ |
| $\pi_\theta(a_t \mid s_t)$ | Probability | How likely was this token given this context? |
| $\log \pi_\theta(a_t \mid s_t)$ | Log-probability | Same info, but in log space (more stable) |
| $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ | Score function* | Gradient of the log-probability; points in the direction that increases this token's probability |
| $R(\tau)$ | Return | Total reward for this trajectory (0 or 1) |

  **Note on terminology**: The name "score function*" comes from statistics, despite "score" sounding like a scalar, the score function is a vector pointing in the direction of steepest increase for the log-probability.

#### The Log-Derivative Trick: Why the Math Works

The magic behind policy gradients is a simple calculus identity:

$$\nabla_\theta P = P \cdot \nabla_\theta \log P$$

This comes from the chain rule for logarithms:

$$\frac{d}{d\theta} \log P = \frac{1}{P} \cdot \frac{d}{d\theta} P$$

Rearranging:

$$\frac{d}{d\theta} P = P \cdot \frac{d}{d\theta} \log P$$

**Why is this useful?** It lets us convert "gradient of an expectation" into "expectation of a gradient"—which we can estimate by sampling!

**Numerical example:**

Suppose $P(a) = 0.3$ is the probability of some action.

Direct gradient: $\nabla_\theta P = 1$ (some value)

Using the trick:
- $\nabla_\theta \log P = \nabla_\theta \log(0.3) = \frac{1}{0.3} \cdot \nabla_\theta P = 3.33 \cdot 1$
- $P \cdot \nabla_\theta \log P = 0.3 \times 3.33 = 1$ ✓

Same answer! The trick is just a rearrangement that makes computation easier.

#### Deriving the Policy Gradient Step-by-Step

Let's derive the REINFORCE equation from scratch.

**Step 1: Write out the expectation explicitly**

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_{\tau} P(\tau | \theta) \cdot R(\tau)$$

This sums over all possible trajectories, weighted by their probability.

**Step 2: Take the gradient**

$$\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau | \theta) \cdot R(\tau)$$

Note: $R(\tau)$ doesn't depend on $\theta$ (it's just "was the answer correct?")

**Step 3: Apply the log-derivative trick**

$$\nabla_\theta J(\theta)= \sum_{\tau} P(\tau | \theta) \cdot \nabla_\theta \log P(\tau | \theta) \cdot R(\tau)$$

**Step 4: Recognize this as an expectation**

$$\nabla_\theta J(\theta)= \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log P(\tau | \theta) \cdot R(\tau)\right]$$

**Step 5: Expand the trajectory probability**

A trajectory's probability is:

$$P(\tau | \theta) = \underbrace{\rho_0(s_0)}_{\text{initial prompt}} \cdot \prod_{t=0}^{T} \underbrace{P(s_{t+1}|s_t, a_t)}_{\text{environment}} \cdot \underbrace{\pi_\theta(a_t|s_t)}_{\text{policy}}$$

Taking the log:

$$\log P(\tau | \theta) = \log \rho_0(s_0) + \sum_t \log P(s_{t+1}|s_t, a_t) + \sum_t \log \pi_\theta(a_t|s_t)$$

When we take $\nabla_\theta$, the first two terms vanish (they don't depend on $\theta$):

$$\nabla_\theta \log P(\tau | \theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**Final result:**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$

#### Intuitive Summary

The policy gradient says:

1. **Sample trajectories** by running your policy
2. **For each token**, compute "how to make it more likely" ($\nabla_\theta \log \pi_\theta$)
3. **Scale by the reward** — good outcomes get reinforced, bad ones don't
4. **Average across trajectories**

### Part 5: Baselines — Reducing the Noise

#### The Problem with Vanilla REINFORCE

Vanilla REINFORCE has **high variance**. Here's why:

Suppose your model already solves 90% of problems. Most trajectories get $R(\tau) = 1$, so the gradient says "increase probability of these tokens!" even for trajectories that succeeded by luck.

The signal is noisy—sometimes you're reinforcing good reasoning, sometimes just lucky guesses.

#### The Solution: Subtract a Baseline

The fix is to subtract a **baseline** $b(s)$ that estimates "what return do we typically get?":

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \underbrace{(R(\tau) - b(s_t))}_{\text{advantage}}\right]$$

The quantity $(R(\tau) - b(s_t))$ is called the **advantage**:

| Advantage | Meaning | Effect |
|-----------|---------|--------|
| Positive | Better than expected | Reinforce these tokens |
| Negative | Worse than expected | Discourage these tokens |
| Zero | Exactly as expected | No change |

#### A Concrete Example

**Without baseline:**

| Trajectory | $R(\tau)$ | Gradient Signal |
|------------|-----------|-----------------|
| Correct | 1 | Make these tokens more likely!|
| Wrong | 0 | Do nothing|

Every correct answer gets the same reinforcement, regardless of difficulty; every wrong answer gets no punishment. The model only learns from successes.
This is actually a key limitation of vanilla REINFORCE with 0/1 rewards! You're not learning what to avoid, only what worked.

**With baseline** (say, $b = 0.9$ because model gets 90% right):

| Trajectory | $R(\tau)$ | Advantage = $R - 0.9$ | Gradient Signal |
|------------|-----------|----------------------|-----------------|
| Correct | 1 | +0.1 | "Slightly reinforce" |
| Wrong | 0 | -0.9 | "Strongly discourage!" |

Now the model can also learn avoiding failures rather than redundantly reinforcing successes!

#### Why Baselines Don't Add Bias

You might worry: "Doesn't subtracting something change the answer?"

No! The baseline term vanishes in expectation. Let's prove it.

**The claim:** For any baseline $b(s)$ that only depends on the state:

$$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = 0$$

**The proof:**

Since $b(s)$ doesn't depend on the action $a$, we can pull it out:

$$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)]= b(s) \cdot \mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)]$$

Now we show the expectation of the score function is zero:

$$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)] = \sum_{a} \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)$$

Using the identity $\nabla_\theta \log P = \frac{\nabla_\theta P}{P}$:

$$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)]= \sum_{a} \pi_\theta(a|s) \cdot \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} = \sum_{a} \nabla_\theta \pi_\theta(a|s)$$

Swapping sum and gradient:

$$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)]= \nabla_\theta \sum_{a} \pi_\theta(a|s) = \nabla_\theta 1 = 0$$

The last step works because probabilities over all possible actions that can be taken sum to 1.

**Concrete example with softmax as the policy function:**

Let's work through a real example. Considering a language model, where token probabilities come from softmax (policy) over logits:

$$\pi(a) = \frac{e^{z_a}}{\sum_k e^{z_k}}$$

The log-probability simplifies nicely:

$$\log \pi(a) = z_a - \log \sum_k e^{z_k}$$

Taking gradients with respect to each logit $z_i$:

| Gradient | Formula | Intuition |
|----------|---------|-----------|
| $\frac{\partial \log \pi(a)}{\partial z_a}$ | $1 - \pi(a)$ | Increasing own logit (or probability) helps (less help if already confident with high probability) |
| $\frac{\partial \log \pi(a)}{\partial z_b}$ | $-\pi(b)$ | Increasing competitor's logit (or probability) hurts |

*Derivation:* For the chosen token, $\frac{\partial}{\partial z_a}[z_a - \log\sum_k e^{z_k}] = 1 - \frac{e^{z_a}}{\sum_k e^{z_k}} = 1 - \pi(a)$. For other tokens, $\frac{\partial}{\partial z_b}[z_a - \log\sum_k e^{z_k}] = 0 - \frac{e^{z_b}}{\sum_k e^{z_k}} = -\pi(b)$.

**Numerical example:**

Suppose we have 3 tokens with probabilities $[\pi(A), \pi(B), \pi(C)] = [0.5, 0.3, 0.2]$.

The score function for each token (as a vector over logits $[z_A, z_B, z_C]$):

| Token | Score function $\nabla_z \log \pi$ | Weighted by $\pi$ |
|-------|-----------------------------------|-------------------|
| A | $[1-0.5, -0.3, -0.2] = [+0.5, -0.3, -0.2]$ | $0.5 \times [+0.5, -0.3, -0.2] = [+0.25, -0.15, -0.10]$ |
| B | $[-0.5, 1-0.3, -0.2] = [-0.5, +0.7, -0.2]$ | $0.3 \times [-0.5, +0.7, -0.2] = [-0.15, +0.21, -0.06]$ |
| C | $[-0.5, -0.3, 1-0.2] = [-0.5, -0.3, +0.8]$ | $0.2 \times [-0.5, -0.3, +0.8] = [-0.10, -0.06, +0.16]$ |
| **Sum** | | $[0, 0, 0]$ ✓ |

Each component sums to zero! For example, the first component: $0.25 - 0.15 - 0.10 = 0$.

The "increase my probability" directions (positive entries) are exactly canceled by the "decrease others' probability" directions (negative entries) when weighted by the policy.

**Why this matters:**

We can subtract any function of the state from our rewards without changing the expected gradient:

$$\mathbb{E}[\nabla_\theta \log \pi_\theta \cdot (R(\tau) - b(s))] = \mathbb{E}[\nabla_\theta \log \pi_\theta \cdot R(\tau)] - \underbrace{\mathbb{E}[\nabla_\theta \log \pi_\theta \cdot b(s)]}_{= 0}$$

We get **lower variance** (because advantages are centered around zero) **without introducing bias**. Free lunch!

### Part 6: Off-Policy Learning — Reusing Old Data

#### The Inefficiency of On-Policy Learning

Vanilla REINFORCE is **on-policy**: you generate rollouts, take one gradient step, then throw away the data and generate fresh rollouts.

```
Generate 1000 responses → one gradient step → discard → Generate 1000 more → ...

┌─────────────────────────────────────────────────────────────────────────────┐
│  ON-POLICY REINFORCE                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: Sample 1000 questions from your question bank                      │
│          (e.g., "What is 2+3?", "Solve x²=4", ...)                          │
│                         │                                                   │
│                         ▼                                                   │
│  Step 2: Current model π_θ GENERATES responses for each question            │
│          (This is expensive! Running inference 1000 times)                  │
│                         │                                                   │
│                         ▼                                                   │
│  Step 3: Check answers → rewards [1, 0, 1, 1, 0, ...]                       │
│                         │                                                   │
│                         ▼                                                   │
│  Step 4: Compute gradient, update θ → θ'                                    │
│                         │                                                   │
│                         ▼                                                   │
│  Step 5: DISCARD all 1000 responses ← This is the wasteful part!            │
│                         │                                                   │
│                         ▼                                                   │
│  Step 6: Go back to Step 1 with the NEW model π_θ'                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
This is wasteful! LLM inference is expensive, and we're only using each sample once.

#### The Solution: Importance Sampling

In **off-policy** learning, we reuse rollouts from a previous policy $\pi_{\theta_{\text{old}}}$ to train the current policy $\pi_\theta$.
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  OFF-POLICY                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: Sample questions                                                   │
│                         │                                                   │
│                         ▼                                                   │
│  Step 2: π_θ_old generates responses (expensive, but done ONCE)             │
│                         │                                                   │
│                         ▼                                                   │
│  Step 3: Check answers → rewards                                            │
│                         │                                                   │
│                         ▼                                                   │
│  ┌─────────────────────────────────────────────┐                            │
│  │  Step 4: Gradient step 1 (with importance   │                            │
│  │          weights to correct for π_θ_old)    │                            │
│  │                     │                       │                            │
│  │                     ▼                       │                            │
│  │  Step 5: Gradient step 2 (same data!)       │  ← Reuse the same          │
│  │                     │                       │    responses multiple      │
│  │                     ▼                       │    times!                  │
│  │  Step 6: Gradient step 3 (same data!)       │                            │
│  │                     │                       │                            │
│  │                     ▼                       │                            │
│  │  ... (typically 4-8 steps per batch)        │                            │
│  └─────────────────────────────────────────────┘                            │
│                         │                                                   │
│                         ▼                                                   │
│  Step 7: NOW discard and generate fresh responses                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

The trick is **importance sampling**: we reweight samples to correct for the mismatch between old and new policies.

$$g_{\text{off-policy}} = \frac{1}{N} \sum_{i=1}^{N} \sum_t \underbrace{\frac{\pi_\theta(a_t^{(i)}|s_t^{(i)})}{\pi_{\theta_{\text{old}}}(a_t^{(i)}|s_t^{(i)})}}_{\text{importance weight}} \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \cdot R(\tau^{(i)})$$

$$N$$ is number of trajectories in the batch (e.g., 1000 responses). 

The importance weight $\rho_t = \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ corrects for the distribution shift.

#### A Concrete Example

Suppose the old policy generated token "Paris" with probability 0.5, but your current policy would generate it with probability 0.7.

**Without correction:** You'd undercount "Paris" because it was sampled from a distribution that liked it less.

**With importance weight:** $\rho = 0.7 / 0.5 = 1.4$

You upweight this sample by 40% to compensate.

| Old Policy | Current Policy | Importance Weight |
|------------|----------------|-------------------|
| P("Paris") = 0.5 | P("Paris") = 0.7 | 0.7/0.5 = 1.4 |
| P("Lyon") = 0.3 | P("Lyon") = 0.1 | 0.1/0.3 = 0.33 |

#### The Catch: Don't Stray Too Far

Importance sampling only works when $\pi_\theta$ and $\pi_{\theta_{\text{old}}}$ are similar. If they diverge:

- Some importance weights explode (e.g., 100×)
- Gradient estimates become unreliable
- Training becomes unstable

This is why algorithms like PPO and GRPO **clip** the importance weights—more on this next!

Now let's put everything together with GRPO, the algorithm used to train DeepSeek R1.

### Part 7: GRPO — Group Relative Policy Optimization

#### The Core Idea: Compare Siblings

Remember, we need a baseline to reduce variance. The standard approach is to train a separate model to predict expected returns—but this is extra work.

**GRPO's insight:** Instead of learning a baseline, sample multiple answers for the same question and compare them to each other!

If you ask the model "What is 2+3?" five times and it gets three right and two wrong, the correct answers are "better than average" and the wrong ones are "worse than average." No separate baseline network needed!

#### Step 1: Sample Multiple Outputs Per Question

For each question $q$, sample $G$ outputs (the "group"):

$$\{o^{(1)}, o^{(2)}, \ldots, o^{(G)}\} \sim \pi_\theta(\cdot | q)$$

**Example with G=5:**

| Question | Output $i$ | Answer | Correct? | Reward $r^{(i)}$ |
|----------|-----------|--------|----------|------------------|
| "What is 15×7?" | 1 | "105" | ✓ | 1 |
| | 2 | "105" | ✓ | 1 |
| | 3 | "112" | ✗ | 0 |
| | 4 | "105" | ✓ | 1 |
| | 5 | "107" | ✗ | 0 |

#### Step 2: Compute Group-Normalized Advantages

The advantage for output $i$ is computed by normalizing within the group:

$$A^{(i)} = \frac{r^{(i)} - \text{mean}(r^{(1)}, \ldots, r^{(G)})}{\text{std}(r^{(1)}, \ldots, r^{(G)}) + \epsilon}$$

**Continuing the example:**

- mean$(r) = (1+1+0+1+0)/5 = 0.6$
- std$(r) = 0.49$

| Output $i$ | $r^{(i)}$ | $A^{(i)} = \frac{r^{(i)} - 0.6}{0.49}$ | Interpretation |
|------------|-----------|----------------------------------------|----------------|
| 1 | 1 | +0.82 | Better than siblings → reinforce |
| 2 | 1 | +0.82 | Better than siblings → reinforce |
| 3 | 0 | -1.22 | Worse than siblings → discourage |
| 4 | 1 | +0.82 | Better than siblings → reinforce |
| 5 | 0 | -1.22 | Worse than siblings → discourage |

**Key insight:** The same advantage applies to **every token** in that output. If output 1 was correct, every token in its reasoning chain gets $A = +0.82$.

#### Step 3: The GRPO-Clip Objective

GRPO combines off-policy learning with **clipping** to stay stable:

$$J_{\text{GRPO-Clip}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o^{(i)}|}\sum_{t} \min\left(\rho_t \cdot A^{(i)}, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \cdot A^{(i)}\right)\right]$$

**Symbol-by-symbol breakdown:**

| Symbol | Name | Meaning |
|--------|------|---------|
| $J_{\text{GRPO-Clip}}(\theta)$ | Objective function | The thing we want to maximize — "how good is our policy?" |
| $\theta$ | Parameters | All the weights in our neural network |
| $\mathbb{E}[\cdot]$ | Expectation | Average over many sampled questions and responses |
| $G$ | Group size | Number of responses we generate per question (e.g., 8) |
| $\frac{1}{G}\sum_{i=1}^{G}$ | Average over group | Average the objective across all G responses for this question |
| $i$ | Response index | Which response in the group (1st, 2nd, ..., G-th) |
| $o^{(i)}$ | Response i | The i-th generated response (sequence of tokens) |
| $\|o^{(i)}\|$ | Response length | Number of tokens in response i |
| $\frac{1}{\|o^{(i)}\|}\sum_{t}$ | Average over tokens | Average the objective across all tokens in this response |
| $t$ | Token index | Which token position in the response |
| $\rho_t$ | Probability ratio | $\frac{\pi_\theta(o_t)}{\pi_{\theta_{old}}(o_t)}$ — how much more/less likely is this token under new vs old policy? |
| $A^{(i)}$ | Advantage | Was response i better or worse than average in its group? |
| $\epsilon$ | Clip parameter | How far we allow the policy to change (typically 0.1–0.2) |
| $\text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)$ | Clipped ratio | Force $\rho_t$ to stay in range $[1-\epsilon, 1+\epsilon]$ |
| $\min(\cdot, \cdot)$ | Minimum | Take the smaller of the two values (conservative update) |

**The probability ratio $\rho_t$ in detail:**

$$\rho_t = \frac{\pi_\theta(o_t^{(i)} | q, o_{<t}^{(i)})}{\pi_{\theta_{\text{old}}}(o_t^{(i)} | q, o_{<t}^{(i)})}$$

| $\rho_t$ value | Meaning |
|----------------|---------|
| 1.0 | Token probability unchanged since we generated the same response |
| 1.5 | New policy is 50% more likely to generate this token |
| 0.7 | New policy is 30% less likely to generate this token |

**The clipping function:**

With $\epsilon = 0.2$, the clip function constrains $\rho_t$ to the range $[0.8, 1.2]$:

```
Input ρ_t:    0.5   0.8   1.0   1.2   1.5   2.0
              ↓     ↓     ↓     ↓     ↓     ↓
Output:       0.8   0.8   1.0   1.2   1.2   1.2
              ↑           ↑           ↑
           clipped     unchanged   clipped
              up                     down
```

**The min operation — being conservative:**

We compute BOTH the clipped and unclipped objectives, then take the minimum:

$$\min\left(\rho_t \cdot A^{(i)}, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \cdot A^{(i)}\right)$$

**Case 1: Positive advantage (good response, $A > 0$)**

| $\rho_t$ | Unclipped $\rho_t \cdot A$ | Clipped | Min (used) |
|----------|---------------------------|---------|------------|
| 0.8 | 0.8A | 0.8A | 0.8A |
| 1.0 | 1.0A | 1.0A | 1.0A |
| 1.2 | 1.2A | 1.2A | 1.2A |
| 1.5 | 1.5A | 1.2A | **1.2A** ← capped! |

Once $\rho_t > 1 + \epsilon$, the objective stops increasing. No more reward for pushing probability higher.

```
Objective
    ▲
    │            ┌────────────── capped at (1+ε)A
    │           /
    │          /
    │         /
    └────────┴─────────────────► ρ_t
           1.0  1.2
```

**Case 2: Negative advantage (bad response, $A < 0$)**

| $\rho_t$ | Unclipped $\rho_t \cdot A$ | Clipped | Min (used) |
|----------|---------------------------|---------|------------|
| 1.2 | -1.2A | -1.2A | -1.2A |
| 1.0 | -1.0A | -1.0A | -1.0A |
| 0.8 | -0.8A | -0.8A | -0.8A |
| 0.5 | -0.5A | -0.8A | **-0.8A** ← capped! |

Once $\rho_t < 1 - \epsilon$, the objective stops decreasing. No more penalty for pushing probability lower.

```
Objective
    ▲
    │   ─────────┐
    │             \
    │              \
    │               \
    └───────────────┴────────► ρ_t
               0.8  1.0
```

**Complete worked example:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRPO-Clip Objective: Step by Step                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Question: "What is 2+3?"                                                  │
│                                                                             │
│   Step 1: Generate G=4 responses from π_θ_old                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Response 1: "Let me think... 2+3 = 5" ✓    reward = 1              │   │
│   │  Response 2: "2 plus 3 equals 6" ✗          reward = 0              │   │
│   │  Response 3: "The answer is 5" ✓            reward = 1              │   │
│   │  Response 4: "I believe it's 7" ✗           reward = 0              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   Step 2: Compute group-normalized advantages                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  mean(rewards) = 0.5,  std(rewards) = 0.5                           │   │
│   │                                                                     │   │
│   │  A⁽¹⁾ = (1 - 0.5) / 0.5 = +1.0   (better than average)              │   │
│   │  A⁽²⁾ = (0 - 0.5) / 0.5 = -1.0   (worse than average)               │   │
│   │  A⁽³⁾ = (1 - 0.5) / 0.5 = +1.0   (better than average)              │   │
│   │  A⁽⁴⁾ = (0 - 0.5) / 0.5 = -1.0   (worse than average)               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   Step 3: For each token, compute clipped objective                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Example: Token "5" in Response 1 (A = +1.0)                        │   │
│   │                                                                     │   │
│   │  π_θ_old("5" | context) = 0.4                                       │   │
│   │  π_θ("5" | context) = 0.6        (after some gradient steps)        │   │
│   │                                                                     │   │
│   │  ρ_t = 0.6 / 0.4 = 1.5                                              │   │
│   │                                                                     │   │
│   │  Unclipped: ρ_t × A = 1.5 × 1.0 = 1.50                              │   │
│   │  Clipped:   clip(1.5, 0.8, 1.2) × A = 1.2 × 1.0 = 1.20              │   │
│   │                                                                     │   │
│   │  min(1.50, 1.20) = 1.20  ← Use the conservative value               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Example: Token "6" in Response 2 (A = -1.0)                        │   │
│   │                                                                     │   │
│   │  π_θ_old("6" | context) = 0.3                                       │   │
│   │  π_θ("6" | context) = 0.15       (model learned to avoid this)      │   │
│   │                                                                     │   │
│   │  ρ_t = 0.15 / 0.3 = 0.5                                             │   │
│   │                                                                     │   │
│   │  Unclipped: ρ_t × A = 0.5 × (-1.0) = -0.50                          │   │
│   │  Clipped:   clip(0.5, 0.8, 1.2) × A = 0.8 × (-1.0) = -0.80          │   │
│   │                                                                     │   │
│   │  min(-0.50, -0.80) = -0.80  ← Use the more negative value           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   Step 4: Average over all tokens and responses → Final objective J(θ)      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Plain English summary:**

The GRPO-Clip objective says:

> 1. **For each question:** Generate G different responses, score them, compute advantages
> 2. **For each token:** Check how much the probability changed ($\rho_t$), multiply by advantage
> 3. **But clip the change:** Don't let the policy move more than $\epsilon$ away from the old policy
> 4. **Average everything:** Over all tokens and all responses

#### Why Clipping Matters

Without clipping, taking many gradient steps on the same batch leads to **overfitting**. Clipping ensures the policy can only move **X% away from the old policy per batch**. This keeps training stable.

#### The GRPO Training Loop Visualized

Here's the complete GRPO workflow showing how it reuses generated responses:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GRPO TRAINING LOOP                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: Sample a batch of questions from dataset                           │
│          ["What is 2+3?", "Solve x²=4", "What is 7×8?", ...]                │
│                         │                                                   │
│                         ▼                                                   │
│  Step 2: Snapshot current model as π_θ_old                                  │
│                         │                                                   │
│                         ▼                                                   │
│  Step 3: Generate G responses per question using π_θ_old                    │
│          (This is EXPENSIVE — full inference G times per question)          │
│                         │                                                   │
│          ┌──────────────┴──────────────┐                                    │
│          │  Question: "What is 2+3?"   │                                    │
│          │  Response 1: "5" ✓          │                                    │
│          │  Response 2: "6" ✗          │                                    │
│          │  Response 3: "5" ✓          │                                    │
│          │  Response 4: "7" ✗          │                                    │
│          └──────────────┬──────────────┘                                    │
│                         │                                                   │
│                         ▼                                                   │
│  Step 4: Compute rewards and group-normalized advantages                    │
│          A⁽¹⁾=+1.0, A⁽²⁾=-1.0, A⁽³⁾=+1.0, A⁽⁴⁾=-1.0                         │
│                         │                                                   │
│                         ▼                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Step 5-8: MULTIPLE gradient steps on the SAME responses              │  │
│  │            (This is where we save compute!)                           │  │
│  │                                                                       │  │
│  │    ┌─────────────────────────────────────────────────────────────┐    │  │
│  │    │  Gradient step 1:                                           │    │  │
│  │    │    - Compute ρ_t = π_θ(token) / π_θ_old(token) for all      │    │  │
│  │    │    - Apply clipping to keep ρ_t in [0.8, 1.2]               │    │  │
│  │    │    - Update θ                                               │    │  │
│  │    └─────────────────────────────────────────────────────────────┘    │  │
│  │                           │                                           │  │
│  │                           ▼                                           │  │
│  │    ┌─────────────────────────────────────────────────────────────┐    │  │
│  │    │  Gradient step 2: (same responses, updated π_θ)             │    │  │
│  │    │    - Recompute ρ_t with new π_θ                             │    │  │
│  │    │    - Clipping prevents ρ_t from going too far               │    │  │
│  │    │    - Update θ again                                         │    │  │
│  │    └─────────────────────────────────────────────────────────────┘    │  │
│  │                           │                                           │  │
│  │                           ▼                                           │  │
│  │    ┌─────────────────────────────────────────────────────────────┐    │  │
│  │    │  Gradient steps 3, 4, ... (typically 4-8 total)             │    │  │
│  │    │    - Eventually ρ_t hits clip boundaries                    │    │  │
│  │    │    - Gradients become zero → time for fresh data            │    │  │
│  │    └─────────────────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                         │                                                   │
│                         ▼                                                   │
│  Step 9: Discard responses, go back to Step 1 with updated π_θ              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why is this more efficient than vanilla REINFORCE?**

| Method | Responses generated | Gradient steps | Efficiency |
|--------|--------------------:|---------------:|-----------:|
| REINFORCE | 1000 | 1 | 1 step per 1000 inferences |
| GRPO | 1000 | 4-8 | 4-8 steps per 1000 inferences |

GRPO extracts 4-8× more learning from each expensive batch of generated responses!

#### The Complete GRPO Algorithm

```
Algorithm: GRPO

Input: initial model π_θ, reward function R, questions D

1:  π_θ ← π_θ_init                        # Start with base model

2:  for step = 1 to n_grpo_steps:          # Main training loop
3:      Sample batch of questions D_b
4:      π_θ_old ← π_θ                      # Snapshot current model
5:      
6:      # Sample G outputs per question
7:      for each question q in D_b:
8:          Sample {o^(1), ..., o^(G)} from π_θ_old
9:          Compute rewards {r^(1), ..., r^(G)}
10:         Compute advantages A^(i) via group normalization
11:     
12:     # Take multiple gradient steps on same rollouts (off-policy)
13:     for train_step = 1 to n_train_steps_per_rollout_batch:
14:         Update π_θ by maximizing GRPO-Clip objective
15:     
16: Output: trained π_θ
```

**Why this works:**

1. **Group normalization** provides a baseline without training a separate network
2. **Off-policy updates** let us take multiple gradient steps per batch (efficient!)
3. **Clipping** prevents the policy from changing too much (stable!)

### Part 8: Putting It All Together

#### Summary Table

| Concept | What It Is | What It Does | Example |
|---------|------------|--------------|---------|
| **Policy** $\pi_\theta$ | Your LLM parameterized by weights $\theta$ | Given context, outputs probability distribution over next tokens | P("Paris" \mid "The capital of France is") = 0.85 |
| **State** $s_t$ | The text generated so far | Provides context for the next decision | "What is 2+3? <think> I need to" |
| **Action** $a_t$ | A single token | The choice made at each step | "add" |
| **Trajectory** $\tau$ | Complete sequence $(s_0, a_0, s_1, a_1, ..., s_T, a_T)$ | One full episode from prompt to end-of-text | Question → reasoning → answer → `</answer>` |
| **Reward** $r_t$ | Scalar feedback signal | Judges quality of action at state | 0 for intermediate steps, 1 if final answer correct |
| **Return** $R(\tau)$ | Sum of rewards over trajectory | Single number measuring "how good was this trajectory?" | R = 1 (correct) or R = 0 (wrong) |
| **Score function** $\nabla_\theta \log \pi_\theta(a \mid s)$ | Gradient of log-probability | Direction in parameter space that increases P(action) | A vector with one entry per model parameter |
| **REINFORCE** | Vanilla policy gradient algorithm | Multiply score function by return, average over trajectories | Good trajectories → reinforce all their tokens |
| **Baseline** $b(s)$ | Estimate of expected return | Subtract from reward to get advantage | If model gets 70% right, b ≈ 0.7 |
| **Advantage** $A = R(\tau) - b$ | "Better or worse than expected?" | Positive → reinforce, Negative → discourage, Zero → no signal | R=1, b=0.7 → A=+0.3 (good!); R=0, b=0.7 → A=-0.7 (bad!) |
| **On-policy** | Generate → one gradient step → discard | Uses fresh data for each update | Wasteful: 1000 inferences for 1 gradient step |
| **Off-policy** | Generate → multiple gradient steps → discard | Reuses data with importance weights $\frac{\pi_\theta}{\pi_{\theta_{old}}}$ | Efficient: 1000 inferences for 4-8 gradient steps |
| **Importance weight** $\rho_t$ | Ratio $\frac{\pi_\theta(a)}{\pi_{\theta_{old}}(a)}$ | Corrects for distribution mismatch when reusing old data | If old P=0.4, new P=0.6, then ρ=1.5 |
| **Clipping** | Constrain $\rho_t$ to $[1-\epsilon, 1+\epsilon]$ | Prevents policy from changing too fast | With ε=0.2, ρ stays in [0.8, 1.2] |
| **GRPO** | Group Relative Policy Optimization | Sample G responses per question, use group statistics as baseline | No need to train separate value network |

#### Key Equations at a Glance

| Equation | Name | Plain English |
|----------|------|---------------|
| $\nabla_\theta J = \mathbb{E}_{\tau}[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau)]$ | Policy Gradient | "Increase probability of tokens that led to high rewards" |
| $\nabla_\theta J = \mathbb{E}_{\tau}[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (R(\tau) - b)]$ | Baselined Policy Gradient | "Reinforce better-than-expected, discourage worse-than-expected" |
| $A^{(i)} = \frac{r^{(i)} - \text{mean}(r)}{\text{std}(r)}$ | GRPO Advantage | "Compare this response to its siblings" |
| $\min(\rho_t \cdot A, \text{clip}(\rho_t) \cdot A)$ | Clipped Objective | "Update conservatively — don't change too much at once" |

#### The PyTorch Connection

When you implement GRPO, the core looks like this:

```python
def compute_grpo_loss(model, batch, old_log_probs, advantages, epsilon=0.2):
    """
    Compute GRPO-Clip loss for a batch of trajectories.
    """
    # Get current log probabilities
    logits = model(batch["input_ids"])
    log_probs = get_log_probs(logits, batch["input_ids"])
    
    # Compute probability ratios: π_θ / π_θ_old
    ratios = torch.exp(log_probs - old_log_probs)
    
    # Clipped objective
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    
    # Take minimum (conservative update)
    loss = -torch.min(unclipped, clipped).mean()
    
    return loss
```

The key steps are:
1. Compute current vs. old log-probabilities
2. Form the probability ratio
3. Clip the ratio
4. Take the minimum of clipped and unclipped objectives
5. Maximize (so we minimize the negative)

The math may look intimidating at first, but the core ideas are simple:
- **Try things** (sample trajectories)
- **See what works** (check rewards)
- **Do more of what works** (policy gradient)
- **Be smart about it** (baselines, clipping, group normalization)

That's really all there is to it!

---

**Resources:**
- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Nathan Lambert's RLHF Book](https://rlhfbook.com/)
- [DeepSeek R1 Paper](https://arxiv.org/abs/2401.02954)
- [DeepSeekMath Paper](https://arxiv.org/abs/2402.03300) — Original GRPO formulation
- [PPO Paper](https://arxiv.org/abs/1707.06347) — Proximal Policy Optimization