---
layout: post
title: "Study Notes: Stanford CS336 Language Modeling from Scratch [8]"
categories: cs336
author: 
- Han Yu
---
## Planning LLM Training: Cross-Entropy Loss, Optimizers, Memory and Computational Cost, and other practical levers

### Table of Contents
1. [Cross-Entropy Loss: Measuring How Wrong We Are](#cross-entropy-loss)
2. [Perplexity: A More Intuitive Metric](#perplexity)
3. [SGD Optimizer: Walking Downhill](#sgd-optimizer)
4. [AdamW: The Smart Optimizer](#adamw)
5. [Memory Requirements: Can It Fit?](#memory-requirements)
6. [Computational Cost: How Long Will This Take?](#computational-cost)
7. [Learning Rate Schedules: Starting Fast, Ending Slow](#learning-rate-schedules)
8. [Gradient Clipping: The Safety Mechanism](#gradient-clipping)

---

### Cross-Entropy Loss: Measuring How Wrong We Are {#cross-entropy-loss}

*A language model is trying to **predict the next word** in a sequence. Noted that we're training a supervised learning model, as we know what is the next correct word in the sequence from the training dataset, and we want to minimize the loss when the model did not predict the correct word.*

#### Simple Example

Imagine you have the sentence: "The cat sat on the ___"

- The model needs to predict what comes next
- Maybe the correct word is "mat"
- The model gives probabilities for all possible words in its vocabulary

#### The Simple Version of the Math

**Step 1: The model outputs "logits"** (raw scores for each possible word)
- Think of logits as unnormalized scores
- Example: "mat" gets score 5.2, "dog" gets 1.3, "table" gets 3.1, etc.

**Step 2: Convert logits to probabilities using softmax**

$$p(\text{word}) = \frac{e^{\text{score of that word}}}{\text{sum of } e^{\text{score}} \text{ for all words}}$$

Example:
- If "mat" has score 5.2: probability = $\frac{e^{5.2}}{e^{5.2} + e^{1.3} + e^{3.1} + ...}$

**Step 3: Calculate the loss**

$$\text{loss} = -\log(p(\text{correct word}))$$

Assume the correct word is "mat", and we want the model to assign high probability to the correct word and low probability to the incorrect word. 

- If the model gives "mat" a probability of 0.8 → loss = -log(0.8) ≈ 0.22 (small loss, good!)
- If the model gives "mat" a probability of 0.1 → loss = -log(0.1) ≈ 2.30 (big loss, bad!)
- If the model gives "mat" a probability of 1.0 → loss = -log(1.0) = 0 (no loss, perfect!)

#### The Full Version of the Math

The complete formula averages this loss over:
- All positions in a sequence (i = 1 to m)
- All sequences in the training dataset D (all x in D)

$$\ell(\theta; D) = \frac{1}{|D|m} \sum_{x \in D} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} | x_{1:i})$$

**Bottom line:** Cross-entropy loss is small when the model assigns high probability to the correct next word, and large when it doesn't. Training tries to minimize this loss!

---

### Perplexity: A More Intuitive Metric {#perplexity}

**Perplexity** is an **evaluation metric** (not a loss function) that provides a more intuitive way to measure how good your language model is. It answers the question: **"On average, how many words is the model confused between?"**  While we **train** using cross-entropy loss, we **report** perplexity to humans because it's easier to interpret.

#### Simple Analogy

Imagine a multiple-choice test:

- **Perplexity = 1**: The model is 100% certain (like having only 1 choice)
- **Perplexity = 10**: The model is as confused as if it had to guess among 10 equally likely options
- **Perplexity = 100**: The model is as confused as if it had to guess among 100 equally likely options

**Lower perplexity = Better model!**

#### The Math

For a sequence where we make m predictions with cross-entropy losses $\ell_1, \ell_2, ..., \ell_m$:

$$\text{perplexity} = \exp\left(\frac{1}{m} \sum_{i=1}^{m} \ell_i\right)$$

This is equivalent to:

$$\text{perplexity} = \exp(\text{average cross-entropy loss})$$

**Breaking it down:**

**Step 1:** Calculate the average cross-entropy loss
- Recall that for each position i: 
    $$\ell_i = -\log p(x_{i+1} | x_{1:i})$$
- Add up all the losses: $\ell_1 + \ell_2 + ... + \ell_m$
- Divide by m (the number of token predictions in the sequence)
- Average loss = $\frac{1}{m} \sum_{i=1}^{m} \ell_i$

**Step 2:** Take the exponential
- Apply $\exp()$ to the average loss
- This "undoes" the log in the cross-entropy formula

#### Why Exponential?

The exponential transformation converts the abstract loss value into an interpretable number:

**Mathematical intuition:**
- Cross-entropy loss: $\ell = -\log p(\text{correct word})$
- Perplexity undoes the log: $\exp(\ell) = \exp(-\log p) = \frac{1}{p}$
- If average probability is 0.1, perplexity ≈ 10 (confused among ~10 words)
- If average probability is 0.01, perplexity ≈ 100 (confused among ~100 words)

#### Concrete Example

Say your model predicts 3 words in a sequence:
- Word 1: $\ell_1 = 0$, probability was 1.0 (perfect!)
- Word 2: $\ell_2 = 2.3$, probability was 0.1
- Word 3: $\ell_3 = 0.69$, probability was 0.5

**Calculation:**

Average loss = $\frac{0 + 2.3 + 0.69}{3} = \frac{2.99}{3} \approx 1.0$

Perplexity = $\exp(1.0) \approx 2.72$

**Interpretation:** On average, the model is as uncertain as if it had to choose uniformly among about **2.72 equally likely words** at each position.

#### Relationship to Training

- **During training:** We minimize cross-entropy loss
- **During evaluation:** We report perplexity for interpretability
- **They're equivalent:** Lower cross-entropy ⟺ Lower perplexity

Since perplexity is just an exponential transformation of cross-entropy, optimizing one automatically optimizes the other. We use cross-entropy for training because it has better mathematical properties for gradient-based optimization. 

#### Key Takeaway

**Perplexity is a user-friendly version of cross-entropy loss:**
- Lower perplexity = model is more confident and accurate
- Higher perplexity = model is confused and uncertain  
- It's **not used for training**, only for **reporting results** in a more interpretable way
- Cross-entropy and perplexity are mathematically equivalent—minimizing one minimizes the other

---

### SGD Optimizer: Which direction to walk to go downhill {#sgd-optimizer}

SGD (Stochastic Gradient Descent) is an algorithm that **adjusts your model's parameters** during the training process to make the loss smaller. Think of it as teaching the model to make better predictions.
#### The Mountain Analogy

Imagine you're standing on a mountain in the fog (you can't see far):
- Your **position** = model parameters (θ)
- Your **altitude** = loss (how bad the model is)
- Your **goal** = get to the bottom of the valley (minimize loss)

**SGD tells you which direction to walk to go downhill!**

#### How SGD Works (Step by Step)

##### Step 1: Start Randomly
- θ₀ = random starting position on the mountain
- You don't know where the bottom is yet

##### Step 2: Look Around (Calculate Gradient)
- ∇L(θₜ; Bₜ) = "Which direction is downhill?"
- The gradient tells you the steepest uphill direction
- So **negative gradient** points downhill!

##### Step 3: Take a Step Downhill

$$\theta_{t+1} = \theta_t - \alpha_t \nabla L(\theta_t; B_t)$$

Let me break down each part:
- **θₜ** = where you are now
- **∇L(θₜ; Bₜ)** = direction of steepest uphill
- **-∇L(θₜ; Bₜ)** = direction of steepest downhill (flip the sign!)
- **αₜ** = learning rate (how big a step to take)
- **θₜ₊₁** = your new position

##### Step 4: Repeat!
- Keep taking steps downhill until you reach the valley (minimum loss)

#### Key Concepts

##### Learning Rate (αₜ)
- **Too large**: You take huge steps and might overshoot the valley
- **Too small**: You take tiny steps and it takes forever
- **Just right**: You make steady progress

**Example:**
- If gradient says "go left by 10 units" and α = 0.1
- You actually move right by: 10 × 0.1 = 1 unit

##### Batch (Bₜ)
- Instead of using ALL your data to calculate the gradient (slow!), use a **random small batch**
- This is the "stochastic" part - it's random! Here "random" means at each training step t, we randomly sample a subset of examples from the full training dataset D.
- **Batch size** = how many examples you use each step (this is fixed during training)

**Why random batches?**
- Much faster! (calculating gradient on 1 million examples is slow)
- Still gives you a good enough direction
- Adds helpful randomness that can escape bad spots

#### Simple Example

Suppose your model has one parameter θ (to keep it simple):

**Initial:** θ₀ = 5, loss = 100

**Step 1:**
- Calculate gradient on a batch: ∇L = 20 (loss increases if we increase θ)
- Learning rate: α = 0.1
- Update: θ₁ = 5 - 0.1 × 20 = 5 - 2 = **3**

**Step 2:**
- New gradient: ∇L = 10
- Update: θ₂ = 3 - 0.1 × 10 = 3 - 1 = **2**

**Step 3:**
- New gradient: ∇L = 2
- Update: θ₃ = 2 - 0.1 × 2 = 2 - 0.2 = **1.8**

You keep going until the loss stops decreasing!

#### Key Takeaway

**SGD is like walking downhill in small steps:**
1. Check which way is uphill (gradient)
2. Take a step in the negative direction (size of the step determined by learning rate)
3. Repeat until you reach the bottom (minimum loss)

The "stochastic" part just means you randomly sample small batches of data instead of using entire dataset when calculate the gradient, making it much faster!

---

### AdamW: The Smart Optimizer {#adamw}

#### What's the Problem with SGD?

Remember SGD takes the same size step (α) for every parameter. But what if:
- Some parameters need **big updates** (they're far from optimal)
- Some parameters need **tiny updates** (they're almost perfect)

**AdamW is smarter** via adapting the step size for each parameter individually.

#### The Big Idea

AdamW keeps track of **two pieces of memory** for each parameter:

1. **m (first moment)**: "Which direction has this parameter been moving lately?" (like momentum)
2. **v (second moment)**: "How much has this parameter been jumping around?" (like volatility)

Then it uses this information to take smarter steps!

#### How AdamW Works (Step by Step)

##### Setup
- **m = 0**: Start with no momentum
- **v = 0**: Start with no volatility estimate
- **β₁ = 0.9**: How much to remember past directions (typically 90%)
- **β₂ = 0.999**: How much to remember past volatility (typically 99.9%)

##### Each Training Step

**Step 1: Calculate gradient** (same as SGD)
- g = ∇ℓ(θ; Bₜ) 
- "Which way should we move?"

**Step 2: Update momentum (first moment)**

$$m = \beta_1 \cdot m + (1-\beta_1) \cdot g$$

Think of this as an **exponential moving average**:
- Keep 90% of the old direction (β₁m)
- Add 10% of the new direction ((1-β₁)g)
- This smooths out noisy gradients!

**Step 3: Update volatility (second moment)**

$$v = \beta_2 \cdot v + (1-\beta_2) \cdot g^2$$

Same idea but for squared gradients:
- Keep 99.9% of old volatility estimate
- Add 0.1% of new squared gradient
- This tracks how "jumpy" the parameter is

**Step 4: Adjust learning rate**

$$\alpha_t = \alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$$

This **bias correction** compensates for starting at m=0 and v=0 (they start biased toward zero!)

**Step 5: Update parameters (the smart part!)**

$$\theta = \theta - \alpha_t \frac{m}{\sqrt{v} + \epsilon}$$

This is where the magic happens:
- **m** tells us which direction to go
- **√v** tells us how confident we should be
- If v is **large** (parameter is jumpy) → take **smaller** steps
- If v is **small** (parameter is stable) → take **larger** steps

**Step 6: Weight decay**

$$\theta = \theta - \alpha\lambda\theta$$

Pull parameters slightly toward zero to prevent them from getting too large (regularization)

#### Simple Example

Imagine two parameters:

**Parameter A:**
- Gradients: [5, 5.1, 4.9, 5, 5] (very stable!)
- v will be small → AdamW takes **bigger** steps
- Makes sense! We're confident about the direction

**Parameter B:**
- Gradients: [5, -4, 6, -3, 5] (super noisy!)
- v will be large → AdamW takes **smaller** steps
- Makes sense! We're uncertain, so be cautious

#### Key Hyperparameters

| Parameter | Typical Value | What it does |
|-----------|---------------|--------------|
| α (learning rate) | 0.001 or 0.0001 | Base step size |
| β₁ | 0.9 | How much momentum to keep |
| β₂ | 0.95-0.999 | How much volatility history to keep |
| λ (weight decay) | 0.01 | How much to pull toward zero |
| ε | 10⁻⁸ | Prevent division by zero |

#### Key Takeaway

**AdamW is like a smart GPS:**
- **SGD**: "Always drive 50 mph, no matter what"
- **AdamW**: "Drive faster on smooth highways, slower on bumpy roads"

It **adapts the step size** for each parameter based on:
1. Recent direction (momentum)
2. Recent stability (volatility)

This makes training **faster and more stable**, which is why all modern language models use it.

---

### Memory Requirements: Can It Fit? {#memory-requirements}

Let's calculate how much memory we need to train a model like GPT-2 XL using AdamW with float32 precision.

#### Setup
- Data type: **float32** = **4 bytes** per number
- Batch size: **B**
- Sequence length: **L** (context_length)
- Model dimension: **d** (d_model)
- Number of layers: **N** (num_layers)
- Number of heads: **H** (num_heads)
- Vocabulary size: **V** (vocab_size)
- Feed-forward dimension: **d_ff = 4d**

### Memory Components

Training a Transformer model requires four main types of memory. **Parameters** store the model's learnable weights—the numbers that define what the model knows. **Gradients** store the *direction* and *magnitude* of how each parameter should change during training, computed during backpropagation. **Optimizer state** keeps AdamW's running statistics: *momentum* (which direction parameters have been moving) and *volatility* (how much parameters have been fluctuating), allowing the optimizer to make smarter, adaptive updates for each parameter. **Activations** store all the intermediate calculations from the forward pass—like attention scores, normalized values, and layer outputs—which must be kept in memory so we can compute gradients during backpropagation. While parameters, gradients, and optimizer state have fixed size (Parameters, gradients, and optimizer state represent the model's internal structure—they exist regardless of what data you feed into the model), activations are the actual values flowing through the network for the particular batch therefore they scale dramatically with both batch size (B) and sequence length (L), particularly the attention scores which grow quadratically as O(BL²). This is why memory, not computation, is often the bottleneck in training large language models—with GPT-2 XL, even an 80GB GPU can only fit a batch size of 3.


#### 1. Parameters Memory

Let's count the learnable parameters in each component of the Transformer.

##### Per Transformer Block:

**A. RMSNorm Layers (2 per block)**

Each RMSNorm layer has a learnable scale parameter for each dimension:
- Pre-attention RMSNorm: **d parameters**
- Pre-FFN RMSNorm: **d parameters**
- **Subtotal: 2d parameters**

**B. Multi-Head Self-Attention**

The attention mechanism consists of four projection matrices. Importantly, **the number of heads H does not affect the parameter count**—we split the d dimensions across heads rather than expanding them.

- Query projection W_Q: (d × d) → **d² parameters**
- Key projection W_K: (d × d) → **d² parameters**
- Value projection W_V: (d × d) → **d² parameters**
- Output projection W_O: (d × d) → **d² parameters**
- **Subtotal: 4d² parameters**

*Note: Modern architectures typically omit bias terms in these projections.*

**C. Feed-Forward Network (FFN)**

The FFN expands to an intermediate dimension d_ff = 4d, then projects back:
- First layer W₁: (d × 4d) → **4d² parameters**
- Activation (SiLU/GELU): **0 parameters** (no learnable weights)
- Second layer W₂: (4d × d) → **4d² parameters**
- **Subtotal: 8d² parameters**

**Total per block: 2d + 4d² + 8d² = 12d² + 2d**

##### All N Transformer Blocks:

**N × (12d² + 2d) = 12Nd² + 2Nd**

##### Additional Components:

**Token Embedding**
- Maps each of V vocabulary tokens to a d-dimensional vector
- Shape: (V × d)
- **Parameters: Vd**

**Final RMSNorm**
- One scale parameter per dimension after the last transformer block
- **Parameters: d**

**Output Projection**
- In modern LLMs (GPT-2, LLaMA, etc.), the output projection **shares weights** with the token embedding (weight tying)
- **Additional parameters: 0**

**Positional Embeddings** (architecture-dependent)
- **Modern models (LLaMA, GPT-3+):** Use RoPE or ALiBi → **0 parameters** ✓
- **Older models (GPT-2, BERT):** Learned positional embeddings → **L_max × d parameters**

For this calculation, we assume modern architecture with no learned positional embeddings.

##### Total Parameters:

**P = 12Nd² + 2Nd + Vd + d**

Which can be factored as:

**P = 12Nd² + d(2N + V + 1)**

**Important notes:**
- The sequence length L does **not** affect parameter count (for modern architectures)
- The number of attention heads H does **not** affect parameter count
- The d² term dominates for large models (quadratic scaling with model dimension)
- The Vd term can be significant for large vocabularies

##### Memory Requirement:

Since each parameter is stored as **float32** (4 bytes):

**Parameters memory = 4P bytes**

**Example (GPT-2 XL):**
- N = 48, d = 1,600, V = 50,257
- P = 12(48)(1,600²) + 1,600(2×48 + 50,257 + 1)
- P ≈ **1,555,126,400 parameters** (~1.56B)
- **Memory = 4 × 1.56B ≈ 6.2 GB**

#### 2. Gradients Memory

During backpropagation, we compute gradients for all parameters. AdamW requires storing these gradients to perform parameter updates.

**Gradients have the same shape as parameters:**
- One gradient value per parameter
- Stored as float32 (4 bytes each)

**Gradients memory = 4P bytes**

#### 3. Optimizer State Memory

AdamW is a **stateful optimizer** that maintains running statistics for each parameter:

**First moment (m):** Exponential moving average of gradients (momentum)
- Shape: same as parameters
- Storage: 4 bytes per parameter (float32)
- **Memory: 4P bytes**

**Second moment (v):** Exponential moving average of squared gradients (volatility)
- Shape: same as parameters
- Storage: 4 bytes per parameter (float32)
- **Memory: 4P bytes**

**Total optimizer state memory = 4P + 4P = 8P bytes**

**Note:** Unlike parameters and gradients which all models need, this 8P overhead is specific to Adam-family optimizers. Simpler optimizers like SGD only need gradient storage (4P), while more complex optimizers may require even more state.


#### 4. Activations Memory

Activations are intermediate values computed during the forward pass that must be stored for backpropagation. This is where batch size (B) and sequence length (L) have major impact.

**Key factors:**
- Activations scale with **B** (batch size) and **L** (sequence length)
- We need to store activations at multiple points for gradient computation
- This is typically the **memory bottleneck** for training

##### Per Transformer Layer Activations:

**A. RMSNorm activations:**
- Input to pre-attention norm: BLd
- Output of pre-attention norm: BLd
- Input to pre-FFN norm: BLd
- **Subtotal: ~3BLd**

**B. Attention intermediate values:**
- Q, K, V projections: 3 × BLd = **3BLd**
- Attention scores (before softmax): B × H × L × L = **BHL²**
- Attention weights (after softmax): B × H × L × L = **BHL²** (needed for softmax backward)
- Attention output: BLd

**Subtotal: ~4BLd + 2BHL²**

**C. Feed-Forward Network activations:**
- W₁ output (before activation): B × L × 4d = **4BLd**
- SiLU/GELU output: B × L × 4d = **4BLd** (needed for activation backward)
- W₂ output: BLd

**Subtotal: ~9BLd**

**Total per layer: 3BLd + 4BLd + 2BHL² + 9BLd ≈ 16BLd + 2BHL²**

*Note: The original formula's 16BLd is an approximation; exact value depends on implementation details like whether certain intermediate values are recomputed vs. stored.*

##### All N Layers:

**N × (16BLd + 2BHL²) ≈ 16NBLd + 2NBHL²**

##### Additional Activations Outside Layers:

**Token embeddings:**
- Embedding lookup output: **BLd**

**Final RMSNorm:**
- Negligible (included in layer activations)

**Output layer (logits):**
- Softmax probabilities: B × L × V = **BLV** (needed for cross-entropy backward)

**Total additional: BLd + BLV**

##### Total Activation Count:

**A = 16NBLd + 2NBHL² + BLd + BLV**

Simplified:
**A ≈ NBLd(16 + 1/N) + 2NBHL² + BLV**

For large N, this is dominated by: **A ≈ 16NBLd + 2NBHL² + BLV**

**Activations memory = 4A bytes** (float32)

**Key observations:**
- **O(BL) scaling:** Most activations scale linearly with batch size and sequence length
- **O(BL²) scaling:** Attention scores create quadratic memory growth with sequence length
- **Bottleneck:** For long sequences, the 2NBHL² term (attention scores) dominates

#### Total Peak Memory

| Component | Memory (bytes) | Notes |
|-----------|----------------|-------|
| **Parameters** | 4P | Model weights |
| **Gradients** | 4P | ∂L/∂θ for all parameters |
| **Optimizer State (m, v)** | 8P | AdamW momentum and variance |
| **Activations** | 4A | Intermediate values for backprop |
| **TOTAL** | **16P + 4A** | Peak during training |

**Breakdown:**
- **Fixed cost (16P):** Independent of batch size and sequence length
- **Variable cost (4A):** Scales with B, L, and L²

#### GPT-2 XL Example

**Model specifications:**
- vocab_size (V) = 50,257
- context_length (L) = 1,024
- num_layers (N) = 48
- d_model (d) = 1,600
- num_heads (H) = 25
- d_ff = 4 × d = 6,400

##### Step 1: Calculate Parameters (P)

**P = 12Nd² + 2Nd + Vd + d**

P = 12(48)(1,600²) + 2(48)(1,600) + 50,257(1,600) + 1,600

**P ≈ 1,555,126,400 ≈ 1.56 × 10⁹ parameters**

##### Step 2: Calculate Fixed Memory (16P)

Fixed memory = 16 × 1,555,126,400 bytes

**Fixed memory ≈ 24.88 GB**

This includes parameters (4P), gradients (4P), and optimizer state (8P).

##### Step 3: Calculate Activation Memory per Batch (4A)

**A = 16NBLd + 2NBHL² + BLd + BLV**

For batch_size = B:

A = 16(48)(B)(1,024)(1,600) + 2(48)(B)(25)(1,024²) + B(1,024)(1,600) + B(1,024)(50,257)

**Activation memory = 4A ≈ 14.3 × B GB**

##### Step 4: Total Memory Formula

**Total Memory = 24.88 + 14.3 × B GB**

##### Step 5: Maximum Batch Size for 80GB GPU

Solving: 24.88 + 14.3B ≤ 80

**B ≤ 3.85**

**Maximum batch_size = 3** (must be integer)

**Key insight:** So on a single A100 80 GB, GPT-2 XL in pure FP32 training fits a batch size of 3 without further memory optimization. This demonstrates why:
1. Large-scale training requires massive GPU clusters
2. Techniques like gradient accumulation, mixed precision (float16/bfloat16), and activation checkpointing are essential
3. Memory, not computation, is often the bottleneck

The estimate above assumes naïve attention that stores full B×H×L×L score and probability tensors, and a non-fused cross-entropy head. Modern implementations cut this drastically:

| Technique                    | What it Removes              | Result (GPT-2 XL, FP32) |
| ---------------------------- | ---------------------------- | ----------------------- |
| **FlashAttention**           | avoids L² attention matrices | ≈ 5 GB per batch       |
| **Fused CE**                 | streams logits → softmax     |  reduce 0.5–1 GB per batch              |
| **Activation checkpointing** | recomputes during backward   | ≈ × 4–6 less            |
| **BF16 / FP16**              | halves memory per value      | ≈ × 2 less              |

With FlashAttention and combined with other memory optimization techniques, batch_size = 12-16 is achievable for GPT-2 XL on an 80GB GPU with FP32, and this can be scaled to 32-40 with BF16 or 60-70 with activation checkpointing.

### Computational Cost: How Long Will This Take? {#computational-cost}

#### The Standard Formula

For Transformer models, there's a widely-used approximation:

**Training FLOPs per token ≈ 6 × number of parameters**

This breaks down as:
- **Forward pass:** 2P FLOPs per token
- **Backward pass:** 4P FLOPs per token (approximately 2× forward)
- **Total:** 6P FLOPs per token

**Why this approximation works:**
- Dominated by matrix multiplications in attention and FFN layers
- For a matrix multiply of (m × k) @ (k × n), we perform 2mkn FLOPs
- The "2" accounts for *multiply* and *add* operations
- Backward pass requires computing gradients for all weight matrices (roughly 2× forward)

**What's excluded:**
- Optimizer computations (~11 FLOPs per parameter, negligible compared to 6P per token)
- Element-wise operations (LayerNorm, activations)
- Attention softmax

These omissions are small compared to the matrix multiplications, making "6P per token" a robust rule of thumb.

#### GPT-2 XL Training Example

**Given:**
- Parameters: P ≈ 1.56 × 10⁹
- Training steps: 400,000
- **Batch size: 1,024 tokens per step** (total tokens processed)
- Hardware: Single NVIDIA A100 GPU (40GB or 80GB)
- Theoretical peak: 19.5 teraFLOP/s (FP32)
- MFU (Model FLOPs Utilization): 50%

**Note on batch size:** "1,024 tokens" typically means the **total number of tokens** processed in one training step. 

**Calculation:**

**Step 1: FLOPs per token**
```
6 × 1.56 × 10⁹ = 9.36 × 10⁹ FLOPs per token
```

**Step 2: FLOPs per training step**
```
9.36 × 10⁹ × 1,024 tokens = 9.585 × 10¹² FLOPs per step
```

**Step 3: Total FLOPs for 400K steps**
```
9.585 × 10¹² × 400,000 = 3.834 × 10¹⁸ FLOPs
```

**Step 4: Effective throughput at 50% MFU**
```
Theoretical: 19.5 × 10¹² FLOP/s
Effective: 19.5 × 10¹² × 0.5 = 9.75 × 10¹² FLOP/s
```

**Step 5: Training time**
```
Time = (3.834 × 10¹⁸) / (9.75 × 10¹²) = 393,231 seconds
       ≈ 109.2 hours
       ≈ 4.55 days
```
#### Key Insights

**Why this matters:**

1. **Single GPU training is impractical for large models**
   - Even "medium-sized" GPT-2 XL takes **109 hours (~4.5 days)** on a top-tier A100
   - Larger models (GPT-3: 175B parameters) would take **months** on a single GPU
   - GPT-3 would require: (175B/1.56B) × 109 hours ≈ 12,200 hours ≈ **1.4 years** on one A100!

2. **Parallelism is essential**
   - **With 100 A100s:** 109.2 / 100 ≈ **1.1 hours** (assuming perfect scaling)
   - **With 1,000 A100s:** 109.2 / 1,000 ≈ **6.6 minutes** (assuming perfect scaling)
   - **Real-world scaling efficiency** is typically 60-90% due to:
     - Communication overhead (gradient synchronization)
     - Load imbalancing
     - Pipeline bubbles
     - Network bandwidth limitations
   - **Realistic with 100 A100s:** 109.2 / (100 × 0.7) ≈ **1.6 hours** (at 70% efficiency)

3. **Cost considerations**
   - **A100 cloud cost:** ~$2-4/hour (varies by provider: AWS, GCP, Azure)
   - **Single A100 training:** 109.2 hours × $2-4 = **$218-437**
   - **100 A100s (70% efficiency):** 
     - Time: ~1.6 hours
     - Cost: 100 GPUs × 1.6 hours × $2-4 = **$320-640**
     - **Trade-off:** Slightly higher cost, but **68× faster!**
   - **Cost scales linearly with GPU count, but time scales sub-linearly** (due to overhead)

4. **Memory vs. compute trade-off**
   - We calculated **batch_size = 3** fits in 80GB memory (using FP32)
   - Larger batches could improve training efficiency (better GPU utilization, more stable gradients)
   - **Solutions to increase effective batch size:**
     - **Gradient accumulation:** Simulate larger batches by accumulating gradients over multiple forward/backward passes before updating
       - Example: Accumulate 32 micro-batches of size 3 → effective batch size of 96
     - **Mixed precision (FP16/BF16):** Reduce memory by 2×, allowing batch_size ≈ 6-8
     - **Gradient checkpointing:** Trade compute for memory (recompute activations during backward pass)
     - **Multi-GPU training:** Distribute batch across GPUs (data parallelism)

5. **Why GPT-3 scale requires massive clusters**
   - GPT-3 (175B parameters): **~112× larger** than GPT-2 XL
   - Single A100 would take: ~1.4 years
   - With **10,000 A100s** (at 60% efficiency): ~12,200 / (10,000 × 0.6) ≈ **2 hours**
   - This explains why frontier models require:
     - Tens of thousands of GPUs
     - Custom datacenters
     - Months of calendar time (even with massive parallelism)
     - Millions of dollars in compute costs

**Summary Table for training GPT-2 XL**

| Configuration | Time | Cost | Notes |
|---------------|------|------|-------|
| **1 A100 (FP32)** | 109 hours | $218-437 | Baseline |
| **1 A100 (FP16)** | ~55 hours | $110-220 | 2× faster with mixed precision |
| **100 A100s (perfect)** | 1.1 hours | $220-440 | Theoretical best case |
| **100 A100s (70% eff.)** | 1.6 hours | $320-640 | Realistic with overhead |
| **1,000 A100s (perfect)** | 6.6 minutes | $220-440 | Theoretical best case |
| **1,000 A100s (60% eff.)** | 11 minutes | $367-733 | Realistic at scale |

**Key takeaway:** Parallelism gives you speed, not cost savings. Using 100 GPUs costs about the same (or slightly more) but finishes **68× faster**, which matters for iteration speed and time-to-market!

---

### Learning Rate Schedules: Starting Fast, Ending Slow {#learning-rate-schedules}

#### Why Do We Need a Schedule?

Imagine you're trying to find the lowest point in a valley while blindfolded:

- **Beginning**: You're far from the goal → take **big steps** to get there quickly
- **Middle**: You're getting close → take **medium steps** to avoid overshooting
- **End**: You're very close → take **tiny steps** to settle into the exact lowest point

The learning rate schedule does exactly this for training!

#### The Problem with Fixed Learning Rate

**Too high throughout training:**
-  Fast at first, but bounces around the minimum at the end
-  Never settles into the best solution

**Too low throughout training:**
-  Slow progress, takes forever to train
-  Might get stuck in bad spots

**Solution: Start high, gradually decrease!**

#### Cosine Annealing Schedule

The schedule has **3 phases**:

##### Phase 1: Warm-up (t < T_w)
**"Ease into it"**

$$\alpha_t = \frac{t}{T_w} \times \alpha_{max}$$

- Start from **0** and **linearly increase** to α_max
- Example: If T_w = 1,000 and α_max = 0.001:
  - Step 0: α = 0
  - Step 500: α = 0.0005 (halfway)
  - Step 1,000: α = 0.001 (full speed!)

**Why warm-up?**
- Prevents unstable updates at the very beginning
- Gives the model time to "orient itself"
- Like warming up before exercise!

##### Phase 2: Cosine Annealing (T_w ≤ t ≤ T_c)
**"Smooth slowdown"**

$$\alpha_t = \alpha_{min} + \frac{1}{2}\left(1 + \cos\left(\frac{t - T_w}{T_c - T_w} \pi\right)\right)(\alpha_{max} - \alpha_{min})$$

This creates a **smooth curve** from α_max down to α_min!

**Breaking it down:**

1. **Progress ratio**: How far through annealing are we? (0 to 1)
2. **Cosine curve**: cos goes from 1 → -1 as we progress
3. **Scale to [0, 1]**: Transform to go from 1 → 0
4. **Final value**: Interpolate between α_max and α_min

**The result: A smooth decrease from α_max to α_min**

##### Phase 3: Post-Annealing (t > T_c)
**"Maintain minimum"**

$$\alpha_t = \alpha_{min}$$

- Keep the learning rate at the minimum value
- Fine-tuning with tiny steps

#### Visual Example

Let's say:
- α_max = 0.001
- α_min = 0.0001
- T_w = 1,000 (warm-up ends)
- T_c = 10,000 (annealing ends)

**Learning rate over time:**

| Step | Phase | Learning Rate |
|------|-------|---------------|
| 0 | Warm-up | 0 |
| 500 | Warm-up | 0.0005 |
| 1,000 | Warm-up → Annealing | 0.001 |
| 3,000 | Annealing | ~0.00085 |
| 5,500 | Annealing | ~0.00055 |
| 8,000 | Annealing | ~0.00025 |
| 10,000 | Annealing → Post | 0.0001 |
| 15,000 | Post-annealing | 0.0001 |

#### Simple Math Example

Let's calculate learning rate at t = 5,500 (middle of annealing):

**Given:**
- T_w = 1,000, T_c = 10,000
- α_max = 0.001, α_min = 0.0001

**Step 1:** Progress = (5,500 - 1,000) / (10,000 - 1,000) = 0.5

**Step 2:** cos(0.5 × π) = 0

**Step 3:** ½(1 + 0) = 0.5

**Step 4:** α = 0.0001 + 0.5 × (0.001 - 0.0001) = **0.00055**

Exactly halfway between min and max!

#### Key Takeaway

**Learning rate schedule = adaptive step size:**
1. **Warm-up**: Gradually increase from 0 → big (safety at start)
2. **Cosine annealing**: Smoothly decrease from big → small (careful landing)
3. **Post-annealing**: Stay small (fine-tuning)

It's like driving: accelerate leaving the driveway, cruise on the highway, then slow down smoothly as you approach your destination!

---

### Gradient Clipping: The Safety Mechanism {#gradient-clipping}

#### The Problem: Exploding Gradients

Imagine you're walking downhill with a GPS that tells you how steep the slope is:

- **Normal case**: "Slope is 5 degrees" → take a reasonable step
- **Bad case**: "Slope is 5,000 degrees!!!" → you'd jump off a cliff!

Sometimes during training, the model encounters weird examples that produce **huge gradients**. If you take a step proportional to these giant gradients, your model parameters can explode and training crashes!

#### What is Gradient Clipping?

Gradient clipping is like having a **speed limiter** on your updates:

**"No matter how steep the slope, I won't step faster than X units"**

#### How It Works (Step by Step)

##### Step 1: Calculate the Gradient Norm

After the backward pass, measure how "big" the gradients are overall:

$$\|g\|_2 = \sqrt{g_1^2 + g_2^2 + g_3^2 + ... + g_n^2}$$

This is the **L2 norm** (Euclidean distance) - just the length of the gradient vector.

**Example:**
- If gradients are [3, 4]: norm = √(9 + 16) = **5**
- If gradients are [30, 40]: norm = √(900 + 1,600) = **50**

##### Step 2: Check Against Maximum

Set a threshold **M** (e.g., M = 1.0):

**Is ∥g∥₂ ≤ M?**

- **YES** → Gradients are reasonable, use them as-is ✓
- **NO** → Gradients are too big, need to clip! ✂️

##### Step 3: Scale Down If Needed

If the norm exceeds M, **rescale** the entire gradient vector:

$$g_{\text{clipped}} = g \times \frac{M}{\|g\|_2 + \epsilon}$$

Where ε ≈ 10⁻⁶ is for numerical stability.

**What this does:**
- Keeps the **direction** the same
- Reduces the **magnitude** to exactly M

#### Simple Example

**Given:**
- Gradient vector: g = [30, 40]
- Maximum norm: M = 1.0
- ε = 10⁻⁶ (negligible)

**Step 1: Calculate norm**
- ∥g∥₂ = √(900 + 1,600) = **50**

**Step 2: Check threshold**
- 50 > 1.0 → **Need to clip!**

**Step 3: Scale down**
- Scaling factor = 1.0 / 50 = **0.02**
- g_clipped = [30, 40] × 0.02 = **[0.6, 0.8]**

**Verify new norm:**
- ∥g_clipped∥₂ = √(0.36 + 0.64) = **1.0** ✓

**Result:** We've scaled from norm 50 to norm 1.0, keeping the same direction!

#### Another Example (No Clipping)

**Given:**
- Gradient: g = [0.3, 0.4]
- Maximum: M = 1.0

**Norm:** √(0.09 + 0.16) = **0.5**

**Check:** 0.5 ≤ 1.0 → **No clipping needed!**

Use gradient as-is: [0.3, 0.4]

#### Why Does This Work?

**Preserves Direction:**
- We still move in the right direction (downhill)
- Just limit how far we jump

**Prevents Instability:**
- Giant gradients can make parameters explode
- Clipping ensures updates stay reasonable

**Training Stability:**
- Without clipping: loss might spike or become NaN
- With clipping: training stays smooth

#### Pseudocode

```python
# After backward pass
gradients = compute_gradients()

# Calculate L2 norm of all gradients
grad_norm = sqrt(sum(g² for all g in gradients))

# Clip if needed
max_norm = 1.0
if grad_norm > max_norm:
    scaling_factor = max_norm / (grad_norm + 1e-6)
    gradients = gradients * scaling_factor

# Use clipped gradients in optimizer
optimizer.step(gradients)
```

#### Key Takeaway

**Gradient clipping = speed limiter for training:**

1. **Measure** how big the gradients are (L2 norm)
2. **Check** if they exceed the maximum allowed
3. **Scale down** if needed (keep direction, reduce magnitude)

**Result:** Training stays stable even when occasional batches produce huge gradients. It's like having a safety governor on a car engine - you can still accelerate and steer normally, but it prevents dangerous speeds that could cause a crash.

---

### Conclusion

Training large language models involves carefully balancing multiple components:

1. **Loss functions** (cross-entropy) measure how bad the model did not predict the correct word
2. **Metrics** (perplexity) make model prediction results more interpretable
3. **Optimizers** (SGD → AdamW) determine how we update model parameters
4. **Memory management** dictates what type of hardware we need
5. **Computational budgets** determine training time, make trade-off between cost and speed 
6. **Learning rate schedules** help us converge smoothly
7. **Safety mechanisms** (gradient clipping) prevent training instability

 While the math can seem complex at first, the underlying intuitions are straightforward: we're teaching a model to predict text by repeatedly showing it examples, measuring its mistakes, and adjusting its parameters to do better next time. The real challenge isn't understanding any single component - it's orchestrating all of them together efficiently at massive scale. That's what makes training models like GPT-3 and GPT-4 such remarkable engineering achievements.

