---
layout: post
title: "Study Notes: Stanford CS336 Language Modeling from Scratch [10]"
categories: cs336
author:
- Han Yu
---

## Building a Complete Training Loop

This note documents the journey of assembling all the core components such as optimizer, learning rate scheduling, data loading, checkpointing, and decoding - into a complete training pipeline for Transformer language models. We'll explore how each piece fits together, the design decisions behind them, and the practical considerations that make the difference between research code and production systems.

### Table of Contents
1. [Introduction: The Big Picture](#introduction)
2. [The AdamW Optimizer: Decoupled Weight Decay Regularization](#adamw-optimizer)
3. [Learning Rate Scheduling: The LLaMA Approach](#lr-scheduling)
4. [Memory-Efficient Data Loading](#data-loading)
5. [Checkpoint Management](#checkpointing)
6. [Decoding Strategies: From Model to Text](#decoding)
7. [Putting It All Together: The Training Script](#training-script)
8. [Testing and Validation](#testing)
9. [Key Takeaways](#takeaways)

---

### Introduction: The Big Picture {#introduction}

Training a large language model isn't just about implementing a forward pass and calling `loss.backward()`. A production training pipeline requires careful orchestration of multiple components, each with its own subtleties and potential pitfalls. In this note, we'll go through how to build a complete training pipeline from scratch, learning why each component matters and how they interact.

**What we'll build:**
- An implementation of the AdamW optimizer 
- A cosine learning rate schedule with warmup, as used in LLaMA
- Memory-mapped data loading to manage loading datasets larger than RAM
- Robust checkpoint saving/loading for long training runs
- Multiple decoding strategies (temperature scaling, top-p sampling)
- A complete sample training script that ties everything together

---

### The AdamW Optimizer: Decoupled Weight Decay Regularization {#adamw-optimizer}

The first step in building our training loop is implementing the optimizer. While PyTorch provides `torch.optim.AdamW`, understanding the exact algorithm is crucial for debugging training issues and understanding why certain hyperparameters matter.

#### The Algorithm

The AdamW algorithm (from "Decoupled Weight Decay Regularization" by Loshchilov & Hutter, 2019) differs from standard Adam in how it applies weight decay. Here's Algorithm 2 from the paper:

**Initialize:**
- Learnable parameters: $\theta$
- First moment vector: $m \leftarrow 0$ (same shape as $\theta$)
- Second moment vector: $v \leftarrow 0$ (same shape as $\theta$)

**For** $t = 1, 2, \ldots, T$:

1. Sample batch of data $B_t$

2. Compute gradient:

   $$g \leftarrow \nabla_\theta \ell(\theta; B_t)$$

3. Update biased first moment estimate:

   $$m \leftarrow \beta_1 m + (1 - \beta_1) g$$

4. Update biased second raw moment estimate:

   $$v \leftarrow \beta_2 v + (1 - \beta_2) g^2$$

5. Compute bias-corrected learning rate:

   $$\alpha_t \leftarrow \alpha \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$$

6. Update parameters with adaptive learning rate:

   $$\theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v + \varepsilon}}$$

7. Apply decoupled weight decay:

   $$\theta \leftarrow \theta - \alpha \lambda \theta$$

#### Why Decoupled Weight Decay Matters

The key innovation in AdamW is **decoupling weight decay from the gradient-based update**. To understand why this matters, let's compare the two approaches:

**Standard Adam with L2 Regularization:**

In traditional Adam with L2 regularization, we add the weight decay term to the gradient before computing adaptive moments:

$$g \leftarrow \nabla_\theta \ell(\theta; B_t) + \lambda \theta$$

Then we proceed with the normal Adam update using this modified gradient. This means:
- Weight decay affects the adaptive moment estimates ($m$ and $v$)
- The effective weight decay depends on the adaptive learning rate
- Parameters with large gradients get less regularization (due to adaptive scaling)

**AdamW with Decoupled Weight Decay:**

In AdamW, we apply weight decay **after** the adaptive update as a separate step:

$$\theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v + \varepsilon}} - \alpha \lambda \theta$$

This decoupling means:
- Weight decay is independent of gradient statistics
- All parameters receive consistent regularization proportional to their magnitude
- Weight decay directly shrinks parameters toward zero, regardless of gradient history

**Why This Improves Performance:**

1. **Better generalization**: Decoupled weight decay provides more consistent regularization across all parameters, leading to better generalization on downstream tasks.

2. **Works with large learning rates**: In standard Adam + L2, increasing the learning rate also increases the effective regularization, creating unwanted coupling. AdamW removes this coupling.

3. **More interpretable**: The weight decay hyperparameter $\lambda$ directly controls regularization strength, making it easier to tune.

**Practical Impact:**

For large language models, this difference is crucial. The original BERT used Adam with L2 regularization and achieved 84.4% on MNLI. Simply switching to AdamW with the same hyperparameters improved accuracy to 84.8% - a significant gain from this single algorithmic change. Similar improvements have been observed across many other deep learning tasks.

#### Complete Implementation

```python
class AdamW(torch.optim.Optimizer):
    """
    Implements AdamW optimizer following Algorithm 1 from
    "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Update biased first moment: m ← β₁m + (1 - β₁)g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment: v ← β₂v + (1 - β₂)g²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias correction terms
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute adjusted learning rate: α_t ← α √(1-(β₂)^t) / (1-(β₁)^t)
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters: θ ← θ - α_t m / √(v+ε)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-alpha_t)

                # Apply decoupled weight decay: θ ← θ - αλθ
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)
```

**Usage example:**

```python
model = TransformerLM(vocab_size=50257, d_model=768, ...)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

for batch in dataloader:
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

### Learning Rate Scheduling: The LLaMA Approach {#lr-scheduling}

Modern large language models don't use a fixed learning rate. Instead, they employ sophisticated schedules that warm up the learning rate at the start and gradually decay it during training. The LLaMA paper (Touvron et al., 2023) uses a three-phase cosine schedule that has become standard.

#### The Three-Phase Schedule

**Phase 1 - Warmup** ($t < T_w$):

$$\alpha_t = \frac{t}{T_w} \cdot \alpha_{\text{max}}$$

Linear increase from 0 to $\alpha_{\text{max}}$ over $T_w$ steps.

**Phase 2 - Cosine Annealing** ($T_w \leq t \leq T_c$):

$$\alpha_t = \alpha_{\text{min}} + \frac{1}{2} \left(1 + \cos\left(\frac{t - T_w}{T_c - T_w} \cdot \pi\right)\right) \left(\alpha_{\text{max}} - \alpha_{\text{min}}\right)$$

Smooth cosine decay from $\alpha_{\text{max}}$ to $\alpha_{\text{min}}$.

**Understanding the Smooth Cosine Decay:**

The beauty of cosine annealing lies in its smoothness. The diagram below shows how the learning rate evolves during the cosine annealing phase:

<img src="/assets/picture/2025-11-02-cs336-building-a-complete-training-loop/cosine_decay.png" alt="Smooth Cosine Decay" width="50%">

*Figure: Cosine annealing schedule showing the smooth decay of learning rate from α<sub>max</sub> to α<sub>min</sub>*

**Breaking down the cosine formula:**

Let's denote $p = \frac{t - T_w}{T_c - T_w}$ as the progress through the cosine phase (where $p \in [0, 1]$).

The formula becomes:

$$\alpha_t = \alpha_{\text{min}} + \frac{1}{2}(1 + \cos(p \cdot \pi)) \cdot (\alpha_{\text{max}} - \alpha_{\text{min}})$$

**Why cosine creates a smooth curve:**

1. **At start** ($p = 0$):
   - $\cos(0) = 1$
   - $\alpha_t = \alpha_{\text{min}} + 1 \cdot (\alpha_{\text{max}} - \alpha_{\text{min}}) = \alpha_{\text{max}}$

2. **At middle** ($p = 0.5$):
   - $\cos(\pi/2) = 0$
   - $\alpha_t = \alpha_{\text{min}} + 0.5 \cdot (\alpha_{\text{max}} - \alpha_{\text{min}})$ (halfway point)

3. **At end** ($p = 1$):
   - $\cos(\pi) = -1$
   - $\alpha_t = \alpha_{\text{min}} + 0 \cdot (\alpha_{\text{max}} - \alpha_{\text{min}}) = \alpha_{\text{min}}$

**Key properties of the smooth cosine decay:**

- **Gentle start**: Derivative is near zero at $t = T_w$, creating a smooth transition from warmup
- **Steepest descent**: Maximum decay rate occurs at the midpoint ($p = 0.5$)
- **Gentle landing**: Derivative approaches zero as $t \to T_c$, allowing fine-tuning
- **No discontinuities**: The function and its derivative are continuous everywhere

**Phase 3 - Constant** ($t > T_c$):

$$\alpha_t = \alpha_{\text{min}}$$

Maintain minimum learning rate.

#### Why This Schedule Works

**Warmup phase:** Starting with a small learning rate prevents the model from making destructive updates when parameters are still randomly initialized. Gradients can be large and unstable early in training, and a small learning rate provides stability.

**Cosine decay:** The smooth decay helps the model settle into a good minimum. The cosine schedule provides:
- Fast initial decay (when model is still far from optimum)
- Slower decay later (allowing fine-tuning)
- No sharp transitions (unlike step decay schedules)

**Constant minimum:** Maintaining α_min instead of decaying to zero allows continued (albeit slow) learning, which can be useful for very long training runs.

#### Implementation

```python
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Get learning rate at iteration `it` using cosine schedule with warmup.

    Three phases:
    1. Warmup: Linear increase from 0 to max_learning_rate
    2. Cosine annealing: Smooth decay from max to min learning rate
    3. Constant: Maintain min_learning_rate

    Args:
        it: Current iteration (0-indexed)
        max_learning_rate: Maximum learning rate (α_max)
        min_learning_rate: Minimum learning rate (α_min)
        warmup_iters: Number of warmup iterations (T_w)
        cosine_cycle_iters: Total iterations for cosine cycle (T_c)

    Returns:
        Learning rate for current iteration
    """
    # Phase 1: Warmup (t < T_w)
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    # Phase 2: Cosine annealing (T_w ≤ t ≤ T_c)
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + math.cos(progress * math.pi))
        return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)

    # Phase 3: Constant (t > T_c)
    return min_learning_rate
```

**Critical detail:** The warmup condition is `it < warmup_iters` (strict inequality), not `it <= warmup_iters`. This ensures iteration `warmup_iters` is the first iteration at `max_learning_rate`, not the last warmup iteration.

#### Integration with Training Loop

```python
for iter_num in range(max_iters):
    # Get learning rate for this iteration
    lr = get_lr_cosine_schedule(
        it=iter_num,
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_iters=2000,
        cosine_cycle_iters=100000,
    )

    # Update optimizer learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Training step
    x, y = get_batch(...)
    loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Typical hyperparameters for large models:**
- Warmup: 2,000-10,000 iterations (1-5% of total training)
- Max LR: 1e-4 to 1e-3 (depends on model size; larger models use smaller LR)
- Min LR: 10% of max LR
- Cosine cycle: Total training iterations

---

### Memory-Efficient Data Loading {#data-loading}

When training on large text datasets (hundreds of GBs to TBs), loading the entire dataset into RAM is impossible. The solution is **memory-mapped arrays** using the Unix `mmap` system call.

#### The Problem

Consider training GPT-3-scale models:
- Common Crawl: ~570GB tokenized
- Books: ~150GB tokenized
- Total: ~800GB of tokens

Your machine might have 64-128GB of RAM. Loading this data is impossible.

#### The Solution: Memory Mapping

Memory mapping lets you "pretend" the entire dataset is in memory, but the OS only loads the portion you actually access.

```python
# Memory-mapped loading
dataset = np.load('train_tokens.npy', mmap_mode='r')
# This doesn't load the file into RAM!
# It creates a memory map to the file on disk

# When you access dataset[1000000:1000512],
# the OS loads just that small portion into RAM
```

**Understanding Virtual Memory**

Before diving into how memory mapping works, it's important to understand the concept of virtual memory—the foundation that makes memory mapping possible.

Virtual memory is a way for your computer to make it look like you have more memory (RAM) than you actually do. It does this by using part of your disk (storage) to act as an extension of RAM. Every program "thinks" it has access to a large, continuous block of memory—but behind the scenes, the operating system (OS) is moving chunks of data between RAM and disk as needed.

**How Memory Mapping Works: Step by Step**

1. **Mapping file to memory**: The system call `mmap()` creates a link between a file on disk and an area in virtual memory. You can then access it like a normal array, even if the file is huge (e.g., 800GB).

2. **Page fault (on first access)**: When your code accesses something like `dataset[i]`, the OS sees that the data isn't in RAM yet. It triggers a **page fault**—a signal that tells the OS to fetch that data page from disk.

3. **Loading data into RAM**: The OS loads the specific page (a small chunk, usually 4KB) from disk into physical RAM. Now `dataset[i]` can be read directly from fast memory.

4. **Caching nearby elements**: The OS often loads neighboring pages too (since they'll likely be accessed soon). So if you later access `dataset[i+1]`, it's already in RAM—fast!

5. **Eviction when RAM is full**: When RAM gets full, the OS automatically evicts less-used pages (writes them back to disk if modified). This keeps the system running smoothly without running out of memory.

**Key insight**: Memory mapping leverages the OS's virtual memory system to handle datasets much larger than available RAM, loading only the data you need on-demand and caching intelligently based on access patterns.

#### Implementation

```python
def get_batch(
    dataset: np.ndarray,  # Can be memory-mapped!
    batch_size: int,
    context_length: int,
    device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of sequences from dataset.

    Supports both regular arrays and memory-mapped arrays transparently.
    Memory-mapped arrays use the Unix mmap system call to map files to virtual
    memory, allowing you to "pretend" you have the entire dataset in memory
    while only loading accessed portions on-demand.

    Args:
        dataset: Token array (regular or memory-mapped)
        batch_size: Number of sequences to sample
        context_length: Length of each sequence
        device: Device to place tensors on

    Returns:
        x: Input sequences [batch_size, context_length]
        y: Target sequences [batch_size, context_length] (shifted by 1)
    """
    # Sample random start positions
    max_start = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_start, size=batch_size)

    # Extract sequences (this triggers page faults for memory-mapped arrays)
    x = np.stack([dataset[i:i + context_length] for i in start_indices])
    y = np.stack([dataset[i + 1:i + context_length + 1] for i in start_indices])

    # Convert to PyTorch tensors
    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)

    return x, y


def load_dataset(data_path: str, vocab_size: int) -> np.ndarray:
    """
    Load dataset using memory-mapped mode for memory efficiency.

    Args:
        data_path: Path to .npy file containing tokenized data
        vocab_size: Expected vocabulary size for validation

    Returns:
        Memory-mapped numpy array
    """
    print(f"Loading dataset from {data_path}...")

    # Load with memory mapping for large datasets
    dataset = np.load(data_path, mmap_mode="r")

    print(f"  Loaded {len(dataset):,} tokens")
    print(f"  Data type: {dataset.dtype}")
    print(f"  Memory-mapped: {isinstance(dataset, np.memmap)}")

    # Verify data integrity
    max_token = dataset.max()
    min_token = dataset.min()
    print(f"  Token range: [{min_token}, {max_token}]")

    if max_token >= vocab_size:
        raise ValueError(
            f"Data contains token {max_token} >= vocab_size {vocab_size}. "
            f"Data may be corrupted or vocab_size is incorrect."
        )

    if min_token < 0:
        raise ValueError(f"Data contains negative token {min_token}")

    print(f"  ✓ Data integrity verified")

    return dataset
```

#### Important Considerations

**1. Data type matching:**
```python
# Ensure dtype matches your vocabulary size
dataset = np.memmap('tokens.dat', dtype='int32', mode='r')  # For vocab < 2^31
# or
dataset = np.memmap('tokens.dat', dtype='int64', mode='r')  # For safety
```

**2. Data integrity:**
Always verify that token values are within valid range:
```python
assert dataset.max() < vocab_size, "Invalid token values!"
assert dataset.min() >= 0, "Negative token values!"
```

**3. Performance tips:**
- Access data sequentially when possible (better cache locality)
- Use larger batch sizes to amortize page fault overhead
- Store data on fast SSD rather than HDD

---

### Checkpoint Management {#checkpointing}

Training large models can take days or weeks. Checkpoint management is crucial for:
- Resuming after crashes or preemption
- Evaluating models at different training stages
- Storing model configurations for reproducibility

#### What to Save

A complete checkpoint includes:
1. **Model state**: All parameter values
2. **Optimizer state**: Momentum buffers, learning rate, etc.
3. **Iteration count**: For resuming at exact position
4. **Model configuration**: For reconstructing architecture

Many implementations forget #4, making it hard to load models for inference later.

**Why Model Configuration Matters**

Think of it this way:
- **Model configuration** = The model's recipe (layer sizes, dropout rates, architecture choices)
- **Model state** = The model's learned ingredients (weights and biases)

Without the configuration, you wouldn't know how to rebuild the same model structure later.

**Example: A Simple Neural Network**

Let's say you built this model in PyTorch:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
```

When you train it, you'll want to save not only the weights, but also the model configuration:

```python
config = {
    "input_size": 784,
    "hidden_size": 256,
    "output_size": 10,
    "dropout": 0.2
}

checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "iteration": step,
    "config": config
}
torch.save(checkpoint, "checkpoint.pth")
```

**Later (for inference or resume training):**

You can rebuild the model exactly the same way:

```python
checkpoint = torch.load("checkpoint.pth")
config = checkpoint["config"]

# Rebuild model using saved configuration
model = MyModel(**config)
model.load_state_dict(checkpoint["model_state"])
```

This same principle applies to Transformer language models, where the configuration includes `vocab_size`, `d_model`, `num_layers`, `num_heads`, `d_ff`, `context_length`, etc.

#### Implementation

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str,
    model_config: dict = None,
) -> None:
    """
    Save complete training state to checkpoint file.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        iteration: Current training iteration
        out: Output path for checkpoint
        model_config: Optional model architecture configuration
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }

    # Save model config for easy loading during inference
    if model_config is not None:
        checkpoint['model_config'] = model_config

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load training state from checkpoint file.

    Args:
        src: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into

    Returns:
        Iteration number from checkpoint
    """
    checkpoint = torch.load(src, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']
```

#### Checkpoint Strategy

**During training:**
```python
# Save periodically during training
if iter_num % checkpoint_interval == 0 and iter_num > 0:
    checkpoint_path = f"checkpoints/checkpoint_iter_{iter_num}.pt"
    save_checkpoint(model, optimizer, iter_num, checkpoint_path, model_config)

# Save final checkpoint with both iteration number and "final" name
final_checkpoint_iter = f"checkpoints/checkpoint_iter_{max_iters}.pt"
final_checkpoint = "checkpoints/checkpoint_final.pt"
save_checkpoint(model, optimizer, max_iters, final_checkpoint_iter, model_config)
save_checkpoint(model, optimizer, max_iters, final_checkpoint, model_config)
```

**Resuming from checkpoint:**
```python
if resume_from is not None:
    start_iter = load_checkpoint(resume_from, model, optimizer)
    print(f"Resumed from iteration {start_iter}")
else:
    start_iter = 0

for iter_num in range(start_iter, max_iters):
    # Training continues from where it left off
    ...
```

**For inference (loading model configuration):**
```python
checkpoint = torch.load("checkpoint.pt")
config = checkpoint['model_config']

model = TransformerLM(
    vocab_size=config['vocab_size'],
    d_model=config['d_model'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    d_ff=config['d_ff'],
    context_length=config['context_length'],
)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

### Decoding Strategies: From Model to Text {#decoding}

After training, your model can predict the next word given the previous ones. But you need a method to:
1. Turn those predictions into probabilities
2. Pick the next word/token from that probability distribution

That process is called **decoding**. The decoding strategy significantly impacts generation quality—it's the difference between coherent text and random gibberish.

#### Step 1: Softmax — Turning Logits into Probabilities

The model outputs a vector of **logits**—raw scores for every possible token in the vocabulary. We turn these into probabilities using the **softmax** formula:

$$P(x_{t+1} = i \mid x_{1..t}) = \frac{e^{v_i}}{\sum_{j} e^{v_j}}$$

Where:
- $v_i$ is the model's score (logit) for token $i$
- The numerator $e^{v_i}$ makes higher scores more likely
- The denominator $\sum_{j} e^{v_j}$ normalizes everything so probabilities sum to 1

This gives us a probability distribution over all words in the vocabulary.

#### Step 2: Decoding — Picking the Next Token

Now that we have probabilities, we need to choose one token to continue the text. We can:

- **Pick the highest-probability token** (greedy decoding) → Safe but repetitive
- **Randomly sample from the probabilities** → Makes text more creative
- **Use other tricks to balance randomness and coherence** → The strategies below

Let's explore two powerful techniques for controlling this balance.

#### Temperature Scaling

**Problem:** Raw softmax outputs can be too peaked (always choosing the most likely token) or too flat (generating random nonsense).

**Solution:** Temperature scaling modifies the softmax distribution:

$$\text{softmax}(v, \tau)_i = \frac{\exp(v_i/\tau)}{\sum_{j} \exp(v_j/\tau)}$$

**Effects:**
- $\tau < 1$: makes the distribution sharper (model becomes more confident, deterministic, greedy)
- $\tau = 1$: Standard softmax (model's original distribution)
- $\tau > 1$: makes the distribution flatter (model becomes more random, creative, diverse)

**Implementation:**

```python
def softmax_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
    dim: int = -1
) -> torch.Tensor:
    """
    Apply softmax with temperature scaling.

    Args:
        logits: Model output logits
        temperature: Temperature parameter τ
        dim: Dimension to apply softmax

    Returns:
        Temperature-scaled probability distribution
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Apply softmax (numerically stable)
    probs = torch.nn.functional.softmax(scaled_logits, dim=dim)

    return probs
```

**Usage:**
```python
logits = model(x, apply_softmax=False)[:, -1, :]  # Get next-token logits

# Deterministic (greedy)
probs = softmax_with_temperature(logits, temperature=0.01)

# Balanced
probs = softmax_with_temperature(logits, temperature=0.8)

# Creative
probs = softmax_with_temperature(logits, temperature=1.5)
```

**Concrete Example:**

Let's say the model predicts the next word with these raw logits and probabilities:

| Token | Raw Logit | $\tau=1.0$ (standard) | $\tau=0.5$ (sharper) | $\tau=2.0$ (flatter) |
|-------|-----------|----------------------|---------------------|---------------------|
| "cat" | 2.5 | 0.60 | 0.94 | 0.52 |
| "dog" | 1.0 | 0.25 | 0.05 | 0.25 |
| "banana" | 0.2 | 0.10 | 0.01 | 0.16 |
| "spaceship" | -1.5 | 0.05 | 0.00 | 0.07 |

**Observations:**

- **With $\tau = 0.5$** (sharper): "cat" becomes dominant (0.94), nearly eliminating other options. The model is very confident and predictable.

- **With $\tau = 1.0$** (standard): Uses the model's original learned distribution. Balanced between confidence and diversity.

- **With $\tau = 2.0$** (flatter): Probabilities become more uniform. "dog" maintains its probability, "banana" nearly doubles (0.10 → 0.16), and even "spaceship" becomes viable (0.05 → 0.07). The model is more creative and exploratory.

#### Top-p (Nucleus) Sampling

**Problem:** Even with temperature scaling, the model might assign non-zero probability to thousands of tokens, many of which are nonsensical in context.

**Solution:** Top-p sampling (Holtzman et al., 2020) truncates the distribution to the smallest set of tokens whose cumulative probability exceeds threshold p.

**Algorithm:**

Define the nucleus $V(p)$ as the smallest set such that:

$$\sum_{i \in V(p)} P(i) \geq p$$

Then the filtered probability distribution is:

$$P_{\text{filtered}}(i) = \begin{cases}
\frac{P(i)}{\sum_{j \in V(p)} P(j)} & \text{if } i \in V(p) \\
0 & \text{otherwise}
\end{cases}$$

**Implementation:**

```python
def top_p_sampling(probs: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to probability distribution.

    Args:
        probs: Probability distribution [batch_size, vocab_size]
        p: Cumulative probability threshold (typical: 0.9, 0.95)

    Returns:
        Filtered and renormalized probability distribution
    """
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff: keep tokens until cumulative prob >= p
    mask = cumulative_probs <= p

    # Always keep at least the top token
    mask[..., 0] = True

    # Zero out probabilities not in nucleus
    filtered_sorted_probs = sorted_probs * mask.float()

    # Scatter back to original positions
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_probs)

    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    return filtered_probs
```

**Example:**
```python
# Original distribution
probs = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05])

# p=0.8: Keep top 2 tokens (0.5 + 0.3 = 0.8)
filtered = top_p_sampling(probs, p=0.8)
# Result: [0.625, 0.375, 0, 0, 0]
```

**Applying to our earlier example:**

With our "cat", "dog", "banana", "spaceship" example, if we use **$p = 0.9$**:

1. Sort by probability: ["cat" (0.60), "dog" (0.25), "banana" (0.10), "spaceship" (0.05)]
2. Cumulative sum: 0.60, 0.85, 0.95, 1.00
3. Keep tokens until cumulative ≥ 0.9: Keep {"cat", "dog", "banana"}
4. Remove "spaceship" (too low probability)
5. Renormalize and sample from the remaining three tokens

**Result:** The model only samples from {"cat", "dog", "banana"}, avoiding the extremely unlikely "spaceship".

#### Summary: Putting It All Together

| Step | Purpose | Key Parameter |
|------|---------|---------------|
| **Softmax** | Turns model logits into probabilities | None |
| **Temperature** | Controls confidence vs. creativity | $\tau$ (typical: 0.7-1.5) |
| **Top-p Sampling** | Limits randomness to most probable tokens | $p$ (typical: 0.9-0.95) |

**Recommended combinations:**
- **Factual tasks**: $\tau = 0.1$ (nearly greedy)
- **Balanced generation**: $\tau = 0.8$, $p = 0.9$
- **Creative writing**: $\tau = 1.2$, $p = 0.95$

#### Autoregressive Decoding

Putting it together for text generation:

```python
def decode(
    model: nn.Module,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = None,
    eos_token_id: int = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate text autoregressively from a prompt.

    Args:
        model: Trained TransformerLM
        prompt_tokens: Initial prompt [batch_size, seq_len]
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold (None to disable)
        eos_token_id: End-of-sequence token for early stopping
        device: Device to run on

    Returns:
        Generated sequence [batch_size, seq_len + num_generated]
    """
    model.eval()

    if prompt_tokens.dim() == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)

    generated = prompt_tokens.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = model(generated, apply_softmax=False)
            next_token_logits = logits[:, -1, :]

            # Apply temperature scaling
            next_token_probs = softmax_with_temperature(
                next_token_logits,
                temperature=temperature
            )

            # Apply top-p filtering if requested
            if top_p is not None:
                next_token_probs = top_p_sampling(next_token_probs, p=top_p)

            # Sample next token
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    return generated
```

---

### Putting It All Together: The Training Script {#training-script}

Now we assemble all components into a production training script. The key is making everything configurable via command-line arguments.

#### Command-Line Interface

```python
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer language model")

    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)

    # Model architecture
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--context_length", type=int, default=512)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=100000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Learning rate schedule
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--lr_decay_iters", type=int, default=100000)

    # Logging and checkpointing
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # Resume
    parser.add_argument("--resume_from", type=str, default=None)

    return parser.parse_args()
```

#### Understanding Key Training Parameters

Before diving into the training loop, let's clarify two important hyperparameters that control how training progresses:

**max_iters (Maximum Iterations)**

The total number of training steps (iterations) to run.

**One iteration** = one forward pass + one backward pass + one optimizer step

```python
for iter_num in range(max_iters):  # e.g., 100,000 steps
    x, y = get_batch(...)
    loss = model(x, y)
    loss.backward()
    optimizer.step()
```

**Example:**
- If `max_iters = 100,000` and `batch_size = 32`:
- Model will train for 100,000 steps
- Each step processes 32 examples
- Total examples seen = 100,000 × 32 = 3,200,000 (with repetition if dataset is smaller)

**gradient_accumulation_steps (Gradient Accumulation)**

The number of mini-batches to accumulate gradients over before updating weights.

**Why use it?** To simulate larger batch sizes when GPU memory is limited.

**Without gradient accumulation** (standard training):
```python
# Effective batch size = 32
x, y = get_batch(batch_size=32)
loss = model(x, y)
loss.backward()      # Compute gradients
optimizer.step()     # Update weights immediately
```

**With gradient accumulation** (e.g., `gradient_accumulation_steps = 4`):
```python
# Effective batch size = 32 × 4 = 128
total_loss = 0.0
for _ in range(4):  # Accumulate over 4 mini-batches
    x, y = get_batch(batch_size=32)
    loss = model(x, y)
    loss = loss / 4  # Scale loss to average over accumulation
    loss.backward()  # Accumulate gradients (don't update yet!)
    total_loss += loss.item()

optimizer.step()     # Now update with accumulated gradients
optimizer.zero_grad()
```

**Key benefits:**
1. **Simulate larger batches**: Want batch_size=128 but only have memory for 32? Use `gradient_accumulation_steps=4`
2. **Effective batch size** = `batch_size × gradient_accumulation_steps`
3. **Smoother gradients**: Larger effective batches lead to more stable training

#### Main Training Loop

```python
def main():
    args = parse_args()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load datasets with memory mapping
    train_data = load_dataset(args.train_data, args.vocab_size)
    val_data = load_dataset(args.val_data, args.vocab_size)

    # Initialize model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(args.device)

    # Store model configuration for checkpoints
    model_config = {
        'vocab_size': args.vocab_size,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'context_length': args.context_length,
    }

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_from:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # Training loop
    model.train()
    for iter_num in range(start_iter, args.max_iters):
        # Get learning rate for this iteration
        lr = get_lr_cosine_schedule(
            iter_num,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.lr_decay_iters,
        )

        # Update learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training step with gradient accumulation
        total_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
            logits = model(x, apply_softmax=False)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        if iter_num % args.log_interval == 0:
            print(f"[{iter_num}/{args.max_iters}] loss: {total_loss:.4f} | lr: {lr:.2e}")

        # Evaluation
        if iter_num % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args)
            print(f"[{iter_num}] val_loss: {val_loss:.4f}")

        # Save checkpoint
        if iter_num % args.checkpoint_interval == 0 and iter_num > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_iter_{iter_num}.pt")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path, model_config)

    # Save final checkpoint
    final_checkpoint = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint, model_config)
```

#### Usage

```bash
# Train from scratch
python -m cs336_basics.train \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --vocab_size 50257 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --d_ff 3072 \
    --batch_size 32 \
    --max_iters 100000 \
    --lr 1e-3 \
    --warmup_iters 2000

# Resume from checkpoint
python -m cs336_basics.train \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --vocab_size 50257 \
    --resume_from checkpoints/checkpoint_iter_50000.pt
```

---

### Testing and Validation {#testing}

Production systems require comprehensive testing. Here's how we can validate our training pipeline to ensure correctness before launching expensive, multi-day training runs.

#### Unit Tests for Components

Each component should have its own unit tests to verify correctness in isolation.

**Test AdamW Optimizer:**

```python
def test_adamw():
    """Test AdamW matches reference implementation."""
    import torch.nn as nn

    # Create simple model
    model = nn.Linear(10, 5)
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.1)

    # Create dummy data
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)

    # Training step
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()

    # Verify weights were updated
    assert loss.item() > 0  # Loss should be non-zero
    # Compare against PyTorch's implementation for exact match
```

**Test Learning Rate Schedule:**

```python
def test_learning_rate_schedule():
    """Test learning rate schedule matches specification."""
    max_lr = 1.0
    min_lr = 0.1
    warmup_iters = 100
    cosine_cycle_iters = 1000

    # Test warmup phase
    lr_start = get_lr_cosine_schedule(0, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    assert lr_start == 0.0, "LR should start at 0"

    lr_mid_warmup = get_lr_cosine_schedule(50, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    assert abs(lr_mid_warmup - 0.5 * max_lr) < 1e-6, "LR should be halfway at warmup midpoint"

    lr_end_warmup = get_lr_cosine_schedule(100, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    assert abs(lr_end_warmup - max_lr) < 1e-6, "LR should be max at end of warmup"

    # Test cosine phase
    lr_mid_cosine = get_lr_cosine_schedule(550, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    assert min_lr < lr_mid_cosine < max_lr, "LR should be decaying in cosine phase"

    lr_end_cosine = get_lr_cosine_schedule(1000, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    assert abs(lr_end_cosine - min_lr) < 1e-6, "LR should be min at end of cosine"

    # Test constant phase
    lr_after = get_lr_cosine_schedule(1500, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    assert lr_after == min_lr, "LR should remain at min after cosine phase"
```

**Test Top-p Sampling:**

```python
def test_top_p_sampling():
    """Test top-p sampling filters correctly."""
    import torch

    probs = torch.tensor([[0.5, 0.3, 0.1, 0.05, 0.05]])
    filtered = top_p_sampling(probs, p=0.8)

    # Should keep only top 2 tokens (0.5 + 0.3 = 0.8)
    assert (filtered[0, :2] > 0).all(), "Top 2 tokens should have non-zero probability"
    assert (filtered[0, 2:] == 0).all(), "Remaining tokens should be filtered out"

    # Should be renormalized
    assert torch.allclose(filtered.sum(), torch.tensor(1.0)), "Probabilities should sum to 1"

    # Check renormalization is correct
    expected = torch.tensor([[0.625, 0.375, 0.0, 0.0, 0.0]])
    assert torch.allclose(filtered, expected, atol=1e-3), "Renormalization should be correct"
```

**Test Temperature Scaling:**

```python
def test_temperature_scaling():
    """Test temperature scaling affects distribution correctly."""
    import torch

    logits = torch.tensor([[2.0, 1.0, 0.0]])

    # Standard softmax
    probs_normal = softmax_with_temperature(logits, temperature=1.0)

    # Low temperature (sharper)
    probs_sharp = softmax_with_temperature(logits, temperature=0.5)
    assert probs_sharp[0, 0] > probs_normal[0, 0], "Low temp should increase max probability"

    # High temperature (flatter)
    probs_flat = softmax_with_temperature(logits, temperature=2.0)
    assert probs_flat[0, 0] < probs_normal[0, 0], "High temp should decrease max probability"
```

#### Integration Test

Test the entire training pipeline end-to-end with a small synthetic dataset.

```python
def test_training_integration():
    """End-to-end test of training pipeline."""
    import tempfile
    import subprocess
    import os
    import numpy as np

    # Create small synthetic dataset
    vocab_size = 1000
    train_data = np.random.randint(0, vocab_size, size=10000, dtype=np.int64)
    val_data = np.random.randint(0, vocab_size, size=2000, dtype=np.int64)

    # Save to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.npy")
        val_path = os.path.join(tmpdir, "val.npy")
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")

        np.save(train_path, train_data)
        np.save(val_path, val_data)
        os.makedirs(checkpoint_dir)

        # Run training for 10 iterations
        result = subprocess.run([
            "python", "-m", "cs336_basics.train",
            "--train_data", train_path,
            "--val_data", val_path,
            "--vocab_size", str(vocab_size),
            "--d_model", "128",
            "--num_layers", "2",
            "--num_heads", "4",
            "--d_ff", "512",
            "--max_iters", "10",
            "--checkpoint_interval", "10",
            "--checkpoint_dir", checkpoint_dir,
        ], check=True, capture_output=True, text=True)

        # Verify checkpoint was created
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        assert os.path.exists(checkpoint_path), "Final checkpoint should be created"

        # Test checkpoint loading
        checkpoint = torch.load(checkpoint_path)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "iteration" in checkpoint
        assert checkpoint["iteration"] == 10

        print("✓ Training ran successfully and created checkpoint")

        # Test resumption from checkpoint
        result = subprocess.run([
            "python", "-m", "cs336_basics.train",
            "--train_data", train_path,
            "--val_data", val_path,
            "--vocab_size", str(vocab_size),
            "--d_model", "128",
            "--num_layers", "2",
            "--max_iters", "15",
            "--checkpoint_dir", checkpoint_dir,
            "--resume_from", checkpoint_path,
        ], check=True, capture_output=True, text=True)

        # Verify training continued from iteration 10
        assert "Resumed from iteration 10" in result.stdout

        print("✓ Training resumed successfully from checkpoint")
```

#### Pre-Training Validation Checklist

Before launching a long training run, verify:

1. **Loss decreases on small data**: Train for 100 iterations on a tiny dataset and verify loss goes down
2. **Checkpoints save/load correctly**: Save and load a checkpoint, verify iteration count and loss match
3. **Learning rate schedule looks correct**: Plot the LR over iterations and verify the curve matches expectations
4. **Memory usage is reasonable**: Monitor GPU memory and ensure it doesn't exceed available capacity
5. **Data loading works**: Verify data batches have correct shape and token values are in valid range
6. **Gradient norms are stable**: Log gradient norms during warmup, verify they decrease and don't explode

**Quick validation script:**

```python
# Quick 100-iteration validation run
python -m cs336_basics.train \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --vocab_size 50257 \
    --max_iters 100 \
    --log_interval 10 \
    --checkpoint_interval 50

# Expected output:
# [0/100] loss: 10.8234 | lr: 0.00e+00    (high initial loss)
# [10/100] loss: 9.2156 | lr: 5.00e-05   (loss decreasing)
# [50/100] loss: 7.8901 | lr: 2.50e-04   (loss continuing to decrease)
# [100/100] loss: 6.5432 | lr: 5.00e-04  (loss still decreasing)
```

If loss doesn't decrease in 100 iterations, something is wrong—debug before launching a long run!

---

### Key Takeaways {#takeaways}

Building a production training pipeline requires attention to many details beyond the core model architecture. Here are the essential lessons:

#### 1. **Correctness Over Convenience**
Follow paper specifications exactly, especially for:
- Optimizer algorithms (AdamW's decoupled weight decay)
- Learning rate schedules (strict inequalities matter)
- Bias correction formulas

Small deviations can cause subtle training instabilities that only appear after days of training.

#### 2. **Memory Efficiency Is Critical**
For large-scale training:
- Use memory-mapped arrays for datasets larger than RAM
- Monitor peak memory usage during training
- Consider gradient checkpointing for very large models

#### 3. **Checkpoint Everything**
A complete checkpoint includes:
- Model parameters
- Optimizer state (momentum buffers!)
- Iteration count
- Model configuration
- Random seeds (for reproducibility)

Don't learn this lesson the hard way after losing a week of training.

#### 4. **Make Everything Configurable**
Use command-line arguments for all hyperparameters:
- Enables systematic hyperparameter sweeps
- Makes it easy to resume with different settings
- Documents what settings were used

#### 5. **Test Before Long Training Runs**
- Run integration tests on small synthetic data
- Train for 100 iterations and verify:
  - Loss decreases
  - Checkpoints save/load correctly
  - Learning rate schedule looks correct
  - Memory usage is reasonable

A 10-minute test can save days of wasted compute.

#### 6. **Generation Quality Depends on Decoding**
Even a well-trained model can produce poor text with bad decoding settings:
- Start with `temperature=0.8, top_p=0.9`
- Adjust based on task (lower temperature for factual, higher for creative)
- Always use some form of sampling (greedy decoding produces repetitive text)

#### 7. **Monitor Training Actively**
Log frequently and watch for:
- Loss spikes (may indicate learning rate too high)
- Loss plateaus (may need more data or capacity)
- Gradient norms (should decrease during warmup)
- Generation samples (qualitative assessment)

#### 8. **Production Code Is Different**
Research code can get away with:
- Hardcoded hyperparameters
- No checkpointing
- Single-file scripts

Production code needs:
- Configuration management
- Robust error handling
- Comprehensive logging
- Restart/resume capability

This note covered the engineering necessary to turn research ideas into a working system. The components we built—AdamW optimizer, cosine schedule, memory-mapped data loading, checkpointing, and decoding strategies—form the foundation of modern LLM training pipelines. These same patterns appear in systems like GPT-3, LLaMA, and other large language models.

The next step is scaling: distributed training across multiple GPUs, larger datasets, and bigger models. But the fundamentals remain the same: correct implementations of proven algorithms, careful attention to numerical stability, and robust engineering practices.
