---
layout: post
title: "Study Notes: Stanford CS336 Language Modeling from Scratch [9]"
categories: cs336
author: 
- Han Yu
---

## Understanding Softmax, Log-Softmax, and Cross-Entropy: A Complete Implementation Guide
This note explains how to implement `Softmax`, `Log-Softmax`, and `Cross-Entropy` from scratch in PyTorch, highlighting key mathematical tricks to ensure numerical stability. It shows why subtracting the maximum logit before exponentiation prevents **overflow** and **underflow**, and walks through essential PyTorch tensor operations—`dim`, `keepdim`, `view`, and `reshape`—that are critical for implementing machine learning algorithms efficiently.

### Table of Contents
1. [Numerical Stability Deep Dive](#numerical-stability-deep-dive)
2. [PyTorch Fundamentals: `dim`, `keepdim`, and `view`](#pytorch-fundamentals)
3. [The Implementation](#the-implementation)
4. [Detailed Explanation of the Implementation](#detailed-explanation)

---

### Numerical Stability Deep Dive {#numerical-stability-deep-dive}

Before diving into the implementation, it's crucial to understand why numerical stability matters. When implementing operations like softmax and cross-entropy, we must carefully handle potential **overflow** and **underflow** issues. Let's explore these challenges and their solutions.

#### Why We Subtract the Maximum: Preventing Overflow and Underflow

When implementing softmax from scratch, you'll encounter a critical numerical stability trick: subtracting the maximum value before computing exponentials. Let's explore why this matters with concrete examples.

#### The Problem: Exponential Overflow

The softmax formula is:

$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$

##### Example 1: Large Numbers Cause Overflow

Suppose you have logits: `[1000, 1001, 1002]`

**Naive approach (without subtracting max):**

$e^{1000} \approx 2 \times 10^{434}$,
$e^{1001} \approx 5 \times 10^{434}$,
$e^{1002} \approx 1.4 \times 10^{435}$

These numbers are astronomically large and exceed what a computer can represent (approximately $10^{308}$ for 64-bit floats). Python/PyTorch returns `inf` (infinity), leading to:

**Result:** `[inf, inf, inf]` → Division gives `[nan, nan, nan]` ❌

**Demonstration in PyTorch:**

```python
import torch

logits = torch.tensor([1000.0, 1001.0, 1002.0])

# Naive implementation (broken!)
exp_vals = torch.exp(logits)
print("Exponentials:", exp_vals)
# Output: tensor([inf, inf, inf])

sum_exp = torch.sum(exp_vals)
print("Sum:", sum_exp)
# Output: tensor(inf)

result = exp_vals / sum_exp
print("Result:", result)
# Output: tensor([nan, nan, nan])  ← BROKEN!
```

##### Example 2: With Numerical Stability Trick

Same logits: `[1000, 1001, 1002]`

**Step 1:** Subtract the maximum value (1002):

$[1000, 1001, 1002] - 1002 = [-2, -1, 0]$

**Step 2:** Compute exponentials on the stable values:

$e^{-2} \approx 0.135$,
$e^{-1} \approx 0.368$,
$e^{0} = 1.0$

These are manageable numbers with no overflow!

**Step 3:** Normalize to get probabilities:

$\text{sum} = 0.135 + 0.368 + 1.0 = 1.503$

$\text{softmax} = \left[\frac{0.135}{1.503}, \frac{0.368}{1.503}, \frac{1.0}{1.503}\right] = [0.090, 0.245, 0.665]$

**Demonstration in PyTorch:**

```python
import torch

logits = torch.tensor([1000.0, 1001.0, 1002.0])

# Stable implementation
max_val = torch.max(logits)
print("Max value:", max_val)
# Output: tensor(1002.)

logits_stable = logits - max_val
print("Stabilized logits:", logits_stable)
# Output: tensor([-2., -1.,  0.])

exp_vals = torch.exp(logits_stable)
print("Exponentials:", exp_vals)
# Output: tensor([0.1353, 0.3679, 1.0000])

sum_exp = torch.sum(exp_vals)
print("Sum:", sum_exp)
# Output: tensor(1.5032)

result = exp_vals / sum_exp
print("Result:", result)
# Output: tensor([0.0900, 0.2447, 0.6652])  ← WORKS!

# Verify probabilities sum to 1
print("Sum of probabilities:", torch.sum(result))
# Output: tensor(1.0000)
```
#### Mathematical Proof: Why This Works

The stability trick is mathematically sound because:

$\text{softmax}(x) = \text{softmax}(x - c)$

for any constant $c$!

**Proof:**

$\frac{e^{x_i - c}}{\sum_{j} e^{x_j - c}} = \frac{e^{x_i} \cdot e^{-c}}{\sum_{j} e^{x_j} \cdot e^{-c}} = \frac{e^{x_i} \cdot e^{-c}}{e^{-c} \cdot \sum_{j} e^{x_j}} = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$

The $e^{-c}$ terms cancel out! By choosing $c = \max(x)$, we ensure the largest exponent becomes 0, preventing overflow while maintaining mathematical correctness.

#### The Underflow Problem

There's also a potential underflow issue with very negative numbers:

$e^{-1000} \approx 0$

This underflows to zero in floating-point arithmetic. However, by subtracting the maximum, the largest value becomes 0 ($e^0 = 1$), and only smaller values might underflow. This is acceptable because extremely small exponentials contribute negligibly to the sum anyway.

**Demonstration:**

```python
import torch

# Very negative logits
logits = torch.tensor([-1000.0, -999.0, -998.0])

# Without stability trick
exp_vals = torch.exp(logits)
print("Exponentials:", exp_vals)
# Output: tensor([0., 0., 0.])  ← All underflow!

# With stability trick
max_val = torch.max(logits)
logits_stable = logits - max_val
exp_vals_stable = torch.exp(logits_stable)
print("Stable exponentials:", exp_vals_stable)
# Output: tensor([0.1353, 0.3679, 1.0000])  ← Works perfectly!

result = exp_vals_stable / torch.sum(exp_vals_stable)
print("Result:", result)
# Output: tensor([0.0900, 0.2447, 0.6652])
```
#### Why This Matters in Deep Learning

In deep learning, logits frequently reach magnitudes of hundreds or thousands, especially in:

- **Language models** with large vocabulary sizes (tens of thousands of classes)
- **Deep networks** where activations accumulate through many layers
- **Unnormalized outputs** before the final softmax layer
- **Training dynamics** where gradients can push logits to extreme values

Without the stability trick, your model would crash with `nan` values during training or inference, making it impossible to:
- Train the model (gradients become `nan`)
- Make predictions (outputs become `nan`)
- Debug issues (everything breaks catastrophically)

This simple technique—subtracting the maximum—keeps all exponentials in a safe computational range (approximately 0 to 1) while computing the exact same mathematical result.

**Real-world example from GPT-2:**

```python
# Typical logits from a language model
logits = torch.randn(1, 50257) * 10  # vocab_size = 50,257
print("Logit range:", logits.min().item(), "to", logits.max().item())
# Output: Logit range: -28.3 to 31.7

# Without stability trick (would overflow!)
# With stability trick (works perfectly)
probs = softmax(logits, dim=-1)
print("Probability range:", probs.min().item(), "to", probs.max().item())
# Output: Probability range: 1.2e-27 to 0.0043
print("Sum:", probs.sum().item())
# Output: Sum: 1.0
```
---

### PyTorch Fundamentals: `dim`, `keepdim`, and `view` {#pytorch-fundamentals}

Now that we understand the importance of numerical stability, we need to master the essential PyTorch operations that enable us to implement these stable operations efficiently. Before diving into the implementation details, it is important to understand the three fundamental PyTorch operations on tensors.

#### 1. Understanding `dim` (Dimension/Axis)

In PyTorch, tensors can have multiple dimensions. The `dim` parameter specifies **which dimension** to operate along.

##### Dimension Indexing

```python
import torch

# A 2D tensor (matrix)
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x.shape)  # torch.Size([2, 3])
# dim=0 → rows (size 2)
# dim=1 → columns (size 3)
```

**Visual representation:**

```
        dim=1 →
       [1, 2, 3]
dim=0  [4, 5, 6]
  ↓
```

##### Operations Along Different Dimensions

```python
# Sum along dim=0 (collapse rows, keep columns)
result_dim0 = torch.sum(x, dim=0)
print(result_dim0)  # tensor([5, 7, 9])
# Adds: [1+4, 2+5, 3+6]

# Sum along dim=1 (collapse columns, keep rows)
result_dim1 = torch.sum(x, dim=1)
print(result_dim1)  # tensor([6, 15])
# Adds: [1+2+3, 4+5+6]
```

**Visual:**

```
Original:          Sum along dim=0:             Sum along dim=1:
[1, 2, 3]          [5, 7, 9]                    [6, 15]
[4, 5, 6]                                       
```

##### 3D Tensor Example

```python
# Shape: [2, 3, 4] means 2 "matrices" of size 3x4
x = torch.randn(2, 3, 4)

print(x.shape)  # torch.Size([2, 3, 4])
# dim=0 → first dimension (size 2)
# dim=1 → second dimension (size 3)
# dim=2 → third dimension (size 4)
# dim=-1 → last dimension (same as dim=2)
# dim=-2 → second to last (same as dim=1)
```

**Negative indexing:**
- `dim=-1` always refers to the **last dimension**
- `dim=-2` refers to the **second to last**, etc.

#### 2. Understanding `keepdim=True`

`keepdim` controls whether the reduced dimension is **kept** or **removed** after an operation.

##### Example: `keepdim=False` (default)

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x.shape)  # torch.Size([2, 3])

# Sum along dim=1 WITHOUT keepdim
result = torch.sum(x, dim=1, keepdim=False)
print(result)        # tensor([6, 15])
print(result.shape)  # torch.Size([2])  ← dimension collapsed!
```

The dimension is **removed**, so shape goes from `[2, 3]` → `[2]`

##### Example: `keepdim=True`

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x.shape)  # torch.Size([2, 3])

# Sum along dim=1 WITH keepdim
result = torch.sum(x, dim=1, keepdim=True)
print(result)        # tensor([[6], [15]])
print(result.shape)  # torch.Size([2, 1])  ← dimension kept!
```

The dimension is **preserved** (but size becomes 1), so shape goes from `[2, 3]` → `[2, 1]`

##### Why `keepdim=True` Matters: Broadcasting

`keepdim=True` is crucial for **broadcasting** operations:

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# Without keepdim
mean_no_keep = torch.mean(x, dim=1, keepdim=False)
print(mean_no_keep.shape)  # torch.Size([2])

# This will fail! Shapes don't match for broadcasting
# x - mean_no_keep  # Error!

# With keepdim
mean_keep = torch.mean(x, dim=1, keepdim=True)
print(mean_keep.shape)  # torch.Size([2, 1])

# This works! Broadcasting happens correctly
normalized = x - mean_keep
print(normalized)
# tensor([[-1., 0., 1.],
#         [-1., 0., 1.]])
```

**Visual explanation:**

```
Original x:        mean (keepdim=True):    Broadcasting x - mean:
[1, 2, 3]          [2]                     [1, 2, 3]   [2]
[4, 5, 6]          [5]                     [4, 5, 6] - [5]
                                                        ↓
Shape [2, 3]       Shape [2, 1]            [1-2, 2-2, 3-2]
                                           [4-5, 5-5, 6-5]
                                            = [[-1, 0, 1],
                                               [-1, 0, 1]]
```

##### More Examples with Different Operations

```python
x = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
print(x.shape)  # torch.Size([2, 2, 2])
# x is a stack of two [2,2] tensors:
# Tensor #0:
# [[1, 2],
#  [3, 4]]

# Tensor #1:
# [[5, 6],
#  [7, 8]]

# Max along dim=0, we're taking the max across the first dimension (dim=0), meaning we compare corresponding elements between the two [2,2] matrices
# [[max(1,5), max(2,6)],
#  [max(3,7), max(4,8)]]
# That gives
# [[5, 6],
#  [7, 8]]

max_vals_no_keep = torch.max(x, dim=0, keepdim=False)[0]
print(max_vals_no_keep.shape)  # torch.Size([2, 2])

max_vals_keep = torch.max(x, dim=0, keepdim=True)[0]
print(max_vals_keep.shape)  # torch.Size([1, 2, 2])
```

#### 3. Understanding `view()` - Reshaping Tensors

`view()` reshapes a tensor **without changing its data** : which means that the underlying values stored in memory stay exactly the same — PyTorch just interprets that same block of memory in a different shape.
##### Basic Reshaping

```python
x = torch.tensor([1, 2, 3, 4, 5, 6])
print(x.shape)  # torch.Size([6])

# Reshape to 2x3
x_2d = x.view(2, 3)
print(x_2d)
# tensor([[1, 2, 3],
#         [4, 5, 6]])
print(x_2d.shape)  # torch.Size([2, 3])

# Reshape to 3x2
x_3d = x.view(3, 2)
print(x_3d)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])
```

**Important:** The **total number of elements must remain the same**!

```python
x = torch.randn(6)
x.view(2, 3)  # ✓ Works: 6 = 2 × 3
x.view(3, 2)  # ✓ Works: 6 = 3 × 2
x.view(1, 6)  # ✓ Works: 6 = 1 × 6
# x.view(2, 2)  # ✗ Error: 6 ≠ 2 × 2
```

##### Using `-1` for Automatic Dimension Inference

You can use `-1` to let PyTorch **automatically calculate** one dimension:

```python
x = torch.randn(12)

# -1 means "figure out this dimension automatically"
x.view(3, -1)   # torch.Size([3, 4]) - PyTorch calculates 12/3 = 4
x.view(-1, 6)   # torch.Size([2, 6]) - PyTorch calculates 12/6 = 2
x.view(-1, 1)   # torch.Size([12, 1])
x.view(-1)      # torch.Size([12]) - flattens to 1D
```

**Rule:** You can only use `-1` for **one dimension** at a time.

##### Flattening Tensors

```python
# Common pattern: flatten all dimensions
x = torch.randn(2, 3, 4)  # Shape: [2, 3, 4]
print(x.shape)  # torch.Size([2, 3, 4])

# Flatten to 1D
x_flat = x.view(-1)
print(x_flat.shape)  # torch.Size([24])  (2*3*4 = 24)

# Flatten batch dimensions but keep last dimension
x_partial_flat = x.view(-1, 4)
print(x_partial_flat.shape)  # torch.Size([6, 4])  (2*3 = 6)
```

##### Practical Example: Batch Processing

```python
# Batch of images: [batch_size, height, width, channels]
images = torch.randn(32, 28, 28, 3)
print(images.shape)  # torch.Size([32, 28, 28, 3])

# Flatten each image for a fully connected layer
# Keep batch dimension, flatten the rest
images_flat = images.view(32, -1)
print(images_flat.shape)  # torch.Size([32, 2352])  (28*28*3 = 2352)

# Or equivalently:
images_flat = images.view(images.size(0), -1)
print(images_flat.shape)  # torch.Size([32, 2352])
```

##### `view()` vs `reshape()`

PyTorch has both `view()` and `reshape()`:

```python
x = torch.randn(2, 3)

# view() - requires contiguous memory
y = x.view(6)

# reshape() - works even if not contiguous (may copy data)
z = x.reshape(6)
```

**Key difference:**
- `view()`: Only works if tensor is **contiguous in memory**, otherwise raises error
- `reshape()` is more flexible:
    - If the tensor is contiguous, it behaves like .view() (no copy).
    - If it isn't contiguous, it will make a copy behind the scenes to ensure the new tensor has contiguous memory. That's why `reshape()` is safer, but sometimes slightly slower (copying costs time & memory).

#### Quick Reference Summary

| Operation | What it does | Example |
|-----------|--------------|---------|
| `dim=0` | Operate along first dimension | `torch.sum(x, dim=0)` |
| `dim=-1` | Operate along last dimension | `torch.max(x, dim=-1)` |
| `keepdim=True` | Keep dimension (size→1) | Shape `[2,3]` → `[2,1]` |
| `keepdim=False` | Remove dimension | Shape `[2,3]` → `[2]` |
| `view(a, b)` | Change the view of the data to `[a, b]` | `x.view(2, 3)` |
| `view(-1)` | Flatten to 1D | Shape `[2,3]` → `[6]` |
| `view(-1, n)` | Auto calculate first dim given `n` | `x.view(-1, 4)` |

---

### The Implementation {#the-implementation}

With a solid understanding of numerical stability concerns and the fundamental PyTorch operations, we're now ready to see how these concepts come together in a complete implementation. Below is a sample code snippet showing the complete implementation.

```python
import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply the softmax operation to a tensor along the specified dimension.

    Uses the numerical stability trick of subtracting the maximum value
    from all elements before applying exponential.

    Args:
        x: torch.Tensor - Input tensor
        dim: int - Dimension along which to apply softmax

    Returns:
        torch.Tensor - Output tensor with same shape as input, with normalized
                      probability distribution along the specified dimension
    """
    # Subtract maximum for numerical stability
    # keepdim=True ensures the shape is preserved for broadcasting
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    x_stable = x - max_vals

    # Apply exponential
    exp_vals = torch.exp(x_stable)

    # Compute sum along the specified dimension
    sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)

    # Normalize to get probabilities
    return exp_vals / sum_exp


def log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply the log-softmax operation to a tensor along the specified dimension.

    log_softmax(x) = log(softmax(x)) = x - log(sum(exp(x)))

    This is more numerically stable than computing log(softmax(x)) separately
    because it cancels the exp and log operations.

    Args:
        x: torch.Tensor - Input tensor
        dim: int - Dimension along which to apply log-softmax

    Returns:
        torch.Tensor - Output tensor with same shape as input, containing
                      log probabilities along the specified dimension
    """
    # Subtract maximum for numerical stability
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    x_stable = x - max_vals

    # Compute log(sum(exp(x_stable)))
    log_sum_exp = torch.log(torch.sum(torch.exp(x_stable), dim=dim, keepdim=True))

    # log_softmax = x_stable - log(sum(exp(x_stable)))
    return x_stable - log_sum_exp


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss given logits and targets.

    The cross-entropy loss is: -log(softmax(logits)[target])

    This implementation uses the log_softmax function which provides
    numerical stability by canceling log and exp operations.

    Args:
        logits: torch.Tensor of shape (..., vocab_size) - Unnormalized logits
        targets: torch.Tensor of shape (...,) - Target class indices

    Returns:
        torch.Tensor - Scalar tensor with average cross-entropy loss across all examples
    """
    # Compute log probabilities using log_softmax (numerically stable)
    log_probs = log_softmax(logits, dim=-1)

    # Get the log probability for the target class for each example
    # Flatten batch dimensions to handle any number of batch dims
    batch_shape = logits.shape[:-1]
    log_probs_flat = log_probs.view(-1, log_probs.size(-1))
    targets_flat = targets.view(-1)

    batch_indices = torch.arange(log_probs_flat.size(0), device=logits.device)
    target_log_probs = log_probs_flat[batch_indices, targets_flat]

    # Reshape back to original batch shape
    target_log_probs = target_log_probs.view(batch_shape)

    # Cross entropy: -log(softmax(o)[target]) = -log_prob[target]
    cross_entropy_loss = -target_log_probs

    # Return average across all batch dimensions
    return cross_entropy_loss.mean()
```

---

### Detailed Explanation of the Implementation  {#detailed-explanation}

Now that we've seen the complete implementation, let's break down each function to understand how they work.

#### Softmax Function

**Goal:** Convert raw scores (logits) into probabilities that sum to 1.

**Formula:** $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$

**Step-by-step:**

```python
# Step 1: Find maximum (for numerical stability)
max_vals = torch.max(x, dim=dim, keepdim=True)[0]
# keepdim=True keeps shape [2,1] instead of [2] → enables broadcasting
# [0] gets values (not indices)

# Step 2: Subtract maximum
x_stable = x - max_vals
# Prevents overflow: softmax(x) = softmax(x - c) mathematically

# Step 3: Exponentiate
exp_vals = torch.exp(x_stable)
# Apply e^x to each element

# Step 4: Sum and normalize
sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)
return exp_vals / sum_exp  # Division broadcasts correctly
```

**Example:** Input `[[1, 2, 3], [4, 5, 6]]` → Output `[[0.09, 0.24, 0.67], [0.09, 0.24, 0.67]]`


#### Log-Softmax Function

**Goal:** Compute $\log(\text{softmax}(x))$ without numerical overflow.

**Formula:** $\log(\text{softmax}(x_i)) = x_i - \log\left(\sum_{j} e^{x_j}\right)$

**Why not just `log(softmax(x))`?** Computing `log(exp(large_number))` can overflow. Log-softmax avoids this by staying in log-space.

**Step-by-step:**

```python
# Step 1 & 2: Subtract maximum (same as softmax)
max_vals = torch.max(x, dim=dim, keepdim=True)[0]
x_stable = x - max_vals

# Step 3: Compute log(sum(exp(x_stable)))
log_sum_exp = torch.log(torch.sum(torch.exp(x_stable), dim=dim, keepdim=True))
# This is the log of the denominator in softmax

# Step 4: Subtract to get log probabilities
return x_stable - log_sum_exp
```

**Example:** Input `[[1, 2, 3], [4, 5, 6]]` → Output `[[-2.41, -1.41, -0.41], [-2.41, -1.41, -0.41]]`


#### Cross-Entropy Function

**Goal:** Measure how well predictions match the correct labels.

**Formula:** $\mathcal{L} = -\log(\text{softmax}(\text{logits})[\text{target}])$

**Intuition:** Penalize low probabilities assigned to the correct class. Lower loss = better prediction.

**Step-by-step:**

```python
# Step 1: Get log probabilities
log_probs = log_softmax(logits, dim=-1)
# Shape: [batch, seq, vocab] → [batch, seq, vocab]

# Step 2: Save batch shape for later
batch_shape = logits.shape[:-1]  # e.g., [2, 2] for shape [2, 2, 3]

# Step 3: Flatten to 2D for easier indexing
log_probs_flat = log_probs.view(-1, log_probs.size(-1))  # [batch*seq, vocab]
targets_flat = targets.view(-1)                           # [batch*seq]
# Example: [2, 2, 3] → [4, 3] and [2, 2] → [4]

# Step 4: Extract log prob for the correct class of each example
batch_indices = torch.arange(log_probs_flat.size(0), device=logits.device)
target_log_probs = log_probs_flat[batch_indices, targets_flat]
# Advanced indexing: log_probs_flat[i, targets_flat[i]] for each i
# Gets the log probability that the model assigned to the correct class

# Step 5: Reshape back to original batch dimensions
target_log_probs = target_log_probs.view(batch_shape)

# Step 6: Apply negative and average
cross_entropy_loss = -target_log_probs
return cross_entropy_loss.mean()
```

**Understanding Advanced Indexing (Step 4):**

This is the trickiest part. Let's break it down with a concrete example:

```python
# Suppose we have:
log_probs_flat = torch.tensor([[-2.4, -1.4, -0.4],  # example 0: log probs for 3 classes
                               [-2.4, -1.4, -0.4],  # example 1
                               [-2.4, -1.4, -0.4],  # example 2
                               [-1.1, -1.1, -1.1]]) # example 3
# Shape: [4, 3]

targets_flat = torch.tensor([2, 1, 0, 1])  # correct class for each example
# Shape: [4]

# We want to extract:
# - example 0, class 2 → log_probs_flat[0, 2] = -0.4
# - example 1, class 1 → log_probs_flat[1, 1] = -1.4
# - example 2, class 0 → log_probs_flat[2, 0] = -2.4
# - example 3, class 1 → log_probs_flat[3, 1] = -1.1
```

**How advanced indexing works:**

When we have `tensor[indices1, indices2]`, PyTorch pairs up elements from both index arrays:
- `tensor[indices1[0], indices2[0]]`
- `tensor[indices1[1], indices2[1]]`
- `tensor[indices1[2], indices2[2]]`
- and so on...

```python
batch_indices = torch.tensor([0, 1, 2, 3])  # row indices
targets_flat = torch.tensor([2, 1, 0, 1])   # column indices

# This pairing happens:
result = log_probs_flat[batch_indices, targets_flat]
# → [log_probs_flat[0,2], log_probs_flat[1,1], log_probs_flat[2,0], log_probs_flat[3,1]]
# → [-0.4, -1.4, -2.4, -1.1]
```

**Why we need `batch_indices`:**

Without it, we'd just have `log_probs_flat[:, targets_flat]`, which tries to select multiple columns for ALL rows—not what we want! We need to select ONE element per row (the correct class for that specific example).

**Visual representation:**

```
log_probs_flat:          targets_flat:         batch_indices:
[[-2.4, -1.4, -0.4]         [2                     [0
 [-2.4, -1.4, -0.4]          1                      1
 [-2.4, -1.4, -0.4]          0                      2
 [-1.1, -1.1, -1.1]]         1]                     3]

Pairing: [0,2] [1,1] [2,0] [3,1]
         ↓     ↓     ↓     ↓
Result: [-0.4, -1.4, -2.4, -1.1]
```

**Summary:**
- Input: `log_probs_flat` shape `[4, 3]`, `targets_flat = [2, 1, 0, 1]`
- Output: `target_log_probs` shape `[4]` with values `[-0.4, -1.4, -2.4, -1.1]`
- Each value is the log probability of the correct class for that example

#### Putting It All Together

Here's a complete workflow showing how these functions work together:

```python
import torch

# Setup: batch=2, sequence=2, vocab=3
logits = torch.tensor([[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]],
                       [[7.0, 8.0, 9.0],
                        [1.0, 1.0, 1.0]]])
targets = torch.tensor([[2, 1],
                        [0, 1]])

# Compute loss
loss = cross_entropy(logits, targets)
print(f"Loss: {loss:.3f}")  # Loss: 1.330
```

**What happens internally:**
1. `log_softmax` converts logits to log probabilities
2. Flatten everything to make indexing easier
3. Extract log probability for each correct class
4. Reshape back and compute mean loss

**Key insight:** The entire pipeline is designed to compute $-\log(P(\text{correct class}))$ efficiently and stably.

---

### Key Takeaways

1. **Numerical stability isn't optional**: It's the difference between code that works and code that fails in production

2. **Always subtract the maximum** before computing softmax or log-softmax

3. **Use log-softmax for cross-entropy**: Computing `log(softmax(x))` separately is both slower and less stable than log-softmax

4. **The math is equivalent**: These tricks don't change the results—they just make them computable

5. **Modern frameworks do this automatically**: PyTorch's `torch.nn.functional.softmax()` and `torch.nn.functional.cross_entropy()`include these optimizations, but understanding them helps you:
   - Debug numerical issues
   - Implement custom loss functions
   - Appreciate the engineering behind deep learning libraries

6. **Test edge cases**: Always test your implementations with extreme values (very large, very small, very negative) to ensure numerical stability
