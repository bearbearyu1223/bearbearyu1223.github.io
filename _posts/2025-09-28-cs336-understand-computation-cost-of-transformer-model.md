---
layout: post
title: "Study Notes: Stanford CS336 Language Modeling from Scratch [7]"
categories: cs336
author: 
- Han Yu
---
## Understanding where the computation really goes in transformer language models*

When we talk about large language models like GPT-4 or Claude, we often hear impressive numbers: "175 billion parameters," "3.5 trillion FLOPs," "trained on thousands of GPUs." But what do these numbers actually mean? Where does all that computation go during inference? And how do these patterns change as models scale up?

In this notes, we'll dissect the computational anatomy of GPT-2 models, from the smallest to the largest variants, to understand exactly where the mathematical heavy lifting happens. By the end, hope you'll have an intuitive understanding of why certain optimization techniques matter more than others, and how the computational landscape shifts dramatically with model size and context length.

*This analysis was based on GPT-2 architectures, but the principles apply broadly to transformer-based language models. The specific percentages may vary, but the scaling laws and optimization insights remain relevant for understanding modern LLMs.*

### Table of Contents

1. [Setting the Stage: GPT-2 XL Under the Microscope](#setting-the-stage-gpt-2-xl-under-the-microscope)
2. [Part 1: Counting Every Parameter](#part-1-counting-every-parameter)
   - [The 1.56 Billion Parameter Breakdown](#the-156-billion-parameter-breakdown)
   - [Memory Requirements](#memory-requirements)
3. [Part 2: Following the FLOPS (Floating Point Operations per Second)](#part-2-following-the-flops-floating-point-operations-per-second)
   - [The 3.5 Trillion FLOPs Journey](#the-35-trillion-flops-journey)
   - [The Computational Cost Hierarchy](#the-computational-cost-hierarchy)
4. [Part 3: How Computational Patterns Change with Model Scale](#part-3-how-computational-patterns-change-with-model-scale)
   - [Scaling Trends](#scaling-trends)
   - [Optimization Strategy by Model Size](#optimization-strategy-by-model-size)
5. [Part 4: The Long Context Revolution](#part-4-the-long-context-revolution)
   - [The 38× Computational Explosion](#the-38-computational-explosion)
   - [The Quadratic Takeover](#the-quadratic-takeover)
   - [Memory Implications](#memory-implications)
6. [Part 5: Understanding Mixture of Experts (MoE)](#part-5-understanding-mixture-of-experts-moe)
   - [The Restaurant Kitchen Analogy](#the-restaurant-kitchen-analogy)
   - [How MoE Works in Practice](#how-moe-works-in-practice)
   - [Why MoE Provides Massive Savings](#why-moe-provides-massive-savings)
7. [Part 6: Key Insights and Practical Implications](#part-6-key-insights-and-practical-implications)
   - [For Model Developers](#for-model-developers)
   - [For Infrastructure Teams](#for-infrastructure-teams)
8. [Conclusion](#conclusion)

### **Setting the Stage: GPT-2 XL Under the Microscope**

Let's start by examining GPT-2 XL, one of the largest publicly available GPT-2 models, with these specifications:

- **Vocabulary size**: 50,257 tokens
- **Context/sequence length**: 1,024 tokens  
- **Layers**: 48 transformer blocks
- **Embedding Model dimension (d_model)**: 1,600
- **Attention heads**: 25 
- **Feed forward dimension (d_ff)**: 6,400 

### **Part 1: Counting Every Parameter**

#### **The 1.56 Billion Parameter Breakdown**

When we say GPT-2 XL has `1.56 billion` parameters, where exactly do they all go?

**Token & Position Embeddings: 82M parameters (5.3%)**
- **Token embeddings**: 50,257 tokens × 1,600 dimensions -> 80.4M parameters
- **Position embeddings**: 1,024 positions × 1,600 dimensions -> 1.6M parameters

*Think of these as lookup tables: each token gets its own 1,600-dimensional vector, and each position (1st word, 2nd word, etc.) gets its own vector too.*

**Transformer Layers: 1.47B parameters (94.7%)**

Each of the 48 layers contains:

* *Multi-Head Attention (10.2M parameters per layer):*
    - **Q, K, V projections**: 3 × (1,600 × 1,600) = 7.68M parameters
    - **Output projection**: 1,600 × 1,600 = 2.56M parameters

* *Feed Forward Network (20.5M parameters per layer):*
    - **Expansion layer (linear projection up)**: 1,600 × 6,400 + 6,400 bias = 10.24M parameters  
    - **Contraction layer (linear project down)**: 6,400 × 1,600 + 1,600 bias = 10.24M parameters

* *Layer Normalization (6,400 parameters per layer):*
    - **Two layer norms (e.g.,RMSNorm)** × 2 parameters each (e.g., two learnable parameters used in RMSNorm) × 1,600 dimensions = 6,400 parameters

**Final Layer Norm (e.g., 1 RMSNorm layer with 2 parameters, 1,600 dimensions): 3,200 parameters**

**Key Insight**: The feed forward networks contain `67%` of parameters per layer, while attention uses `33%`. This `2:1` ratio will become important when we analyze computation.

#### **Memory Requirements**

With single-precision floating point (each parameter is represented by 32 bits which is 4 bytes per parameter):
- **Total memory**: `1.56B × 4 bytes` -> **`6.2 GB`**

This is just for storing the model weights—actual inference requires additional memory for activations, gradients (if training), intermediate computations, and other overhead for using frameworks.

### **Part 2: Following the FLOPS (Floating Point Operations per Second)**

Now comes the crucial question: during a forward pass with 1,024 input tokens, where does the computational work actually happen?

#### **The 3.5 Trillion FLOPs Journey**
A FLOP stands for Floating Point Operation. It means one basic arithmetic operation (addition, subtraction, multiplication, division, etc.) performed on floating-point numbers (like 32-bit floats in neural nets).

Example:
>3.14×2.71 → 1 FLOP
>
>(3.14 × 2.71) + 1.23 → 2 FLOPs

FLOPS vs FLOP:

> FLOP = one operation.
>
> FLOPS = FLOPs per second → a measure of compute speed.

Example: 
> A GPU that can perform 1×10^12 FLOPs per second = 1 TFLOPS.

When we talk about FLOPs of a model: it means the total number of floating-point operations required for one forward pass (sometimes forward + backward during training).

Example: 
> A Transformer block does a lot of matrix multiplications (like attention and feedforward layers). Counting their FLOPs helps estimate compute cost and compare model efficiency.

Note: 
> For matrix multiplication (M × N) @ (N × K), the FLOPs = 2 × M × N × K

**Feed Forward Networks: 2.01 TFLOPs (57.4%)**
- Two matrix multiplications per layer: d_model ↔ d_ff expansion and contraction
- Why so expensive: The `4×` expansion (1,600 → 6,400), then contraction (6,400 → 1,600) creates huge matrix operations
- Per layer cost: 2 × 1,600 × 6,400 + 2 × 6,400 × 1,600 -> 41.9 GFLOPs 
- Total across all layers: 41.9 GFLOPs × 48 layers -> 2.01 TFLOPs

**Attention Linear Projections: 1.01 TFLOPs (28.7%)**  
- Four projections per layer: Query, Key, Value, and Output matrices
- Each projection: (1,024 × 1,600) @ (1,600 × 1,600) matrix multiplication
- Per projection cost: 2 × 1,024 × 1,600 × 1,600 = 5.24 GFLOPs
- Per layer cost (4 projections): 4 × 5.24 GFLOPs = 20.97 GFLOPs
- Total across all layers: 20.97 GFLOPs × 48 layers = 1.007 TFLOPs

**Attention Computation: 0.32 TFLOPs (9.2%)**
- **Q@K^T**: Computing attention scores
    - Creates the attention matrix that determines which tokens attend to which
    - Matrix multiplication: (1,024 × 1,600) @ (1,600 × 1,024) 
    - FLOPs per layer = 2 × 1024 × 1600 × 1024 ->  3,355,443,200 FLOPs per layer
    - Total across all layers: 3,355,443,200 × 48 layers -> 0.16 TFLOPs
- **Attention@V**: Applying attention weights to values 
    - Applies attention weights to value vectors to get final attended representations
    - Matrix multiplication: (1,024 × 1,024)  @  (1,024 × 1,600) 
    - FLOPs per layer = 2 × 1024 × 1024 × 1600 ->  3,355,443,200 FLOPs per layer
    - Total across all layers: 3,355,443,200 × 48 layers -> 0.16 TFLOPs 
- **Currently small** but *grows quadratically with longer sequences*

**Output Projection: 0.16 TFLOPs (4.7%)**
- Final projection from hidden states to 50,257 vocabulary logits
    - Input shape: (1,024 × 1,600) hidden states
    - Weight shape: (1,600 × 50,257) vocabulary projection
    - Output shape: (1,024 × 50,257) logits for each position
    - FLOPs: 2 × 1,024 × 1,600 × 50,257 -> 0.16 TFLOPs
- Large vocabulary makes this significant despite being a single operation

#### **The Computational Cost Hierarchy**

**🥇 Feed Forward is the King (57.4%)**

This is the most important finding: *feed forward networks consume more computation than everything else combined*. The 4× expansion factor creates the largest matrix operations in the entire model.

**🥈 Attention Linear Projections Runner-Up (28.7%)**

The four linear projections (Q, K, V, O) that prepare attention computations use nearly 30% of all FLOPs.

**🥉 The Famous Attention Mechanism (9.2%)**

Despite getting most research attention, the actual attention computation (the part that makes transformers special) uses less than 10% of computation for typical sequence lengths. But, it can *grow quadratically with longer sequences or context window*. 

**🏅 Vocabulary Bottleneck (4.7%)**

The final projection to vocabulary logits is notable due to GPT-2 XL's large vocabulary.

**Why This Distribution Matters**

This analysis reveals that **feed forward networks are the computational elephant in the room**. For current sequence lengths, optimizing feed forward networks provides bigger computational savings than optimizing attention mechanisms.

### **Part 3: How Computational Patterns Change with Model Scale**

Let's examine how FLOP distribution evolves across the entire GPT-2 family:

| Model | Layers | d_model | Total FLOPs | Feed Forward % | Attention Linear Proj % | Attention Computation % | Output Projection % |
|-------|--------|---------|-------------|------|-------------|-------------|----------|
| **Small** | 12 | 768 | 0.29 TFLOPs | 39.8% | 19.9% | 13.3% | **27.1%** |
| **Medium** | 24 | 1024 | 0.83 TFLOPs | 49.9% | 24.9% | 12.5% | 12.7% |
| **Large** | 36 | 1280 | 1.78 TFLOPs | 54.5% | 27.2% | 10.9% | 7.4% |
| **XL** | 48 | 1600 | 3.51 TFLOPs | **57.4%** | **28.7%** | 9.2% | 4.7% |

#### **Scaling Trends**

**📈 Feed Forward Dominance Grows**
- Small: 39.8% → XL: 57.4% (+17.7 percentage points)
- **Why**: Scales as O(d_model² × layers), growing faster than other components, and *growing quadratically with larger embedding model*.

**📉 Output Projection Becomes Negligible**  
- Small: 27.1% → XL: 4.7% (-22.4 percentage points)
- **Why**: Scales as O(d_model) while vocabulary size stays constant

**📉 Attention Computation Relatively Shrinks**
- Small: 13.3% → XL: 9.2% (-4.1 percentage points)  
- **Why**: Scales as O(d_model × layers), but growing only linear with larger embedding model. 

**📈 Attention Projections Grow Steadily**
- Small: 19.9% → XL: 28.7% (+8.8 percentage points)
- **Why**: Scales same as feed forward: O(d_model² × layers), and *growing quadratically with larger embedding model*.

#### **Optimization Strategy by Model Size**

**Small Models**: Output projection matters most (27% of computation)
- Focus on vocabulary efficiency and embedding optimizations

**Large Models**: Feed forward networks dominate (57% of computation)  
- Focus on Mixture of Experts (MoE), pruning, and quantization

This scaling analysis explains why techniques like **Mixture of Experts**—which primarily optimize feed forward networks—become increasingly important for large models (we will explain that later in this notes).

### **Part 4: The Long Context Revolution**

What happens when we extend GPT-2 XL's context from 1,024 to 16,384 tokens? The results are dramatic.

#### **The 38× Computational Explosion**

| Component | 1K Context | 16K Context | Scaling | Change |
|-----------|------------|-------------|---------|---------|
| **Attention Computation** | **9.2%** | **61.8%** | **256×** | **+52.6 pts** |
| **Feed Forward** | **57.4%** | **24.1%** | **16×** | **-33.3 pts** |
| **Attention Projections** | **28.7%** | **12.1%** | **16×** | **-16.6 pts** |
| **Output Projection** | **4.7%** | **2.0%** | **16×** | **-2.7 pts** |

#### **The Quadratic Takeover**

**Total FLOPs**: 3.5 TFLOPs → 133.4 TFLOPs (38× increase)

The most shocking change: **attention computation explodes from 9% to 62%** of total computation. Here's why:

- **Sequence length scaling**: 1,024 → 16,384 tokens (16× increase)
- **Linear components** (feed forward, projections): Scale by 16×
- **Quadratic components** (attention computation): Scale by 16² = 256×

The mathematical culprits are the `Q@K^T` and `Attention@V` operations, which both scale as `O(sequence_length²)`.

#### **Memory Implications**

The memory story is even more dramatic:
- **Attention matrices**: 50M → 12,885M elements (256× increase)
- **Storage requirement**: Each attention head must store a 16K×16K matrix

This reveals why **long context is the next major frontier** in LLM optimization, requiring techniques like:
- **Flash Attention**: Memory-efficient attention computation
- **Sparse Attention**: Only compute attention for relevant tokens  
- **Linear Attention**: Approximate attention with linear complexity
- **Sliding Window**: Limit attention to recent tokens

### **Part 5: Understanding Mixture of Experts (MoE)**

Given that feed forward networks dominate computation (57% in large models), let's understand why **Mixture of Experts** has become a game-changing optimization technique.

#### **The Restaurant Kitchen Analogy**

**Traditional Model**: One super-chef tries to cook everything—pizza, sushi, pasta, desserts. The chef gets exhausted and slower as the menu grows.

**MoE Model**: Multiple specialist chefs (pizza expert, sushi master, pasta chef, dessert specialist) with a smart dispatcher who sends each order to the right specialist. Only the relevant chef works on each dish.

#### How MoE Works in Practice
Instead of one giant feedforward network, we have many smaller expert networks (say 64 FFNs). A router (small gating network) decides which subset of experts (often 1 or 2) each token should use. Thus, for each token: Only ~1–2 experts are activated
The others are inactive (no compute for them).
#### Why MoE Provides Massive Savings
**Computational Savings**: 
- **Before**: Use 100% of the giant network for every input
- **After**: Use only 10-20% of total network capacity  
- **Result**: 5-10× faster inference with same quality

**Specialization Benefits**:
- **Expert 1**: Math and science
- **Expert 2**: Creative writing
- **Expert 3**: Code and programming  
- **Expert 4**: Languages and translation

**Scale Without Pain**:
- **Traditional**: 2× bigger model = 2× more computation
- **MoE**: 2× more experts ≈ same computation (since only 1-2 active)

Since feed forward networks use 57% of computation in large models, MoE can reduce this to 6-12%, eliminating the primary bottleneck.

### **Part 6: Key Insights and Practical Implications**

#### **For Model Developers**

**Small Models (< 1B parameters)**:
- Output projection optimization matters most
- Vocabulary efficiency and embedding techniques provide biggest gains
- Feed forward optimization secondary

**Large Models (> 10B parameters)**:  
- Feed forward networks are the primary target (MoE, quantization, pruning)
- Attention projection optimizations become important
- Output projection becomes negligible

**Long Context Models**:
- Attention computation becomes dominant bottleneck
- Memory optimization equally critical as computation optimization
- Linear attention mechanisms essential

#### **For Infrastructure Teams**

**Hardware Requirements Scale Predictably**:

Computational Scaling Patterns:

- Feed forward: Scales as O(d_model² × layers)
    - GPT-2 Small → XL: d_model grows 2.1x, layers grow 4x → FF computation grows ~17x
    - Dominates short-sequence/context window workloads (57% of computation)
    - Requires high tensor core utilization for matrix multiplication
- Attention: Scales as O(sequence_length² × layers)
    - 1K → 16K context window: sequence_length grows 16x → attention grows 256x
    - Becomes dominant for long sequences (62% computation at 16K context)
    - Requires specialized attention kernels and memory optimization

Memory Scaling Reality:
- Model weights: Static ~6 GB for GPT-2 XL (predictable)
- Activations: Variable based on sequence length
    - Short sequences (1K): ~0.5 GB activations
    - Long sequences (16K): ~57 GB activations (100x more!)
- Attention matrices: The memory killer at long sequences
    - 1K context: 0.2 GB attention matrices
    - 16K context: 51.5 GB attention matrices (256x increase)

*Activations* are the intermediate values computed during the forward pass that must be stored in memory for:
- Computing the next layer's input
- Backpropagation (during training)
- Gradient computation

**Optimization Priorities**:
- **Short sequences**: Focus on feed forward efficiency
- **Long sequences**: Focus on attention efficiency
- **Both**: Memory bandwidth becomes critical

### **Conclusion**

Large language models may seem like black boxes, but their computational patterns follow clear mathematical principles:

1. **Feed forward networks dominate** computation in most scenarios (57% for large models)
2. **Model scaling** predictably shifts optimization priorities from vocabulary to feed forward efficiency  
3. **Long context** fundamentally changes the game, making attention the primary bottleneck
4. **Memory requirements** often exceed computational requirements for optimization

Understanding these patterns isn't just academic—it directly informs optimization strategies, hardware requirements, and research directions. As we push toward even larger models and longer contexts, these computational realities will increasingly determine what's possible and what's practical in the world of large language models.