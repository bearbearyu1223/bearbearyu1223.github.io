---
layout: post
title: "Study Notes: Stanford CS336 Language Modeling from Scratch [12]"
categories: cs336
author:
- Han Yu
---

## From MacBook to Cloud: A Practical Guide to Developing and Scaling LLM Training Code

When developing machine learning training pipelines, there's often a disconnect between local development environments and production-scale cloud infrastructure. You might prototype on your laptop (say, a MacBook with Apple Silicon), only to discover that your code breaks on CUDA GPUs, or that patterns that worked locally don't scale in the cloud.

In this note, I'll share my workflow for developing Supervised Fine-Tuning (SFT) code on a MacBook with Apple Silicon, testing it locally, then seamlessly deploying to <img src="https://colab.research.google.com/img/colab_favicon_256px.png" height="20" style="vertical-align: middle;"/> [Google Colab](https://colab.research.google.com/) or multi-GPU cloud instances like <img src="https://lambdalabs.com/favicon.ico" height="20" style="vertical-align: middle;"/> [Lambda Labs](https://lambdalabs.com/).

*This workflow was developed while implementing SFT for Qwen2.5-Math-1.5B on the MATH dataset (for CS336 Assignment 5), but the principles apply broadly to any PyTorch-based training pipeline development.*

### Table of Contents
- [From MacBook to Cloud: A Practical Guide to Developing and Scaling LLM Training Code](#from-macbook-to-cloud-a-practical-guide-to-developing-and-scaling-llm-training-code)
  - [Table of Contents](#table-of-contents)
  - [The Challenge: Bridging Local and Cloud Development](#the-challenge-bridging-local-and-cloud-development)
  - [**Part 1: Setting Up Local Development Environment**](#part-1-setting-up-local-development-environment)
    - [**Why Apple Silicon for ML Development?**](#why-apple-silicon-for-ml-development)
    - [**Project Structure and Package Management**](#project-structure-and-package-management)
  - [**Part 2: Writing Device-Agnostic Training Code**](#part-2-writing-device-agnostic-training-code)
    - [**Handling Device Detection**](#handling-device-detection)
    - [**Numerical Precision Considerations**](#numerical-precision-considerations)
    - [**Gradient Accumulation for Memory Efficiency**](#gradient-accumulation-for-memory-efficiency)
  - [Part 3: Local Testing and Validation](#part-3-local-testing-and-validation)
    - [Quick Sanity Checks](#quick-sanity-checks)
    - [Inference Engine: Local vs Cloud](#inference-engine-local-vs-cloud)
    - [**Verifying Gradient Accumulation**](#verifying-gradient-accumulation)
  - [**Part 4: Packaging for Cloud Deployment**](#part-4-packaging-for-cloud-deployment)
    - [**Repository Structure**](#repository-structure)
    - [**Dependency Management with uv**](#dependency-management-with-uv)
  - [**Part 5: Deploying to Google Colab**](#part-5-deploying-to-google-colab)
    - [**Single GPU Training on Google Colab**](#single-gpu-training-on-google-colab)
    - [Colab-Specific Considerations](#colab-specific-considerations)
  - [**Part 6: Scaling to Multi-GPU with Accelerate**](#part-6-scaling-to-multi-gpu-with-accelerate)
    - [**Why HuggingFace Accelerate**](#why-huggingface-accelerate)
    - [**Code Changes for Multi-GPU Support**](#code-changes-for-multi-gpu-support)
    - [**Lambda Labs Deployment**](#lambda-labs-deployment)
  - [**Part 7: Practical Recommendations and Lessons Learned**](#part-7-practical-recommendations-and-lessons-learned)
    - [**Development Workflow Summary**](#development-workflow-summary)
    - [**Common Pitfalls and Solutions**](#common-pitfalls-and-solutions)
    - [**Performance Comparison**](#performance-comparison)
  - [Conclusion](#conclusion)

### The Challenge: Bridging Local and Cloud Development

My typical ML development workflow faces a fundamental tension—I use a MacBook Pro with M4 chips for personal side projects, which creates some tradeoffs:

| Environment | Pros | Cons |
|-------------|------|------|
| **Local (MacBook)** | Fast iteration, no cost, familiar tools | Limited memory, slower training, no CUDA (many GPU acceleration frameworks only support CUDA) |
| **Cloud (Colab/Lambda)** | Powerful GPUs, scalable, CUDA support | Setup overhead, costs money, less interactive |

The ideal workflow would let me:
1. **Develop locally** with fast feedback loops
2. **Test easily** before committing cloud resources
3. **Deploy seamlessly** without rewriting code
4. **Scale horizontally** when more compute is available

This note presents a battle-tested approach to achieving all four.

### **Part 1: Setting Up Local Development Environment**

#### **Why Apple Silicon for ML Development?**

Beyond personal preference (I've been an Apple product fan since grad school), Apple Silicon Macs offer a genuinely compelling development environment:

- **Unified Memory Architecture**: 16–64GB RAM shared between CPU and GPU
- **Metal Performance Shaders (MPS)**: PyTorch backend for GPU acceleration
- **Power Efficiency**: Extended battery life for portable development
- **Native ARM**: Fast Python and native tool execution

However, there are important limitations:

| Feature | CUDA (NVIDIA) | MPS (Apple Silicon) |
|---------|---------------|---------------------|
| Float16 Training | Stable with gradient scaling | Often causes NaN losses |
| BFloat16 | Full support (Ampere+) | Not supported |
| Multi-GPU | NCCL, NVLink | Single GPU only |
| Flash Attention | Available | Not available |
| Memory | Dedicated VRAM | Shared system RAM |

**Key Insight**: MPS ( Metal Performance Shaders—Apple's GPU-accelerated compute framework for macOS and iOS) is excellent for development and testing but usually requires float32 precision for numerical stability. I need plan for this difference when writing device-agnostic code.

#### **Project Structure and Package Management**

I use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python package management. Here's how I set up my local dev environment for CS336 Assignment 5.

**Install uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Project structure:**
```
assignment5-alignment/
├── cs336_alignment/
│   ├── __init__.py
│   ├── sft.py              # Main training code
│   └── prompts/
│       └── r1_zero.prompt  # Prompt template
├── scripts/
│   ├── run_sft.py          # Training entry point
│   ├── download_model.py   # Model downloader
│   └── download_math.py    # Data downloader
├── notebooks/
│   └── sft_training_colab.ipynb
├── pyproject.toml          # Dependencies
└── uv.lock                 # Locked versions
```

**pyproject.toml** with optional CUDA dependencies:
```toml
[project]
name = "alignment"
requires-python = ">=3.11,<3.13"
dependencies = [
    "accelerate>=1.5.2",
    "torch",
    "transformers>=4.50.0",
    "tqdm>=4.67.1",
    "matplotlib>=3.8.0",
]

[project.optional-dependencies]
cuda = [
    "flash-attn==2.7.4.post1",
]
```

**Local installation:**
```bash
uv sync              # Basic install (Mac/CPU)
uv sync --extra cuda # With CUDA extras (Linux with GPU)
```

### **Part 2: Writing Device-Agnostic Training Code**

The key to seamless local-to-cloud transitions is writing code that adapts to available hardware without manual changes.

#### **Handling Device Detection**

```python
def get_device(device_str: str = "auto") -> str:
    """Get the best available device for training."""
    if device_str != "auto":
        return device_str
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

#### **Numerical Precision Considerations**

This is where many developers encounter their first "works locally, fails on cloud" (or vice versa) bug:

```python
def get_dtype_and_precision(device: str) -> tuple[torch.dtype, str]:
    """
    Determine appropriate dtype and mixed precision setting.

    Critical insight: MPS does NOT support float16 training reliably.
    Using float16 on MPS often results in NaN losses due to lack of
    proper mixed-precision support and gradient scaling.
    """
    if device == "cuda":
        # CUDA: Use bfloat16 if available (Ampere+), else float16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"
        else:
            return torch.float16, "fp16"
    else:
        # MPS and CPU: Use float32 for numerical stability
        return torch.float32, "no"
```

**Why This Matters**:

| Device | Recommended Dtype | Reason |
|--------|------------------|--------|
| CUDA (Ampere+) | bfloat16 | Best balance of speed and stability |
| CUDA (older) | float16 | With gradient scaling |
| MPS | float32 | float16 may cause NaN losses |
| CPU | float32 | No mixed precision benefit |

I learned this the hard way when my training showed `loss: nan` on MPS after working fine conceptually. The fix was simple once identified:

```python
# Before (broken on MPS)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

# After (works everywhere)
dtype = torch.float32 if device in ["mps", "cpu"] else torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
```

#### **Gradient Accumulation for Memory Efficiency**

With limited memory on laptops (even 32GB unified memory), gradient accumulation is essential:

```python
# Effective batch size = batch_size * gradient_accumulation_steps
# Example: batch_size=1, grad_accum=8 -> effective batch of 8

for step, batch in enumerate(dataloader):
    # Forward pass
    loss = compute_loss(model, batch)

    # Scale loss for gradient accumulation
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    # Only update weights every N steps
    if (step + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```
**Memory Scaling Recommendations For CS336 Assignment 5 SFT on Qwen2.5-Math-1.5B with the MATH dataset**

| Device | Chip Generations | Typical Memory | Found In | batch_size | gradient_accumulation_steps | Effective Batch |
|--------|------------------|----------------|----------|------------|----------------------------|-----------------|
| Apple M-series (base) | M1, M2, M3, M4 | 8–16GB | MacBook Air, 13" MacBook Pro | 1 | 16 | 16 |
| Apple M-series Pro | M1, M2, M3, M4 | 18–48GB | 14"/16" MacBook Pro | 2–4 | 4–8 | 16 |
| Apple M-series Max | M1, M2, M3, M4 | 36–128GB | 14"/16" MacBook Pro (high-end) | 4–8 | 2–4 | 16 |
| Apple M-series Ultra | M1, M2 | 64–192GB | Mac Studio, Mac Pro | 8–16 | 1–2 | 16 |
| NVIDIA A100 (40GB) | — | 40GB | Cloud (Lambda, GCP, AWS) | 8 | 2 | 16 |
| NVIDIA A100 (80GB) | — | 80GB | Cloud (Lambda, GCP, AWS) | 16 | 1 | 16 |

*Effective batch = batch_size × gradient_accumulation_steps. Larger batch sizes reduce training time but require more memory.*

**Key insights:**

- **Memory constrains batch size, not effective batch size.** When GPU memory is limited, reduce `batch_size` and increase `gradient_accumulation_steps` to maintain the same effective batch size. The model sees identical gradients either way—accumulation just trades memory for time.

- **Gradient accumulation is a memory-saving trick.** Instead of computing gradients on 16 samples at once (which requires storing all intermediate activations), you process 1 sample 16 times, accumulating gradients before each optimizer step. This uses ~1/16th the memory at the cost of ~16× more forward/backward passes.

- **Effective batch size should stay constant across devices.** Notice that all rows target an effective batch of 16. This ensures consistent training dynamics regardless of hardware—important for reproducibility when moving between local development and cloud training.

- **Diminishing returns on large batch sizes.** Beyond a certain point, larger batch sizes don't proportionally speed up training due to memory bandwidth limits (GPUs can only move data so fast, once your batch is large enough to fully utilize the GPU, making it bigger just creates a queue—the GPU can't process it any faster) and reduced gradient noise (which can actually help optimization).

### Part 3: Local Testing and Validation

Before deploying to cloud, thorough local testing saves time and money.

#### Quick Sanity Checks
```bash
# Test with minimal samples to verify pipeline works
uv run python scripts/run_sft.py \
    --model-name-or-path models/qwen2.5-math-1.5b \
    --train-data-path data/math/train.jsonl \
    --output-dir outputs/sft_test \
    --num-samples 10 \
    --num-epochs 1 \
    --batch-size 1 \
    --gradient-accumulation-steps 2
```

**What to verify:**
1. Model loads without errors
2. Data pipeline produces valid batches
3. Loss decreases (not NaN or constant)
4. Checkpoints save correctly
5. Model can be reloaded from checkpoint

#### Inference Engine: Local vs Cloud

A key challenge when developing on Apple Silicon is that [vLLM](https://github.com/vllm-project/vllm)—the go-to inference engine for fast LLM serving—requires CUDA and doesn't run on Macs. This means I need two inference backends during the initial development phase:

| Environment | Inference Backend | Why |
|-------------|-------------------|-----|
| Local (MPS) | HuggingFace Transformers | Pure PyTorch, runs anywhere |
| Cloud (CUDA) | vLLM | Optimized kernels, PagedAttention, 10–20× faster |

**My approach**: Write a simple abstraction layer that switches backends based on the available hardware:
```python
def get_inference_backend(model_path: str, device: str):
    """Return appropriate inference backend for the current environment."""
    if device == "cuda" and is_vllm_available():
        from vllm import LLM
        return VLLMBackend(LLM(model=model_path))
    else:
        # Fallback to HuggingFace for MPS/CPU
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return TransformersBackend(model, tokenizer)
```

**What this enables:**
- **Local development**: Test generation logic, prompt templates, and output parsing using the Transformers backend on my Mac
- **Cloud deployment**: Automatically switch to vLLM for fast, batched inference without changing my evaluation code

**Trade-off to keep in mind**: my local inference is much slower than cloud. For local testing, I need to use small sample sizes (10–50 examples) to validate correctness, before move to run full evaluations on cloud.

#### **Verifying Gradient Accumulation**

A common bug is incorrect gradient accumulation scaling. Here's a verification approach:

```python
def verify_gradient_accumulation():
    """
    Verify that accumulated gradients match single large batch.

    The gradients should be identical (within floating point tolerance)
    whether we:
    1. Process 8 samples in one batch, or
    2. Process 1 sample 8 times with gradient accumulation
    """
    model_single = create_model()
    model_accum = create_model()

    # Copy weights
    model_accum.load_state_dict(model_single.state_dict())

    # Method 1: Single large batch
    large_batch = get_batch(size=8)
    loss = compute_loss(model_single, large_batch)
    loss.backward()
    grad_single = get_gradients(model_single)

    # Method 2: Accumulated small batches
    for i in range(8):
        small_batch = get_batch(size=1)
        loss = compute_loss(model_accum, small_batch) / 8  # Scale!
        loss.backward()
    grad_accum = get_gradients(model_accum)

    # Verify they match
    assert torch.allclose(grad_single, grad_accum, rtol=1e-4)
```

### **Part 4: Packaging for Cloud Deployment**

#### **Repository Structure**

Push my code to GitHub for easy cloud access, for example

```bash
git add cs336_alignment/ scripts/ notebooks/ pyproject.toml uv.lock
git commit -m "Add SFT training pipeline"
git push origin main
```

#### **Dependency Management with uv**

The `uv.lock` file ensures reproducible environments:
```bash
# Generate lock file locally
uv lock

# On cloud, install exact versions
uv sync  # Reads uv.lock automatically
```

**Why uv over pip/conda/poetry?**

| Aspect | pip | conda | Poetry | uv |
|--------|-----|-------|--------|-----|
| Speed | Moderate | Slow | Slow | Very fast (Rust-based) |
| Lock file | ❌ (requires pip-tools) | ❌ (manual export) | ✅ | ✅ |
| PyTorch/CUDA handling | Manual | Good | Finicky | Smooth |
| Mac → Linux portability | Poor | Poor | Good | Excellent |
| Dependency resolution | Basic | Solver can be slow | Good but slow | Fast and reliable |

**Why this matters for ML workflows:**

- **Speed**: ML projects have heavy dependencies (PyTorch, Transformers, flash-attn). Poetry can take 30–60s to resolve; uv takes 1–5s.

- **PyTorch complexity**: PyTorch has separate wheels for CPU, CUDA 11.8, CUDA 12.1, etc. Poetry often requires manual configuration with custom sources. uv handles this automatically.

- **Cross-platform**: I am developing on Mac (ARM) and deploying to Linux (x86 + CUDA). uv's lock file captures platform-specific metadata, so `uv sync` installs the correct versions on each platform without separate environment files.

**When you might still choose Poetry:**
- Publishing packages to PyPI (Poetry has built-in support)
- Your team already uses it and has established workflows
- You need Poetry's plugin ecosystem

For ML development workflows like this one, uv's speed and PyTorch handling are significant wins.

### **Part 5: Deploying to Google Colab**

#### **Single GPU Training on Google Colab**
[Google Colab](https://colab.research.google.com/) provides easy access to cloud GPUs without any setup. With your packaged repo, you can create a notebook with the following cells to run training on Colab:
```python
# Cell 1: Clone and setup
!git clone https://github.com/YOUR_USERNAME/assignment5-alignment.git
%cd assignment5-alignment
!git checkout main

# Cell 2: Install uv and dependencies
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ['PATH'] = f"{os.path.expanduser('~')}/.local/bin:{os.environ['PATH']}"
!uv sync --extra cuda

# Cell 3: Download model and data
!uv run python scripts/download_model.py --model-name Qwen/Qwen2.5-Math-1.5B
!uv run python scripts/download_math.py

# Cell 4: Run training
!uv run python scripts/run_sft.py \
    --model-name-or-path models/qwen2.5-math-1.5b \
    --train-data-path data/math/train.jsonl \
    --output-dir outputs/sft_model \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --device cuda
```

#### Colab-Specific Considerations

| Aspect | Recommendation |
|--------|----------------|
| **Runtime selection** | Runtime → Change runtime type → Select GPU (T4 for free tier, A100 for Pro+) |
| **Session timeout** | Save checkpoints every 1–2 epochs; free tier can preempt without warning |
| **Persistence** | Mount Google Drive for outputs to survive session resets |
| **Memory limits** | T4 has 16GB VRAM—use `batch_size=2` with gradient accumulation |
| **Background execution** | Pro+ only—training continues after closing browser |

**Google Drive mounting:**
```python
from google.colab import drive
drive.mount('/content/drive')

# Save outputs to Drive
output_dir = '/content/drive/MyDrive/sft_outputs'
```

**Saving to Google Drive**:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r outputs/sft_model/final /content/drive/MyDrive/sft_model
```

### **Part 6: Scaling to Multi-GPU with Accelerate**

#### **Why HuggingFace Accelerate**

Google Colab typically provides only 1 GPU. For multi-GPU training (Lambda Labs, AWS, etc.), we can use HuggingFace Accelerate:

| Feature | Manual DDP | Accelerate |
|---------|------------|------------|
| Code changes | Significant | Minimal |
| Device placement | Manual | Automatic |
| Gradient sync | Manual | Automatic |
| Mixed precision | Manual setup | One flag |
| Single/Multi GPU | Different code paths | Same code |

#### **Code Changes for Multi-GPU Support**

The key changes to support multi-GPU:

```python
from accelerate import Accelerator

def train_sft(config):
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if device == "cuda" else "no",
    )

    # Prepare model, optimizer, dataloader
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    for batch in dataloader:
        # Use accelerator's gradient accumulation context
        with accelerator.accumulate(model):
            loss = compute_loss(model, batch)
            accelerator.backward(loss)  # Instead of loss.backward()

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Save only on main process
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        save_model(unwrapped_model, output_dir)
```

**Key Accelerate Patterns**:

1. **`accelerator.prepare()`**: Wraps objects for distributed training
2. **`accelerator.accumulate()`**: Handles gradient accumulation correctly
3. **`accelerator.backward()`**: Syncs gradients across devices
4. **`accelerator.sync_gradients`**: True when accumulation cycle completes
5. **`accelerator.is_main_process`**: Only one process logs/saves

#### **Lambda Labs Deployment**

```bash
# SSH into Lambda instance
ssh ubuntu@your-instance-ip

# Setup
git clone https://github.com/YOUR_USERNAME/assignment5-alignment.git
cd assignment5-alignment

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv sync --extra cuda

# Download model and data
uv run python scripts/download_model.py
uv run python scripts/download_math.py

# Multi-GPU training (auto-detects available GPUs)
uv run accelerate launch --multi_gpu scripts/run_sft.py \
    --model-name-or-path models/qwen2.5-math-1.5b \
    --batch-size 4 \
    --gradient-accumulation-steps 2
```

**Scaling Guide**:

| GPUs | batch_size | grad_accum | Effective Batch | Command |
|------|------------|------------|-----------------|---------|
| 1 | 4 | 4 | 16 | `uv run python scripts/run_sft.py` |
| 2 | 4 | 2 | 16 | `accelerate launch --num_processes 2` |
| 4 | 4 | 1 | 16 | `accelerate launch --num_processes 4` |
| 8 | 4 | 1 | 32 | `accelerate launch --num_processes 8` |

### **Part 7: Practical Recommendations and Lessons Learned**

#### **Development Workflow Summary**

```
+------------------------------------------------------------------+
|                    LOCAL DEVELOPMENT (Mac)                        |
+------------------------------------------------------------------+
|  1. Write code with device-agnostic patterns                     |
|  2. Test with small samples (--num-samples 10)                   |
|  3. Verify loss decreases, no NaN                                |
|  4. Run unit tests (pytest)                                      |
|  5. Commit and push to GitHub                                    |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    CLOUD VALIDATION (Colab)                       |
+------------------------------------------------------------------+
|  1. Clone repo, install dependencies                             |
|  2. Quick test with 100 samples                                  |
|  3. Verify CUDA path works correctly                             |
|  4. Check memory usage fits GPU                                  |
|  5. Save checkpoint to Google Drive                              |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                PRODUCTION TRAINING (Lambda/Cloud)                 |
+------------------------------------------------------------------+
|  1. Use accelerate launch for multi-GPU                          |
|  2. Full dataset training                                        |
|  3. Monitor with logging/wandb                                   |
|  4. Save final model and metrics                                 |
+------------------------------------------------------------------+
```

#### **Common Pitfalls and Solutions**

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Float16 on MPS | `loss: nan` | Use float32 on MPS |
| Wrong grad accumulation | Gradients don't match | Divide loss by accumulation steps |
| Missing `is_main_process` check | Duplicate logs/saves | Guard with `accelerator.is_main_process` |
| Hardcoded device | Crashes on different hardware | Use `get_device("auto")` |
| No checkpoint saving | Lost progress on timeout | Save every N steps |

#### **Performance Comparison**

From my experiments with Qwen2.5-Math-1.5B on MATH dataset:

| Environment | Device | batch_size x grad_accum | Time per 100 steps |
|-------------|--------|------------------------|-------------------|
| MacBook M2 Pro | MPS | 1 x 8 | ~45 min |
| Colab Free | T4 | 2 x 8 | ~12 min |
| Colab Pro | A100 | 8 x 2 | ~3 min |
| Lambda (4x A100) | 4x A100 | 4 x 1 (per GPU) | ~1 min |

### Conclusion

Developing ML training code that works seamlessly from a MacBook to multi-GPU cloud instances requires intentional design:

1. **Device-agnostic code**: Abstract device selection and dtype handling
2. **Numerical stability**: Use float32 on MPS, mixed precision on CUDA
3. **Memory efficiency**: Implement gradient accumulation from the start
4. **Reproducible environments**: Use `uv` with lock files
5. **Distributed-ready**: Integrate Accelerate for painless multi-GPU scaling

The workflow I've shared—develop locally on MacBook, validate on Colab, scale on cloud with distributed training—provides fast iteration during development while enabling production-scale training when needed. The key insight is that **the code should adapt to the hardware, not the other way around**.

I hope this empowers you to develop confidently on your laptop, knowing that deploying to powerful cloud GPUs is a matter of changing a single command—not rewriting your training pipeline.

---

**Resources**:
- [HuggingFace Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [uv Package Manager](https://github.com/astral-sh/uv)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [Lambda Labs GPU Cloud](https://lambdalabs.com/)
