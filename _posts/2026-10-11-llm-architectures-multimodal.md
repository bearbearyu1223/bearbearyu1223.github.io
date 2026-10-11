---
title: "LLM Architecture Refresh [7]: Teaching a Language Model to See"
date: 2026-10-11 09:00:00 -0700
categories: [LLM Architecture Refresh, Multimodal]
tags: [multimodal, vision-language, vlm, llava, siglip, vision-transformer, lora, qwen, pytorch]
description: >-
  Building a small vision-language model from three parts, a frozen image
  encoder, a trained connector and a frozen language model, then training it in
  two stages and checking, with control rows, whether it actually looks at the
  picture.
math: true
published: false  # draft: not yet reviewed
---

## The first post with a picture in it

Six posts in, everything has been text.

[Post 1](/posts/llm-architectures-attention-and-rope/) took a transformer block apart and put a cost on every piece. [Post 2](/posts/llm-architectures-kv-cache/) cached keys and values so a model stops redoing work for every new token. [Post 3](/posts/llm-architectures-flash-attention/) computed attention in tiles to move fewer bytes. [Post 4](/posts/llm-architectures-quantization/) stored each weight in fewer bits. [Post 5](/posts/llm-architectures-mixture-of-experts/) built a model that runs only some of its parameters for each token. [Post 6](/posts/llm-architectures-lora/) fine-tuned a model by training a small adapter beside its frozen weights, **0.89%** of its parameters.

Every one of those models took text in and gave text out. This post gives one a picture.

A model that takes more than one kind of input, here pictures and text, is called **multimodal**, and one that reads pictures and writes text is a **vision-language model**, or VLM. The surprising thing about how most of them work is how little of the language model has to change. The picture gets turned into a short run of vectors, the same width as the vectors the language model already uses for words, and the language model reads those vectors as if they were words. Getting there takes three parts and two rounds of training, and this post builds all of it on a laptop:

<!-- FIGURE: vlm-overview -->
![A vision-language model from three parts: a picture goes through a frozen vision encoder into 196 vectors, a trained connector turns them into 49 vectors as wide as the language model's word vectors, and those sit in the sequence between the text tokens that the frozen language model reads](/assets/picture/2026-10-11-llm-architectures-multimodal/vlm-overview-light.png){: .light width="1000" height="488" }
![A vision-language model from three parts: a picture goes through a frozen vision encoder into 196 vectors, a trained connector turns them into 49 vectors as wide as the language model's word vectors, and those sit in the sequence between the text tokens that the frozen language model reads](/assets/picture/2026-10-11-llm-architectures-multimodal/vlm-overview-dark.png){: .dark width="1000" height="488" }

### The short version {#the-short-version}

<!-- TODO-RESULTS: short version -->

### Setup

Everything below is measured, and you can re-run all of it:

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo07
```

The code is in [`demos/d07_multimodal.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d07_multimodal.py), and the model itself is in [`vlm.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/vlm.py): the patch cutting, the connector, and the code that splices a picture into a sentence are all written out rather than imported, because this series builds the thing it's explaining. Every number this post quotes is printed by that program, and each block of its output, which I call a **receipt**, names the function that printed it.

<!-- TODO-RESULTS: runtime and reproducibility -->

**The three parts.**

- The **vision encoder** is [SigLIP-base](https://huggingface.co/google/siglip-base-patch16-224) ([paper](https://arxiv.org/abs/2303.15343)), a 12-layer transformer that was trained on pictures and their captions. It stays frozen the whole time.
- The **language model** is **Qwen2.5-0.5B**, the same model [post 4](/posts/llm-architectures-quantization/) quantized and [post 6](/posts/llm-architectures-lora/) fine-tuned ([the weights](https://huggingface.co/Qwen/Qwen2.5-0.5B), [the report](https://arxiv.org/abs/2412.15115)). It's a plain text model; nothing about it was built for pictures. It stays frozen too, except for the LoRA adapters one of the stage-2 runs adds.
- The **connector** is the only new part. It starts random and is trained from scratch.

Both models download on the first run. Everything runs in fp32, as in post 6, so that two training runs can only differ in what they train.

#### The pictures, and what the model is asked {#the-pictures}

The demo draws its own pictures: up to four colored shapes on a white square, each a circle, a square or a triangle in one of six colors. Here are the kinds of things it's taught to say about them:

| | what it's given | what it must write |
|---|---|---|
| **stage 1: describe** | the picture, then `Caption:` | every shape, left to right: `an orange circle, a yellow circle and a purple circle` |
| **stage 2: count** | `How many shapes are there?` | `three` |
| **stage 2: binding** | `What color is the triangle?` (asked only when there is one triangle) | `purple` |
| **stage 2: leftmost** | `What color is the leftmost shape?` | `orange` |

**Binding** is the name for tying two facts about one object together: it's not enough to see that there's a triangle and that there's purple somewhere, the model has to know the purple belongs to the triangle.

Generated pictures are a deliberate choice, and the biggest limitation of this post. Every answer is exact, so a score of 87 of 100 means 87 answers were exactly right and nothing else. The test pictures are new ones drawn from a different random seed, so a score is a score on pictures the model has never seen. No dataset has to be downloaded or cleaned. The price is that these pictures are far easier than photographs, so the numbers here say how the mechanism behaves, not how good a VLM this is.

### Table of Contents

Skip to [the short version](#the-short-version) for the findings without the derivations.

1. [From pixels to patches](#pixels-to-patches)
2. [The connector](#the-connector)
3. [Stage 1: learning to describe](#stage-one)
4. [Stage 2: learning to answer](#stage-two)
5. [Is it actually looking?](#is-it-looking)
6. [What an image costs](#what-an-image-costs)
7. [Other ways to connect a picture](#other-designs)
8. [What follows from all this](#what-follows)
9. [Sidebar: the probe](#sidebar-the-probe)

Plus [all notation](#appendix-all-notation) at the end.

---

### 1. From pixels to patches {#pixels-to-patches}

A language model never sees text either. It sees a sequence of vectors, one per token, each looked up from a table ([post 1](/posts/llm-architectures-attention-and-rope/) walked through it). So the question for a picture is the same one: how do you turn it into a sequence of vectors?

The answer most models use comes from the **vision transformer**, or ViT, of [Dosovitskiy et al.](https://arxiv.org/abs/2010.11929), whose title gives the idea away: *An Image is Worth 16x16 Words*. Cut the picture into a grid of small squares, called **patches**, and treat each patch the way a language model treats a token.

<!-- FIGURE: image-to-patches -->
![A picture cut into a 14 by 14 grid of patches; one patch enlarged to its 16 by 16 pixels, flattened into 768 numbers, multiplied by one matrix and given its position, becoming one 768-wide vector](/assets/picture/2026-10-11-llm-architectures-multimodal/image-to-patches-light.png){: .light width="1000" height="416" }
![A picture cut into a 14 by 14 grid of patches; one patch enlarged to its 16 by 16 pixels, flattened into 768 numbers, multiplied by one matrix and given its position, becoming one 768-wide vector](/assets/picture/2026-10-11-llm-architectures-multimodal/image-to-patches-dark.png){: .dark width="1000" height="416" }

Take it one step at a time.

1. **A picture is a grid of numbers.** Each pixel is three numbers, how much red, green and blue it has. SigLIP takes pictures 224 pixels on a side.
2. **Cut it into patches.** At 16 × 16 pixels per patch, a 224-pixel side holds 14 patches, so the picture becomes a 14 × 14 grid of 196 patches.
3. **Flatten each patch.** Write a patch's pixels out in one long row: 16 × 16 pixels × 3 colors = 768 numbers.
4. **Multiply by one learned matrix.** That turns the 768 pixel numbers into a 768-wide vector. The ViT paper describes this step as mapping the flattened patches "to D dimensions with a trainable linear projection" (§3.1). It's the picture's version of looking up a word's embedding: same idea, pixels in place of a token id.
5. **Add a position.** Like words, patches need to know where they are. A learned vector for each of the 196 grid positions is added on.

Then the encoder's 12 transformer layers do what a language model's layers do: every patch attends to every other patch, so by the end each patch vector carries information about the whole picture, not just its own 16 × 16 square.

Here are the numbers, from the real encoder ([`pixels_to_patches`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d07_multimodal.py)):

<!-- RECEIPT: s1 -->

Two things in that receipt are worth reading slowly.

**The patch embedding is a convolution in the code and a matrix multiply in the math, and they're the same thing.** SigLIP's code stores step 4 as a convolution with a 16 × 16 window that jumps 16 pixels at a time, so the windows never overlap. A window that never overlaps sees each patch exactly once, which is all a matrix multiply on each flattened patch does. The demo runs both and they agree to rounding, the "max difference" line.

**Not all of the encoder is used.** SigLIP ends with a small **pooling head** that squeezes the 196 vectors into one, which is what SigLIP needs for its own training: it scores whether a picture and a caption belong together. A VLM wants all 196 vectors, one per patch, so the head is skipped. The encoder that makes the patch vectors is the remaining 85.8 million parameters.

#### Why the encoder already knows what it's looking at {#why-frozen}

A freshly initialized encoder would turn every picture into noise. SigLIP isn't fresh: it was trained on a large collection of pictures paired with their captions, to tell matching pairs from mismatched ones. Its paper's change from the earlier [CLIP](https://arxiv.org/abs/2103.00020) is the loss: instead of comparing every picture in a batch against every caption at once, SigLIP "processes every image-text pair independently, effectively turning the learning problem into the standard binary classification" (§3.2): does this caption go with this picture, yes or no. To get good at that question, an encoder has to learn which patches show what, so its 196 output vectors already carry "orange", "round", "top left" in some form.

That's why it can stay frozen, and why the whole recipe is cheap. In the demo it never trains, so its output for a given picture never changes, and the demo runs it once per picture at the start and keeps the result. Every training step after that skips the encoder entirely.

---

### 2. The connector {#the-connector}

The encoder hands over 196 vectors, each 768 wide. The language model expects vectors 896 wide, and more importantly, it expects them to mean what its own word vectors mean. SigLIP's numbers and Qwen's numbers were learned separately, by different models with different goals, so a SigLIP vector dropped into Qwen's sequence is noise to it. The **connector** translates.

This post's connector is two steps, following the design [Qwen2-VL](https://arxiv.org/abs/2409.12191) uses:

<!-- FIGURE: connector -->
![The connector: four neighbouring patch vectors glued end to end into one 3,072-wide vector, then a trained two-layer MLP turns it into one 896-wide image token](/assets/picture/2026-10-11-llm-architectures-multimodal/connector-light.png){: .light width="1000" height="390" }
![The connector: four neighbouring patch vectors glued end to end into one 3,072-wide vector, then a trained two-layer MLP turns it into one 896-wide image token](/assets/picture/2026-10-11-llm-architectures-multimodal/connector-dark.png){: .dark width="1000" height="390" }

1. **Merge each 2 × 2 block of neighbouring patches into one vector.** Four 768-wide vectors are glued end to end into one 3,072-wide vector. The 14 × 14 grid becomes a 7 × 7 grid, so 196 vectors become 49. Nothing is averaged: every number still reaches the next step. What shrinks is how many positions the language model has to read, which [§6](#what-an-image-costs) puts a price on. Qwen2-VL's paper describes the same move: "a simple MLP layer is employed after the ViT to compress adjacent 2 × 2 tokens into a single token" (§2.1).
2. **Pass each merged vector through a small MLP.** Two layers: 3,072 numbers in, 896 out, then 896 to 896, with a GELU between them (a smooth version of "keep positives, zero negatives", close cousin of the SiLU in [post 1's FFN](/posts/llm-architectures-attention-and-rope/#the-ffn)). The output is 896 wide, the width of one of Qwen's word embeddings. This two-layer shape is the one [LLaVA-1.5](https://arxiv.org/abs/2310.03744) found worked better than the single linear layer of the original [LLaVA](https://arxiv.org/abs/2304.08485) (§3.3).

([`the_connector`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d07_multimodal.py))

<!-- RECEIPT: s2a -->

The connector is 3.56 million parameters, about 0.7% of the language model, and it's the only part stage 1 trains.

#### Where the picture goes {#where-the-picture-goes}

Then comes the step that makes the whole thing work, and it's simpler than you might expect: the 49 image vectors are placed in the sequence where words would be.

<!-- FIGURE: sequence -->
![The sequence the language model reads for one question: two text tokens, 49 image vectors, then the question's 13 tokens, after which it predicts the answer](/assets/picture/2026-10-11-llm-architectures-multimodal/sequence-light.png){: .light width="1000" height="246" }
![The sequence the language model reads for one question: two text tokens, 49 image vectors, then the question's 13 tokens, after which it predicts the answer](/assets/picture/2026-10-11-llm-architectures-multimodal/sequence-dark.png){: .dark width="1000" height="246" }

The language model is given the vectors for `Image:`, then the picture's 49 vectors, then the vectors for the question's tokens, and asked to continue. The word vectors are looked up from Qwen's vocabulary table as always. The image vectors come out of the connector instead. From the language model's side they're all just positions in one sequence, and its attention treats them all the same way: the token `leftmost` can attend to an image position exactly as it would attend to an earlier word.

<!-- RECEIPT: s2b -->

**The picture is never turned into words.** There's no hidden caption in the middle. The language model gets 49 vectors that don't correspond to any word in its vocabulary, and has to learn, through the connector, to make sense of them.

---

### 3. Stage 1: learning to describe {#stage-one}

<!-- FIGURE: training-stages -->
![The two training stages side by side: in stage 1 only the connector trains, on captions; in stage 2 it trains on questions, alone in one run and with LoRA adapters on the language model in the other](/assets/picture/2026-10-11-llm-architectures-multimodal/training-stages-light.png){: .light width="1000" height="517" }
![The two training stages side by side: in stage 1 only the connector trains, on captions; in stage 2 it trains on questions, alone in one run and with LoRA adapters on the language model in the other](/assets/picture/2026-10-11-llm-architectures-multimodal/training-stages-dark.png){: .dark width="1000" height="517" }

<!-- TODO-RESULTS: stage 1 -->

---

### 4. Stage 2: learning to answer {#stage-two}

<!-- TODO-RESULTS: stage 2 -->

---

### 5. Is it actually looking? {#is-it-looking}

<!-- TODO-RESULTS: is it looking -->

---

### 6. What an image costs {#what-an-image-costs}

<!-- TODO-RESULTS: costs -->

---

### 7. Other ways to connect a picture {#other-designs}

This post's model belongs to one family. It's worth knowing the others, because "multimodal LLM" covers several designs that differ exactly where this post's connector sits.

<!-- FIGURE: connector-designs -->
![Four ways to give a language model a picture: project every patch into the sequence, summarize it with learned queries into a fixed number of tokens, let cross-attention layers read it from the side, or turn it into discrete tokens from a codebook](/assets/picture/2026-10-11-llm-architectures-multimodal/connector-designs-light.png){: .light width="1000" height="785" }
![Four ways to give a language model a picture: project every patch into the sequence, summarize it with learned queries into a fixed number of tokens, let cross-attention layers read it from the side, or turn it into discrete tokens from a codebook](/assets/picture/2026-10-11-llm-architectures-multimodal/connector-designs-dark.png){: .dark width="1000" height="785" }

**1. Project every patch into the sequence.** This post's design, and [LLaVA](https://arxiv.org/abs/2304.08485)'s. LLaVA used a frozen CLIP encoder and "a simple linear layer" as its connector, trained in two stages that this post copies: first "we keep both the visual encoder and LLM weights frozen" and train only the projection, on 595,000 picture-caption pairs; then "we always keep the visual encoder weights frozen, and continue to update both the pre-trained weights of the projection layer and LLM" (§4.2). The one difference from this post is that LLaVA's second stage trained every weight of its language model, not a LoRA adapter. [Qwen2-VL](https://arxiv.org/abs/2409.12191) adds the 2 × 2 merge and handles pictures of any size by letting the number of patches vary. The appeal is simplicity: the language model sees every patch. The cost is tokens, which grow with the picture's size.

**2. Summarize the picture into a fixed number of tokens.** [Flamingo](https://arxiv.org/abs/2204.14198)'s Perceiver Resampler and [BLIP-2](https://arxiv.org/abs/2301.12597)'s Q-Former both use a small set of learned **queries**, vectors that are trained to ask the patches useful questions through attention. Whatever the picture's size, the output is the same number of vectors: Flamingo "produces a fixed number of visual outputs (64)" (§2.1), BLIP-2 uses "32 queries" (§3.1). [Idefics2](https://arxiv.org/abs/2405.02246) compared the two on its own model and found that this kind of learned pooling, cutting 729 visual tokens to 64, "increases the performance by 8.5 points on average" (§3.3).

**3. Let the language model look across at the picture.** Here the picture never enters the sequence at all. New **cross-attention** layers are inserted between the language model's own layers: the text attends to the image vectors from the side, through layers trained for the purpose, while the original layers stay frozen. Flamingo does this with a gate that starts closed, multiplying each new layer's output by $\tanh(\alpha)$ with $\alpha$ "initialized to 0" (§2.2), so on day one the model is exactly the text model it started as. Meta's [Llama 3 vision models](https://arxiv.org/abs/2407.21783) use the same idea, with cross-attention "applied after every fourth self-attention layer" (§7.2), and leave the language model's own weights alone: "we intentionally do not update the language-model parameters" (§1).

**4. Turn the picture into tokens from a vocabulary.** [Chameleon](https://arxiv.org/abs/2405.09818) quantizes each picture into discrete tokens, 1,024 tokens per 512 × 512 picture from a codebook of 8,192 (§2.1), and trains one model on text tokens and image tokens together from the start. Since pictures are tokens like any other, the same model can also *write* them, which none of the other three designs can.

Which is best? There's no single answer. Idefics2's most useful finding is a conditional one: a cross-attention design "performs better than the fully autoregressive one when unimodal pre-trained backbones are kept frozen", but once the backbones are trained too, putting the picture in the sequence wins (Finding 2). This post's frozen-backbone, small-scale setup is exactly the regime where design 3 might have done better, and that's a fair caveat on everything above.

---

### 8. What follows from all this {#what-follows}

<!-- TODO-RESULTS: what follows -->

---

### 9. Sidebar: the probe {#sidebar-the-probe}

<!-- TODO-RESULTS: probe -->

### What's next {#whats-next}

<!-- TODO: next post -->

### Appendix: all notation {#appendix-all-notation}

The terms this post leans on; anything not listed is defined where it first appears. [Post 1's appendix](/posts/llm-architectures-attention-and-rope/#appendix-all-notation) covers attention's notation and [post 6's](/posts/llm-architectures-lora/#appendix-all-notation) covers LoRA's.

| Term | Means | In this post |
| --- | --- | --- |
| multimodal | taking in more than one kind of input, here pictures and text | |
| VLM | vision-language model: reads pictures and text, writes text | §1 to §6 |
| ViT | vision transformer: a transformer whose tokens are patches of a picture | SigLIP-base, 12 layers |
| patch | one square of the picture, treated like one token | 16 × 16 pixels, 196 per picture |
| patch embedding | the learned matrix that turns a flattened patch's pixel numbers into a vector | 768 × 768 |
| vision encoder | the ViT that turns a picture into patch vectors | frozen throughout |
| pooling head | SigLIP's final layer that squeezes all patch vectors into one; unused here | |
| connector | the trained part between encoder and language model | 2 × 2 merge + 2-layer MLP, 3,557,120 parameters |
| merge | gluing each 2 × 2 block of patch vectors end to end | 196 vectors of 768 → 49 of 3,072 |
| image tokens | the connector's output vectors, placed in the sequence where word embeddings would go | 49, each 896 wide |
| stage 1 | training the connector alone, on captions | |
| stage 2 | training on questions, with the connector alone or the connector plus LoRA | |
| binding | tying two facts about one object together, such as a color to a shape | the `binding` task |
| exact match | the series' scoring rule: greedy decoding, trim spaces at the ends, compare | |
| control | the same question scored with a picture that can't contain the answer: blank, or another scene's | §5 |
| KV cache | the keys and values a model keeps for every position it has read ([post 2](/posts/llm-architectures-kv-cache/)) | 12,288 bytes per position in bf16 |
| cross-attention | attention where one sequence's queries read another sequence's keys and values | §7's design 3 |

### References

- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (2020) — the vision transformer. §3.1 is §1's recipe: the image is reshaped "into a sequence of flattened 2D patches", which are mapped "to D dimensions with a trainable linear projection", with "standard learnable 1D position embeddings" added.
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021) — CLIP: an image encoder and a text encoder trained together on "400 million (image, text) pairs" (§2.2) to predict which pictures go with which captions (§2.3). The encoder most VLMs started from.
- Zhai et al., [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) (2023) — SigLIP, this post's encoder. It replaces CLIP's batch-wide comparison with a loss that "processes every image-text pair independently" (§3.2), which "does not require a global view of the pairwise similarities for normalization" (abstract).
- Liu et al., [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) (2023) — LLaVA, the recipe this post follows. A frozen CLIP ViT-L/14 encoder, "a simple linear layer" as the connector (§4.1), and two stages (§4.2): the projection alone on 595K picture-caption pairs filtered from CC3M with encoder and language model frozen, then the projection and the whole language model on instruction data.
- Liu et al., [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744) (2023) — LLaVA-1.5. A two-layer MLP connector in place of the linear one, and 336-pixel input (§3.3). The paper doesn't state a token count; §6's 576 is $(336/14)^2$ from its encoder's patch size.
- Wang et al., [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191) (2024) — the 2 × 2 merge this post's connector uses: "a simple MLP layer is employed after the ViT to compress adjacent 2 × 2 tokens into a single token" (§2.1), and pictures of any size turned into a varying number of tokens.
- Alayrac et al., [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) (2022) — §7's designs 2 and 3: a Perceiver Resampler that "produces a fixed number of visual outputs (64)" (§2.1), and gated cross-attention layers inserted between frozen language-model layers, gated by $\tanh(\alpha)$ with $\alpha$ "initialized to 0" (§2.2), every fourth layer in Flamingo-9B (§3.3).
- Li et al., [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) (2023) — the Q-Former: "32 queries" (§3.1) that read a frozen encoder's output and hand a frozen language model a fixed-size summary.
- Grattafiori et al. (Llama Team), [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (2024) — §7's design 3 at scale: cross-attention layers "applied after every fourth self-attention layer" (§7.2), the language model kept frozen on purpose (§1), the image encoder trained (§7.2).
- Team Chameleon, [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818) (2024) — §7's design 4: images quantized into discrete tokens, 1,024 per 512 × 512 picture from a codebook of 8,192 (§2.1), in one model trained on both.
- Laurençon et al., [What matters when building vision-language models?](https://arxiv.org/abs/2405.02246) (2024) — Idefics2's design study. Cross-attention wins with frozen backbones and loses once they train (Finding 2); learned pooling from 729 to 64 visual tokens "increases the performance by 8.5 points on average" (§3.3); and on training the language model, "Leveraging LoRA still adds expressivity to the training and stabilizes it" (Finding 3).
- Karamcheti et al., [Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models](https://arxiv.org/abs/2402.07865) (2024) — found the connector-only first stage "unnecessary" at 7B and 13B, where single-stage training improved results and saved "20-25% of training cost" (§4.1). Their second stage trains the whole language model, not LoRA.
- Chen et al., [Are We on the Right Way for Evaluating Large Vision-Language Models?](https://arxiv.org/abs/2403.20330) (2024) — MMStar. §5's control rows at benchmark scale: "visual content is unnecessary for many samples", with more than half of ScienceQA's questions and a fifth of MMMU's answerable by language models with no picture at all.
- Rahmanzadehgervi et al., [Vision language models are blind: Failing to translate detailed visual features into words](https://arxiv.org/abs/2407.06581) (2024; title and numbers from the v6 revision) — seven simple tasks (counting overlapping shapes, line intersections, nested squares and others) on which strong VLMs fail, while linear probes show the encoders "contain sufficient visual information".
- Tong et al., [Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](https://arxiv.org/abs/2401.06209) (2024) — "CLIP-blind pairs": "images that CLIP perceives as similar despite their clear visual differences", and the MMVP benchmark built from them.
- Marafioti et al., [SmolVLM: Redefining small and efficient multimodal models](https://arxiv.org/abs/2504.05299) (2025) — pixel shuffle, a merge like §2's, reduces visual tokens "by a factor of r²" (§2.2), and small models "benefit from more aggressive visual token compression" (Finding 3).
- Qwen Team, [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) (2024) — the language model: 24 layers, 14 query heads and 2 key/value heads (Table 1). Its width, 896, is from the model's config.
