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

I built a small vision-language model from three parts: a frozen picture encoder (SigLIP-base), a frozen text model (Qwen2.5-0.5B, the model from posts 4 and 6), and a small **connector** between them, the only part trained from scratch. It was taught on generated pictures of colored shapes in two stages, first to describe them and then to answer questions about them. Every number is measured by the demo:

| measured | result |
|---|---|
| one picture becomes | **49 image tokens** |
| connector size | 3,557,120, **0.7%** of the LM |
| stage 1 captions exactly right | **87 of 120** (blank: 1) |
| "color of the triangle?", no stage 1 | **63** of 100 |
| the same, after stage 1 | **97** of 100 |
| looking adds, over the controls | **44 to 81** points |
| KV cache per picture (bf16) | **0.57 MiB** (196 tokens: 2.30) |

**How it works.** A picture enters a transformer the way text does, as a sequence of vectors. The encoder cuts it into 16 × 16-pixel patches and turns each into a vector ([§1](#pixels-to-patches)). The connector merges neighbouring patches and reshapes them to the language model's width, and those vectors go into the sequence where words would be ([§2](#the-connector)). The language model doesn't change at all to read them.

**What training buys.**

- **The connector alone gets a long way.** With both big models frozen, 3.6 million trained parameters teach the model to caption new pictures exactly right 87 times out of 120 ([§3](#stage-one)).
- **Captions first is what teaches binding.** Tying a color to the right shape reached a median of 97 of 100 after a captioning stage, against 63 when the same number of steps went to the questions alone. Counting didn't need it ([§4](#stage-two)).
- **LoRA added little here.** Once stage 1 was done, letting the language model adapt moved two of the three questions by 2 points or less, and position by 10 at the median, within the spread of three seeds ([§4](#stage-two)).

**Living with it.**

- **A score means nothing without a control.** Asked with a blank picture, the models still got about 29 of 100 counting questions right by guessing. Every number here is read against that ([§5](#is-it-looking)).
- **Pictures cost tokens.** 49 tokens is sixteen times as many positions as this picture's three-word caption. Merging 196 patches into 49 made a training step 3× faster, and here it didn't cost accuracy ([§6](#what-an-image-costs)).
- **This is one design of four.** Others summarize the picture into a fixed number of tokens, read it through cross-attention, or turn it into a vocabulary of image tokens ([§7](#other-designs)).

The pictures are simple, generated shapes, so these numbers describe how the mechanism behaves, not how good the model is on photographs.

### Setup

Everything below is measured, and you can re-run all of it:

```bash
git clone https://github.com/bearbearyu1223/llm-architectures-refresher
cd llm-architectures-refresher
uv sync && uv run demo07
```

The code is in [`demos/d07_multimodal.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d07_multimodal.py), and the model itself is in [`vlm.py`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/vlm.py): the patch cutting, the connector, and the code that splices a picture into a sentence are all written out rather than imported, because this series builds the thing it's explaining. Every number this post quotes is printed by that program, and each block of its output, which I call a **receipt**, names the function that printed it.

A warning about time: the demo *trains*, eleven times, and takes about **two hours** on my Mac. Most of that is stage 2's three seeds of three versions, and one stage 1 run on 196 tokens per picture, which is three times slower than the rest. `uv run demo07 --figures` redraws every figure from the last run's saved scores in seconds.

Each training run starts from a fixed seed, so it's repeatable on the same machine: I stopped and restarted the demo partway through, and stage 1 and the first stage-2 run printed identical scores both times. Expect a different machine to move some scores by a point or two, since floating-point sums can be ordered differently.

**The three parts.**

- The **vision encoder** is [SigLIP-base](https://huggingface.co/google/siglip-base-patch16-224) ([paper](https://arxiv.org/abs/2303.15343)), a 12-layer transformer that was trained on pictures and their captions. It stays frozen the whole time.
- The **language model** is **Qwen2.5-0.5B**, the same model [post 4](/posts/llm-architectures-quantization/) quantized and [post 6](/posts/llm-architectures-lora/) fine-tuned ([the weights](https://huggingface.co/Qwen/Qwen2.5-0.5B), [the report](https://arxiv.org/abs/2412.15115)). It's a plain text model; nothing about it was built for pictures. It stays frozen too, except for the LoRA adapters one of the stage-2 runs adds.
- The **connector** is the only new part. It starts random and is trained from scratch.

Both models download on the first run. Everything runs in fp32, as in post 6, so that two training runs can only differ in what they train.

#### The pictures, and what the model is asked {#the-pictures}

The demo draws its own pictures: up to four colored shapes on a white square, each a circle, a square or a triangle in one of six colors. Here's what it's taught to say about them, using the picture from the figure above:

- **Stage 1, describe.** Given the picture and `Caption:`, write every shape from left to right: `an orange circle, a yellow circle and a purple circle`.
- **Stage 2, count.** `How many shapes are there?` is answered `three`.
- **Stage 2, binding.** `What color is the triangle?` is answered with that triangle's color. It's only asked when the picture has exactly one triangle, so there's always one right answer. (The picture in the figures has three circles, so it never gets this question.)
- **Stage 2, leftmost.** `What color is the leftmost shape?` is answered `orange`.

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
```text
  step                                              result
  ----------------------------------------------------------
  a picture               224 x 224 pixels, 3 numbers each
  numbers in the picture                           150,528
  patch size                                16 x 16 pixels
  patches                                    14 x 14 = 196
  numbers in one patch                   16 x 16 x 3 = 768

  patch embedding weight             768 x 768 = 589,824
    as a convolution                 (768, 3, 16, 16), stride 16
  conv vs one matmul per patch       max difference 2.3e-06
  position embedding                 (196, 768), one per patch

  encoder layers                     12, width 768, 12 heads
  encoder parameters                 92,884,224
    pooling head (not used here)     7,087,104
    used to make the patch vectors   85,797,120
  encoder output for one picture     196 vectors, each 768 wide

  >> A picture reaches the transformer as 196 patch vectors, and the patch
     embedding is an ordinary matrix multiply on each patch's pixels: the
     same first step as looking up a word's embedding, with pixels in
     place of a token id.
```

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
```text
  step                                          shape
  -----------------------------------------------------
  encoder output                    196 vectors x 768
  merge each 2 x 2 block           49 vectors x 3,072
  projector layer 1                      3,072 -> 896
  projector layer 2                        896 -> 896
  what the language model reads      49 vectors x 896
  one word's embedding, for scale      1 vector x 896

  projector parameters, after merge  3,557,120
    without the merge (768 in)       1,492,736
```

The connector is 3.56 million parameters, about 0.7% of the language model, and it's the only part stage 1 trains.

#### Where the picture goes {#where-the-picture-goes}

Then comes the step that makes the whole thing work, and it's simpler than you might expect: the 49 image vectors are placed in the sequence where words would be.

<!-- FIGURE: sequence -->
![The sequence the language model reads for one question: two text tokens, 49 image vectors, then the question's 13 tokens, after which it predicts the answer](/assets/picture/2026-10-11-llm-architectures-multimodal/sequence-light.png){: .light width="1000" height="246" }
![The sequence the language model reads for one question: two text tokens, 49 image vectors, then the question's 13 tokens, after which it predicts the answer](/assets/picture/2026-10-11-llm-architectures-multimodal/sequence-dark.png){: .dark width="1000" height="246" }

The language model is given the vectors for `Image:`, then the picture's 49 vectors, then the vectors for the question's tokens, and asked to continue. The word vectors are looked up from Qwen's vocabulary table as always. The image vectors come out of the connector instead. From the language model's side they're all just positions in one sequence, and its attention treats them all the same way: the token `leftmost` can attend to an image position exactly as it would attend to an earlier word.

<!-- RECEIPT: s2b -->
```text
  piece         positions   kind
  --------------------------------
  'Image:'              2   text
  the picture          49  image
  the question         13   text
  TOTAL                64
```

**The picture is never turned into words.** There's no hidden caption in the middle. The language model gets 49 vectors that don't correspond to any word in its vocabulary, and has to learn, through the connector, to make sense of them.

---

### 3. Stage 1: learning to describe {#stage-one}

The model now has all its parts, and two of them already know a lot: the encoder knows what's in a picture, and the language model knows how to write. The connector between them knows nothing. It starts random, so the 49 vectors it hands the language model are noise.

Training happens in two stages, the recipe [LLaVA](https://arxiv.org/abs/2304.08485) made standard:

<!-- FIGURE: training-stages -->
![The two training stages side by side: in stage 1 only the connector trains, on captions; in stage 2 it trains on questions, alone in one run and with LoRA adapters on the language model in the other](/assets/picture/2026-10-11-llm-architectures-multimodal/training-stages-light.png){: .light width="1000" height="517" }
![The two training stages side by side: in stage 1 only the connector trains, on captions; in stage 2 it trains on questions, alone in one run and with LoRA adapters on the language model in the other](/assets/picture/2026-10-11-llm-architectures-multimodal/training-stages-dark.png){: .dark width="1000" height="517" }

**Stage 1 trains only the connector, on captions.** The encoder and the language model are both frozen. The model is shown a picture and the word `Caption:`, and its target is the caption: every shape, named left to right. LLaVA's paper calls this stage "pre-training for feature alignment", and describes it the same way: "we keep both the visual encoder and LLM weights frozen, and maximize the likelihood ... with trainable parameters θ = W (the projection matrix) only" (§4.2).

Why captions first? Because a caption is the most direct test of the connector's one job. The language model already knows how to write `a yellow circle`. It only needs to be handed vectors that make *that* the likely thing to write, and the only part allowed to change is the one producing those vectors. So stage 1 teaches the connector to speak the language model's language, without the language model learning anything that could cover for a bad connector.

**How it learns.** Training works as in [post 6](/posts/llm-architectures-lora/#loss-and-exact-match): the model predicts each caption token, the **loss** measures how surprised it was by the right one, and each step nudges the trainable parameters to make it less surprised next time. Only the caption's tokens are scored; the picture and the prompt are context. And as in post 6, freezing a part doesn't mean the training signal skips it. To reach the connector, the gradient has to travel backwards through all 24 of the language model's frozen layers, because that's how the connector's output affected the loss. So a step costs a full forward and backward pass through the language model even though none of its weights change.

Here's stage 1, from [`stage_one`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d07_multimodal.py):

<!-- RECEIPT: s3 -->
```text
  trained: the connector only        3,557,120 parameters
  frozen                             vision encoder and language model
  training                           400 steps x 16 captions, lr 0.001
  loss at step 1                     3.8180
  loss at the end (mean of last 10)  0.0899
  time per step                      1.02 s

  Captions exactly right, on 120 new pictures:

                  right picture  blank picture  wrong picture
  -------------------------------------------------------------
  before stage 1              0              0              0
  after stage 1              87              1              0
```

Read it top to bottom. The loss starts at 3.82, which means that, on average, the model gave the right next word of a caption a probability of about 2% ($e^{-3.82}$); it ends near 0.09, about 91% ($e^{-0.09}$). Before training it gets **no** caption right. That's what a random connector should do: the language model is reading 49 vectors of noise. After 400 steps it gets **87 of 120** exactly right on pictures it never saw, every shape named in the right order with the right color. Exact match is strict: one wrong word in `a blue triangle, a green square and a purple square` and the whole caption counts as wrong.

The two control columns say the 87 isn't luck. With a blank picture the model gets 1 caption right, and with another scene's picture none. A caption names every shape and its position, so there's nothing to guess from: the model either reads the picture or fails.

Remember what changed to get here: 3.6 million numbers in the connector. The encoder and the language model are bit-for-bit the models that were downloaded. The connector learned to hand Qwen vectors that make it write `a green square` when there's a green square in the picture, from 400 batches of examples.

The first three test pictures, as the model describes them:

<!-- RECEIPT: s3b -->
```text
  truth: a red triangle
  model: a red triangle

  truth: a blue triangle, a green square and a purple square
  model: a blue triangle, a green square and a purple square

  truth: a purple circle and a green triangle
  model: a purple circle and a green triangle
```

All three are exactly right, including the three-shape picture with two squares, where the colors have to stay attached to the right shapes. That's binding, and the next section shows it's the thing stage 1 teaches best. The demo prints only these three, so the 33 misses aren't shown.


---

### 4. Stage 2: learning to answer {#stage-two}

Stage 1 taught the model to say what's in a picture. Stage 2 teaches it to answer a question about one. The three questions are counting, binding and finding the leftmost shape, introduced in [Setup](#the-pictures), on pictures with two to four shapes.

Notice that a good caption already contains every answer. `an orange circle, a yellow circle and a purple circle` has three shapes in it, the first one named is the leftmost, and each color sits right next to the shape it belongs to. So a model that captions well has, in a sense, already seen what the questions ask about. Stage 2 has to teach it to *answer*: to read the question, pick out the part that matters, and say it in one word.

The demo trains four versions and scores each on 100 new pictures per question:

- **Stage 1 only.** The stage-1 model, asked the questions without any training on them. The base language model has never seen this question format with a picture in it, so this is the starting point.
- **No stage 1.** A connector trained from random on the questions alone, skipping captions. It gets **800 steps**, as many as stages 1 and 2 together, so if it falls short it isn't for lack of training. This is the control for stage 1 itself: without it, there's no way to tell how much of stage 2's result came from stage 1.
- **Connector only.** The stage-1 connector, trained 400 more steps on the questions. The language model stays frozen.
- **Connector + LoRA.** The same, plus a rank-8 LoRA adapter on all seven kinds of matrix in the language model, exactly as in [post 6](/posts/llm-architectures-lora/#where-the-adapters-go): 4,399,104 more trainable parameters. Now the language model itself can change a little.

Each trained version is run three times from different random starts (the **seeds**), because post 6 learned the hard way that a difference between two single runs can be luck. Tables show the median of the three, with the lowest and highest in brackets.

([`stage_two`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d07_multimodal.py))

<!-- RECEIPT: s4a -->
```text
  batch                              16 questions, seeds 0, 1, 2

  model             starts from  steps  trainable  s/step
  ---------------------------------------------------------
  no stage 1             random    800  3,557,120    0.91
  connector only        stage 1    400  3,557,120    0.92
  connector + LoRA      stage 1    400  7,956,224    1.11
```

<!-- RECEIPT: s4b -->
```text
  task                 model  right picture       blank  wrong picture
  ----------------------------------------------------------------------
  count         stage 1 only              0           0              0
                  no stage 1    98 (98-100)  29 (29-29)     35 (34-35)
              connector only     96 (91-99)   29 (0-29)     34 (34-34)
            connector + LoRA     95 (91-96)  29 (29-29)     34 (34-35)
  binding       stage 1 only             15           0              4
                  no stage 1     63 (49-66)  15 (12-21)     18 (15-19)
              connector only     97 (96-97)   14 (0-21)     16 (16-18)
            connector + LoRA    99 (95-100)   13 (8-17)     18 (17-21)
  leftmost      stage 1 only             23           0              2
                  no stage 1     72 (69-73)  15 (15-18)       9 (7-13)
              connector only     82 (81-95)   13 (0-18)        6 (5-7)
            connector + LoRA     92 (90-93)   18 (0-18)        7 (6-8)
```

<!-- FIGURE: stage-two-results -->
![Stage 2 scores for three questions, each model scored with the right picture, a blank picture and another scene's picture](/assets/picture/2026-10-11-llm-architectures-multimodal/stage-two-results-light.png){: .light width="1000" height="477" }
![Stage 2 scores for three questions, each model scored with the right picture, a blank picture and another scene's picture](/assets/picture/2026-10-11-llm-architectures-multimodal/stage-two-results-dark.png){: .dark width="1000" height="477" }

Read the table one question at a time, because the three disagree.

**Counting is easy whichever way it's trained.** Every trained version gets almost all 100 right, median 95 to 98, including the one that never saw a caption. How many things there are is something the encoder's patch vectors carry plainly, and a connector trained on nothing but counting questions learns to pass it on.

**Binding is where stage 1 matters.** Trained on questions alone, for 800 steps, the connector gets a median of **63** of 100 colors right; after 400 steps of captions and 400 of questions, the same total, it gets **97**. Telling the model *which shape* the purple belongs to is exactly what a left-to-right caption drills thousands of times: `a purple triangle` puts the color next to its shape every time. Questions alone teach it far less well, even given as many steps as both stages together. This is the post's clearest result, and it cuts against a finding at much larger scale: the [Prismatic VLMs](https://arxiv.org/abs/2402.07865) study found the connector-only first stage "unnecessary" for 7B and 13B models (§4.1). The difference is plausibly in what stage 2 trains. Theirs trains every weight of the language model, which can learn binding itself; this post's connector-only arm has 3.6 million parameters to learn it with, and a frozen 0.5B model behind them.

**Position benefits from both.** Leftmost improves with stage 1 (median 72 without it, 82 with it) and again with LoRA (92). Its spread is the widest in the table: the connector-only runs scored 81, 82 and 95 depending on the seed, which is why a single run would have told two different stories here.

**What LoRA adds is small.** With stage 1 done, letting the language model adapt through a rank-8 adapter leaves count where it was and moves binding by 2. Leftmost gains 10 at the median and its runs become far more consistent, 90 to 93 against 81 to 95, though one connector-only run beat all three LoRA runs, so three seeds can't call it settled. That's a narrower result than "fine-tune the language model too", which is what LLaVA does. A likely reason is the questions: these have one-word answers and a fixed format, so once the connector hands over the right information, the frozen language model already knows how to say `purple`. Questions that need the language model to *reason* about the picture would give an adapter more to do, and [Idefics2](https://arxiv.org/abs/2405.02246)'s experiments found LoRA useful for exactly that kind of training at scale: "Leveraging LoRA still adds expressivity to the training and stabilizes it" (Finding 3).

---

### 5. Is it actually looking? {#is-it-looking}

Every score in §4 comes with two more columns, and they're the most important numbers in this post.

A model can answer a question about a picture correctly without looking at it. If most pictures have three shapes, "three" is a good guess for any counting question. If the language model has a favorite color, it'll be right whenever that's the answer. A score with nothing to compare it to can't tell real looking from a good guess, and this isn't a theoretical worry: the [MMStar](https://arxiv.org/abs/2403.20330) authors found that on real benchmarks "visual content is unnecessary for many samples", with more than half of one benchmark's questions answerable by a language model that was never shown the picture.

So every question is asked three ways:

- **With the right picture.** The real score.
- **With a blank picture.** An all-white square, so the model sees nothing. Whatever it scores here, it scores by guessing from the question.
- **With the wrong picture**, the next test scene's picture, which has its own shapes and colors. A model that really reads the picture should now give *that* picture's answer, which is usually wrong for the question's scene. Right answers here are the ones two scenes happen to share.

The gap between the right picture and the better of the two controls is how much of the score came from looking ([`is_it_looking`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d07_multimodal.py)):

<!-- RECEIPT: s5 -->
```text
  task                 model  right picture  best control  gap
  --------------------------------------------------------------
  count           no stage 1             98            35   63
              connector only             96            34   62
            connector + LoRA             95            34   61
  binding         no stage 1             63            19   44
              connector only             97            18   79
            connector + LoRA             99            18   81
  leftmost        no stage 1             72            15   57
              connector only             82            13   69
            connector + LoRA             92            18   74
```

Every trained model is reading the picture: the smallest gap is 44 points. But the controls are far from zero, and they're worth reading.

**Blank pictures still score.** Shown nothing, the trained models answer about 29 of 100 counting questions right. Pictures have two, three or four shapes, so a model that answers the same number every time is right about a third of the time; with no picture to read, the models fall back on a guess like that. On the two color questions, a blank picture scores 13 to 18, close to what guessing one of six colors would get, about 17.

**Wrong pictures score for a different reason.** A different scene shares the count with the question's scene about a third of the time, so 34 to 35 right on counting is the model correctly reading the *wrong* picture. On leftmost, wrong pictures score only 6 to 9, below the blank picture's 13 to 18, which is exactly what reading the picture predicts: the model answers with the wrong picture's leftmost color, and that's rarely the right scene's.

**The gap shows what stage 1 did.** On binding, the model trained without stage 1 is 44 points above its controls, the stage-1 models 79 and 81. The no-stage-1 model looks at the picture too, it just reads less from it. And the stage-1 model asked questions it was never trained on (`stage 1 only` in §4) scores 15 on binding and 23 on leftmost, against 0 with a blank picture: it already reads the colors and positions, it just hasn't learned to answer in one word. On counting it scores 0, whatever it writes, it isn't the single word `three`.

The same check at benchmark scale is where it gets uncomfortable. The MMStar authors report that GeminiPro scored 42.9% on the MMMU benchmark "without any visual input". This post's controls are the cheap version of their test, and any evaluation of a VLM should run one.

---

### 6. What an image costs {#what-an-image-costs}

A picture isn't free once it's in the sequence. Every position the language model reads costs attention work, and, as [post 2](/posts/llm-architectures-kv-cache/) measured, memory in the **KV cache**: the keys and values the model keeps for every position so it doesn't recompute them for each new token. A picture's positions count exactly like a word's.

For Qwen2.5-0.5B the per-position cost is small and fixed: keys and values, for each of 24 layers, for each of its 2 key/value heads, 64 numbers each, at 2 bytes per number in bf16 ([`what_an_image_costs`](https://github.com/bearbearyu1223/llm-architectures-refresher/blob/main/src/llmrefresher/demos/d07_multimodal.py)):

<!-- RECEIPT: s6a -->
```text
  KV cache per token, in bf16        12,288 bytes
    = 2 (K and V) x 24 layers x 2 heads x 64 dims x 2 bytes

  the same picture as   tokens  KV cache
  ----------------------------------------
  196 patch vectors        196  2.30 MiB
  49 merged vectors         49  0.57 MiB
  its caption in words       3  0.04 MiB
  caption: "a red triangle"
```

A picture costs the same no matter what's in it. The demo's first test picture is one red triangle, whose caption is three tokens; its 49 image tokens take sixteen times as many positions. At 196 tokens it would be 65 times as many.

That's small per picture for a 0.5B model, but it adds up the way post 2's cache did: a conversation with ten pictures at 196 tokens holds 23 MiB of cache for the pictures alone, and the cost per token grows with the model. Real models use far more tokens per picture than this one. LLaVA-1.5's encoder takes 336-pixel pictures in 14-pixel patches, which is $(336/14)^2 = 576$ tokens per picture, the last line of the receipt. That's why so much design effort goes into cutting the count: Qwen2-VL's 2 × 2 merge, which this post copies, SmolVLM's "pixel shuffle", which cuts tokens "by a factor of r²" and finds small models "benefit from more aggressive visual token compression", and the fixed-size summaries of [§7](#other-designs).

Does merging cost accuracy? The demo trains stage 1 again on all 196 patch vectors, same seed, same data, same 400 steps:

<!-- RECEIPT: s6b -->
```text
                             49 tokens  196 tokens
  --------------------------------------------------
  connector parameters       3,557,120   1,492,736
  captions right, of 120            87          70
  final loss (last-10 mean)     0.0899      0.1205
  seconds per step                1.02        3.08

  LLaVA-1.5: 336 px, 14 px patches   (336 / 14)^2 = 576 tokens
```

The merged model did *better*: 87 captions right against 70, with a lower final loss, at a third of the time per step. But that's not a clean comparison, and the reason is in the first row. Without the merge, the connector's first layer takes 768 numbers per vector instead of 3,072, so it has 1.49 million parameters instead of 3.56 million. The comparison changed two things at once, the number of tokens and the size of the connector, and one run can't say which one mattered. What it does show is the thing that matters in practice: cutting the picture to a quarter of the tokens cost nothing here, and made training three times faster. [Idefics2](https://arxiv.org/abs/2405.02246) reports the same direction at scale, where learned pooling from 729 to 64 tokens "increases the performance by 8.5 points on average" (§3.3).

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

Six things, in the order the post found them.

1. **A picture is just more vectors.** A vision encoder turns patches into vectors the way an embedding table turns tokens into vectors, and after that it's a transformer like any other. Nothing about the language model had to change to read a picture.
2. **The connector is the whole adaptation.** 3.6 million trained parameters, about 0.7% of the language model, were enough to get captions exactly right 87 times in 120. The two big models stayed exactly as they were downloaded.
3. **What you train on first matters more than what you train.** Captions taught binding, a color attached to its shape, far better than the questions themselves did with the same number of steps: 97 against 63. At this scale the first stage wasn't a formality.
4. **LoRA was the smallest lever here.** After stage 1 it moved scores by a few points, within or near the spread between seeds. That's a statement about one-word answers on simple pictures, where the frozen language model already knows how to say the answer, not about LoRA in general.
5. **Controls are part of the result.** A blank picture still answered counting questions right 29 times in 100. Without that column, a 50% score would look like partial vision instead of a guess.
6. **Tokens are the cost.** Every picture is a fixed number of positions in the sequence and the KV cache, and the field's designs are largely different answers to how few positions a picture can be squeezed into.

And three limits, stated plainly. The pictures are generated shapes, much easier than photographs, so the scores say how the mechanism behaves, not how good the model is. Everything is one small language model and one small encoder, trained for minutes. And the encoder was frozen throughout, so this post says nothing about what training it would add, which some large models do ([Llama 3](https://arxiv.org/abs/2407.21783) trains its image encoder, §7.2).

---

### 9. Sidebar: the probe {#sidebar-the-probe}

One question to close on, the kind this material gets asked, and what separates an answer that sounds right from one that is.

> **"We attached a vision encoder to our language model and fine-tuned it on question-answer pairs about our product photos. It scores 90% on our test set. Is it ready?"**

**The tempting answer:** *"90% is good. Ship it, and fine-tune more if users find problems."*

A score alone can't answer this, and there are three things to check before it means anything.

**1. Score it without the picture.** Ask every test question again with a blank image, and again with another item's photo. If the blank-picture score is high, the model is answering from the question's wording, and the 90% is partly a measure of how predictable your questions are. This post's §5 does this for every number it reports, and real benchmarks fail it: MMStar found more than half of one benchmark's questions answerable with no picture at all. The useful number is the gap between the right picture and the best control, not the score.

**2. Break the score down by what the question needs.** One average hides everything. Here, three questions about the same pictures behaved completely differently: counting was learned almost however the model was trained, while binding a color to a shape depended on whether the model had first learned to describe pictures. A model that's 90% overall can be near-perfect on "what color is it?" and near-chance on "which one is on the left?", and your users will find the second kind.

**3. Ask where a failure lives before training more.** A VLM has three parts and a wrong answer can come from any of them: the encoder may not represent the detail, the connector may not pass it on, or the language model may not use it. More question-answer data trains the last two at best. This post's binding result is a small example: the questions alone, given as many steps as both stages together, didn't teach binding nearly as well as captions did. And research on much larger models ([*Vision language models are blind*](https://arxiv.org/abs/2407.06581)) finds the encoders often hold information the language model fails to use.

What the question is really testing is whether "90% on our test set" is being read as "it understands the photos", or as what it is: a number that means little until you know what the model scores without looking.

### What's next {#whats-next}

Post 8 is **QLoRA**, the post this one jumped ahead of. It's [post 6](/posts/llm-architectures-lora/) and [post 4](/posts/llm-architectures-quantization/) at the same time: store the frozen base model in 4 bits, then train a LoRA adapter on top of it. This post's stage 2 did exactly the first half of that, an adapter on a frozen model, with the frozen model in full fp32. QLoRA asks what happens when the thing you froze is also compressed: whether gradients can flow back through quantized weights to reach the adapter, why the adapter stays in higher precision when the model under it doesn't, and whether the two approximations add up or stay out of each other's way.

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
