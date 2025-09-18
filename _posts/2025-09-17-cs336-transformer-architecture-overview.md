---
layout: post
title: "Study Notes: Stanford CS336 Language Modeling from Scratch [6]"
categories: cs336
author: 
- Han Yu
---
## An Overview of Popular Transformer Architectures

While working on the Transformer LM assignments, I realized it would be helpful to also step back and look at some of the most popular Transformer architectures. Here are my notes and takeaways.

### Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Encoder-Decoder Transformers](#encoder-decoder-transformers)
4. [Decoder-Only Transformers](#decoder-only-transformers)
5. [Encoder-Only Transformers](#encoder-only-transformers)
6. [Comparison Summary](#comparison-summary)
7. [Modern Trends and Applications](#modern-trends-and-applications)

### Introduction

Transformer architectures have revolutionized natural language processing and machine learning. Since the original "Attention is All You Need" paper in 2017, three main architectural variants have emerged, each optimized for different types of tasks:

- **Encoder-Decoder**: Sequence-to-sequence transformations
- **Decoder-Only**: Autoregressive text generation
- **Encoder-Only**: Text understanding and classification

This note provides an overview of how each architecture works, their training methodologies, evaluation approaches, and practical applications.

---

### Architecture Overview

#### Core Components

All transformer architectures share fundamental building blocks:

- **Self-Attention Mechanism**: Allows tokens to attend to other tokens
- **Feed-Forward Networks**: Position-wise processing layers
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Enables deep architectures
- **Positional Encodings**: Provides sequence position information

#### Multi-Head Self-Attention Deep Dive

The multi-head self-attention mechanism is the core innovation of transformers. Here's how it works in detail:

<img src="/assets/picture/2025-09-17-cs336-transformer-architecture-overview/encoder-multi-head-self-attention.png" alt="Multi-Head Self-Attention Mechanism" width="1080">

**Key Steps:**
1. **Linear Projections**: Input embeddings are transformed into Query (Q), Key (K), and Value (V) matrices
2. **Head Splitting**: Q, K, V matrices are reshaped and split into multiple attention heads
3. **Parallel Attention**: Each head computes attention independently using scaled dot-product attention
4. **Concatenation**: All head outputs are concatenated back together
5. **Final Projection**: A final linear layer projects the concatenated result back to the model dimension

This parallel processing allows the model to attend to different types of relationships simultaneously - some heads might focus on syntactic relationships while others capture semantic connections.

#### Key Differences

The main distinction lies in the **attention patterns**:

| Architecture | Attention Pattern | Primary Use Case |
|--------------|-------------------|------------------|
| Encoder-Decoder | Bidirectional (encoder) + Causal (decoder) | Sequence-to-sequence tasks |
| Decoder-Only | Causal only | Autoregressive generation |
| Encoder-Only | Bidirectional only | Understanding and classification |

---

### Encoder-Decoder Transformers

#### Architecture Design

The encoder-decoder architecture consists of two separate stacks connected through cross-attention:

<img src="/assets/picture/2025-09-17-cs336-transformer-architecture-overview/encoder-decoder-transformer-architecture.png" alt="Encoder-Decoder Transformer Architecture" width="1080">

**Key Components:**
- **Encoder**: Uses bidirectional self-attention to process input sequence with full context
- **Decoder**: Uses causal self-attention + cross-attention to generate output sequence
- **Cross-Attention**: Allows decoder to attend to encoder representations at each layer
- **Layer-by-Layer Processing**: Each decoder layer receives information from the corresponding encoder layer

**Key Features:**
- **🔄 Bidirectional Encoder**: Full context understanding for source sequence
- **🔗 Cross-Attention**: Decoder attends to encoder representations
- **📝 Sequence-to-Sequence**: Perfect for translation, summarization, and question answering

This architecture excels at tasks requiring structured input-output transformations where the model needs to understand the entire input before generating the output.

#### Training Methodology

**Objective**: Learn to map input sequences to output sequences

**Training Process:**
1. **Teacher Forcing**: Use ground truth target tokens as decoder input
2. **Parallel Training**: All target positions trained simultaneously
3. **Cross-Entropy Loss**: Computed over target vocabulary

```python
# Training pseudocode
def train_encoder_decoder(model, dataloader):
    for batch in dataloader:
        src_tokens = batch['source']      # Input sequence
        tgt_tokens = batch['target']      # Target sequence
        
        # Teacher forcing setup
        tgt_input = tgt_tokens[:-1]       # Decoder input
        tgt_output = tgt_tokens[1:]       # Expected output
        
        # Forward pass
        logits = model(src_tokens, tgt_input)
        
        # Compute loss
        loss = cross_entropy(logits, tgt_output)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
```

**Training Data Requirements:**
- **Parallel Corpora**: Paired input-output sequences
- **Domain-Specific**: Task-dependent datasets
- **Quality**: High-quality alignments crucial for performance

#### Evaluation Methods

**Generation-Based Evaluation:**

1. **Automatic Metrics**:
   - **BLEU**: N-gram overlap for translation
   - **ROUGE**: Recall-oriented for summarization
   - **METEOR**: Semantic similarity measures
   - **BERTScore**: Contextual embeddings comparison

2. **Human Evaluation**:
   - **Fluency**: How natural the output sounds
   - **Adequacy**: How well meaning is preserved
   - **Faithfulness**: Accuracy to source content

**Task-Specific Benchmarks:**
- **Translation**: WMT datasets, OPUS corpora
- **Summarization**: CNN/DailyMail, XSum
- **Question Answering**: SQuAD variants

#### Use Cases and Applications

**Primary Applications:**
- **Machine Translation**: Language pair transformations
- **Text Summarization**: Document to summary conversion
- **Dialogue Systems**: Context-aware response generation
- **Code Translation**: Between programming languages
- **Data-to-Text**: Structured data to natural language

**Examples:**
- Google Translate (earlier versions)
- T5 (Text-to-Text Transfer Transformer)
- BART (Bidirectional and Auto-Regressive Transformers)
- mT5 (Multilingual T5)

---

### Decoder-Only Transformers

#### Architecture Design

Decoder-only models use a single stack with causal attention:

<img src="/assets/picture/2025-09-17-cs336-transformer-architecture-overview/decoder-only-transformer-lm.png" alt="Decoder-Only Transformer Architecture" width="880">

**Key Characteristics:**
- **Causal Masking**: Prevents attention to future tokens during training and inference
- **Autoregressive Generation**: Produces one token at a time during generation
- **Unified Architecture**: Same model architecture handles various tasks through different prompting strategies
- **Scalability**: Architecture scales well to very large model sizes (billions of parameters)

**Key Features:**
- **🔒 Causal Masking**: Can only attend to previous tokens
- **🔄 Autoregressive**: Generates tokens one at a time
- **💬 Text Generation**: Chat, completion, and code generation

This architecture has become the foundation for modern large language models like GPT, excelling at open-ended text generation and few-shot learning through prompting.

#### Training Methodology

**Objective**: Learn to predict the next token given previous context

**Training Process:**
1. **Next Token Prediction**: Core training objective
2. **Causal Masking**: Maintains autoregressive property during training
3. **Large-Scale Data**: Trained on massive text corpora

```python
# Training pseudocode
def train_decoder_only(model, dataloader):
    for batch in dataloader:
        # Sequence: "The cat sat on the mat"
        input_tokens = batch['tokens'][:-1]   # "The cat sat on the"
        target_tokens = batch['tokens'][1:]   # "cat sat on the mat"
        
        # Forward pass with causal masking
        logits = model(input_tokens)
        
        # Next token prediction loss
        loss = cross_entropy(logits, target_tokens)
        
        loss.backward()
        optimizer.step()
```

**Multi-Stage Training:**

1. **Pre-training**:
   - **Data**: Large-scale web text, books, articles
   - **Objective**: Next token prediction
   - **Scale**: Billions to trillions of tokens

2. **Instruction Fine-tuning**:
   - **Data**: Human-written instruction-response pairs
   - **Objective**: Follow instructions accurately
   - **Benefits**: Improved task performance

3. **Reinforcement Learning from Human Feedback (RLHF)**:
   - **Data**: Human preference comparisons
   - **Objective**: Align with human values
   - **Benefits**: Safer, more helpful responses

#### Evaluation Methods

**Multiple Evaluation Paradigms:**

1. **Perplexity Measurement**:
```python
def compute_perplexity(model, test_data):
    total_loss = 0
    total_tokens = 0
    
    for sequence in test_data:
        loss = model.compute_loss(sequence)
        total_loss += loss * len(sequence)
        total_tokens += len(sequence)
    
    return exp(total_loss / total_tokens)
```

2. **Generation Quality**:
   - **Human Evaluation**: Coherence, relevance, helpfulness
   - **Automatic Metrics**: Diversity, repetition, toxicity
   - **Task-Specific**: BLEU for translation, ROUGE for summarization

3. **Benchmark Evaluation**:
   - **GLUE/SuperGLUE**: General language understanding
   - **MMLU**: Massive multitask language understanding
   - **HumanEval**: Code generation capabilities
   - **HellaSwag**: Commonsense reasoning

#### Use Cases and Applications

**Primary Applications:**
- **Text Generation**: Creative writing, content creation
- **Conversational AI**: Chatbots, virtual assistants
- **Code Generation**: Programming assistance
- **Question Answering**: Information retrieval and reasoning
- **Few-Shot Learning**: Task adaptation through prompting

**Examples:**
- GPT family (GPT-2, GPT-3, GPT-4)
- LLaMA (Large Language Model Meta AI)
- PaLM (Pathways Language Model)
- Claude (Anthropic's assistant)
- ChatGPT and GPT-4

---

### Encoder-Only Transformers

#### Architecture Design

Encoder-only models use bidirectional attention for understanding:

<img src="/assets/picture/2025-09-17-cs336-transformer-architecture-overview/encoder-only-transformer-lm.png" alt="Encoder-Only Transformer Architecture" width="880">

**Key Characteristics:**
- **Bidirectional Context**: Can attend to all positions in the sequence simultaneously
- **Rich Representations**: Deep contextual understanding from both left and right context
- **Task Adaptation**: Requires fine-tuning for downstream tasks but excels at understanding
- **Special Tokens**: Uses [CLS] and [SEP] tokens for sequence classification and separation

**Key Features:**
- **🔄 Bidirectional Attention**: Full context understanding from both directions
- **🧠 Understanding Tasks**: Classification, extraction, comprehension
- **📚 Pre-training + Fine-tuning**: Masked language modeling then task-specific training

This architecture excels at tasks requiring deep understanding of text, where the model benefits from seeing the entire context before making predictions. The bidirectional nature makes it particularly powerful for classification and extraction tasks.

#### Training Methodology

**Pre-training Objective**: Masked Language Modeling (MLM)

**Training Process:**
1. **Token Masking**: Randomly mask 15% of input tokens
2. **Bidirectional Processing**: Full context available for predictions
3. **Mask Prediction**: Reconstruct original tokens

```python
# Training pseudocode
def train_encoder_only(model, dataloader):
    for batch in dataloader:
        # Original: "The cat sat on the mat"
        # Masked:   "The [MASK] sat on the [MASK]"
        
        masked_tokens = batch['masked_input']
        original_tokens = batch['original_input']
        mask_positions = batch['mask_positions']
        
        # Bidirectional encoding
        hidden_states = model(masked_tokens)
        
        # Predict only at masked positions
        masked_predictions = hidden_states[mask_positions]
        masked_targets = original_tokens[mask_positions]
        
        loss = cross_entropy(masked_predictions, masked_targets)
        
        loss.backward()
        optimizer.step()
```

**Masking Strategy:**
- **80%**: Replace with [MASK] token
- **10%**: Replace with random token
- **10%**: Keep original token

**Fine-tuning for Downstream Tasks:**

After pre-training, models are fine-tuned for specific applications:

```python
def finetune_classification(pretrained_model, task_data):
    # Add task-specific classification head
    classifier = Linear(hidden_size, num_classes)
    
    for batch in task_data:
        inputs, labels = batch['text'], batch['labels']
        
        # Get contextual representations
        hidden_states = pretrained_model(inputs)
        
        # Use [CLS] token for classification
        cls_representation = hidden_states[:, 0]
        
        # Classification prediction
        logits = classifier(cls_representation)
        loss = cross_entropy(logits, labels)
        
        loss.backward()
        optimizer.step()
```

#### Evaluation Methods

**Task-Specific Evaluation:**

1. **Classification Tasks**:
   - **Accuracy**: Percentage of correct predictions
   - **F1-Score**: Harmonic mean of precision and recall
   - **Matthews Correlation**: Balanced measure for imbalanced data

2. **Token-Level Tasks**:
   - **Named Entity Recognition**: Entity-level F1
   - **Part-of-Speech Tagging**: Token-level accuracy
   - **Dependency Parsing**: Unlabeled/labeled attachment scores

3. **Span-Based Tasks**:
   - **Question Answering**: Exact match and F1 scores
   - **Reading Comprehension**: Answer extraction accuracy

**Benchmark Suites:**
- **GLUE**: General Language Understanding Evaluation
- **SuperGLUE**: More challenging language understanding tasks
- **SentEval**: Sentence representation evaluation

#### Use Cases and Applications

**Primary Applications:**
- **Text Classification**: Sentiment analysis, topic classification
- **Named Entity Recognition**: Information extraction
- **Question Answering**: Extractive QA systems
- **Semantic Similarity**: Text matching and retrieval
- **Language Understanding**: Intent classification, slot filling

**Examples:**
- BERT (Bidirectional Encoder Representations from Transformers)
- RoBERTa (Robustly Optimized BERT Pretraining Approach)
- DeBERTa (Decoding-enhanced BERT with Disentangled Attention)
- ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)

---

### Comparison Summary

#### Architecture Comparison

The three transformer architectures shown in the diagrams above have distinct characteristics that make them suitable for different tasks:

| Aspect | Encoder-Decoder | Decoder-Only | Encoder-Only |
|--------|-----------------|--------------|--------------|
| **Attention Pattern** | Bidirectional + Causal | Causal Only | Bidirectional Only |
| **Primary Strength** | Seq2seq transformation | Text generation | Text understanding |
| **Training Data** | Parallel sequences | Raw text | Raw text + labels |
| **Evaluation Focus** | Generation quality | Perplexity + tasks | Task performance |
| **Inference** | Autoregressive | Autoregressive | Single forward pass |
| **Architecture Complexity** | Most complex (2 stacks) | Simple (1 stack) | Simple (1 stack) |
| **Cross-Attention** | ✅ Required | ❌ None | ❌ None |

#### Training Requirements

| Requirement | Encoder-Decoder | Decoder-Only | Encoder-Only |
|-------------|-----------------|--------------|--------------|
| **Data Quantity** | Moderate (paired data) | Large (raw text) | Moderate (raw + labeled) |
| **Data Quality** | High (alignment crucial) | Variable (web-scale) | High (clean text) |
| **Compute Cost** | Moderate | Very High | Moderate |
| **Training Time** | Days to weeks | Weeks to months | Days to weeks |

#### Use Case Suitability

| Task Type | Best Architecture | Rationale |
|-----------|------------------|-----------|
| **Translation** | Encoder-Decoder | Structured input-output mapping with cross-attention |
| **Text Generation** | Decoder-Only | Autoregressive nature with causal masking |
| **Classification** | Encoder-Only | Bidirectional understanding with task-specific heads |
| **Summarization** | Encoder-Decoder / Decoder-Only | Both work well - encoder-decoder for extractive, decoder-only for abstractive |
| **Question Answering** | All three | Encoder-only for extractive, decoder-only for generative, encoder-decoder for complex reasoning |
| **Dialogue** | Decoder-Only | Generative conversation with context understanding |
| **Code Generation** | Decoder-Only | Sequential token generation with programming syntax |
| **Sentiment Analysis** | Encoder-Only | Classification task with bidirectional context |
| **Named Entity Recognition** | Encoder-Only | Token-level classification with full context |

#### Architecture Selection Guide

**Choose Encoder-Decoder when:**
- You have paired input-output data (parallel corpora)
- Tasks require understanding input completely before generating output
- You need structured transformations (translation, summarization)
- Cross-attention between source and target is beneficial

**Choose Decoder-Only when:**
- You want a unified model for multiple tasks
- Open-ended text generation is the primary goal
- You have large amounts of raw text data
- You want to leverage in-context learning and prompting

**Choose Encoder-Only when:**
- Understanding and classification are the primary goals
- You don't need to generate long sequences
- You have labeled data for fine-tuning
- Bidirectional context improves performance significantly

This is just a quick note 📝 — to dive into the details, you’d probably need to read some relevant papers 📚, but I hope it still shared something useful ✨