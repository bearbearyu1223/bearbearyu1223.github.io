---
title: "Study Notes: Stanford CS336 Language Modeling from Scratch [4]"
date: 2025-08-10 00:00:00 -0700
categories: [CS336, Tokenization]
tags: [cs336, bpe, pretokenization, regex, gpt-2, multilingual]
description: >-
  Demystifying GPT-2's pre-tokenization regex pattern — how one
  carefully crafted regex handles text from multiple languages,
  scripts, and symbol sets for BPE tokenization.
redirect_from:
  - /cs336/2025/08/10/cs336-gpt2-regex-for-pretokenization-explaind.html
---
## Demystifying GPT-2's Pre-Tokenization: How One Regex Pattern Handles the World's Languages

While working on **Assignment 1** of *Stanford’s CS336: Language Modeling from Scratch*, I came across a deceptively simple — yet remarkably powerful — regex pattern used in the pre-tokenization stage of the BPE algorithm.  

I thought it would be worthwhile to share my notes and walk through how this single pattern can handle text from **multiple languages, scripts, and symbol sets** with precision.

You can find my full BPE assignment implementation here:  
- **BPE training algorithm:** [bpe.py](https://github.com/bearbearyu1223/assignment1-basics/blob/main/cs336_basics/bpe.py)  
- **Tokenizer class:** [tokenizer.py](https://github.com/bearbearyu1223/assignment1-basics/blob/main/cs336_basics/tokenizer.py)

---

### 🔧 How to Run the BPE Training Process

1. **Clone the repository**  
   ```bash
   git clone https://github.com/bearbearyu1223/assignment1-basics.git
   cd assignment1-basics
   ```

2. **Set up the local development environment**  
   Follow the instructions in the [developer_guide.md](https://github.com/bearbearyu1223/assignment1-basics/blob/main/developer_guide.md).

3. **Run BPE training**  
   ```bash
   uv run cs336_basics/train_bpe_example.py
   ```
   This will train a BPE tokenizer using the `TinyStoriesV2-GPT4-train.txt` dataset, with 10,000 vocabulary size and with special token `"<|endoftext|>"`.

---

### 🧪 How to Test the Tokenizer

Run:
```bash
uv run pytest tests/test_train_bpe.py
```

This will validate the tokenizer’s functionality and ensure the pre-tokenization regex behaves as expected.


### 📜 The GPT-2 Split Pattern

```python
import regex

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

This single (but mighty) regex is responsible for splitting text into meaningful segments — **words, numbers, punctuation, symbols, whitespace** — in a way that is consistent across languages and scripts.

---

### 🔍 Pattern Breakdown

#### 1. Contractions
```regex
'(?:[sdmt]|ll|ve|re)
```
Matches **common English contractions** starting with an apostrophe:  
`'s, 'd, 'm, 't, 'll, 've, 're`

**Examples:**
- `"don't"` → `["don", "'t"]`
- `"we're"` → `["we", "'re"]`

---

#### 2. Letters (Any Language)
```regex
 ?\p{L}+
```
Matches **letters** from **any Unicode language** (with optional leading space).

- `\p{L}` = Unicode “Letter” category  
- Covers: English, Chinese, Arabic, accented characters, and more.

**Examples:**
- `"hello world"` → `["hello", " world"]`
- `"café 北京"` → `["café", " 北京"]`

---

#### 3. Numbers (Any Script)
```regex
 ?\p{N}+
```
Matches **numbers** from any writing system (with optional leading space).

- `\p{N}` = Unicode “Number” category  
- Covers: Arabic numerals (`0–9`), Roman numerals (`Ⅰ, Ⅱ, Ⅲ`), Arabic-Indic (`٠١٢`), etc.

**Examples:**
- `"I have 5 items"` → `["I", " have", " 5", " items"]`
- `"Ⅲ winners"` → `["Ⅲ", " winners"]`

---

#### 4. Punctuation / Symbols
```regex
 ?[^\s\p{L}\p{N}]+
```
Matches **punctuation or symbols** (with optional leading space).

- `[^\s\p{L}\p{N}]` = NOT whitespace, NOT letters, NOT numbers  
- Captures: `!@#$%^&*()_+-=[]{}|;:'",./<>?`

**Examples:**
- `"Wow!!!"` → `["Wow", "!!!"]`
- `" $100"` → `[" $", "100"]`

---

#### 5. Trailing Whitespace
```regex
\s+(?!\S)
```
Matches **whitespace at the end** of text or before more whitespace.  
This ensures trailing spaces are preserved as tokens.

---

#### 6. General Whitespace
```regex
\s+
```
Matches **any remaining whitespace**.

---

### 🛠 Testing the Pattern

Here’s a helper function to test how this regex splits different inputs.

```python
def test_regex(text, description=""):
    """Test the regex pattern and display results clearly"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"INPUT: '{text}'")
    print(f"{'='*60}")

    matches = regex.findall(GPT2_SPLIT_PATTERN, text)

    print(f"TOKENS ({len(matches)}):")
    for i, token in enumerate(matches, 1):
        print(f"  {i:2d}: {repr(token)}") 

    reconstructed = ''.join(matches)
    print(f"\nRECONSTRUCTION CHECK: {'✓ PASS' if reconstructed == text else '✗ FAIL'}")
    return matches
```

---

### 🧪 Real-World Test Cases

Below are **diverse examples** — from contractions to Unicode scripts, punctuation to code.

```python
test_cases = [
    ("I can't believe it's working!", "Basic contractions"),
    ("You're right, they'll see we've done it.", "Multiple contractions"),
    ("Hello 世界! Café français 🌍", "Unicode letters and emoji"),
    ("I have 5 cats, ٧ dogs, and Ⅲ birds.", "Various number systems"),
    ("Wait... What?!? $100.50 (seriously)!!!", "Complex punctuation"),
    ("  Multiple   spaces   everywhere  ", "Multiple spaces"),
    ("She's got $1,000.50 in café № ٧... Amazing!!! 🚀", "Complex mixed text"),
    ("'s'd'm't'll've're", "Contraction edge cases"),
    ("!@#$%^&*()_+-=[]{}|;:",./<>?", "Pure punctuation"),
    ("   \t\n  ", "Pure whitespace"),
    ("", "Empty string"),
    ("a 1 ! '", "Single characters"),
    ("我有3只猫，很可爱！", "Chinese with numbers"),
    ("مرحبا بالعالم ١٢٣", "Arabic text with numbers"),
    ("def hello_world(): return 'Hello, World!'", "Code-like text"),
    ("Visit https://example.com or email test@domain.co.uk", "URLs and emails"),
]

for text, description in test_cases:
    test_regex(text, description)
```

Running these cases produces token lists that **perfectly reconstruct the original text**, see the test results below:
```
    ============================================================
    TEST: Basic contractions
    INPUT: 'I can't believe it's working!'
    ============================================================
    TOKENS (8):
       1: 'I'
       2: ' can'
       3: "'t"
       4: ' believe'
       5: ' it'
       6: "'s"
       7: ' working'
       8: '!'
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Multiple contractions
    INPUT: 'You're right, they'll see we've done it.'
    ============================================================
    TOKENS (12):
       1: 'You'
       2: "'re"
       3: ' right'
       4: ','
       5: ' they'
       6: "'ll"
       7: ' see'
       8: ' we'
       9: "'ve"
      10: ' done'
      11: ' it'
      12: '.'
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Unicode letters and emoji
    INPUT: 'Hello 世界! Café français 🌍'
    ============================================================
    TOKENS (6):
       1: 'Hello'
       2: ' 世界'
       3: '!'
       4: ' Café'
       5: ' français'
       6: ' 🌍'
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Various number systems
    INPUT: 'I have 5 cats, ٧ dogs, and Ⅲ birds.'
    ============================================================
    TOKENS (12):
       1: 'I'
       2: ' have'
       3: ' 5'
       4: ' cats'
       5: ','
       6: ' ٧'
       7: ' dogs'
       8: ','
       9: ' and'
      10: ' Ⅲ'
      11: ' birds'
      12: '.'
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Complex punctuation
    INPUT: 'Wait... What?!? $100.50 (seriously)!!!'
    ============================================================
    TOKENS (11):
       1: 'Wait'
       2: '...'
       3: ' What'
       4: '?!?'
       5: ' $'
       6: '100'
       7: '.'
       8: '50'
       9: ' ('
      10: 'seriously'
      11: ')!!!'
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Multiple spaces
    INPUT: '  Multiple   spaces   everywhere  '
    ============================================================
    TOKENS (7):
       1: ' '
       2: ' Multiple'
       3: '  '
       4: ' spaces'
       5: '  '
       6: ' everywhere'
       7: '  '
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Complex mixed text
    INPUT: 'She's got $1,000.50 in café № ٧... Amazing!!! 🚀'
    ============================================================
    TOKENS (17):
       1: 'She'
       2: "'s"
       3: ' got'
       4: ' $'
       5: '1'
       6: ','
       7: '000'
       8: '.'
       9: '50'
      10: ' in'
      11: ' café'
      12: ' №'
      13: ' ٧'
      14: '...'
      15: ' Amazing'
      16: '!!!'
      17: ' 🚀'
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Contraction edge cases
    INPUT: ''s'd'm't'll've're'
    ============================================================
    TOKENS (7):
       1: "'s"
       2: "'d"
       3: "'m"
       4: "'t"
       5: "'ll"
       6: "'ve"
       7: "'re"
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Pure punctuation
    INPUT: '!@#$%^&*()_+-=[]{}|;:",./<>?'
    ============================================================
    TOKENS (1):
       1: '!@#$%^&*()_+-=[]{}|;:",./<>?'
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Pure whitespace
    INPUT: '   	
      '
    ============================================================
    TOKENS (1):
       1: '   \t\n  '
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Empty string
    INPUT: ''
    ============================================================
    TOKENS (0):
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Single characters
    INPUT: 'a 1 ! ''
    ============================================================
    TOKENS (4):
       1: 'a'
       2: ' 1'
       3: ' !'
       4: " '"
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Chinese with numbers
    INPUT: '我有3只猫，很可爱！'
    ============================================================
    TOKENS (6):
       1: '我有'
       2: '3'
       3: '只猫'
       4: '，'
       5: '很可爱'
       6: '！'
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Arabic text with numbers
    INPUT: 'مرحبا بالعالم ١٢٣'
    ============================================================
    TOKENS (3):
       1: 'مرحبا'
       2: ' بالعالم'
       3: ' ١٢٣'
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: Code-like text
    INPUT: 'def hello_world(): return 'Hello, World!''
    ============================================================
    TOKENS (11):
       1: 'def'
       2: ' hello'
       3: '_'
       4: 'world'
       5: '():'
       6: ' return'
       7: " '"
       8: 'Hello'
       9: ','
      10: ' World'
      11: "!'"
    
    RECONSTRUCTION CHECK: ✓ PASS
    
    ============================================================
    TEST: URLs and emails
    INPUT: 'Visit https://example.com or email test@domain.co.uk'
    ============================================================
    TOKENS (15):
       1: 'Visit'
       2: ' https'
       3: '://'
       4: 'example'
       5: '.'
       6: 'com'
       7: ' or'
       8: ' email'
       9: ' test'
      10: '@'
      11: 'domain'
      12: '.'
      13: 'co'
      14: '.'
      15: 'uk'
    
    RECONSTRUCTION CHECK: ✓ PASS
```

---

### 💡 Why This Matters

BPE pre-tokenization **sets the stage** for the tokenizer’s merge rules to apply. A well-designed split pattern ensures:

- **Language independence** — Works with English, Arabic, Chinese, emoji, etc.
- **Symbol awareness** — Keeps punctuation and symbols intact.
- **Whitespace fidelity** — Preserves exact spacing for reversible tokenization.
- **Downstream accuracy** — Reduces surprises during model training or inference.

---

### 🚀 Takeaways

- This regex is a **core building block** of GPT-2’s tokenization process.
- It’s **language-agnostic**, **Unicode-friendly**, and **precise** in splitting.
- Understanding it helps when **building custom tokenizers** or adapting GPT-2 BPE to new domains.

If you’re working with **LLMs, tokenization, or multilingual NLP**, knowing the details behind this pattern will help you **debug**, **customize**, and **optimize** your preprocessing pipeline.