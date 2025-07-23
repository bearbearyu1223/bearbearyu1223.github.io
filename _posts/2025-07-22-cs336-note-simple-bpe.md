---
layout: post
title: "Study Notes: Stanford CS336 Language Modeling from Scratch [2]"
categories: cs336
author: 
- Han Yu
---
# Byte Pair Encoding (BPE) Tokenizer in a Nutshell
## Key Terms

| Concept | Description |
|:--------|:------------|
| Unicode | System that assigns every character a unique codepoint (e.g., 'A' → 65) |
| UTF-8 | A way to encode those codepoints into 1-4 bytes |
| Byte | 8 bits; one byte can hold values from 0 to 255 |
| Tokenization | Breaking text corpus input into manageable pieces (tokens) for a model |

Let us take the following string as a simple example to illustrate the concept.

## Example: Encoding 'A😊'

### Step 1: Get the Unicode Codepoints

```python
text = 'A😊'
codepoints = [ord('A'), ord('😊')]
print(codepoints)  # [65, 128522]
```

Output:
```
[65, 128522]
```

### Step 2: UTF-8 Encoding (Turn Codepoints into Bytes)

```python
utf8_bytes = text.encode("utf-8")
print(tuple(utf8_bytes))  # (65, 240, 159, 152, 138)
```

Output:
```
(65, 240, 159, 152, 138)
```

Here's what happened:
- 'A' is encoded using one byte: 65
- '😊' is encoded using four bytes: [240, 159, 152, 138]
- 'A😊' is encoded as the sequence [65, 240, 159, 152, 138]

## Why Using UTF-8 for Encoding is Helpful

Instead of dealing with hundreds of thousands of possible codepoints (Unicode has more than 150,000 codepoints) or millions of words/subwords in vocabulary, we can model text using sequences of bytes. Each byte can be represented by an integer from 0 to 255, so we only need a vocabulary of size 256 to model input text. This approach is simple and complete—any character in any language can be represented as bytes, eliminating out-of-vocabulary token concerns.

## Tokenization Spectrum

| Tokenization Level | Example Tokens | Pros | Cons |
|:-------------------|:---------------|:-----|:-----|
| **Word-level** | `["unbelievable"]` | Human-readable, efficient | OOV (out-of-vocabulary) issues |
| **Subword-level** (BPE) | `["un", "believ", "able"]` | Handles rare words, compact | Requires training |
| **Byte-level** | `[117, 110, 98, 101, ...]` (bytes) | No OOV, simple | Longer sequences, less semantic meaning |

## Why Subword Tokenization is the Middle Ground

**Subword tokenization** with **Byte Pair Encoding (BPE)** provides a balance between the other approaches:

- **Word-level tokenization** struggles with rare or unseen words (e.g., "unbelievable" might be unknown even if "believe" is known)
- **Byte-level tokenization** avoids unknown token issues but creates long, inefficient sequences
- **Subword tokenization** (BPE):
  1. Breaks rare words into familiar pieces (subwords)
  2. Retains compactness for common words
  3. Is learnable from corpus statistics

## Byte Pair Encoding (BPE) Algorithm Overview

BPE starts from characters and iteratively **merges the most frequent adjacent pairs** into longer tokens. 

### Example Training Corpus

Consider this toy training corpus:

```
"low"     (5 times)  
"lowest"  (2 times)  
"newest"  (6 times)  
"wider"   (3 times)
```

We want to learn a compact subword vocabulary that reuses frequent patterns like "low" and "est".

### Step-by-Step BPE Process

#### Step 0: Preprocess as Characters
Each word is broken into characters with an end-of-word marker `</w>`:

```
"l o w </w>"        (5)
"l o w e s t </w>"  (2)
"n e w e s t </w>"  (6)
"w i d e r </w>"    (3)
```

#### Step 1: Count Adjacent Pairs
Compute most frequent adjacent pairs across all words:

```
('e', 's') appears 8 times
('s', 't') appears 8 times
('l', 'o') appears 7 times
('o', 'w') appears 7 times
```

#### Step 2: Merge 'e' + 's' → 'es'
Update vocabulary:

```
"l o w </w>"          (5)
"l o w es t </w>"     (2)
"n e w es t </w>"     (6)
"w i d e r </w>"      (3)
```

#### Step 3: Merge 'es' + 't' → 'est'

```
"l o w </w>"         (5)
"l o w est </w>"     (2)
"n e w est </w>"     (6)
"w i d e r </w>"     (3)
```

#### Step 4: Merge 'l' + 'o' → 'lo', then 'lo' + 'w' → 'low'

```
"low </w>"          (5)
"low est </w>"      (2)
"n e w est </w>"    (6)
"w i d e r </w>"    (3)
```

#### Continue merging...
Eventually we learn useful building blocks like 'low', 'est', and 'new'. After training, "newest" would tokenize to `['new', 'est', '</w>']`.

## BPE Implementation

Below is a complete implementation demonstrating the BPE algorithm on the corpus:

```
"low low low low low lower lower widest widest widest newest newest newest newest newest newest"
```

### Key Components

1. **Initialization**: Creates vocabulary with `<|endoftext|>` special token and all 256 byte values
2. **Pre-tokenization**: Splits text on whitespace and converts words to byte tuples
3. **Pair Frequency Counting**: Counts all adjacent byte pairs across the corpus
4. **Merging**: Merges the most frequent pair (lexicographically largest in case of ties)
5. **Tokenization**: Uses learned merges to tokenize new words

### How It Works

1. **Pre-tokenization**: Converts `"low low low..."` into `{(l,o,w): 5, (l,o,w,e,r): 2, ...}`
2. **Merge Selection**: Finds most frequent pairs like `('s','t')` and `('e','s')`, chooses lexicographically larger `('s','t')`
3. **Iterative Merging**: Continues merging until desired number of merges is reached
4. **Tokenization**: Applies learned merges in order to tokenize new words

### Expected Output

With 6 merges, the algorithm produces:
- **Merges**: `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`
- **Final vocabulary**: `<|endoftext|>`, 256 byte chars, `st`, `est`, `ow`, `low`, `west`, `ne`
- **"newest" tokenizes as**: `['ne', 'west']`

Below is one implementation for Algorithm 1 of [Sennrich et al. [2016]](https://arxiv.org/abs/1508.07909). 
```python
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Union

class BPEEncoder:
    def __init__(self):
        # Initialize vocabulary with special token and 256 byte values
        self.vocab = {"<|endoftext|>": 0}
        # Add all possible byte values (0-255) to vocabulary
        for i in range(256):
            self.vocab[i] = i + 1

        self.merges = []  # List of (token1, token2) pairs that were merged
        self.merge_tokens = {}  # Maps (token1, token2) -> new_token_id
        self.token_names = {}  # Maps token_id -> readable name
        self.next_token_id = 257

    def pre_tokenize(self, text: str) -> Dict[Tuple[int, ...], int]:
        """
        Pre-tokenize text by splitting on whitespace and convert to byte tuples.
        Returns frequency count of each word as tuple of byte integers.
        For example, converts "low low low..." into {(l,o,w): 5, (l,o,w,e,r): 2, ...}
        """
        words = text.split()
        word_freq = Counter(words)

        # Convert to tuple of byte integers
        byte_word_freq = {}
        for word, freq in word_freq.items():
            byte_tuple = tuple(word.encode('utf-8'))
            byte_word_freq[byte_tuple] = freq

        return byte_word_freq

    def get_pair_frequencies(self, word_freq: Dict[Tuple[Union[int, str], ...], int]) -> Dict[Tuple[Union[int, str], Union[int, str]], int]:
        """
        Count frequency of all adjacent token pairs across all words.
        """
        pair_freq = defaultdict(int)

        for word, freq in word_freq.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] += freq

        return dict(pair_freq)

    def merge_pair(self, word_freq: Dict[Tuple[Union[int, str], ...], int],
                   pair_to_merge: Tuple[Union[int, str], Union[int, str]],
                   new_token: str) -> Dict[Tuple[Union[int, str], ...], int]:
        """
        Merge the specified pair in all words where it appears.
        """
        new_word_freq = {}

        for word, freq in word_freq.items():
            new_word = []
            i = 0
            while i < len(word):
                # Check if current position matches the pair to merge
                if (i < len(word) - 1 and
                    word[i] == pair_to_merge[0] and
                    word[i + 1] == pair_to_merge[1]):
                    new_word.append(new_token)
                    i += 2  # Skip both tokens of the pair
                else:
                    new_word.append(word[i])
                    i += 1

            new_word_freq[tuple(new_word)] = freq

        return new_word_freq

    def train(self, text: str, num_merges: int) -> List[str]:
        """
        Train BPE on the given text for specified number of merges.
        Returns list of merge operations performed.
        """
        # Pre-tokenize text
        word_freq = self.pre_tokenize(text)
        print(f"Initial word frequencies: {self._format_word_freq(word_freq)}")

        merges_performed = []

        for merge_step in range(num_merges):
            # Get pair frequencies
            pair_freq = self.get_pair_frequencies(word_freq)

            if not pair_freq:
                break

            # Find most frequent pair (lexicographically largest in case of tie)
            max_freq = max(pair_freq.values())
            most_frequent_pairs = [pair for pair, freq in pair_freq.items() if freq == max_freq]

            # Sort pairs lexicographically - convert to comparable format
            def pair_sort_key(pair):
                def token_to_str(token):
                    if isinstance(token, int):
                        return chr(token)
                    return str(token)
                return (token_to_str(pair[0]), token_to_str(pair[1]))

            # Take lexicographically largest (max)
            best_pair = max(most_frequent_pairs, key=pair_sort_key)

            print(f"\nMerge {merge_step + 1}:")
            print(f"Pair frequencies: {self._format_pair_freq(pair_freq)}")
            print(f"Most frequent pair: {self._format_pair(best_pair)} (freq: {max_freq})")

            # Create new token name
            new_token = f"merge_{self.next_token_id}"

            # Perform the merge
            word_freq = self.merge_pair(word_freq, best_pair, new_token)

            # Record the merge
            token1_name = self._token_to_str(best_pair[0])
            token2_name = self._token_to_str(best_pair[1])
            merge_str = f"{token1_name} {token2_name}"
            merges_performed.append(merge_str)

            # Store merge information
            self.merges.append(best_pair)
            self.merge_tokens[best_pair] = new_token
            self.vocab[new_token] = self.next_token_id
            self.token_names[new_token] = merge_str
            self.next_token_id += 1

            print(f"After merge: {self._format_word_freq(word_freq)}")

        return merges_performed

    def tokenize(self, word: str) -> List[str]:
        """
        Tokenize a word using the learned BPE merges.
        """
        # Start with individual bytes as integers
        tokens = list(word.encode('utf-8'))

        # Apply merges in order
        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == merge_pair[0] and
                    tokens[i + 1] == merge_pair[1]):
                    # Replace with the merged token name
                    merged_token = self.merge_tokens[merge_pair]
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Convert to readable format
        result = []
        for token in tokens:
            if isinstance(token, int):
                result.append(chr(token))
            elif isinstance(token, str) and token.startswith('merge_'):
                # Convert back to the original characters this merge represents
                result.append(self.token_names[token])
            else:
                result.append(str(token))

        return result

    def _format_word_freq(self, word_freq: Dict[Tuple[Union[int, str], ...], int]) -> str:
        """Format word frequency dictionary for readable output."""
        formatted = {}
        for word_tuple, freq in word_freq.items():
            tokens = [self._token_to_str(token) for token in word_tuple]
            word_str = '(' + ','.join(tokens) + ')'
            formatted[word_str] = freq
        return str(formatted)

    def _format_pair_freq(self, pair_freq: Dict[Tuple[Union[int, str], Union[int, str]], int]) -> str:
        """Format pair frequency dictionary for readable output."""
        formatted = {}
        for pair, freq in pair_freq.items():
            first = self._token_to_str(pair[0])
            second = self._token_to_str(pair[1])
            pair_str = first + second
            formatted[pair_str] = freq
        return str(formatted)

    def _format_pair(self, pair: Tuple[Union[int, str], Union[int, str]]) -> str:
        """Format a pair for readable output."""
        first = self._token_to_str(pair[0])
        second = self._token_to_str(pair[1])
        return f"({first}, {second})"

    def _token_to_str(self, token: Union[int, str]) -> str:
        """Convert a token to readable string."""
        if isinstance(token, int):
            return chr(token)
        elif isinstance(token, str) and token.startswith('merge_'):
            return self.token_names.get(token, token)
        else:
            return str(token)

# Example usage
if __name__ == "__main__":
    # Initialize BPE encoder
    bpe = BPEEncoder()

    # Example corpus
    corpus = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

    print("BPE Training on Corpus:")
    print(f"Corpus: {corpus}")
    print("=" * 50)

    # Train with 6 merges
    merges = bpe.train(corpus, num_merges=6)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Merges performed: {merges}")

    # Test tokenization
    print("\n" + "=" * 50)
    print("Tokenization Examples:")
    test_words = ["newest", "lower", "widest", "low"]
    for word in test_words:
        tokens = bpe.tokenize(word)
        print(f"'{word}' -> {tokens}")

    # Show final vocabulary (subset)
    print("\n" + "=" * 50)
    print("New Vocabulary (merged tokens only):")
    for token, token_id in bpe.vocab.items():
        if isinstance(token, str) and token.startswith('merge_'):
            description = bpe.token_names[token]
            print(f"Token ID {token_id}: '{description}'")
```

## Sample Output

When you run this code, you'll see output like:

```
    BPE Training on Corpus:
    Corpus: low low low low low lower lower widest widest widest newest newest newest newest newest newest
    ==================================================
    Initial word frequencies: {'(l,o,w)': 5, '(l,o,w,e,r)': 2, '(w,i,d,e,s,t)': 3, '(n,e,w,e,s,t)': 6}
    
    Merge 1:
    Pair frequencies: {'lo': 7, 'ow': 7, 'we': 8, 'er': 2, 'wi': 3, 'id': 3, 'de': 3, 'es': 9, 'st': 9, 'ne': 6, 'ew': 6}
    Most frequent pair: (s, t) (freq: 9)
    After merge: {'(l,o,w)': 5, '(l,o,w,e,r)': 2, '(w,i,d,e,s t)': 3, '(n,e,w,e,s t)': 6}
    
    Merge 2:
    Pair frequencies: {'lo': 7, 'ow': 7, 'we': 8, 'er': 2, 'wi': 3, 'id': 3, 'de': 3, 'es t': 9, 'ne': 6, 'ew': 6}
    Most frequent pair: (e, s t) (freq: 9)
    After merge: {'(l,o,w)': 5, '(l,o,w,e,r)': 2, '(w,i,d,e s t)': 3, '(n,e,w,e s t)': 6}
    
    Merge 3:
    Pair frequencies: {'lo': 7, 'ow': 7, 'we': 2, 'er': 2, 'wi': 3, 'id': 3, 'de s t': 3, 'ne': 6, 'ew': 6, 'we s t': 6}
    Most frequent pair: (o, w) (freq: 7)
    After merge: {'(l,o w)': 5, '(l,o w,e,r)': 2, '(w,i,d,e s t)': 3, '(n,e,w,e s t)': 6}
    
    Merge 4:
    Pair frequencies: {'lo w': 7, 'o we': 2, 'er': 2, 'wi': 3, 'id': 3, 'de s t': 3, 'ne': 6, 'ew': 6, 'we s t': 6}
    Most frequent pair: (l, o w) (freq: 7)
    After merge: {'(l o w)': 5, '(l o w,e,r)': 2, '(w,i,d,e s t)': 3, '(n,e,w,e s t)': 6}
    
    Merge 5:
    Pair frequencies: {'l o we': 2, 'er': 2, 'wi': 3, 'id': 3, 'de s t': 3, 'ne': 6, 'ew': 6, 'we s t': 6}
    Most frequent pair: (w, e s t) (freq: 6)
    After merge: {'(l o w)': 5, '(l o w,e,r)': 2, '(w,i,d,e s t)': 3, '(n,e,w e s t)': 6}
    
    Merge 6:
    Pair frequencies: {'l o we': 2, 'er': 2, 'wi': 3, 'id': 3, 'de s t': 3, 'ne': 6, 'ew e s t': 6}
    Most frequent pair: (n, e) (freq: 6)
    After merge: {'(l o w)': 5, '(l o w,e,r)': 2, '(w,i,d,e s t)': 3, '(n e,w e s t)': 6}
    
    ==================================================
    Training Complete!
    Merges performed: ['s t', 'e s t', 'o w', 'l o w', 'w e s t', 'n e']
    
    ==================================================
    Tokenization Examples:
    'newest' -> ['n e', 'w e s t']
    'lower' -> ['l o w', 'e', 'r']
    'widest' -> ['w', 'i', 'd', 'e s t']
    'low' -> ['l o w']
    
    ==================================================
    New Vocabulary (merged tokens only):
    Token ID 257: 's t'
    Token ID 258: 'e s t'
    Token ID 259: 'o w'
    Token ID 260: 'l o w'
    Token ID 261: 'w e s t'
    Token ID 262: 'n e'
```

This implementation demonstrates how BPE learns to represent text efficiently by identifying and merging frequently occurring character patterns, creating a vocabulary that balances between the simplicity of byte-level tokenization and the efficiency of word-level tokenization.