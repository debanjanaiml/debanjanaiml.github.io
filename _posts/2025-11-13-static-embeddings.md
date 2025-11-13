---
title: "🧱 Static Embeddings: The Foundations of Text Vectorization"
date: 2025-11-13
categories:
  - posts
tags:
  - nlp
  - embeddings
toc: true
toc_label: "Table of Contents"
toc_icon: "bookmark"
excerpt: "`Embeddings` are the foundations of advanced NLP, where texts are converted into numeric vectors, and various ML techniques are applied on them. Static embedding techniques serves as the foundation for understanding how words in a sentence can be used to derive semantic understanding"
header:
  teaser: "/assets/images/portfolio/finetuning-bert.png"
---
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/19m8uP3Z5TMCtx4gQSzrqD7bOhMg6u5h8)

# A - Introduction

## Why Do Computers Need Numbers to Understand Words?

**The Core Problem: Language is Qualitative**

Imagine you are building a system to automatically categorize emails as "Spam" or "Not Spam," or perhaps determining the sentiment of a product review ("positive" or "negative"). Computers, at their core, are excellent at processing numbers, math, and logic, but they don't natively understand words, grammar, or meaning.

We cannot feed the raw string "This product is amazing!" directly into a mathematical model. We must first convert the word (or the entire sentence) into a numerical representation — a vector. This process is called **Text Vectorization** or **Embedding**.

## Static vs. Contextual Embeddings

The techniques explored in this article are classified as Static Embeddings:

1. **Static (Context-Independent)**: Every unique word in the entire dictionary (vocabulary) gets one fixed, unchanging vector. The word "bank" has the same vector whether it appears in the sentence "I went to the bank to deposit money" or "I sat on the river bank."

2. **Contextual (Modern NLP)**: Newer models (like BERT) generate a vector for a word that changes based on the surrounding sentence.

Static embeddings are the foundation. They are simpler, faster, and still incredibly useful for many applications like document classification, spam filtering, and clustering.


# Count-Based Methods (The Bag-of-Words Family)

The simplest way to turn words into numbers is by counting them. This approach is called the **Bag-of-Words (BoW)** model.

The name "Bag-of-Words" is very descriptive: we treat a document like a simple bag of marbles. The color of the marbles represents the unique words, and we only care about how many of each color are in the bag. We completely ignore the order of the words (which is why it's context-independent).

## 1. Count Vectorizer

The Count Vectorizer is the direct implementation of the Bag-of-Words model using Raw Frequency.

* It creates a dictionary (vocabulary) of every unique word across all documents. For any given document, it counts how many times each word from the dictionary appears.
* *It tells you *which words are present* and *how frequently they occur*. Documents with similar counts of the same words are considered similar.
* This method suffers from the issue that very common words like "the," "a," and "is" (called *stop words*) often have the highest counts, but contribute little to the document's actual meaning.

### Code

We will use a small list of sentences (our 'corpus') to see exactly how the counts are generated and what the resulting matrix looks like. Later we shall generalize this to larger text based datasets

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# --- 1.1. Prepare the Data (Our Corpus) ---
# A small collection of documents (sentences)
corpus = [
    'The quick brown fox jumps over the lazy dog.',
    'A lazy fox is not a quick fox.',
    'The dog and the fox are running quickly.',
    'The cat sat on the mat.'
]

print("--- Original Corpus ---")
for i, doc in enumerate(corpus):
    print(f"Document {i+1}: {doc}")
print("-" * 30)
```

```
--- Original Corpus ---
Document 1: The quick brown fox jumps over the lazy dog.
Document 2: A lazy fox is not a quick fox.
Document 3: The dog and the fox are running quickly.
Document 4: The cat sat on the mat.
------------------------------
```

```python
# --- 1.2. Initialize and Fit the Vectorizer ---
# We use CountVectorizer to build the vocabulary and the counting logic
vectorizer = CountVectorizer()

# fit_transform performs two steps:
# 1. 'fit': scans the entire corpus to build the vocabulary (assigning an index to each unique word).
# 2. 'transform': converts each document into a vector of word counts.
count_matrix = vectorizer.fit_transform(corpus)

# --- 1.3. Inspect the Results ---

# 3a. The Vocabulary Mapping (The Dictionary)
print("--- 3a. Vocabulary (Word to Index Mapping) ---")
# The vocabulary_ attribute shows which index (column) corresponds to which word.
# Note that all words are converted to lowercase.
vocabulary = vectorizer.vocabulary_
print(vocabulary)
print(f"Total unique words (dimensions): {len(vocabulary)}")
print("-" * 30)


# 1.3b. The Document-Term Matrix (The Counts)
print("--- 3b. Document-Term Matrix (Raw Counts) ---")
# Convert the sparse matrix output to a dense array for easier viewing
count_array = count_matrix.toarray()

# Create a DataFrame for a clean, visual representation
feature_names = vectorizer.get_feature_names_out()
df_counts = pd.DataFrame(data=count_array, columns=feature_names, index=[f"Doc {i+1}" for i in range(len(corpus))])

# Display the resulting matrix
print(df_counts)
print("-" * 30)
```

```
--- 3b. Document-Term Matrix (Raw Counts) ---
       and  are  brown  cat  dog  fox  is  jumps  lazy  mat  not  on  over  \
Doc 1    0    0      1    0    1    1   0      1     1    0    0   0     1   
Doc 2    0    0      0    0    0    2   1      0     1    0    1   0     0   
Doc 3    1    1      0    0    1    1   0      0     0    0    0   0     0   
Doc 4    0    0      0    1    0    0   0      0     0    1    0   1     0   

       quick  quickly  running  sat  the  
Doc 1      1        0        0    0    2  
Doc 2      1        0        0    0    0  
Doc 3      0        1        1    0    2  
Doc 4      0        0        0    1    2  
------------------------------
```

```
--- 1.4. Interpretation ---

Look at 'Doc 1': 'The quick brown fox jumps over the lazy dog.'
Word 'the' has a count of 2.
Word 'fox' has a count of 1.
Word 'dog' has a count of 1.
This is reflected in the first row of the DataFrame.

Look at 'Doc 2': 'A lazy fox is not a quick fox.'
Word 'fox' has a count of 2.
Word 'lazy' has a count of 1.
The resulting vector for Doc 2 is simply a histogram of word occurrences, ignoring the original word order.
```

In this final section, we shall try to create a class SimpleCountVectorizer which mimicks the CountVectorizer class from sklearn to really understand what is happening under the hood.

```python
import re
import numpy as np
import pandas as pd

# --- 1.5a. Define the SimpleCountVectorizer Class ---
class SimpleCountVectorizer:
    """
    A simplified implementation of the Count Vectorizer (Bag-of-Words).
    It takes a corpus, builds a vocabulary, and converts documents into 
    vectors of raw word counts.
    """
    def __init__(self):
        # Stores the mapping from word to its unique integer index (column number)
        self.vocabulary_ = {}
        # Stores the list of feature names (words) in order of their index
        self.feature_names_out = []

    def _tokenize(self, text):
        """
        Simple text cleaning and tokenization. 
        1. Convert to lowercase.
        2. Remove punctuation (keeping only letters and spaces).
        3. Split by space.
        """
        # Convert to lowercase
        text = text.lower()
        # Remove all non-alphanumeric characters except spaces
        text = re.sub(r'[^a-z\s]', '', text)
        # Split into tokens (words)
        tokens = text.split()
        return tokens

    def fit(self, corpus):
        """
        Learns the unique vocabulary from the entire corpus.
        """
        unique_words = set()
        for doc in corpus:
            tokens = self._tokenize(doc)
            # Add all unique tokens from the document to our master set
            unique_words.update(tokens)

        # Create the word-to-index mapping (the vocabulary)
        # We sort the words alphabetically to ensure a consistent, ordered output
        sorted_words = sorted(list(unique_words))
        
        for index, word in enumerate(sorted_words):
            self.vocabulary_[word] = index
        
        # Store the feature names for easy inspection
        self.feature_names_out = sorted_words
        
        return self

    def transform(self, corpus):
        """
        Transforms the documents into the Bag-of-Words count matrix 
        using the learned vocabulary.
        """
        # Determine the size of the vocabulary (number of columns)
        vocab_size = len(self.vocabulary_)
        # Initialize an empty matrix (rows = documents, columns = words)
        # using numpy for efficient numerical array handling
        count_matrix = np.zeros((len(corpus), vocab_size), dtype=int)

        for doc_index, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            # Count the occurrences of each word in the current document
            word_counts = {}
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
            
            # Populate the row in the count matrix
            for word, count in word_counts.items():
                # Only use words that are actually in our learned vocabulary
                if word in self.vocabulary_:
                    col_index = self.vocabulary_[word]
                    count_matrix[doc_index, col_index] = count
        
        return count_matrix

    def fit_transform(self, corpus):
        """
        A convenience method that calls fit() and then transform().
        """
        self.fit(corpus)
        return self.transform(corpus)

```

I strongly recommend you to go through the notebook to see the actual implementation and it's application on a common text dataset called 20 newsgroups dataset available as part of sklearn datasets, which consists of approximately 20,000 newsgroup documents evenly distributed across 20 different topics. It is then fed into a  Multinomial Naive Bayes algorithm to determine the accuracy of the classification using Count Vectorizer. While CountVectorizer is the most simple and naive technique using raw frequency counts, it is still able to achieve 91% accuracy in classifying news documents.


## 2. TF-IDF (Term Frequency-Inverse Document Frequency)

While raw counts (**Bag-of-Words**) are simple, they suffer from a major flaw: 
common words like 'the', 'a', and 'is' dominate the vectors, but they carry 
almost no unique information about the document's content.

**TF-IDF (Term Frequency-Inverse Document Frequency)** is designed to solve this. 
It’s not just about how often a word appears in a document (*Term Frequency*), 
but how rare that word is across the entire collection of documents (*Inverse 
Document Frequency*).

The final TF-IDF score for a word in a document is calculated as:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

where:
* $t$ is the term (word).
* $d$ is the document.
* $D$ is the corpus (collection of all documents).

### 2.A. Term Frequency (TF)
The TF part is what we've already done with the Count Vectorizer! It measures 
how often a word appears in a document. A common formula is:
$$\text{TF}(t, d) = \text{Raw Count} \text{ of } t \text{ in } d$$

### 2.B. Inverse Document Frequency (IDF)
The IDF part is the '*magic*' that penalizes common words and rewards rare ones. 
It measures how much information a word provides. If a word appears in *every* document, its IDF score will be close to zero, effectively silencing it.

The standard formula, including Laplace smoothing (adding 1 to avoid division by zero):
$$\text{IDF}(t, D) = \log \left(\frac{1 + N}{1 + \text{DF}(t)} \right) + 1$$
where:
* $N$ is the total number of documents in the corpus $D$.
* $\text{DF}(t)$ (Document Frequency) is the number of documents in $D$ that contain the term $t$.
* The final $+1$ ensures that words that appear in all documents still have a positive weight.

### 2.C. The Final Weighting
When TF is multiplied by IDF, the result is:
* **High TF-IDF:** Given to a word that appears frequently in a **specific document** (high TF) but rarely in the **entire corpus** (high IDF). These are the words that define the document's topic.
* **Low TF-IDF:** Given to words like 'the', 'a', or 'is' that are common everywhere.

#### The Result: A Semantic Vector
The TF-IDF vector is no longer a simple count histogram; it's a **semantic vector** where the magnitude of each dimension (word) reflects its importance to that specific document's meaning within the context of the entire corpus. This often leads to superior performance in classification and clustering tasks compared to raw counts.
"""
