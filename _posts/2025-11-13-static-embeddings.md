---
title: "🧱 Static Embeddings: The Foundations of Text Vectorization"
date: 2025-11-13
categories:
  - posts
tags:
  - nlp
  - embeddings
use_math: true
toc: true
toc_label: "Table of Contents"
toc_icon: "bookmark"
excerpt: "`Embeddings` are the foundations of advanced NLP, where texts are converted into numeric vectors, and various ML techniques are applied on them. Static embedding techniques serves as the foundation for understanding how words in a sentence can be used to derive semantic understanding"
header:
  teaser: "/assets/images/blogs/static_embeddings.png"
---
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/19m8uP3Z5TMCtx4gQSzrqD7bOhMg6u5h8)

# Introduction

**Why Do Computers Need Numbers to Understand Words?**

**The Core Problem: Language is Qualitative**

Imagine you are building a system to automatically categorize emails as "Spam" or "Not Spam," or perhaps determining the sentiment of a product review ("positive" or "negative"). Computers, at their core, are excellent at processing numbers, math, and logic, but they don't natively understand words, grammar, or meaning.

We cannot feed the raw string "This product is amazing!" directly into a mathematical model. We must first convert the word (or the entire sentence) into a numerical representation — a vector. This process is called **Text Vectorization** or **Embedding**.

**Static vs. Contextual Embeddings**

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

# --- 1.1. Prepare the Toy Data (Our Corpus) ---
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

# 1.3a. The Vocabulary Mapping (The Dictionary)
print("--- 1.3a. Vocabulary (Word to Index Mapping) ---")
# The vocabulary_ attribute shows which index (column) corresponds to which word.
# Note that all words are converted to lowercase.
vocabulary = vectorizer.vocabulary_
print(vocabulary)
print(f"Total unique words (dimensions): {len(vocabulary)}")
print("-" * 30)


# 1.3b. The Document-Term Matrix (The Counts)
print("--- 1.3b. Document-Term Matrix (Raw Counts) ---")
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
--- 1.3b. Document-Term Matrix (Raw Counts) ---
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

**2.A. Term Frequency (TF)**
The TF part is what we've already done with the Count Vectorizer! It measures 
how often a word appears in a document. A common formula is:
$$\text{TF}(t, d) = \text{Raw Count} \text{ of } t \text{ in } d$$

**2.B. Inverse Document Frequency (IDF)**
The IDF part is the '*magic*' that penalizes common words and rewards rare ones. 
It measures how much information a word provides. If a word appears in *every* document, its IDF score will be close to zero, effectively silencing it.

The standard formula, including Laplace smoothing (adding 1 to avoid division by zero):
$$\text{IDF}(t, D) = \log \left(\frac{1 + N}{1 + \text{DF}(t)} \right) + 1$$
where:
* $N$ is the total number of documents in the corpus $D$.
* $\text{DF}(t)$ (Document Frequency) is the number of documents in $D$ that contain the term $t$.
* The final $+1$ ensures that words that appear in all documents still have a positive weight.

**2.C. The Final Weighting**
When TF is multiplied by IDF, the result is:
* **High TF-IDF:** Given to a word that appears frequently in a **specific document** (high TF) but rarely in the **entire corpus** (high IDF). These are the words that define the document's topic.
* **Low TF-IDF:** Given to words like 'the', 'a', or 'is' that are common everywhere.

**The Result: A Semantic Vector**
The TF-IDF vector is no longer a simple count histogram; it's a **semantic vector** where the magnitude of each dimension (word) reflects its importance to that specific document's meaning within the context of the entire corpus. This often leads to superior performance in classification and clustering tasks compared to raw counts.

### Code

Here we start again with our toy dataset (our corpus) try to understand how TF-IDF works on it. Then we shall try to create TF-IDF from scratch and completely understand what is happening under the hood. Finally we shall apply TF-IDF to the same newsgroup datasets.

```python
# --- 2.1. Define the Toy Corpus ---
corpus = [
    'The quick brown fox jumps over the lazy dog.',
    'A lazy fox is not a quick fox.',
    'The dog and the fox are running quickly.',
    'The cat sat on the mat.'
]

print("--- 2.1. Corpus Data ---")
for i, doc in enumerate(corpus):
    print(f"Doc {i+1}: {doc}")
print("\n" + "="*40 + "\n")
```

```
--- 2.1. Corpus Data ---
Doc 1: The quick brown fox jumps over the lazy dog.
Doc 2: A lazy fox is not a quick fox.
Doc 3: The dog and the fox are running quickly.
Doc 4: The cat sat on the mat.

========================================
```

```python
# --- 2.2. Initialize and Apply TfidfVectorizer ---
# The TfidfVectorizer handles tokenization, counting (TF),
# and calculation of Inverse Document Frequency (IDF) in a single step.
vectorizer = TfidfVectorizer(
    lowercase=True, 
    # Stop words are highly common words (like 'the', 'a', 'is')
    # Ignoring them improves the quality of the vectors.
    stop_words='english' 
)

# fit_transform() learns the vocabulary and IDF weights from the corpus (fit),
# and then computes the TF-IDF scores for the documents (transform).
tfidf_matrix = vectorizer.fit_transform(corpus)

print("--- 2.2. Resulting TF-IDF Matrix Shape ---")
print(f"Documents: {tfidf_matrix.shape[0]} (Rows)")
print(f"Features (Unique Words): {tfidf_matrix.shape[1]} (Columns)")
print("\n" + "="*40 + "\n")
```

```
--- 2.2. Resulting TF-IDF Matrix Shape ---
Documents: 4 (Rows)
Features (Unique Words): 11 (Columns)

========================================
```

```python
# --- 2.3. Inspect the Results ---

# Get the feature names (vocabulary)
feature_names = vectorizer.get_feature_names_out()

# Convert the sparse matrix to a dense numpy array for display
# (Sparse matrices are memory-efficient for large datasets, but dense is easier to read)
tfidf_array = tfidf_matrix.toarray()

# Create a DataFrame for a clean, visual representation
df_tfidf = pd.DataFrame(data=tfidf_array, columns=feature_names, index=[f"Doc {i+1}" for i in range(len(corpus))])

print("--- 2.3. Document-Term Matrix (TF-IDF Scores) ---")
# Display the matrix, rounding to 4 decimal places for readability
with pd.option_context('display.max_columns', None):
    print(df_tfidf.round(4))

print("\n" + "="*40 + "\n")
```

```
--- 2.3. Document-Term Matrix (TF-IDF Scores) ---
        brown     cat     dog     fox   jumps    lazy     mat   quick  \
Doc 1  0.4838  0.0000  0.3814  0.3088  0.4838  0.3814  0.0000  0.3814   
Doc 2  0.0000  0.0000  0.0000  0.7532  0.0000  0.4652  0.0000  0.4652   
Doc 3  0.0000  0.0000  0.4530  0.3667  0.0000  0.0000  0.0000  0.0000   
Doc 4  0.0000  0.5774  0.0000  0.0000  0.0000  0.0000  0.5774  0.0000   

       quickly  running     sat  
Doc 1   0.0000   0.0000  0.0000  
Doc 2   0.0000   0.0000  0.0000  
Doc 3   0.5746   0.5746  0.0000  
Doc 4   0.0000   0.0000  0.5774  

========================================
```


2.4. Explanation of Key Scores

Find the index of a common word like 'fox' and a rare word like 'jumps' 
NOTE: Stop words like 'the' and 'a' were removed by the vectorizer setting.

2.4a. Example 1: 'fox' (appears in multiple documents, medium IDF)

```python
try:
    fox_index = vectorizer.vocabulary_['fox']
    fox_scores = df_tfidf['fox'].round(4).to_list()
    print("2.4a. Word 'fox': Appears in 3 out of 4 documents.")
    print(f"\tTF-IDF Scores in each document: {fox_scores}")
except KeyError:
    print("2.4a. Word 'fox' not found (check stop words/tokenization).")
```

```
2.4a. Word 'fox': Appears in 3 out of 4 documents.
	TF-IDF Scores in each document: [0.3088, 0.7532, 0.3667, 0.0]
```

The score is moderate. It's high in documents where it appears (Doc 1, 2, 3), but penalized slightly because it's a common word across the corpus.

2.4b. Example 2: 'jumps' (appears in only one document, high IDF)

```python
try:
    jumps_index = vectorizer.vocabulary_['jumps']
    jumps_scores = df_tfidf['jumps'].round(4).to_list()
    print("\n2.4b. Word 'jumps': Appears in only 1 out of 4 documents.")
    print(f"\tTF-IDF Scores in each document: {jumps_scores}")
except KeyError:
    print("2.4b. Word 'jumps' not found (check stop words/tokenization).")
```

```
2.4b. Word 'jumps': Appears in only 1 out of 4 documents.
	TF-IDF Scores in each document: [0.4838, 0.0, 0.0, 0.0]
```

The score for 'jumps' in Doc 1 (0.4838) is much higher than 'fox' (0.3088) even though both appeared once. This is because 'jumps' is a rare word, giving it a much higher IDF multiplier, which signals its importance.

**2.5 sklearn TF-IDF (Baseline)**

```python
import re
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the categories for newsgroup 20 dataset
CATEGORIES = [
    'alt.atheism', 
    'comp.graphics', 
    'rec.sport.baseball', 
    'sci.med'
]

print("#"*40)
print("2.5: SKLEARN TF-IDF (Baseline)")
print("#"*40 + "\n")
```

```python
# --- 2.5.1. Load Dataset (20 Newsgroups Subset) ---
print(f"Loading 20 Newsgroups data for categories: {CATEGORIES}...")

newsgroups_data = fetch_20newsgroups(
    categories=CATEGORIES, 
    subset='all', 
    remove=('headers', 'footers', 'quotes')
)

X = newsgroups_data.data # The text documents
y = newsgroups_data.target # The category labels
```

```
Loading 20 Newsgroups data for categories: ['alt.atheism', 'comp.graphics', 'rec.sport.baseball', 'sci.med']...
```

```python
# --- 2.5.2. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

print(f"\nTraining set size: {len(X_train)} documents")
print(f"Testing set size: {len(X_test)} documents")
```

```
Training set size: 3004 documents
Testing set size: 752 documents
```

```python
# --- 2.5.3. Vectorize the Data using Sklearn TfidfVectorizer ---
print("\nVectorizing data using sklearn.TfidfVectorizer...")

# Use default settings (including L2 normalization and Laplace smoothing)
sklearn_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

X_train_tfidf_sklearn = sklearn_vectorizer.fit_transform(X_train)
X_test_tfidf_sklearn = sklearn_vectorizer.transform(X_test)

print(f"Sklearn TF-IDF Training Matrix Shape: {X_train_tfidf_sklearn.shape}")
```

```
Vectorizing data using sklearn.TfidfVectorizer...
Sklearn TF-IDF Training Matrix Shape: (3004, 31102)
```

```python
# --- 2.5.4. Train Multinomial Naive Bayes Model ---
print("\nTraining Multinomial Naive Bayes Classifier with Sklearn TF-IDF...")
nb_classifier_sklearn = MultinomialNB()
nb_classifier_sklearn.fit(X_train_tfidf_sklearn, y_train)
```

```python
# --- 2.5.5. Predict and Evaluate ---
y_pred_sklearn = nb_classifier_sklearn.predict(X_test_tfidf_sklearn)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print("\n--- Sklearn TF-IDF Performance ---")
print(f"Test Accuracy: {accuracy_sklearn:.4f}")
print("----------------------------------")
```

```
--- Sklearn TF-IDF Performance ---
Test Accuracy: 0.9255
----------------------------------
```

In this section we shall try to create the TF-IDF Vectorizer from scratch and see how it fares with the implementation from sklarn

```python
class SimpleTfidfVectorizer:
    """
    A simplified implementation of the TF-IDF Vectorizer.
    This calculates Term Frequency (TF) and Inverse Document Frequency (IDF) 
    to create weighted document vectors.
    """
    def __init__(self):
        self.vocabulary_ = {} # Word -> Index mapping
        self.idf_weights_ = {} # Word -> IDF score mapping
        self.feature_names_out = [] # List of words
        self.doc_count = 0 # Total number of documents (N)

    def _tokenize(self, text):
        """Simple text cleaning and tokenization."""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        return tokens

    def fit(self, corpus):
        """
        Learns the vocabulary and calculates the IDF weights.
        """
        self.doc_count = len(corpus)
        unique_words = set()
        
        # 1. Build Vocabulary and Calculate Document Frequency (DF)
        df_counts = {} # Word -> number of documents containing the word
        
        for doc in corpus:
            tokens = self._tokenize(doc)
            # Find unique words in this specific document for DF calculation
            unique_tokens_in_doc = set(tokens)
            
            unique_words.update(unique_tokens_in_doc)
            
            for token in unique_tokens_in_doc:
                df_counts[token] = df_counts.get(token, 0) + 1

        # Sort the words to create a consistent index mapping
        sorted_words = sorted(list(unique_words))
        
        for index, word in enumerate(sorted_words):
            self.vocabulary_[word] = index
        
        self.feature_names_out = sorted_words
        
        # 2. Calculate IDF Weights
        # Using the standard sklearn formula (smooth=idf): 
        # IDF(t) = log((1 + N) / (1 + DF(t))) + 1
        for word, df in df_counts.items():
            # N is self.doc_count
            idf_score = math.log((1 + self.doc_count) / (1 + df)) + 1
            self.idf_weights_[word] = idf_score
        
        return self

    def transform(self, corpus):
        """
        Transforms the documents into the TF-IDF matrix using 
        the learned vocabulary and IDF weights.
        """
        vocab_size = len(self.vocabulary_)
        # Initialize an empty matrix for the TF-IDF scores
        tfidf_matrix = np.zeros((len(corpus), vocab_size), dtype=float)

        for doc_index, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            
            # 1. Calculate Term Frequency (TF - raw count)
            word_counts = {}
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
            
            # 2. Apply TF * IDF and store in the matrix
            for word, count in word_counts.items():
                if word in self.vocabulary_:
                    # Raw Count (TF)
                    tf = count
                    # IDF weight from the fitted model
                    idf = self.idf_weights_.get(word, 1.0) # Default to 1.0 if word is new (out-of-vocabulary)
                    
                    tfidf_score = tf * idf
                    
                    col_index = self.vocabulary_[word]
                    tfidf_matrix[doc_index, col_index] = tfidf_score
        
        # NOTE: Standard sklearn TfidfVectorizer applies L2 normalization (unit vector).
        # We skip that here to keep the implementation simple and focus on the core TF*IDF product.
        
        return tfidf_matrix

    def fit_transform(self, corpus):
        """
        A convenience method that calls fit() and then transform().
        """
        self.fit(corpus)
        return self.transform(corpus)

```


```python
print("\n" + "#"*40)
print("2.7: Custom TF-IDF Application and Comparison")
print("#"*40 + "\n")

# --- 2.7.1. Vectorize the Data using SimpleTfidfVectorizer ---
print("Vectorizing data using SimpleTfidfVectorizer...")
custom_vectorizer = SimpleTfidfVectorizer()

# Note: Our simple vectorizer does not handle stop words, so the vocabulary 
# size will be larger than the sklearn version.
X_train_tfidf_custom = custom_vectorizer.fit_transform(X_train)
X_test_tfidf_custom = custom_vectorizer.transform(X_test)

print(f"Custom TF-IDF Training Matrix Shape: {X_train_tfidf_custom.shape}")
```

```
Vectorizing data using SimpleTfidfVectorizer...
Custom TF-IDF Training Matrix Shape: (3004, 31765)
```

```python
# --- 2.7.2. Train Multinomial Naive Bayes Model ---
# MultinomialNB is very forgiving and often works well even without L2 normalization, 
# provided the features are non-negative.
print("\nTraining Multinomial Naive Bayes Classifier with Custom TF-IDF...")
nb_classifier_custom = MultinomialNB()
nb_classifier_custom.fit(X_train_tfidf_custom, y_train)
```

```python
# --- 2.7.3. Predict and Evaluate ---
y_pred_custom = nb_classifier_custom.predict(X_test_tfidf_custom)
accuracy_custom = accuracy_score(y_test, y_pred_custom)

print("\n--- Custom TF-IDF Performance ---")
print(f"Test Accuracy: {accuracy_custom:.4f}")
print("---------------------------------")
    
print(f"\nSummary of Results:")
print(f"Sklearn TF-IDF Accuracy: {accuracy_sklearn:.4f}")
print(f"Custom TF-IDF Accuracy: {accuracy_custom:.4f}")
```

```
--- Custom TF-IDF Performance ---
Test Accuracy: 0.9335
---------------------------------

Summary of Results:
Sklearn TF-IDF Accuracy: 0.9255
Custom TF-IDF Accuracy: 0.9335
```

The custom implementation, even without L2 normalization or advanced preprocessing, achieves a better result, validating the core TF-IDF weighting logic.

## 3. N-grams

While TF-IDF tells us which individual words are important, N-grams allow us to capture the importance of phrases and context, often leading to better classification performance.

An N-gram is a sequence of $N$ words. For instance, if $N=2$ (a bigram), we look at sequences like "quick brown" or "lazy dog". If we use an ngram_range of (1, 2), our feature set will include all single words (unigrams) and all two-word phrases (bigrams).

### Code

Let's look at the below Python implementation for understanding how it actually works:

```python
# Understanding N-Grams (The Core Logic)

def _get_ngrams(tokens, n_min, n_max):
    """
    Generates all n-grams (phrases) for a list of tokens within a specified range.
    
    Example: 
    Tokens = ['the', 'quick', 'brown']
    If n_min=1, n_max=2:
    Result = ['the', 'quick', 'brown', 'the_quick', 'quick_brown']
    """
    ngrams = []
    # Loop from the minimum N to the maximum N (inclusive)
    for n in range(n_min, n_max + 1):
        # The range ensures we stop before the index goes out of bounds
        for i in range(len(tokens) - n + 1):
            # Join the words with an underscore to treat the phrase as a single feature
            ngrams.append('_'.join(tokens[i:i+n]))
    return ngrams
```

```python
print("\n" + "="*40)
print("3: Understanding N-Grams Demonstration")
print("="*40 + "\n")

sample_tokens = ['this', 'is', 'a', 'short', 'sentence']
print(f"Sample Tokens: {sample_tokens}")

# Demonstrate Unigrams (1-grams)
unigrams = _get_ngrams(sample_tokens, 1, 1)
print(f"Unigrams (1-grams): {unigrams}")

# Demonstrate Bigrams (2-grams)
bigrams = _get_ngrams(sample_tokens, 2, 2)
print(f"Bigrams (2-grams): {bigrams}")

# Demonstrate Mixed (1-gram and 2-gram)
mixed_ngrams = _get_ngrams(sample_tokens, 1, 2)
print(f"Mixed N-grams (1-2): {mixed_ngrams}\n")
```

```
========================================
3: Understanding N-Grams Demonstration
========================================

Sample Tokens: ['this', 'is', 'a', 'short', 'sentence']
Unigrams (1-grams): ['this', 'is', 'a', 'short', 'sentence']
Bigrams (2-grams): ['this_is', 'is_a', 'a_short', 'short_sentence']
Mixed N-grams (1-2): ['this', 'is', 'a', 'short', 'sentence', 'this_is', 'is_a', 'a_short', 'short_sentence']
```

`sklearn` already provides support ngram into `TfidfVectorizer` by simply setting `ngram_range=(1, 2)` to include **unigrams** (single words) and **bigrams** (two-word phrases) in the feature set.

In order to apply TFIDF with ngram we can use something like:
```python
print("\nVectorizing data using sklearn.TfidfVectorizer with ngrams...")

sklearn_vectorizer = TfidfVectorizer(
    stop_words='english', 
    lowercase=True,
    ngram_range=(1, 2) 
)

X_train_tfidf_ng_sklearn = sklearn_vectorizer.fit_transform(X_train)
X_test_tfidf_ng_sklearn = sklearn_vectorizer.transform(X_test)

print(f"Sklearn TF-IDF Training Matrix Shape: {X_train_tfidf_ng_sklearn.shape}")
```

```
Vectorizing data using sklearn.TfidfVectorizer with ngrams...
Sklearn TF-IDF Training Matrix Shape: (3004, 236326)
```

Note: Because we included n-gram range 1 to 2, which includes both unigrams and bigrams, our feature set has exploded.

We can then train the Multinomial NB classifier with this dataset as
```python
# --- 3.4. Train Multinomial Naive Bayes Model ---
print("\nTraining Multinomial Naive Bayes Classifier with Sklearn TF-IDF...")
nb_classifier_sklearn = MultinomialNB()
nb_classifier_sklearn.fit(X_train_tfidf_ng_sklearn, y_train)

# --- 3.5. Predict and Evaluate ---
y_pred_sklearn = nb_classifier_sklearn.predict(X_test_tfidf_ng_sklearn)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print("\n--- Sklearn TF-IDF + N-grams Performance ---")
print(f"Test Accuracy: {accuracy_sklearn:.4f}")
print("---------------------------------------------")
```

```
Training Multinomial Naive Bayes Classifier with Sklearn TF-IDF...

--- Sklearn TF-IDF + N-grams Performance ---
Test Accuracy: 0.9242
---------------------------------------------
```

And in-order to include ngrams into our custom TFIDF implementation we simply add the function and use it in our tokenization as:

```python
class SimpleTfidfNgramsVectorizer:
    ...
    def _get_ngrams(self, tokens, n_min, n_max):
        """Generates n-grams (phrases) for a list of tokens."""
        ngrams = []
        for n in range(n_min, n_max + 1):
            # The range ensures we stop before the index goes out of bounds
            for i in range(len(tokens) - n + 1):
                # Join the words with an underscore to treat the phrase as a single feature
                ngrams.append('_'.join(tokens[i:i+n]))
        return ngrams

    def _tokenize(self, text):
        """
        Simple text cleaning, tokenization, and N-gram generation.
        NOTE: This simple tokenizer does not remove stop words, 
        which will lead to many common bigrams (e.g., 'the_quick').
        """
        text = text.lower()
        # Remove all non-alphanumeric characters except spaces
        text = re.sub(r'[^a-z\s]', '', text)
        # Split into unigrams
        tokens = text.split()
        
        # Generate N-grams based on the set range
        return self._get_ngrams(tokens, self.ngram_range[0], self.ngram_range[1])
    
    # rest of the functions remains the same
```

```python
print("\n" + "#"*40)
print("Part 3: Custom TF-IDF + N-grams Application and Comparison")
print("#"*40 + "\n")

# --- 3.6. Vectorize the Data using SimpleTfidfNgramsVectorizer ---
print("Vectorizing data using SimpleTfidfNgramsVectorizer...")
# CRITICAL CHANGE: Initialize with bigrams
custom_vectorizer = SimpleTfidfNgramsVectorizer(ngram_range=(1, 2)) 

X_train_tfidf_ng_custom = custom_vectorizer.fit_transform(X_train)
X_test_tfidf_ng_custom = custom_vectorizer.transform(X_test)

print(f"Custom TF-IDF + N-grams Training Matrix Shape: {X_train_tfidf_ng_custom.shape}")
```

```
########################################
Part 3: Custom TF-IDF + N-grams Application and Comparison
########################################

Vectorizing data using SimpleTfidfNgramsVectorizer...
Custom TF-IDF + N-grams Training Matrix Shape: (3004, 257636)
```

```python
# --- 3.7. Train Multinomial Naive Bayes Model ---
print("\nTraining Multinomial Naive Bayes Classifier with Custom TF-IDF...")
nb_classifier_custom = MultinomialNB()
nb_classifier_custom.fit(X_train_tfidf_ng_custom, y_train)

# --- 3.8. Predict and Evaluate ---
y_pred_custom = nb_classifier_custom.predict(X_test_tfidf_ng_custom)
accuracy_custom = accuracy_score(y_test, y_pred_custom)

print("\n--- Custom TF-IDF + N-grams Performance ---")
print(f"Test Accuracy: {accuracy_custom:.4f}")
print("-------------------------------------------")
print(f"\nSummary of Results:")
print(f"Sklearn TF-IDF + N-grams Accuracy: {accuracy_sklearn:.4f}")
print(f"Custom TF-IDF + N-grams Accuracy: {accuracy_custom:.4f}")
```

```
Training Multinomial Naive Bayes Classifier with Custom TF-IDF...

--- Custom TF-IDF + N-grams Performance ---
Test Accuracy: 0.9242
-------------------------------------------

Summary of Results:
Sklearn TF-IDF + N-grams Accuracy: 0.9242
Custom TF-IDF + N-grams Accuracy: 0.9242
```

The inclusion of bigrams dramatically increases the feature space (vocabulary size) but often captures richer semantic information, potentially boosting classification accuracy, however, in this case, it suffers from the curse of dimensionality and results in a drop in accuracy.

## 4. Hashing Vectorizer

The Hashing Vectorizer, also known as the "**hashing trick**", introduces a critical concept in scalable NLP: feature engineering without an explicit, fixed vocabulary. The **Hashing Vectorizer** provides a highly memory-efficient way to convert text documents into a matrix of token occurrences, similar to Count Vectorization, but without storing a vocabulary dictionary. This is ideal for very large datasets.

**Key Concepts:**
1.  **Hash Function:** Instead of maintaining a dictionary mapping 'word' -> index, 
    it uses a mathematical hash function (like `MurmurHash3`) to convert the word 
    string directly into a numerical feature index.
2.  **Fixed Feature Size:** You must pre-specify the final size of the feature vector, 
    often a large power of 2 (e.g., 2**18). This fixes the memory footprint 
    regardless of the corpus size.
3.  **Collision Trade-off:** Since the number of unique words is usually far greater 
    than the fixed feature size, different words can occasionally map to the 
    same index. This is called a **hash collision**. In practice, the impact of 
    these collisions is often negligible.
4.  **Sign Trick:** To help mitigate collisions, a second hash is often used 
    to determine the sign (+1 or -1) of the resulting count, allowing the 
    contribution of different words to (partially) cancel out at the same index.

### Code

Let's look at the below Python implementation of simple hashing first to understand how hashing works:

```python
# =========================================================================
# Demonstration of the Hashing Concept
# =========================================================================

def simple_hashing_demonstration(token, n_features=10):
    """
    Simulates the core concept of the Hashing Trick: 
    mapping a string (token) to a fixed-size index using Python's built-in hash.
    
    NOTE: sklearn uses a more robust hash (MurmurHash3) and a sign trick.
    """
    # 1. Get a numerical hash value for the token
    hash_value = hash(token)
    
    # 2. Map the hash value to a fixed index (0 to n_features-1)
    # The absolute value handles Python's negative hash results gracefully.
    index = abs(hash_value) % n_features
    
    # 3. Determine the sign (very simplified stand-in for the sign trick)
    sign = 1 if hash_value >= 0 else -1
    
    print(f"Token: '{token}'")
    print(f"  Hash: {hash_value}")
    print(f"  Index (0-{n_features-1}): {index}")
    print(f"  Sign: {sign}")
    
    return index, sign
```

```python
print("\n" + "="*50)
print("Part 2: Simple Hashing Demonstration")
print("="*50 + "\n")

# Example 1: Mapping different tokens
simple_hashing_demonstration("computer", n_features=10)
simple_hashing_demonstration("graphics", n_features=10)

# Example 2: Demonstrating a potential collision 
# (These indices might change across Python sessions/versions, but the concept is clear)
print("\nChecking for potential collisions:")
simple_hashing_demonstration("apple", n_features=5)
simple_hashing_demonstration("banana", n_features=5)
```

```
==================================================
Simple Hashing Demonstration
==================================================

Token: 'computer'
  Hash: -6524747023095029386
  Index (0-9): 6
  Sign: -1
Token: 'graphics'
  Hash: 3457090053383372878
  Index (0-9): 8
  Sign: 1

Checking for potential collisions:
Token: 'apple'
  Hash: 8988249274639096007
  Index (0-4): 2
  Sign: 1
Token: 'banana'
  Hash: 8778862295073914582
  Index (0-4): 2
  Sign: 1
(2, 1)
```

As how hashing works is clear now, let's look at an application using Sklearn HashingVectorizer

```python
# --- 4.3. Vectorize the Data using Sklearn HashingVectorizer ---
from sklearn.feature_extraction.text import HashingVectorizer
print("\nVectorizing data using sklearn.feature_extraction.text.HashingVectorizer...")

# We specify a fixed feature size (2**18 = 262,144 features)
# The vectorizer automatically handles tokenization, n-grams, and the sign trick.
# NOTE: HashingVectorizer is often configured for TF-IDF scaling after hashing.
# We keep it simple for now (equivalent to count vectorization via hashing).
hashing_vectorizer = HashingVectorizer(
    n_features=2**18, 
    stop_words='english', 
    ngram_range=(1, 2), # Can handle n-grams just like TF-IDF
    alternate_sign=True # Ensures the use of the sign trick
)

# Unlike CountVectorizer/TfidfVectorizer, HashingVectorizer does NOT need a 
# .fit() step, as it builds no vocabulary. It can be transformed immediately.
X_train_hashed = hashing_vectorizer.transform(X_train)
X_test_hashed = hashing_vectorizer.transform(X_test)

# Notice the feature size is exactly the n_features we specified.
print(f"Hashed Training Matrix Shape: {X_train_hashed.shape}") 
```

```
Vectorizing data using sklearn.feature_extraction.text.HashingVectorizer...
Hashed Training Matrix Shape: (3004, 262144)
```

```python
# --- 4.4. Train Linear Support Vector Classifier (LinerSVC) ---
from sklearn.svm import LinearSVC
print("\nTraining Linear Support Vector Classifier (LinearSVC) with Hashed Features...")

# We must use a classifier that can handle the negative values generated by the Hashing Trick.
classifier_hashed = LinearSVC(max_iter=1000) 
classifier_hashed.fit(X_train_hashed, y_train)

# --- 4.5. Predict and Evaluate ---
y_pred_hashed = classifier_hashed.predict(X_test_hashed)
accuracy_hashed = accuracy_score(y_test, y_pred_hashed)

print("\n--- Hashing Vectorizer + LinearSVC Performance ---")
print(f"Test Accuracy: {accuracy_hashed:.4f}")
print("--------------------------------------------------")
```

```
Training Linear Support Vector Classifier (LinearSVC) with Hashed Features...

--- Hashing Vectorizer + LinearSVC Performance ---
Test Accuracy: 0.9189
--------------------------------------------------
```

**Note:**
* As we are using `HashingVectorizer(..., alternate_sign=True)`: The alternate_sign trick is crucial for collision mitigation, but it produces negative feature values in the output matrix.
* The MultinomialNB classifier is based on probability distributions of counts and requires non-negative input data.
* To fix this, we need to switch to a classifier that can handle the negative feature weights produced by the Hashing Trick, such as a Linear Support Vector Classifier (LinearSVC).


# Distributional and Shallow Embedding Methods

The Bag-of-Words family gave us a great foundation, but those methods treat words as *isolated entities*. Now, we're moving into Distributional Semantics, where we capture the actual meaning and relationships between words.
These methods represent words as **dense vectors** by analyzing the co-occurrence of words in a large corpus. The principle is: "*A word is characterized by the company it keeps.*" They are *semantic* but still *context-independent*.

## Word2Vec

While Count/TF-IDF vectorizers are fast and simple, they suffer from two major drawbacks:
1.  **Sparsity:** The feature space is enormous (millions of dimensions) and mostly zeros.
2.  **No Semantic Meaning:** They cannot tell that 'dog' and 'canine' are related.

**Word2Vec** solves this using the **Distributional Hypothesis** - '*A word is characterized by the company it keeps.*'

This means that words that appear in similar contexts tend to have similar meanings. *Word2Vec* trains a small neural network to predict words based on their neighbors, resulting in *dense*, *low-dimensional* vectors (e.g., 100 or 300 dimensions) where semantic relationships are encoded **spatially**.

**Core Architectures:**

1.  **CBOW (Continuous Bag-of-Words):** Predicts the current word based on its surrounding context words. This method is generally faster to train.

2.  **Skip-gram:** Predicts the surrounding context words given the current word. This method is generally slower but performs better with smaller amounts of training data and captures rare words more effectively. 

The resulting vectors are the '*embeddings*'—numerical representations where vector arithmetic reveals semantic and syntactic relationships (e.g., *Vector('King') - Vector('Man') + Vector('Woman') ≈ Vector('Queen')*).

### Gensim Implementation

Word2vec was developed by Tomáš Mikolov, Kai Chen, Greg Corrado, Ilya Sutskever and Jeff Dean at Google, and published in 2013. It is widely distributed under the library `gensim`, which can be installed as:

```bash
pip install --upgrade gensim -q
```

Let's look at how Word2Vec tokenizes words using a toy example:

```python
import logging
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
# Note: For production use, nltk.download('punkt') might be required once.

# Set up logging for gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("#"*50)
print("5: Word2Vec: Distributional and Shallow Embeddings")
print("#"*50)

print("\n" + "="*50)
print("5.1: Preparing the Corpus (Tokenization)")
print("="*50 + "\n")

# A small, clean corpus for demonstration
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A lazy cat sleeps next to the big dog.",
    "Dogs and foxes are animals, but cats are small.",
    "The brown fox is quicker than the lazy cat.",
    "Dogs often jump and run."
]

def preprocess_and_tokenize(text_list):
    """Simple tokenizer for Word2Vec training."""
    tokenized_sentences = []
    for sentence in text_list:
        # Simple cleaning and lowercasing
        sentence = re.sub(r'[^\w\s]', '', sentence.lower())
        # Tokenize using nltk (or a simple split for small examples)
        tokens = sentence.split()
        tokenized_sentences.append(tokens)
    return tokenized_sentences

# Preprocess the corpus
tokenized_corpus = preprocess_and_tokenize(corpus)

print("Sample Tokenized Sentence:")
print(tokenized_corpus[0])

```

```
##################################################
5: Word2Vec: Distributional and Shallow Embeddings
##################################################

==================================================
5.1: Preparing the Corpus (Tokenization)
==================================================

Sample Tokenized Sentence:
['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
```

```python
# =========================================================================
# 5.2: Training Word2Vec using Gensim
# =========================================================================

print("\n" + "#"*50)
print("5.2: Training Word2Vec with Gensim")
print("#"*50 + "\n")

# --- Training the Model ---
# Key Parameters:
# vector_size (or size): Dimensionality of the word vectors (e.g., 100).
# window: Maximum distance between the current and predicted word within a sentence.
# min_count: Ignores all words with a total frequency lower than this.
# sg: 0 for CBOW, 1 for Skip-gram (we'll use Skip-gram here).
model = Word2Vec(
    sentences=tokenized_corpus, 
    vector_size=10,        # Small vector size for demonstration
    window=2,              # Look 2 words left and right
    min_count=1,           # Include all words
    sg=1,                  # Use Skip-gram architecture
    epochs=100             # Increase epochs for better training on small data
)

print(f"Vocabulary Size: {len(model.wv.key_to_index)}")
print(f"Vector Dimensionality: {model.vector_size}")
```

```
##################################################
5.2: Training Word2Vec with Gensim
##################################################

Vocabulary Size: 28
Vector Dimensionality: 10
```

```python
# 5.3 Inspecting the Vectors and Relationships

print("\n--- Word Vectors ---")
# Access the vector for a word
word_vector = model.wv['fox']
print(f"Vector for 'fox' (first 5 dimensions): {word_vector[:5]}")


print("\n--- Semantic Similarity ---")
# Find words most similar to 'dog'
similar_words = model.wv.most_similar('dog', topn=3)
print(f"Words similar to 'dog': {similar_words}")


print("\n--- Analogies (Semantic Arithmetic) ---")
# Example: What is to 'dog' what 'cat' is to 'sleeps'? (Simplified for small corpus)
# Analogy: 'dog' - 'lazy' + 'quick'
try:
    analogy_result = model.wv.most_similar(
        positive=['dog', 'quick'], 
        negative=['lazy'], 
        topn=1
    )
    print(f"Analogy: 'dog' - 'lazy' + 'quick' is most similar to: {analogy_result}")
except Exception as e:
    print(f"Could not run complex analogy (corpus too small): {e}")

```

```
--- Word Vectors ---
Vector for 'fox' (first 5 dimensions): [ 0.05278032  0.08030616 -0.01256576 -0.09743598  0.04672134]

--- Semantic Similarity ---
Words similar to 'dog': [('dogs', 0.6270390748977661), ('next', 0.5659131407737732), ('big', 0.5416152477264404)]

--- Analogies (Semantic Arithmetic) ---
Analogy: 'dog' - 'lazy' + 'quick' is most similar to: [('cats', 0.5542731285095215)]
```

**5.4 Interpretting the results:**

* Words with similar meanings or contexts will have vectors pointing in similar directions (i.e., they will be close together in the vector space)
* It uses Cosine Similarity to measure the angular distance between the vector for 'dog' and all other word vectors in the vocabulary. A score of `1.0` means they are **identical**; `0.0` means they are **orthogonal** (unrelated).
* `Words similar to 'dog': ('dogs', 0.627...)`: Since the singular ('dog') and plural ('dogs') often appear in identical contexts, Word2Vec correctly places them very close together, showing it captures basic syntactic similarity.
* `('next', 0.565...)` and `('big', 0.541...)`: These results are due to our tiny, limited corpus: "A lazy cat sleeps **next** to the **big dog**." Because our model only learned from five sentences, 'next' and 'big' are high-frequency neighbors of 'dog', causing them to be artificially similar in the resulting small vector space. In a real-world corpus, you would expect semantically related words like 'cat', 'leash', or 'pet' to appear here.
* Word2Vec has **vector arithmetic capabilities** - it tries to find the word vector that is closest to the resulting arithmetic vector: $V_{result} \approx V_{dog} - V_{lazy} + V_{quick}$. We take the vector for 'dog', subtract the vector for 'lazy' (removing the "laziness" feature), and add the vector for 'quick' (imparting the "quickness" feature).
* In our corpus, both 'dogs' and 'cats' are the central "animal" entities that interact with properties like 'lazy' and 'quick' (The lazy dog, the lazy cat, the quick fox). By swapping out the 'lazy' context for the 'quick' context, the resulting vector points toward the other main domestic animal mentioned in the sentences, showing a basic shift in context from one animal to another.

### Continuous Bag-of-Words (CBOW) from scratch

Let's try to implement CBOW from scratch to understand the mechanics of Word2Vec. We won't train a deep model, but rather focus on setting up the architecture, weights, and loss function using basic NumPy, which is sufficient to illustrate the training process on a toy dataset.

Let's take a toy dataset
```python
corpus = [
    "A dog runs fast",
    "A cat jumps high",
    "The brown dog runs and plays",
    "The tiny kitten jumps up",
    "I drink hot coffee every morning",
    "She likes tea, not coffee",
    "He reads books and drinks tea"
]
```

There are some helper functions I shall be using in this, which can be found in the notebook attached to this blog post. I am not providing those codes here for brevity. Avid readers are encouraged to try out the Colab notebook.

Implementing CBOW from scratch
```python
# =========================================================================
# Custom CBOW Implementation
# =========================================================================
class CBOW:
    """
    **CBOW** predicts the **target word** from its surrounding **context words**. 
    Input is the *average* of the context word vectors.
    """
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.V = vocab_size
        self.D = embedding_dim
        self.lr = learning_rate
        
        # W1: Input-to-Hidden weights (V x D). These are the initial word vectors.
        self.W1 = np.random.uniform(-1, 1, (self.V, self.D)) * 0.1
        # W2: Hidden-to-Output weights (D x V). These are the context vectors.
        self.W2 = np.random.uniform(-1, 1, (self.D, self.V)) * 0.1

    def forward(self, context_oh):
        """
        context_oh is the average of the one-hot vectors for context words.
        Shape: (V,)
        """
        # h (Hidden Layer): D-dimensional vector. W1^T * context_oh
        h = np.dot(context_oh, self.W1)
        
        # u (Output Layer Pre-Softmax): V-dimensional vector. W2^T * h
        u = np.dot(h, self.W2)
        
        # y (Output Layer Post-Softmax): Probability distribution over vocab
        y = softmax(u)
        
        return y, h, u

    def backward(self, y_pred, target_oh, h, context_oh):
        """
        Backpropagation to update weights W1 and W2.
        """
        # E: Error at output layer (Predicted - Actual One-Hot)
        E = y_pred - target_oh
        
        # Backpropagate error to Hidden->Output weights (W2)
        dL_dW2 = np.outer(h, E)
        
        # Backpropagate error to Input->Hidden weights (W1)
        dL_dh = np.dot(self.W2, E) # Error at hidden layer
        
        # Update W1: W1 update is proportional to the context and the hidden error
        dL_dW1 = np.outer(context_oh, dL_dh)

        # Gradient Descent Update
        self.W2 -= self.lr * dL_dW2
        self.W1 -= self.lr * dL_dW1
        
    def get_embedding(self, word_idx):
        # The embedding is the row vector in W1 corresponding to the word index
        return self.W1[word_idx]
```

We shall also need to prepare the dataset for training a CBOW model, the codes to which can be found in the notebook. Post training the model, check for similarity between pairs of words with similar semantic context such as:
```
word_pairs = [
    ('dog', 'kitten'), # Related (Animals)
    ('coffee', 'tea'), # Related (Drinks)
    ('runs', 'jumps'), # Related (Actions)
    ('dog', 'morning')  # Unrelated
]
```
```
--- CBOW Cosine Similarities ---
Similarity('dog', 'kitten'): 0.5547
Similarity('coffee', 'tea'): 0.6967
Similarity('runs', 'jumps'): 0.0119
Similarity('dog', 'morning'): 0.4651
```

We plot them in a 2D graph to visualize how CBOW represents them, and we see that it is able to identify natural clusters, where words with similar semantic meaning have been paired together.

<img src="https://debanjanaiml.github.io/assets/images/blogs/cbow.png" width="500">{: .align-center}


### Skip-gram from scratch

Skip-gram is another architecture of Word2Vec where it tries to predict the context words from the words surrounding it. Here we shall be using Negative Sampling technique to simplify this implementation

```python
print("\n" + "#"*50)
print("Part 3: Skip-gram Model (Negative Sampling Simplified)")
print("#"*50 + "\n")

class SkipGram:
    """
    **Skip-gram** predicts the **context words** from the **target word**. 
    We use a simplified Negative Sampling approach for the output layer 
    to make the training feasible.
    """
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.V = vocab_size
        self.D = embedding_dim
        self.lr = learning_rate
        
        # W1: Input-to-Hidden weights (V x D). These are the word vectors (center word).
        self.W1 = np.random.uniform(-1, 1, (self.V, self.D)) * 0.1
        # W2: Hidden-to-Output weights (D x V). These are the context vectors (outside word).
        self.W2 = np.random.uniform(-1, 1, (self.D, self.V)) * 0.1

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def train_pair(self, target_idx, context_idx, is_positive):
        """
        Trains the model on a single (target, context) pair 
        using a simplified binary objective.
        """
        
        # 1. Forward Pass
        # h (Hidden Layer): The vector for the target word
        h = self.W1[target_idx] # Shape (D,)
        
        # Output Score: Dot product between target vector (h) and context vector (W2[:, context_idx])
        u_score = np.dot(h, self.W2[:, context_idx]) 
        y_pred = self.sigmoid(u_score)
        
        # 2. Compute Error and Gradient
        
        # Label: 1 for positive context, 0 for negative sample
        target_label = 1 if is_positive else 0
        
        # Error (Sigmoid Cross-Entropy derivative)
        error = y_pred - target_label
        
        # Gradient for W2 (context vector)
        dL_dW2_col = error * h 
        
        # Gradient for W1 (target vector)
        dL_dh = error * self.W2[:, context_idx]
        
        # 3. Update Weights
        
        # Update W2 (Context Vector): Only update the column corresponding to the context_idx
        self.W2[:, context_idx] -= self.lr * dL_dW2_col
        
        # Update W1 (Target Vector): Update the row corresponding to the target_idx
        self.W1[target_idx] -= self.lr * dL_dh

    def get_embedding(self, word_idx):
        return self.W1[word_idx]
```

Here as well, we compute the cosine similarities and plot the embeddings in a 2D graph post training

```
--- Skip-gram Cosine Similarities ---
Similarity('dog', 'kitten'): 0.6345
Similarity('coffee', 'tea'): -0.5171
Similarity('runs', 'jumps'): -0.4653
Similarity('dog', 'morning'): -0.8592
```

<img src="https://debanjanaiml.github.io/assets/images/blogs/skipgram.png" width="500">{: .align-center}


**Conclusion from CBOW and Skipgram (Word2Vec) techniques**

While we had a deeper look into the CBOW and Skipgram techniques, we finally arrived at the following conclusion
* **Semantic Clustering**: The 2D plots visually demonstrated that words with related meanings (e.g., 'dog' and 'kitten', or 'coffee' and 'tea') were mapped to points that are close together in the vector space.

* **Quantifiable Similarity**: The Cosine Similarity scores validated this, showing high positive scores for related word pairs and low, near-zero scores for unrelated pairs (like 'dog' and 'morning').

* **Dimensionality Reduction**: The process successfully converted sparse, high-dimensional inputs (one-hot vectors) into dense, low-dimensional vectors (2D), without losing the underlying semantic relationships.

* **Enables Transfer Learning**: Pre-trained embeddings (like those from a large corpus) can be immediately used as features in a new task (e.g., classifying movie reviews), saving significant time and improving performance.


## GloVe (Global Vectors for Word Representation)

GloVe, developed at Stanford, is neither a purely count-based method (like TF-IDF) nor a purely predictive neural network method (like Word2Vec). It's a **count-based model with an optimization objective** that combines the efficiency of count statistics with the semantic encoding of distributed representations.

The core idea is: **Instead of predicting the next word, GloVe tries to predict the logarithm of the co-occurrence probability.**

**The Co-occurrence Matrix (X)**

GloVe starts by generating a massive **Word-Word Co-occurrence Matrix (X)**. 

* **Rows/Columns:** Each word in the vocabulary.
* **Entry $X_{ij}$:** The number of times word $i$ and word $j$ appear together within a specified window size in the entire corpus.

**The GloVe Objective Function**

GloVe trains vectors ($w_i$ and $\tilde{w}_j$) by minimizing this weighted least-squares objective function:

$$
J = \sum_{i=1}^{V} \sum_{j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2
$$

Where:
* $w_i$ and $\tilde{w}_j$ are the word vectors and context word vectors.
* $b_i$ and $\tilde{b}_j$ are bias terms.
* $X_{ij}$ is the co-occurrence count.
* $f(X_{ij})$ is a **weighting function** that gives less importance to very rare (low count) and very frequent (high count) co-occurrences.

### **Generating the Co-occurrence Matrix**

Let's generate the input data structure for GloVe (the Co-occurrence Matrix) using our toy corpus.

```python
# --- GloVe Data Generator ---
def build_co_occurrence_matrix(tokens, word_to_idx, vocab_size, window_size=1):
    """
    Builds the word-word co-occurrence matrix X.
    This matrix is the primary input for the GloVe model training.
    """
    X = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for i, center_word in enumerate(tokens):
        center_idx = word_to_idx[center_word]
        
        # Define context boundaries
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:
                context_word = tokens[j]
                context_idx = word_to_idx[context_word]
                
                # Increment the co-occurrence count
                # X[i, j] is the number of times word i is found near word j
                X[center_idx, context_idx] += 1
                
    return X
```

```python
# --- Run GloVe Data Generation Example ---
tokens, vocab, word_to_idx, idx_to_word = preprocess_and_build_vocab(corpus)
VOCAB_SIZE = len(vocab)
WINDOW_SIZE = 2

X = build_co_occurrence_matrix(tokens, word_to_idx, VOCAB_SIZE, window_size=WINDOW_SIZE)

print(f"Vocabulary Size (V): {VOCAB_SIZE}")
print(f"Co-occurrence Matrix Shape (V x V): {X.shape}")

# Print a small section of the matrix for inspection
# We'll select indices for 'dog' and 'cat' to see their relationship
words_to_check = ['dog', 'cat', 'runs', 'jumps']
indices_to_check = [word_to_idx[w] for w in words_to_check if w in word_to_idx]

if indices_to_check:
    print("\n--- Example Co-occurrence Matrix Entries ---")
    print(f"Words: {words_to_check}")
    
    # Create a small sub-matrix for display
    sub_matrix = X[np.ix_(indices_to_check, indices_to_check)]
    
    df_x = pd.DataFrame(
        sub_matrix, 
        index=[f"Center: {w}" for w in words_to_check], 
        columns=[f"Context: {w}" for w in words_to_check]
    )
    print(df_x)
```

```
Vocabulary Size (V): 28
Co-occurrence Matrix Shape (V x V): (28, 28)

--- Example Co-occurrence Matrix Entries ---
Words: ['dog', 'cat', 'runs', 'jumps']
               Context: dog  Context: cat  Context: runs  Context: jumps
Center: dog               0             0              2               0
Center: cat               0             0              0               1
Center: runs              2             0              0               0
Center: jumps             0             1              0               0
```

The GloVe methodology is a powerful fusion of count-based and predictive methods:

* **Global Context:** Unlike Word2Vec, which only sees local windows, GloVe uses the entire, comprehensive co-occurrence matrix.
* **Dimensionality Reduction:** The final embedding vectors are derived by mathematically factoring (decomposing) this matrix.
* **Increased Stability:** Factoring global statistics often leads to more stable and faster training convergence than relying on stochastic gradient descent (SGD) on local samples (Word2Vec).


## FastText: The Power of Sub-word Information

FastText (developed by Facebook's AI Research lab) is an extension of the Word2Vec model, but with one critical difference: **it represents a word as a bag of character n-grams (sub-words).**

**Why Sub-word Embeddings?**

1.  **Handling Out-of-Vocabulary (OOV) Words:** If a word like "unbelievable" was not in the training vocabulary, Word2Vec/GloVe cannot generate a vector for it. FastText, however, can calculate a vector for "unbelievable" by averaging the vectors of its constituent character n-grams (e.g., `<unb`, `nbe`, `bel`, ..., `ble>`, `ble`).
2.  **Morphology/Relatedness:** It naturally captures morphological similarities. "running" and "runs" share the sub-word `run`, so their vectors will be inherently closer than two completely unrelated words.
3.  **Better Embeddings for Rare Words:** Rare words often have insufficient data to train a stable Word2Vec vector, but they still share sub-words with common words, giving them a more robust embedding.

**Training FastText**

We will train a FastText model on our toy corpus and demonstrate its ability to guess the vector for a word it has *never* seen.

```python
# A larger, illustrative corpus for demonstration (clear semantic groups)
# This corpus contains the root 'run' but not the plural 'runs' (which we will test as OOV)
corpus_sentences = [
    "A dog runs fast",
    "A cat jumps high",
    "The brown dog quickly moves and plays",
    "The tiny kitten jumps up",
    "I drink hot coffee every morning",
    "She likes tea, not coffee",
    "He reads books and drinks tea",
    "The runner has a marathon", # Contains 'run' as sub-word
    "We need to speed up the process" # Contains 'speed'
]

# Gensim utility function to preprocess (tokenize, lowercase) the sentences
tokenized_corpus = [simple_preprocess(sentence) for sentence in corpus_sentences]

print(f"Tokenized Corpus Example (First sentence): {tokenized_corpus[0]}")
print(f"Total tokens for training: {sum(len(s) for s in tokenized_corpus)}")
```

```
Tokenized Corpus Example (First sentence): ['dog', 'runs', 'fast']
Total tokens for training: 45
```

```python
# --- Train FastText Model ---

# We use a small vector size and short character n-gram range for demonstration.
# min_n=3, max_n=6: FastText will consider character sequences from 3 to 6 characters long.
EMBEDDING_DIM = 10
MIN_N = 3
MAX_N = 6

model_fasttext = FastText(
    tokenized_corpus, 
    vector_size=EMBEDDING_DIM, 
    window=4, 
    min_count=1, 
    min_n=MIN_N, 
    max_n=MAX_N, 
    epochs=100
)

print(f"\nModel trained with character n-gram range: ({MIN_N}, {MAX_N})")
```

```python
# --- FastText Results: OOV and Similarity ---

# 7.1. Similarity for known words (e.g., 'coffee' vs 'tea')
similarity_known = model_fasttext.wv.similarity('coffee', 'tea')
print(f"\n7.1. Similarity for known words ('coffee', 'tea'): {similarity_known:.4f}")
```

```
7.1. Similarity for known words ('coffee', 'tea'): 0.1100
```

```python
# 2. Testing OOV Handling (The main feature!)
# We will test a word that did NOT appear in the corpus: 'running'

OOV_WORD = 'running'
KNOWN_RELATED_WORD = 'runner'

# Check if OOV_WORD is actually in the vocabulary (it should not be if min_count=1)
if OOV_WORD not in model_fasttext.wv.key_to_index:
    print(f"\n7.2. Testing OOV: The word '{OOV_WORD}' is NOT in the vocabulary, but its vector can still be generated.")
    
    # Get the vector for the OOV word
    oov_vector = model_fasttext.wv[OOV_WORD]
    print(f"   Vector for OOV word '{OOV_WORD}' (first 5 dims): {oov_vector[:5]}")

    # Check the similarity between the OOV word and a known related word
    similarity_oov = model_fasttext.wv.similarity(OOV_WORD, KNOWN_RELATED_WORD)
    print(f"   Similarity('{OOV_WORD}', '{KNOWN_RELATED_WORD}'): {similarity_oov:.4f}")
    
    # Check the similarity between the OOV word and an unrelated word
    similarity_unrelated = model_fasttext.wv.similarity(OOV_WORD, 'morning')
    print(f"   Similarity('{OOV_WORD}', 'morning'): {similarity_unrelated:.4f}")
```

```
7.2. Testing OOV: The word 'running' is NOT in the vocabulary, but its vector can still be generated.
   Vector for OOV word 'running' (first 5 dims): [-0.00965883 -0.00111312  0.01884038  0.01726348 -0.00672889]
   Similarity('running', 'runner'): 0.4531
   Similarity('running', 'morning'): 0.0951
```

```python
# 7.3. Finding nearest neighbors for the OOV word
print(f"\n7.3. Nearest neighbors for the OOV word '{OOV_WORD}':")
# This demonstrates that the vector for 'running' is meaningful and relates to other 'motion' words
neighbors = model_fasttext.wv.most_similar(OOV_WORD, topn=3)
for word, sim in neighbors:
    print(f"   - {word} (Similarity: {sim:.4f})")
```

```
7.3. Nearest neighbors for the OOV word 'running':
   - she (Similarity: 0.5078)
   - need (Similarity: 0.4691)
   - runner (Similarity: 0.4531)
```

FastText solves the limitations of Word2Vec and GloVe by incorporating sub-word information. 
The key takeaway is its ability to **generate a vector for any word, regardless of whether it was seen during training**, provided it is composed of known character n-grams. This makes it highly robust for datasets with specialized terminology, misspellings, or highly inflected languages.

# Applications

Now that we have covered almost all the commonly used static embedding techniques, and given their drawbacks, the prime of which includes static embeddings have fixed vectors for a given word, i.e, the vectors don't change with relation to the context surrounding it, one might wonder what is the actual use-case for these techniques given we can use contextual embeddings which can capture more semantic meaning for a given context. Well, that is true that the contextual embeddings are far superior to static ones, however, static embeddings find their use in the following cases:

1. **Extreme Latency Constraints (Real-Time Systems)**: Contextual models (BERT) are too slow, often requiring specialized hardware (GPUs/TPUs) for inference. Static embeddings (like Hashing Vectorizer, TF-IDF) convert text to vectors instantly (especially Hashing or pre-computed TF-IDF) on a CPU, which is crucial for high-throughput, low-latency APIs. 
2. **Cold Start Problem in Recommenders**: In collaborative filtering, you often need to represent new, unseen items (e.g., a news article with text) quickly. You can compute a FastText vector on the fly, as it doesn't need to be in the vocabulary, and use that vector immediately to seed recommendations.
3. **Memory/Storage Constraints (Edge Devices)**: Full contextual models weigh hundreds of megabytes or even gigabytes. Static embeddings (Word2Vec, GloVe) often result in model files (vocabulary and weights) that are only a few megabytes in size. This is essential for applications running on mobile devices or IoT platforms.
4. **Feature Engineering for Classic ML**: If you are using traditional machine learning models (e.g., Random Forest, Gradient Boosting) which do not handle sequence data, you must use a fixed-length feature vector. Averaged static embeddings (like an average Word2Vec vector per document) provide the perfect, efficient input features.
5. **Languages with High Inflection (Morphology)**: For languages like Turkish, Finnish, or Arabic, words can have thousands of forms (e.g., conjugation). FastText's sub-word mechanism natively handles this complexity and provides a robust vector for any new inflection, performing better than a context-agnostic Word2Vec model trained on those languages.FastText
6. **Simple Topic Modeling & Clustering**: For quick, unsupervised tasks where general document similarity is needed (e.g., clustering news articles), TF-IDF provides highly interpretable feature weights. You can easily inspect the TF-IDF features to understand what defines each cluster, which is difficult with dense, abstract contextual vectors.TF-IDF
7. **Search bar applications**: It has been noted that if a search query takes longer times to load the results, customer often leave the app, causing something known as dropoff. In such cases you need a lightweight embedding technique which can quickly convert the search query to embeddings, perform the similarity search and yeild the top-k matching results quickly.


# Conclusion

We've covered the full evolution of static embeddings! We started with simple Counts and TF-IDF, learned how to scale up with the Hashing Trick, and then made the jump to dense semantic representations with Word2Vec and GloVe. Finally, we saw how FastText makes these models robust by understanding sub-words.

These methods are the backbone of classical NLP. Now that you understand how they capture meaning, you're ready for the modern era of contextual models!

If you found this series useful, please consider giving it a like, sharing it with a friend, and subscribing for more deep dives into machine learning! Happy coding!