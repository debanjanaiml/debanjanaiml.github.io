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

#### 2.5 sklearn TF-IDF (Baseline)

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
