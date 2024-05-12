# Introduction

Have you ever tried ChatGPT and been amazed by its capabilities?

I think what makes it super cool is being able to not just generate text, but also to have a conversation with it. It remembers what you said and can respond in context. That's what I think makes it super helpful.

In this post, I wanna dive into what Large Language Models (LLMs) are and how they evolved over time.

# What is natural language processing (NLP)?

Natural language processing (NLP) is a part of artificial intelligence that helps computers interact with humans using everyday language. NLP allows computers to understand, interpret, and create human language.

# What is a vector?

A vector is a mathematical object that has both a size, called [magnitude](https://mathinsight.org/definition/magnitude_vector) and a [direction](https://www.cuemath.com/direction-of-a-vector-formula/). It is used in different fields like physics, computer science, and machine learning.

In machine learning, a vector is a one-dimensional array of numbers. For instance, `[1, 2, 3]` is a vector of length 3.

Vectors are important in machine learning for representing data. For example, in natural language processing, words are represented as vectors. These word vectors help capture the meanings of words.

```python
# A vector with 4 elements
my_vector = [2.5, 3.1, 0.8, 1.2]

# Accessing elements by index
print(my_vector[0]) # 2.5
print(my_vector[2]) # 0.8
```

# What is a sparse vector?

A sparse vector is a type of vector where most of the elements are zero. It's used to represent data with many dimensions in a way that saves memory and computing power. However, sparse vectors don't hold as much information as dense vectors.

```python
# A sparse vector with 10 elements, mostly zeros
my_sparse_vector = [0, 0, 0, 2.5, 0, 0, 0, 0.8, 0, 0]

# Accessing non-zero elements by index
print(my_sparse_vector[3]) # 2.5
print(my_sparse_vector[7]) # 0.8
```

The reason they hold less information is because they contain many zeros. For instance, if you have a sparse vector with 1000 elements and only 10 are non-zero, you're missing a lot of information.

# What is a dense vector?

A dense vector is a vector with many non-zero elements. This means most of its elements are non-zero. Dense vectors hold more information than sparse vectors, but they also use more memory and computational resources.

```python
# A dense vector with 10 elements, mostly non-zeros
my_dense_vector = [2.5, 3.4, 1.1, 4.6, 0.8, 5.3, 2.2, 1.7, 1.2, 0.9]

# Accessing non-zero elements by index
print(my_dense_vector[0]) # 2.5
print(my_dense_vector[8]) # 1.2
```

The reason they require more memory and computation is that you need to store and process all the non-zero elements.

# N-Grams

N-Grams are the simplest form of language models. An n-gram is a sequence of n words. Common types of n-grams:

- Unigram: Single word
- Bigram: Two words
- Trigram: Three words

N-grams capture the local context of words but fail to capture the global context.

For example, the bigrams of the sentence "The quick brown fox" are: "The quick", "quick brown", and "brown fox".

As you can see, bigrams don't capture the global context of the sentence. That's why we needed more progress in language models.

## Limitations of N-Grams

- Can't capture global context.
- Lack of semantic meaning. Words with similar meanings have different n-grams.
- High-dimensional sparse vectors, which in simpler terms means that the vectors are very large and have a lot of zeros.

# Word Embeddings

Word embeddings are more advanced than n-grams. They aim to learn dense, low-dimensional representations of words. Word embeddings understand the meanings of words. For instance, "king" and "queen" will have similar embeddings.

The most popular word embedding model is Word2Vec.

```python
# Example word2vec embeddings
king = [0.2, 0.5, 0.9, 0.1, ...]
queen = [0.3, 0.6, 0.8, 0.2, ...]
man = [0.4, 0.3, 0.7, 0.3, ...]
woman = [0.5, 0.4, 0.6, 0.4, ...]

# Arithmetic on word vectors can reveal relationships
king - man + woman ≈ queen
```

## Limitations of Word Embeddings

- Each word has a fixed-size vector representation. This means that the model can't learn different meanings of the same word because it has only one vector for each word.
- Word embeddings are static. They don't change based on the context of the sentence.
- They don't capture the order of words in a sentence.

# Neutral Networks

## What are they?

Neural networks are a kind of machine learning algorithm modeled after the human brain. They are made up of linked nodes, called "neurons," arranged in layers:

- An input layer that takes in data.
- One or more hidden layers that handle the data.
- An output layer that delivers the final outcome.

## What is a weight?

In neural networks, a "weight" is a number linked to the connection between two neurons. It shows how strong the influence of one neuron is on another.

During training, the network changes these weights to reduce errors and get better at making predictions. Weights are important for the network to learn from the data.

## GPUs changed it all?

Neural networks have existed for a long time, but they became more mainstream with the rise of GPUs.

Training neural networks is computationally intensive. GPUs are much better than CPUs at handling the [matrix](<https://en.wikipedia.org/wiki/Matrix_(mathematics)>) operations that neural networks require.

Why GPUs are better for neural networks:

- GPUs have thousands of cores, while CPUs have only a few.
- GPUs are designed for parallel processing. This means they can handle many tasks at once, which is good for training neural networks because they involve many matrix operations.
- GPUs can handle large amounts of data more efficiently than CPUs.

# Before Transformers: RNNs and LSTMs

## Recurrent Neural Networks (RNNs)

[RNNs (Recurrent Neural Networks)](https://en.wikipedia.org/wiki/Recurrent_neural_network) are a type of neural network made for dealing with data that comes in a sequence, like text or speech. They have a "memory" that lets them remember past information as they process new data in the sequence.

## Long Short-Term Memory (LSTM)

[LSTMs (Long Short-Term Memory)](https://en.wikipedia.org/wiki/Long_short-term_memory) are a type of RNN that can remember information for long periods. They're implemented in a way (via gates) that lets them selectively remember or forget information.

## Limitations of RNNs and LSTMs

- RNNs and LSTMs struggle with long sequences.
- RNNs and LSTMs process data one step at a time, which is slow and inefficient.
- LSTMs have better memory than RNNs but are still limited.
- LSTMs are hard to train and require a lot of computational resources.

# The Transformer Revolution

Did you know the T in GPT stands for "Transformer"?

GPT stands for "Generative Pre-trained Transformer." The Transformer model revolutionized natural language processing. It's based on the "attention" mechanism, which lets it focus on different parts of the input text.

Since I'm taking notes for myself, I'm gonna keep it simple:

- Transformers can process data in parallel, making them faster than RNNs and LSTMs.
- Transformers are more effective at capturing long-range dependencies in data.
- Transformers use multi-head attention which allows them to focus on different parts of the input text at the same time.

# What is an LLM?

Let's summarize what an LLM is:

- LLMs are language models that use deep learning to understand and generate human language.
- They're trained on large amounts of text data to learn the patterns and structures of language.
- They're based on the Transformer architecture, which allows them to process data in parallel and capture long-range dependencies.
- LLMs can generate human-like text, answer questions, and perform other language-related tasks.

# Current limitations

Despite how powerful LLMs are, they have several limitations:

- **Hallucinations**: LLMs can generate text that's not factually accurate.
- **Bias**: LLMs could generate biased text based on the data they were trained on.
- **Computational Limitations**: Training LLMs require a lot of computational power and resources.
