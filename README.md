# TETRA: TExtual TRust Analyzer
This library provides tools for analyzing the trust between two people in a conversation analyzing their messages. The library focuses on evaluating manner, similarity, and quantity for each text. It leverages various NLP libraries and APIs, including Google's Perspective API, NLTK, Presidio, and Sentence Transformers.

### Features

- **Manner Analysis**: Evaluates the manner of text using the Perspective API, checking for harmful content indicators such as toxicity, profanity, threats, etc.
- **Similarity Analysis**: Computes the cosine similarity between two pieces of text using a pre-trained sentence transformer model.
- **Quantity Analysis**: Measures the informational density of text, taking into account named entities and coherence with another piece of text.

