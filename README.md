# TETRA: TExtual TRust Analyzer
This library provides tools for analyzing the trust between two people in a conversation analyzing their messages. The library focuses on evaluating manner, similarity, and quantity for each text. It leverages various NLP libraries and APIs, including Google's Perspective API, Detoxify, NLTK, Presidio, and Sentence Transformers.

### Features

- **Manner Analysis**: Evaluates the manner of text using either Detoxify or Google's Perspective API, checking for harmful content indicators such as toxicity, profanity, threats, etc.
- **Similarity Analysis**: Computes the cosine similarity between two pieces of text using a pre-trained sentence transformer model.
- **Quantity Analysis**: Measures the informational density of text, taking into account named entities and coherence with another piece of text.

## Installation
You can install TETRA by using pip: 

```python
pip install tetra-textual-trust-analyzer
```

## Quickstart
You can import the Trust class using
```python
from tetra.trust_main import Trust
```
Then, you create a Trust object and set your Perspective API key. If you don't already have a Perspective API key, you can visit this [link](https://perspectiveapi.com/) and create your own key. Otherwise, Tetra will default to Detoxify.

```python
trust_analyzer = Trust('<insert-your-Perspective-API-key>')
```
To obtain the trust scores between two sentences you need to use the method ```get_trust()``` and provide the sentences as parameters in the following way:

```python
scores = trust_analyzer.get_trust(
    "This is the main comment, of which we are analyzing the trust.",
    "And this is the comment it is responding to. For example, it could be a question.")
```

This method returns as output the trust scores, i.e. manner, similarity and quantity scores.

```python
for maxim, score in zip(['Similarity', 'Quantity', 'Manner'], scores):
    print (maxim, score)
```

## Examples

You can find a notebook example in the [notebook](https://github.com/simonamazzarino/TETRA/tree/main/example_notebook) folder. 




