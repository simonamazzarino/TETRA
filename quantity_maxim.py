import pandas as pd
import numpy as np
import matplotlib_inline as plt
import seaborn as sns
import datasets
import nltk

from presidio_analyzer import AnalyzerEngine
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

analyzer = AnalyzerEngine()

def extract_named_entities(text):
    results = analyzer.analyze(text,entities=['DATE_TIME', 'NRP', 'LOCATION', 'PERSON', 'PHONE_NUMBER'], language='en')
    results = [result.entity_type for result in results]
    return results

def count_words(text):
    words = word_tokenize(text)
    return len(words)

def count_elements(lst):
    return len(lst)

def calculate_overlap_metrics(question, response):
    question_tokens = set(token.lower() for token in nltk.word_tokenize(question) if token.lower() not in stopwords.words('english'))
    response_tokens = set(token.lower() for token in nltk.word_tokenize(response) if token.lower() not in stopwords.words('english'))

    overlap_words = len(question_tokens.intersection(response_tokens))

    return overlap_words

def calculate_information_density(response):
    response_tokens = nltk.word_tokenize(response.lower())
    stop_words = set(stopwords.words('english'))
    content_tokens = [token for token in response_tokens if token.isalnum() and token not in stop_words]

    information_density = len(content_tokens) / len(response_tokens) if len(response_tokens) > 0 else 0

    return information_density
