from typing import Any, Dict, List, Union, Optional

import pandas as pd
import numpy as np
import matplotlib_inline as plt
import seaborn as sns
import datasets
import nltk

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

analyzer = AnalyzerEngine()
anonimyzer = AnonymizerEngine()



def extract_named_entities(text):
    results = analyzer.analyze(text,entities=['DATE_TIME', 'NRP', 'LOCATION', 'PERSON', 'PHONE_NUMBER'], language='en')
    results = [result.entity_type for result in results]
    return results

def anonymized_text(text):
    results = analyzer.analyze(text,entities=['DATE_TIME', 'NRP', 'LOCATION', 'PERSON', 'PHONE_NUMBER'], language='en')

    operators = {
    "DATE_TIME": OperatorConfig("replace", {"new_value": "DATE_TIME"}),
    "NRP": OperatorConfig("replace",{"new_value": "NRP"},),
    "LOCATION": OperatorConfig("replace", {"new_value": "LOCATION"}),
    "PERSON": OperatorConfig("replace", {"new_value": "PERSON"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "PHONE_NUMBER"})}
    results = anonimyzer.anonymize(text, analyzer_results=results, operators=operators)

    return results.text



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

def calculate_information_density(n_entities, text_anonymized, cohere_who, cohere_when, cohere_where):
    response_tokens = nltk.word_tokenize(text_anonymized.lower())
    stop_words = set(stopwords.words('english'))
    content_tokens = [token for token in response_tokens if token.isalnum() and token not in stop_words]

    words = 0
    boost_cohere = 0
    for word in content_tokens:
            words += 1

    if n_entities != 0:

        boost = n_entities / len(content_tokens) if len(content_tokens) > 0 else 0
    
        if ((cohere_who == 1) or (cohere_where == 1) or (cohere_when == 1)):
            boost_cohere = 1 / len(content_tokens) if len(content_tokens) > 0 else 0 #faccio sempre così anche se c'è più di una NER che soddisfa la domanda. A me quello che interessa è che ce ne sia almeno una.
            boost += boost_cohere
        
        
        words += boost * n_entities + boost_cohere

    information_density = words / len(response_tokens) if len(response_tokens) > 0 else 0

    return information_density

#def calculate_information_density(response):
#    response_tokens = nltk.word_tokenize(response.lower())
#    stop_words = set(stopwords.words('english'))
#    content_tokens = [token for token in response_tokens if token.isalnum() and token not in stop_words]
#
#    information_density = len(content_tokens) / len(response_tokens) if len(response_tokens) > 0 else 0
#
#    return information_density

def cohere_who(text, has_person_entity):
    contains_when = 'Who' in text.lower()
    
    if contains_when and has_person_entity:
        return 1
    elif contains_when and not has_person_entity:
        return -1
    elif not contains_when and not has_person_entity:
        return 0
    elif not contains_when and has_person_entity:
        return 0
    

def cohere_where(text, has_location_entity):
    contains_when = 'Where' in text.lower()
    
    if contains_when and has_location_entity:
        return 1
    elif contains_when and not has_location_entity:
        return -1
    elif not contains_when and not has_location_entity:
        return 0
    elif not contains_when and has_location_entity:
        return 0
    


def cohere_when(text, has_datetime_entity):
    contains_when = 'When' in text.lower()
    
    if contains_when and has_datetime_entity:
        return 1
    elif contains_when and not has_datetime_entity:
        return -1
    elif not contains_when and not has_datetime_entity:
        return 0
    elif not contains_when and has_datetime_entity:
        return 0
    



def get_quantity(text_A, text_B = None):
    if text_B == None:
        n_entities_A = extract_named_entities(text_A)
        text_A_anonymized = anonymized_text(text_A)

    
        tokens = nltk.word_tokenize(text_A_anonymized.lower())
        stop_words = set(stopwords.words('english'))
        content_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

        words = 0
        for word in content_tokens:
                words += 1

        if n_entities_A != 0:
                boost = len(n_entities_A) / len(content_tokens) if len(content_tokens) > 0 else 0
                words += boost * n_entities_A

        information_density = words / len(tokens) if len(tokens) > 0 else 0

        return information_density
    
    else:
        n_entities_B = extract_named_entities(text_B)
        text_B_anonymized = anonymized_text(text_B)

        response_tokens = nltk.word_tokenize(text_B_anonymized.lower())
        stop_words = set(stopwords.words('english'))
        content_tokens = [token for token in response_tokens if token.isalnum() and token not in stop_words]

        words = 0
        for word in content_tokens:
                words += 1

        boost_cohere=0
        if n_entities_B != 0:
            boost = len(n_entities_B) / len(content_tokens) if len(content_tokens) > 0 else 0
            if ((cohere_who(text_A, n_entities_B) == 1) or (cohere_where(text_A, n_entities_B) == 1) or (cohere_when(text_A, n_entities_B) == 1)):
                 boost_cohere = 1 / len(content_tokens) if len(content_tokens) > 0 else 0 #faccio sempre così anche se c'è più di una NER che soddisfa la domanda. A me quello che interessa è che ce ne sia almeno una.
            boost += boost_cohere
        
        
        words += boost * len(n_entities_B) + boost_cohere

        information_density = words / len(response_tokens) if len(response_tokens) > 0 else 0

        return information_density

