import nltk
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from nltk.corpus import stopwords

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
stop_words = set(stopwords.words('english'))

operators = {
    "DATE_TIME": OperatorConfig("replace", {"new_value": "DATE_TIME"}),
    "NRP": OperatorConfig("replace", {"new_value": "NRP"}),
    "LOCATION": OperatorConfig("replace", {"new_value": "LOCATION"}),
    "PERSON": OperatorConfig("replace", {"new_value": "PERSON"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "PHONE_NUMBER"})
}

def extract_named_entities(text):
    results = analyzer.analyze(text, entities=['DATE_TIME', 'NRP', 'LOCATION', 'PERSON', 'PHONE_NUMBER'], language='en')
    return [result.entity_type for result in results], anonymizer.anonymize(text, analyzer_results=results, operators=operators).text

def cohere_who(text, has_person_entity):
    contains_who = 'who' in text.lower()
    return 1 if contains_who and 'PERSON' in has_person_entity else 0

def cohere_where(text, has_location_entity):
    contains_where = 'where' in text.lower()
    return 1 if contains_where and 'LOCATION' in has_location_entity else 0

def cohere_when(text, has_datetime_entity):
    contains_when = 'when' in text.lower()
    return 1 if contains_when and 'DATE_TIME' in has_datetime_entity else 0

def get_quantity_opt(text_B, text_A=None):
    
    entities_B, text_B_anonymized = extract_named_entities(text_B)
    n_entities_B = len(entities_B)

    tokens = nltk.word_tokenize(text_B_anonymized.lower())
    content_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    words = len(content_tokens)

    if n_entities_B != 0:
        boost = n_entities_B / len(content_tokens) if content_tokens else 0
        words += (boost * n_entities_B)

    if text_A is not None:
        boost_cohere = sum([cohere_who(text_A, entities_B), cohere_where(text_A, entities_B), 
                   cohere_when(text_A, entities_B)]) / len(content_tokens) if content_tokens else 0
        words += boost_cohere

    information_density = min(words / len(tokens), 1) if tokens else 0
    
    return information_density
