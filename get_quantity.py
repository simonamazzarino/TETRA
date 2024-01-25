
import nltk

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


def extract_named_entities(text):
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text,entities=['DATE_TIME', 'NRP', 'LOCATION', 'PERSON', 'PHONE_NUMBER'], language='en')
    results = [result.entity_type for result in results]
    return results

def anonymized_text(text):
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    results = analyzer.analyze(text,entities=['DATE_TIME', 'NRP', 'LOCATION', 'PERSON', 'PHONE_NUMBER'], language='en')

    operators = {
    "DATE_TIME": OperatorConfig("replace", {"new_value": "DATE_TIME"}),
    "NRP": OperatorConfig("replace",{"new_value": "NRP"},),
    "LOCATION": OperatorConfig("replace", {"new_value": "LOCATION"}),
    "PERSON": OperatorConfig("replace", {"new_value": "PERSON"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "PHONE_NUMBER"})}
    results = anonymizer.anonymize(text, analyzer_results=results, operators=operators)

    return results.text

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
    

#text_B origine
#text_A
def get_quantity(text_B, text_A = None):
    entities_B = extract_named_entities(text_B)
    n_entities_B = len(entities_B)
    text_B_anonymized = anonymized_text(text_B)


    tokens = nltk.word_tokenize(text_B_anonymized.lower())
    stop_words = set(stopwords.words('english'))
    content_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    words = 0
    for word in content_tokens:
        words += 1
    
    if n_entities_B != 0:
            boost = n_entities_B / len(content_tokens) if len(content_tokens) > 0 else 0
            words += boost * n_entities_B

    


    if text_A is not None:
        entities_A = extract_named_entities(text_A)
        n_entities_A = len(entities_A)
        text_A_anonymized = anonymized_text(text_A)

        tokens = nltk.word_tokenize(text_A_anonymized.lower())
        stop_words = set(stopwords.words('english'))
        content_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

        words = 0
        for word in content_tokens:
            words += 1

        if n_entities_A != 0:
            boost = n_entities_A / len(content_tokens) if len(content_tokens) > 0 else 0
            if ((cohere_who(text_B, entities_A) == 1) or (cohere_where(text_B, entities_A) == 1) or (cohere_when(text_B, entities_A) == 1)):
                boost_cohere = 1 / len(content_tokens) if len(content_tokens) > 0 else 0 #faccio sempre così anche se c'è più di una NER che soddisfa la domanda. A me quello che interessa è che ce ne sia almeno una.
                boost += boost_cohere
        
        
            words += boost * n_entities_A + boost_cohere

        information_density = words / len(tokens) if len(tokens) > 0 else 0

        return information_density
    
    else:
        entities_B = extract_named_entities(text_B)
        n_entities_B = len(entities_B)
        text_B_anonymized = anonymized_text(text_B)


        tokens = nltk.word_tokenize(text_B_anonymized.lower())
        stop_words = set(stopwords.words('english'))
        content_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

        words = 0
        for word in content_tokens:
            words += 1

        if n_entities_A != 0:
            boost = n_entities_A / len(content_tokens) if len(content_tokens) > 0 else 0
            words += boost * n_entities_A

        information_density = words / len(tokens) if len(tokens) > 0 else 0

        return information_density
        
    