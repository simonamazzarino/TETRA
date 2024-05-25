from googleapiclient import discovery
from sentence_transformers import SentenceTransformer, util
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from nltk.corpus import stopwords

import nltk



### Manner
def perspective(text, client, features = ['TOXICITY', 'SEVERE_TOXICITY', 'INSULT', 
          'SEXUALLY_EXPLICIT', 'PROFANITY', 'LIKELY_TO_REJECT', 'THREAT']):
    
    
    analyze_request = {
      'comment': { 'text': text},
      "languages":["en"],
      'requestedAttributes': {}
    }
    
    for f in features:
        analyze_request['requestedAttributes'][f] = {}
    
    scores = client.comments().analyze(body=analyze_request).execute()
    return scores

def get_score(scores, feat):
    return scores['attributeScores'][feat]['summaryScore']['value']

def get_manner(text, client, f='SEVERE_TOXICITY'):
    values = perspective(text, client, [f])
    score = get_score(values, f)
    return score 

### Similarity
def get_sim(model, comment, reply_to):
    embeddings1 = model.encode([comment], convert_to_tensor=True)
    embeddings2 = model.encode([reply_to], convert_to_tensor=True) 

    return float(util.cos_sim(embeddings1, embeddings2))

### Quantity

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


### Trust Class

class Trust:
    def __init__(self, API_KEY=None):

        self.API_KEY = API_KEY
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_model.max_seq_length = 512


        if API_KEY is not None:
            self.perspective_client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        
    def get_trust(self, comment, reply_to=None, mode='SEVERE_TOXICITY'):

        if reply_to is not None:
            similarity = get_sim(self.similarity_model, comment, reply_to)
        else:
            similarity = None

        quantity = get_quantity_opt(comment, reply_to)

        if self.API_KEY is not None:
            manner = 1 - get_manner(comment, self.perspective_client, mode)
        else: 
            manner = None
            
        return [similarity, quantity, manner]
        
    
