from googleapiclient import discovery
from sentence_transformers import SentenceTransformer, util


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

def get_quantity(comment, reply_to):
    return 1

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

        quantity = get_quantity(comment, reply_to)

        if self.API_KEY is not None:
            manner = 1 - get_manner(comment, self.perspective_client, mode)
        else: 
            manner = None

        return [similarity, quantity, manner]
        
    
