from typing import Any, Dict, List, Optional, Tuple

from googleapiclient import discovery

import nltk
from nltk.corpus import stopwords

from presidio_analyzer import AnalyzerEngine

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from sentence_transformers import SentenceTransformer, util


# Manner
def perspective(
    text: str,
    client: Any,
    features: List[str] = [
        "TOXICITY",
        "SEVERE_TOXICITY",
        "INSULT",
        "SEXUALLY_EXPLICIT",
        "PROFANITY",
        "LIKELY_TO_REJECT",
        "THREAT",
    ],
) -> Dict[str, Any]:
    """
    Analyzes the given text using the Perspective API and returns the scores for the specified features.

    Args:
        text (str): The text to be analyzed.
        client (Any): The Perspective API client.
        features (List[str]): A list of features to be analyzed. Defaults to common harmful content indicators.

    Returns:
        Dict[str, Any]: The response from the Perspective API containing scores for each requested feature.
    """

    analyze_request = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {},
    }

    for f in features:
        analyze_request["requestedAttributes"][f] = {}
    scores = client.comments().analyze(body=analyze_request).execute()
    return scores


def get_score(scores: Dict[str, Any], feat: str) -> float:
    """
    Retrieves the summary score for a specific feature from the Perspective API response.

    Args:
        scores (Dict[str, Any]): The response from the Perspective API containing scores for various features.
        feat (str): The feature for which to retrieve the summary score.

    Returns:
        float: The summary score for the specified feature.
    """
    return scores["attributeScores"][feat]["summaryScore"]["value"]


def get_manner(text: str, client: Any, f: str = "SEVERE_TOXICITY") -> float:
    """
    Analyzes the given text using the Perspective API and returns the score for a specific feature.

    Args:
        text (str): The text to be analyzed.
        client (Any): The Perspective API client.
        f (str): The feature to be analyzed. Defaults to "SEVERE_TOXICITY".

    Returns:
        float: The score for the specified feature.
    """
    values = perspective(text, client, [f])
    score = get_score(values, f)
    return score


# Similarity
def get_sim(model: SentenceTransformer, comment: str, reply_to: str) -> float:
    """
    Computes the cosine similarity between the embeddings of two text inputs using a given model.

    Args:
        model (SentenceTransformer): The SentenceTransformer model used to encode the text.
        comment (str): The first text input to be compared.
        reply_to (str): The second text input to be compared.

    Returns:
        float: The cosine similarity score between the two text embeddings.
    """
    embeddings1 = model.encode([comment], convert_to_tensor=True)
    embeddings2 = model.encode([reply_to], convert_to_tensor=True)

    return (float(util.cos_sim(embeddings1, embeddings2)) + 1)/2


# Quantity

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
stop_words = set(stopwords.words("english"))

operators = {
    "DATE_TIME": OperatorConfig("replace", {"new_value": "DATE_TIME"}),
    "NRP": OperatorConfig("replace", {"new_value": "NRP"}),
    "LOCATION": OperatorConfig("replace", {"new_value": "LOCATION"}),
    "PERSON": OperatorConfig("replace", {"new_value": "PERSON"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "PHONE_NUMBER"}),
}


def extract_named_entities(text: str) -> Tuple[List[str], str]:
    """
    Extracts named entities from the given text and anonymizes them.

    Args:
        text (str): The text to be analyzed.

    Returns:
        Tuple[List[str], str]: A tuple containing a list of identified entity types and the anonymized text.
    """
    results = analyzer.analyze(
        text,
        entities=["DATE_TIME", "NRP", "LOCATION", "PERSON", "PHONE_NUMBER"],
        language="en",
    )
    return [result.entity_type for result in results], anonymizer.anonymize(
        text, analyzer_results=results, operators=operators
    ).text


def cohere_who(text: str, has_person_entity: List[str]) -> int:
    """
    Determines if the text contains the word 'who' and has a 'PERSON' entity.

    Args:
        text (str): The text to be analyzed.
        has_person_entity (List[str]): A list of entity types identified in the text.

    Returns:
        int: Returns 1 if the text contains 'who' and has a 'PERSON' entity, otherwise returns 0.
    """
    contains_who = "who" in text.lower()
    return 1 if contains_who and "PERSON" in has_person_entity else 0


def cohere_where(text: str, has_location_entity: List[str]) -> int:
    """
    Determines if the text contains the word 'where' and has a 'LOCATION' entity.

    Args:
        text (str): The text to be analyzed.
        has_location_entity (List[str]): A list of entity types identified in the text.

    Returns:
        int: Returns 1 if the text contains 'where' and has a 'LOCATION' entity, otherwise returns 0.
    """
    contains_where = "where" in text.lower()
    return 1 if contains_where and "LOCATION" in has_location_entity else 0


def cohere_when(text: str, has_datetime_entity: List[str]) -> int:
    """
    Determines if the text contains the word 'when' and has a 'DATE_TIME' entity.

    Args:
        text (str): The text to be analyzed.
        has_datetime_entity (List[str]): A list of entity types identified in the text.

    Returns:
        int: Returns 1 if the text contains 'when' and has a 'DATE_TIME' entity, otherwise returns 0.
    """
    contains_when = "when" in text.lower()
    return 1 if contains_when and "DATE_TIME" in has_datetime_entity else 0


def get_quantity_opt(text_B: str, text_A: Optional[str] = None) -> float:
    """
    Calculates the information density of the given text, optionally boosting based on coherence with another text.

    Args:
        text_B (str): The main text to be analyzed.
        text_A (Optional[str]): An optional text to check for coherence boosts.

    Returns:
        float: The calculated information density.
    """
    entities_B, text_B_anonymized = extract_named_entities(text_B)
    n_entities_B = len(entities_B)

    tokens = nltk.word_tokenize(text_B_anonymized.lower())
    content_tokens = [
        token for token in tokens if token.isalnum() and token not in stop_words
    ]

    words = len(content_tokens)

    if n_entities_B != 0:
        boost = n_entities_B / len(content_tokens) if content_tokens else 0
        words += boost * n_entities_B

    if text_A is not None:
        boost_cohere = (
            sum(
                [
                    cohere_who(text_A, entities_B),
                    cohere_where(text_A, entities_B),
                    cohere_when(text_A, entities_B),
                ]
            )
            / len(content_tokens)
            if content_tokens
            else 0
        )
        words += boost_cohere

    information_density = min(words / len(tokens), 1) if tokens else 0

    return information_density


# Trust Class


class Trust:
    def __init__(self, API_KEY: Optional[str] = None) -> None:
        """
        Initializes the Trust class with an optional API key for Perspective API and a pre-trained sentence transformer model.

        Args:
            API_KEY (Optional[str]): The API key for accessing the Perspective API. If not provided, manner analysis will be disabled.
        """
        self.API_KEY = API_KEY
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.similarity_model.max_seq_length = 512

        if API_KEY is not None:
            self.perspective_client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=API_KEY,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )

    def get_trust(self, comment: str, reply_to: Optional[str] = None, mode: str = "SEVERE_TOXICITY") -> List[Optional[float]]:
        """
        Evaluates the trustworthiness of a comment by analyzing its similarity to a reply, its quantity of informative content, and its manner using Perspective API.

        Args:
            comment (str): The comment text to be analyzed.
            reply_to (Optional[str]): An optional text to which the comment is a reply. If provided, similarity will be calculated.
            mode (str): The mode for Perspective API analysis. Defaults to "SEVERE_TOXICITY".

        Returns:
            List[Optional[float]]: A list containing similarity score, quantity score, and manner score.
        """
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
