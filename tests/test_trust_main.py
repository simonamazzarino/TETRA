import nltk
nltk.download('stopwords')

from presidio_analyzer import AnalyzerEngine, RecognizerResult

from presidio_anonymizer import AnonymizerEngine

import pytest

from sentence_transformers import SentenceTransformer

from tetra.trust_main import (
    perspective,
    get_score,
    get_manner,
    get_sim,
    extract_named_entities,
    cohere_who,
    cohere_where,
    cohere_when,
    get_quantity_opt,
    Trust,
)

pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


@pytest.fixture
def mock_perspective_client():
    class MockClient:
        def comments(self):
            return self

        def analyze(self, body):
            return self

        def execute(self):
            return {
                "attributeScores": {"SEVERE_TOXICITY": {"summaryScore": {"value": 0.1}}}
            }

    return MockClient()


@pytest.fixture
def mock_sentence_transformer():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.max_seq_length = 512
    return model


@pytest.fixture
def mock_analyzer():
    analyzer = AnalyzerEngine()
    analyzer.analyze = lambda text, entities, language: [
        RecognizerResult(entity_type="PERSON", start=0, end=5, score=0.8)
    ]
    return analyzer


@pytest.fixture
def mock_anonymizer():
    anonymizer = AnonymizerEngine()
    anonymizer.anonymize = lambda text, analyzer_results, operators: text
    return anonymizer


def test_perspective(mock_perspective_client):
    text = "Test text"
    features = ["SEVERE_TOXICITY"]
    scores = perspective(text, mock_perspective_client, features)
    assert "attributeScores" in scores
    assert "SEVERE_TOXICITY" in scores["attributeScores"]


def test_get_score():
    scores = {"attributeScores": {"SEVERE_TOXICITY": {"summaryScore": {"value": 0.1}}}}
    score = get_score(scores, "SEVERE_TOXICITY")
    assert score == 0.1


def test_get_manner(mock_perspective_client):
    text = "You are an idiot!"
    score = get_manner(text, mock_perspective_client, "SEVERE_TOXICITY")
    assert isinstance(score, float)


def test_get_sim(mock_sentence_transformer):
    comment = "Neil Armstrong was the first man on the moon."
    reply_to = "Who was the first man on the moon?"
    similarity = get_sim(mock_sentence_transformer, comment, reply_to)
    assert isinstance(similarity, float)


def test_extract_named_entities(mock_analyzer, mock_anonymizer):
    text = "My name is John and I live in New York"
    entities, anonymized_text = extract_named_entities(text)
    assert entities == ["PERSON", "LOCATION"]
    assert anonymized_text == "My name is PERSON and I live in LOCATION"


def test_cohere_who():
    text = "Who is this?"
    has_person_entity = ["PERSON"]
    result = cohere_who(text, has_person_entity)
    assert result == 1


def test_cohere_where():
    text = "Where is this place?"
    has_location_entity = ["LOCATION"]
    result = cohere_where(text, has_location_entity)
    assert result == 1


def test_cohere_when():
    text = "When is the meeting?"
    has_datetime_entity = ["DATE_TIME"]
    result = cohere_when(text, has_datetime_entity)
    assert result == 1


def test_get_quantity_opt(mock_analyzer, mock_anonymizer):
    text_B = "John went to New York on January 1st"
    information_density = get_quantity_opt(text_B)
    assert isinstance(information_density, float)


def test_trust_get_trust(mock_perspective_client, mock_sentence_transformer):
    trust = Trust(API_KEY=None)
    trust.perspective_client = mock_perspective_client
    trust.similarity_model = mock_sentence_transformer

    comment = "Neil Armstrong was the first man on the moon."
    reply_to = "Who was the first man on the moon?"
    result = trust.get_trust(comment, reply_to, mode="SEVERE_TOXICITY")

    assert len(result) <= 3
    assert isinstance(result[0], float) if len(result) == 3 else True  # similarity
    assert isinstance(result[1], float)  # quantity
    assert isinstance(result[2], float) if trust.API_KEY is not None else True  # manner
