import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from src.search.api import app, SearchRequest, openai_client, index


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def mock_openai():
    with patch('src.search.api.openai_client') as mock:
        # Mock embedding creation
        mock.embeddings.create.return_value.data = [Mock(embedding=[0.1] * 3072)]
        yield mock


@pytest.fixture
def mock_pinecone():
    with patch('src.search.api.index') as mock:
        # Mock successful query response
        mock.query.return_value = {
            "matches": [
                {
                    "id": "test-1",
                    "score": 0.95,
                    "metadata": {
                        "text": "This is a test chunk",
                        "program": "Workshop",
                        "title": "Test Workshop",
                        "speakers": ["Test Speaker"],
                        "date": "2025-09-01"
                    }
                }
            ]
        }
        yield mock


def test_search_endpoint_success(test_client, mock_openai, mock_pinecone):
    """Test successful search request"""
    response = test_client.post(
        "/search",
        json={"question": "test query", "top_k": 5}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    
    result = data["results"][0]
    assert result["id"] == "test-1"
    assert result["score"] == 0.95
    assert result["text"] == "This is a test chunk"
    assert result["metadata"]["program"] == "Workshop"
    assert result["metadata"]["title"] == "Test Workshop"
    assert result["metadata"]["speakers"] == ["Test Speaker"]
    assert result["metadata"]["date"] == "2025-09-01"

    # Verify OpenAI embedding was called
    mock_openai.embeddings.create.assert_called_once_with(
        model="text-embedding-3-large",
        input="test query"
    )

    # Verify Pinecone query was called
    mock_pinecone.query.assert_called_once_with(
        vector=[0.1] * 3072,
        top_k=5,
        include_metadata=True
    )


def test_search_empty_query(test_client):
    """Test search with empty query"""
    response = test_client.post(
        "/search",
        json={"question": "   ", "top_k": 5}
    )
    assert response.status_code == 400
    assert "Empty query" in response.json()["detail"]


def test_search_pinecone_connection_error(test_client, mock_openai, mock_pinecone):
    """Test handling of Pinecone connection error"""
    # Make the test query fail
    mock_pinecone.query.side_effect = Exception("Connection refused")
    
    response = test_client.post(
        "/search",
        json={"question": "test query", "top_k": 5}
    )
    
    assert response.status_code == 503
    assert "Service is warming up" in response.json()["detail"]


def test_search_openai_error(test_client, mock_openai):
    """Test handling of OpenAI API error"""
    # Make the embedding creation fail
    mock_openai.embeddings.create.side_effect = Exception("OpenAI API error")
    
    response = test_client.post(
        "/search",
        json={"question": "test query", "top_k": 5}
    )
    
    assert response.status_code == 503
    assert "Service error" in response.json()["detail"]


def test_search_with_custom_top_k(test_client, mock_openai, mock_pinecone):
    """Test search with custom top_k parameter"""
    response = test_client.post(
        "/search",
        json={"question": "test query", "top_k": 10}
    )
    
    assert response.status_code == 200
    
    # Verify Pinecone was called with correct top_k
    mock_pinecone.query.assert_called_once_with(
        vector=[0.1] * 3072,
        top_k=10,
        include_metadata=True
    )


def test_search_default_top_k(test_client, mock_openai, mock_pinecone):
    """Test search with default top_k"""
    response = test_client.post(
        "/search",
        json={"question": "test query"}  # No top_k specified
    )
    
    assert response.status_code == 200
    
    # Verify Pinecone was called with default top_k
    mock_pinecone.query.assert_called_once_with(
        vector=[0.1] * 3072,
        top_k=5,  # Default value
        include_metadata=True
    )


def test_search_with_missing_metadata(test_client, mock_openai, mock_pinecone):
    """Test handling of missing or None metadata"""
    # Mock Pinecone response with missing metadata
    mock_pinecone.query.return_value = {
        "matches": [
            {
                "id": "test-missing",
                "score": 0.85,
                "metadata": None
            }
        ]
    }
    
    response = test_client.post(
        "/search",
        json={"question": "test query"}
    )
    
    assert response.status_code == 200
    data = response.json()
    result = data["results"][0]
    
    # Check default values are used
    assert result["text"] == ""
    assert result["metadata"]["program"] is None
    assert result["metadata"]["title"] is None
    assert result["metadata"]["speakers"] == []
    assert result["metadata"]["date"] is None


def test_search_with_invalid_score(test_client, mock_openai, mock_pinecone):
    """Test handling of invalid score values"""
    # Mock Pinecone response with invalid score
    mock_pinecone.query.return_value = {
        "matches": [
            {
                "id": "test-score",
                "score": "invalid",
                "metadata": {
                    "text": "Test text",
                    "program": "Test Program",
                    "title": "Test Title",
                    "speakers": "Single Speaker",  # Test single speaker string
                    "date": "2024-01-01"
                }
            }
        ]
    }
    
    response = test_client.post(
        "/search",
        json={"question": "test query"}
    )
    
    assert response.status_code == 200
    data = response.json()
    result = data["results"][0]
    
    # Check score is defaulted to 0.0
    assert result["score"] == 0.0
    # Check single speaker is converted to list
    assert result["metadata"]["speakers"] == ["Single Speaker"]


def test_search_with_none_values(test_client, mock_openai, mock_pinecone):
    """Test handling of None values in metadata"""
    mock_pinecone.query.return_value = {
        "matches": [
            {
                "id": "test-none",
                "score": None,
                "metadata": {
                    "text": None,
                    "program": None,
                    "title": None,
                    "speakers": None,
                    "date": None
                }
            }
        ]
    }
    
    response = test_client.post(
        "/search",
        json={"question": "test query"}
    )
    
    assert response.status_code == 200
    data = response.json()
    result = data["results"][0]
    
    # Check default values for None
    assert result["score"] == 0.0
    assert result["text"] == ""
    assert result["metadata"]["program"] is None
    assert result["metadata"]["title"] is None
    assert result["metadata"]["speakers"] == []
    assert result["metadata"]["date"] is None
