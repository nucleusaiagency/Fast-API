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
        yield mock


def test_search_response_format_validation(test_client, mock_openai, mock_pinecone):
    """Test that response format strictly matches the expected schema"""
    # Setup mock response with all possible metadata fields
    mock_pinecone.query.return_value = {
        "matches": [
            {
                "id": "test-1",
                "score": 0.95,
                "metadata": {
                    "text": "Test content",
                    "program": "Workshop",
                    "title": "Test Workshop",
                    "speakers": ["Speaker 1", "Speaker 2"],
                    "date": "2025-09-01",
                    "extra_field": "Should not appear in response"
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
    
    # Check structure
    assert "results" in data
    assert len(data["results"]) == 1
    result = data["results"][0]
    
    # Check exact fields in result
    expected_fields = {"id", "score", "text", "metadata"}
    assert set(result.keys()) == expected_fields
    
    # Check metadata fields
    expected_metadata_fields = {"program", "title", "speakers", "date"}
    assert set(result["metadata"].keys()) == expected_metadata_fields
    
    # Verify extra fields are not included
    assert "extra_field" not in result["metadata"]


def test_search_empty_response_handling(test_client, mock_openai, mock_pinecone):
    """Test handling of empty search results"""
    mock_pinecone.query.return_value = {"matches": []}

    response = test_client.post(
        "/search",
        json={"question": "test query"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 0


def test_search_metadata_edge_cases(test_client, mock_openai, mock_pinecone):
    """Test handling of missing or malformed metadata"""
    mock_pinecone.query.return_value = {
        "matches": [
            {
                "id": "test-1",
                "score": 0.95,
                "metadata": None
            },
            {
                "id": "test-2",
                "score": 0.85,
                "metadata": {}
            },
            {
                "id": "test-3",
                "score": 0.75,
                "metadata": {
                    "text": None,
                    "program": "",
                    "speakers": None,
                    "date": "invalid-date"
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
    assert len(data["results"]) == 3
    
    # Check None metadata handling
    assert data["results"][0]["metadata"] == {
        "program": None,
        "title": None,
        "speakers": [],
        "date": None
    }
    
    # Check empty metadata handling
    assert data["results"][1]["metadata"] == {
        "program": None,
        "title": None,
        "speakers": [],
        "date": None
    }
    
    # Check invalid metadata handling
    assert data["results"][2]["metadata"] == {
        "program": None,
        "title": None,
        "speakers": [],
        "date": "invalid-date"
    }


def test_search_large_response_handling(test_client, mock_openai, mock_pinecone):
    """Test handling of large response payload"""
    # Create a large number of results
    matches = []
    for i in range(100):  # Test with 100 results
        matches.append({
            "id": f"test-{i}",
            "score": 0.99 - (i * 0.01),
            "metadata": {
                "text": "Test content " * 100,  # Large text field
                "program": "Workshop",
                "title": "Test Workshop",
                "speakers": ["Speaker 1", "Speaker 2"],
                "date": "2025-09-01"
            }
        })
    
    mock_pinecone.query.return_value = {"matches": matches}

    response = test_client.post(
        "/search",
        json={"question": "test query", "top_k": 100}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 100
    
    # Verify that all results are properly formatted
    for result in data["results"]:
        assert "id" in result
        assert "score" in result
        assert "text" in result
        assert "metadata" in result
        assert isinstance(result["metadata"]["speakers"], list)


def test_search_score_validation(test_client, mock_openai, mock_pinecone):
    """Test validation of search result scores"""
    mock_pinecone.query.return_value = {
        "matches": [
            {
                "id": "test-1",
                "score": "0.95",  # String instead of float
                "metadata": {"text": "content"}
            },
            {
                "id": "test-2",
                "score": None,  # Missing score
                "metadata": {"text": "content"}
            },
            {
                "id": "test-3",
                # Missing score field entirely
                "metadata": {"text": "content"}
            }
        ]
    }

    response = test_client.post(
        "/search",
        json={"question": "test query"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # All scores should be converted to float
    assert isinstance(data["results"][0]["score"], float)
    assert data["results"][0]["score"] == 0.95
    
    # Missing scores should default to 0.0
    assert data["results"][1]["score"] == 0.0
    assert data["results"][2]["score"] == 0.0
