"""Tests for the PaperPusher class."""

import datetime
from unittest.mock import Mock

import numpy as np
import pytest
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.embedding import Embedding

from paperpusher import PaperPusher


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    mock_client = Mock()
    # Mock embedding response
    mock_embedding = Mock(spec=Embedding)
    mock_embedding.embedding = [0.1, 0.2, 0.3]  # Simple test embedding

    mock_response = Mock(spec=CreateEmbeddingResponse)
    mock_response.data = [mock_embedding]

    mock_client.embeddings.create.return_value = mock_response
    return mock_client


@pytest.fixture
def paper_pusher(mock_openai_client):
    """Create a PaperPusher instance with mocked OpenAI client."""
    return PaperPusher(openai_client=mock_openai_client)


def test_initialization(paper_pusher):
    """Test PaperPusher initialization."""
    assert paper_pusher.similarity_threshold == 0.5
    assert paper_pusher.openai_embedding_model == "text-embedding-3-small"
    assert isinstance(paper_pusher.index, dict)
    assert isinstance(paper_pusher.values, dict)


def test_save_and_get_value(paper_pusher):
    """Test saving and retrieving information."""
    # Test data
    key = "test-key"
    description = "Test description"
    intended_use = "Test intended use"
    value = "Test value"
    authored_by = "test-user"
    agent_id = "agent-1"

    # Save information
    paper_pusher.save(
        key=key,
        description=description,
        intended_use=intended_use,
        value=value,
        authored_by=authored_by,
    )

    # Verify value can be retrieved
    retrieved_value = paper_pusher.get_value(agent_id, key)
    assert retrieved_value == value

    # Verify metadata was stored
    metadata_found = False
    for _, metadata in paper_pusher.index.items():
        if metadata["key"] == key:
            metadata_found = True
            assert metadata["description"] == description
            assert metadata["intended_use"] == intended_use
            assert metadata["authored_by"] == authored_by
            assert isinstance(metadata["created_at"], datetime.datetime)
            assert agent_id in metadata["accessed_by"]
            assert isinstance(metadata["accessed_by"][agent_id], datetime.datetime)

    assert metadata_found


def test_get_value_nonexistent_key(paper_pusher):
    """Test retrieving value with non-existent key raises KeyError."""
    with pytest.raises(KeyError):
        paper_pusher.get_value("agent-1", "nonexistent-key")


def test_search(paper_pusher):
    """Test searching for information."""
    # Add test data
    paper_pusher.save(
        key="meeting-notes",
        description="Q1 planning meeting notes",
        intended_use="Reference for decisions",
        value="Meeting content",
        authored_by="alice",
    )

    # Perform search
    results = paper_pusher.search("Q1 planning")

    assert len(results) > 0
    # First result should be a tuple of (similarity_score, metadata)
    score, metadata = results[0]
    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert metadata["key"] == "meeting-notes"


def test_empty_search(paper_pusher):
    """Test searching with empty index."""
    results = paper_pusher.search("test query")
    assert len(results) == 0


def test_cosine_similarity(paper_pusher):
    """Test cosine similarity calculation."""
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    c = np.array([0, 1, 0])

    # Same vectors should have similarity 1
    assert paper_pusher._cosine_similarity(a, b) == pytest.approx(1.0)
    # Orthogonal vectors should have similarity 0
    assert paper_pusher._cosine_similarity(a, c) == pytest.approx(0.0)


def test_tool_schemas(paper_pusher):
    """Test getting tool schemas."""
    schemas = paper_pusher.get_tool_schemas()
    assert isinstance(schemas, list)
    assert len(schemas) == 3  # Should have 3 tool schemas

    # Verify schema structure
    for schema in schemas:
        assert "type" in schema
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


def test_execute_tool(paper_pusher):
    """Test tool execution."""
    agent_id = "test-agent"

    # Test store_information tool
    store_result = paper_pusher.execute_tool(
        agent_id,
        "store_information",
        {"key": "tool-test", "description": "Test description", "intended_use": "Testing", "value": "Test content"},
    )
    assert isinstance(store_result, str)
    assert "tool-test" in store_result

    # Test invalid tool
    with pytest.raises(ValueError):
        paper_pusher.execute_tool(agent_id, "invalid_tool", {})


def test_save_and_load_file(paper_pusher, tmp_path):
    """Test saving and loading state to/from file."""
    # Add some test data
    paper_pusher.save(
        key="test-save",
        description="Test save/load",
        intended_use="Testing persistence",
        value="Test content",
        authored_by="test-user",
    )

    # Save state to temporary file
    test_file = tmp_path / "test_state.pkl"
    paper_pusher.save_to_file(str(test_file))

    # Create new instance and load state
    new_pusher = PaperPusher()
    new_pusher.load_from_file(str(test_file))

    # Verify data was preserved
    assert "test-save" in new_pusher.values
    assert new_pusher.values["test-save"] == "Test content"
    assert len(new_pusher.index) == len(paper_pusher.index)
