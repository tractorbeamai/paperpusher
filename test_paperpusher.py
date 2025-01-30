"""Tests for the PaperPusher class."""

import datetime
import random
import string
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

    def create_random_embedding():
        # Create a realistic 1536-dimensional embedding vector (matches OpenAI's models)
        vector = np.random.normal(0, 1, 1536)
        # Normalize to unit length like real embeddings
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()

    def mock_create(**kwargs):
        mock_embedding = Mock(spec=Embedding)
        mock_embedding.embedding = create_random_embedding()
        mock_response = Mock(spec=CreateEmbeddingResponse)
        mock_response.data = [mock_embedding]
        return mock_response

    mock_client.embeddings.create = mock_create
    return mock_client


@pytest.fixture
def paper_pusher(mock_openai_client):
    """Create a PaperPusher instance with mocked OpenAI client."""
    return PaperPusher(openai_client=mock_openai_client)


def test_initialization(paper_pusher):
    """Test PaperPusher initialization."""
    assert paper_pusher.similarity_threshold == 0.5
    assert paper_pusher.openai_embedding_model == "text-embedding-3-small"
    assert isinstance(paper_pusher.index, list)
    assert isinstance(paper_pusher.values, dict)
    assert len(paper_pusher.index) == 0
    assert len(paper_pusher.values) == 0


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
    assert len(paper_pusher.index) == 1
    embedding, metadata = paper_pusher.index[0]
    assert isinstance(embedding, np.ndarray)
    assert metadata["key"] == key
    assert metadata["description"] == description
    assert metadata["intended_use"] == intended_use
    assert metadata["authored_by"] == authored_by
    assert isinstance(metadata["created_at"], datetime.datetime)
    assert agent_id in metadata["accessed_by"]
    assert isinstance(metadata["accessed_by"][agent_id], datetime.datetime)


def test_get_value_nonexistent_key(paper_pusher):
    """Test retrieving value with non-existent key raises KeyError."""
    with pytest.raises(KeyError):
        paper_pusher.get_value("agent-1", "nonexistent-key")


def test_large_scale_search(paper_pusher, benchmark):
    """Benchmark search performance with many documents and different search patterns."""

    def random_string(length, char_set=None):
        if char_set is None:
            char_set = string.ascii_letters + string.digits + " "
        return "".join(random.choices(char_set, k=length))

    # Generate test documents with different characteristics
    docs = []

    # Documents with random content
    for i in range(500):
        docs.append(
            {
                "key": f"random-{i}",
                "description": random_string(50),
                "intended_use": random_string(30),
                "value": random_string(200),
                "authored_by": f"user-{random.randint(1,10)}",
            }
        )

    # Documents with common prefixes/patterns
    common_prefixes = ["report-", "meeting-", "analysis-", "data-"]
    for prefix in common_prefixes:
        for i in range(100):
            docs.append(
                {
                    "key": f"{prefix}{i}",
                    "description": f"{prefix} document {random_string(30)}",
                    "intended_use": f"Reference for {prefix} tasks",
                    "value": random_string(200),
                    "authored_by": f"user-{random.randint(1,10)}",
                }
            )

    # Documents with specific topics
    topics = ["finance", "technology", "marketing", "research"]
    for topic in topics:
        for i in range(50):
            docs.append(
                {
                    "key": f"{topic}-doc-{i}",
                    "description": f"{topic} related content {random_string(20)}",
                    "intended_use": f"{topic} reference material",
                    "value": random_string(200),
                    "authored_by": f"user-{random.randint(1,10)}",
                }
            )

    # Store all documents
    for doc in docs:
        paper_pusher.save(**doc)

    def run_searches():
        # Test different search patterns
        search_patterns = [
            # Random searches
            [random_string(10) for _ in range(5)],
            # Prefix searches
            [prefix + random_string(5) for prefix in common_prefixes],
            # Topic searches
            topics,
            # Longer, more specific queries
            [f"document about {topic} for reference" for topic in topics],
            # Mixed case and special characters
            [random_string(15, string.ascii_letters + string.digits + "!@#$%^&* ").upper() for _ in range(5)],
        ]

        for pattern_group in search_patterns:
            for query in pattern_group:
                results = paper_pusher.search(query)
                assert isinstance(results, list)

                # Verify results are sorted by similarity
                if len(results) > 1:
                    for i in range(len(results) - 1):
                        assert results[i][0] >= results[i + 1][0]

                # Verify similarity scores are within valid range
                for similarity, _ in results:
                    assert 0 <= similarity <= 1

    # Benchmark the search operation
    benchmark.pedantic(run_searches, iterations=3, rounds=50)


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
    new_pusher = PaperPusher(openai_client=paper_pusher.openai_client)  # Use same mock client
    new_pusher.load_from_file(str(test_file))

    # Verify data was preserved
    assert "test-save" in new_pusher.values
    assert new_pusher.values["test-save"] == "Test content"
    assert len(new_pusher.index) == len(paper_pusher.index)
    # Verify first item's metadata
    _, orig_metadata = paper_pusher.index[0]
    _, loaded_metadata = new_pusher.index[0]
    assert orig_metadata["key"] == loaded_metadata["key"]
    assert orig_metadata["description"] == loaded_metadata["description"]
