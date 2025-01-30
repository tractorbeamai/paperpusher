"""PaperPusher provides semantic storage and retrieval of information blocks.

This class implements a semantic information store that allows storing, searching and retrieving
text content based on semantic similarity. It uses OpenAI embeddings to create vector representations
of content metadata, enabling natural language search capabilities.

Key features:
- Store text content with descriptive metadata
- Search stored information using natural language queries
- Track content access patterns
- Persist state to disk
- OpenAI-compatible tool schemas for LLM integration

Example usage:
    pusher = PaperPusher()

    # Store some information
    pusher.save(
        key="meeting-2024-01",
        description="Q1 planning meeting notes",
        intended_use="Reference for action items and decisions",
        value="Meeting minutes...",
        authored_by="alice"
    )

    # Search for relevant information
    results = pusher.search("Q1 planning decisions")
    # Results will be a list of tuples with (similarity_score, metadata_dict):
    # [
    #     (0.89, {
    #         "key": "meeting-2024-01",
    #         "description": "Q1 planning meeting notes",
    #         "intended_use": "Reference for action items and decisions",
    #         "authored_by": "alice",
    #         "created_at": "2024-01-15T10:30:00"
    #     }),
    #     (0.76, {...}),
    # ]

    # Retrieve specific content
    notes = pusher.get_value("meeting-2024-01")
    # Returns: "Meeting minutes..."
"""

import datetime
import pickle
from typing import Any, Union

import numpy as np
import openai

type Embedding = np.ndarray[float, np.dtype[np.float64]]


class PaperPusher:
    """Semantic information store for text content.

    This class provides methods to store, search and retrieve text content using semantic
    similarity matching powered by OpenAI embeddings. Content is stored with descriptive
    metadata that enables natural language search.

    Attributes:
        similarity_threshold (float): Minimum similarity score (0-1) for search results.
            Default 0.5.
        openai_embedding_model (str): OpenAI model ID to use for embeddings.
            Default "text-embedding-3-small".
    """

    def __init__(self, openai_client: openai.OpenAI = None):
        """Initialize an empty PaperPusher instance with default settings.

        Args:
            openai_client: Optional OpenAI client instance. If not provided, a default client will be created.
        """
        self.similarity_threshold = 0.5
        self.openai_client = openai_client or openai.OpenAI()
        self.openai_embedding_model = "text-embedding-3-small"
        self.index: dict[Embedding, dict[str, Any]] = {}
        self.values: dict[str, str] = {}

    def _get_embedding_from_openai(self, text: str) -> Embedding:
        """Get embedding vector from OpenAI API for given text.

        Args:
            text: Text content to embed.

        Returns:
            numpy.ndarray: Embedding vector from OpenAI.
        """
        response = self.openai_client.embeddings.create(input=text, model=self.openai_embedding_model)
        embedding = np.array(response.data[0].embedding)
        return embedding

    def _get_embedding_for_metadata(self, description: str, intended_use: str) -> Embedding:
        """Get embedding vector for content metadata.

        Args:
            description: Content description text.
            intended_use: Description of content's intended use.

        Returns:
            numpy.ndarray: Combined embedding for metadata.
        """
        metadata_text = f"{description} {intended_use}"
        return self._get_embedding_from_openai(metadata_text)

    def _cosine_similarity(self, a: Embedding, b: Embedding):
        """Calculate cosine similarity between two embedding vectors.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def save(
        self,
        key: str,
        description: str,
        intended_use: str,
        value: str,
        authored_by: str,
    ):
        """Save information to storage.

        Args:
            key: Unique identifier for this content.
            description: Brief description of the content.
            intended_use: Description of how this content should be used.
            value: The actual content text to store.
            authored_by: Identifier of the agent/user storing this content.
        """
        metadata = {
            "key": key,
            "description": description,
            "intended_use": intended_use,
            "authored_by": authored_by,
            "created_at": datetime.datetime.now(),
            "accessed_by": {},
        }

        embedding = self._get_embedding_for_metadata(description, intended_use)
        self.index[embedding] = metadata
        self.values[key] = value

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search for information similar to query string.

        Performs semantic search using cosine similarity between the query embedding
        and stored content metadata embeddings.

        Args:
            query: Natural language search query.

        Returns:
            List of tuples containing (similarity_score, metadata_dict), sorted by
            descending similarity score. Only results above similarity_threshold are included.
        """
        query_embedding = self._get_embedding_from_openai(query)

        if not self.index:
            return []

        similarities = [
            (self._cosine_similarity(query_embedding, emb), metadata) for emb, metadata in self.index.items()
        ]

        filtered_similarities = [
            (score, metadata) for score, metadata in similarities if score >= self.similarity_threshold
        ]

        return sorted(filtered_similarities, key=lambda x: x[0], reverse=True)

    def get_value(self, agent_identifier: str, key: str) -> str:
        """Retrieve content value by key.

        Args:
            agent_identifier: Identifier of the agent/user accessing the content.
            key: Unique key of the content to retrieve.

        Returns:
            The stored content text.

        Raises:
            KeyError: If no content exists with the given key.
        """
        if key not in self.values:
            raise KeyError(f"No information found with key: {key}")

        # Find the metadata for this key
        for _, metadata in self.index.items():
            if metadata["key"] == key:
                # Update accessed_by with current timestamp
                metadata["accessed_by"][agent_identifier] = datetime.datetime.now()
                break

        return self.values[key]

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for public methods.

        Returns schemas that can be used to expose PaperPusher functionality to
        OpenAI's function calling API.

        Returns:
            List of tool schema dictionaries compatible with OpenAI API.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "store_information",
                    "description": "Store a new piece of information in the system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Unique identifier for this information",
                            },
                            "description": {
                                "type": "string",
                                "description": "Brief description of what this information contains",
                            },
                            "intended_use": {
                                "type": "string",
                                "description": "Description of how this information should be used",
                            },
                            "value": {
                                "type": "string",
                                "description": "The actual information content to store",
                            },
                        },
                        "required": ["key", "description", "intended_use", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_information",
                    "description": "Search for stored information using semantic similarity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text to find relevant information",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_information",
                    "description": "Retrieve the content of stored information by its key",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Key identifying the information to retrieve",
                            }
                        },
                        "required": ["key"],
                    },
                },
            },
        ]

    def execute_tool(
        self, agent_identifier: str, tool_name: str, tool_args: dict[str, Any]
    ) -> Union[str, list[dict[str, Any]], dict[str, Any]]:
        """Execute a tool function with the given arguments.

        Provides a unified interface for executing PaperPusher operations through
        OpenAI's function calling API.

        Args:
            agent_identifier: Identifier of the agent/user executing the tool.
            tool_name: Name of the tool to execute.
            tool_args: Dictionary of arguments for the tool.

        Returns:
            Tool-specific return value (string, list, or dict).

        Raises:
            ValueError: If tool_name is not recognized.
        """
        if tool_name == "store_information":
            self.save(
                key=tool_args["key"],
                description=tool_args["description"],
                intended_use=tool_args["intended_use"],
                value=tool_args["value"],
                authored_by=agent_identifier,
            )
            return f"Information stored with key: {tool_args['key']}"

        elif tool_name == "semantic_search":
            results = self.search(tool_args["query"])
            return [
                {
                    "key": metadata["key"],
                    "description": metadata["description"],
                    "intended_use": metadata["intended_use"],
                }
                for _, metadata in results
            ]

        elif tool_name == "retrieve_information":
            return self.get_value(tool_args["key"])

        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def save_to_file(self, filename: str):
        """Save the current state to a file.

        Args:
            filename: Path where state should be saved.
        """
        with open(filename, "wb") as f:
            pickle.dump((self.index, self.values), f)

    def load_from_file(self, filename: str):
        """Load a previously saved state from a file.

        Args:
            filename: Path to saved state file.
        """
        with open(filename, "rb") as f:
            self.index, self.values = pickle.load(f)
