# PaperPusher

PaperPusher is a lightweight, single-file semantic information store that acts as an AI-powered filing cabinet. It enables natural language storage and retrieval of information between AI agents and humans, making it perfect for local-first AI agent workforces.

Think of it as passing notes and memos in an office, but with the power of semantic search built in.

## Features

- 🔍 Semantic search using OpenAI embeddings
- 📝 Store text content with rich metadata
- 🤖 OpenAI-compatible tool schemas for LLM integration
- 📊 Track content access patterns
- 💾 Simple persistence to disk
- 🪶 Single file, zero dependencies (besides OpenAI and numpy)

## Getting Started

1. Copy `paper_pusher.py` into your project
2. Install the requirements:
   ```bash
   pip install openai numpy
   ```
3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=<your-api-key>
   ```

## Usage

```python
from paper_pusher import PaperPusher

# Initialize the store
pusher = PaperPusher() 

# (optionally, use an existing OpenAI client)
# pusher = PaperPusher(openai_client=openai.OpenAI())

# Store information
pusher.save(
    key="meeting-2024-01",
    description="Q1 planning meeting notes",
    intended_use="Reference for action items and decisions",
    value="Meeting minutes...",
    authored_by="alice"
)

# Search using natural language (returns top 5 results by default)
results = pusher.search("Q1 planning decisions")
# Returns list of (similarity_score, metadata) tuples
for score, metadata in results:
    print(f"Score: {score:.2f}, Key: {metadata['key']}")

# Search with custom number of results
results = pusher.search("Q1 planning decisions", k=10)  # Get top 10 results

# Retrieve specific content
notes = pusher.get_value("agent1", "meeting-2024-01")

# Save state to disk
pusher.save_to_file("paper_pusher_state.pkl")

# Load state from disk
pusher.load_from_file("paper_pusher_state.pkl")
```

## LLM Integration

PaperPusher provides OpenAI-compatible tool schemas for easy integration with LLMs:

```python
# Get OpenAI tool schemas
tool_schemas = pusher.get_tool_schemas()

# Execute tools through a unified interface
result = pusher.execute_tool(
    agent_identifier="agent1",
    tool_name="store_information",
    tool_args={
        "key": "task-123",
        "description": "Customer feedback analysis",
        "intended_use": "Improve product features",
        "value": "Detailed analysis..."
    }
)

# Search with custom k parameter
results = pusher.execute_tool(
    agent_identifier="agent1",
    tool_name="search_information",
    tool_args={
        "query": "customer feedback",
        "k": 10  # Get top 10 results
    }
)
```

## Configuration

- `openai_embedding_model`: OpenAI model for embeddings. Default: "text-embedding-3-small"

```python
pusher = PaperPusher()
pusher.openai_embedding_model = "text-embedding-3-large"  # Use larger model
```

## How It Works

PaperPusher uses OpenAI embeddings to create vector representations of content metadata, enabling semantic search capabilities. When you store information, it generates embeddings for the metadata (description and intended use). When you search, it converts your query to an embedding and finds the most similar content using cosine similarity, returning the top k results.

## Limitations

- Requires an OpenAI API key and internet connection for embeddings
- Stores everything in memory (with optional persistence to disk)
- Not optimized for very large datasets
- No built-in access control or security features

## Contributing

This is a proof-of-concept implementation. Feel free to fork and adapt it to your needs!

## License

MIT License - feel free to use this in your projects!
