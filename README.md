# NovelLLM

A powerful AI agent that combines language model capabilities with vector search for intelligent document querying and analysis.

## Features

- Advanced vector search using ChromaDB
- Intelligent document chunking and processing
- Context-aware querying with source tracking
- Timestamp-based content organization
- Efficient metadata handling

## Setup

1. Clone the repository:
```bash
git clone https://github.com/pavelnovel/novelllm.git
cd novelllm
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main agent:
```bash
python main_agent.py
```

## Project Structure

- `main_agent.py`: Main application entry point
- `query_chunks.py`: Vector search and query handling
- `scripts/`: Utility scripts for document processing
  - `chunk_md.py`: Document chunking utilities
  - `embed_chunks.py`: Document embedding utilities
  - `query_chunks.py`: Query processing utilities

## Development

The project uses two main branches:
- `main`: Production-ready code
- `dev`: Development branch for new features

## License

MIT License - see [LICENSE](LICENSE) for details 