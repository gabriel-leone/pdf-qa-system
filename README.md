# PDF Q&A System

A web-based application that enables users to upload PDF documents and ask natural language questions about their contents using retrieval-augmented generation (RAG).

## Features

- Upload and process PDF documents
- Extract text and create semantic chunks
- Generate vector embeddings for semantic search
- Ask questions in natural language
- Get accurate answers with source references
- Multilingual support (Portuguese and English)

## Setup

### Prerequisites

- Python 3.12+
- Virtual environment (recommended)

### Installation

1. Clone the repository and navigate to the project directory

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy environment configuration:
```bash
cp .env.example .env
```

5. Edit `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Application

```bash
python main.py
```

The application will be available at `http://localhost:8000`

### API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

### Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
├── api/                 # API endpoints and controllers
├── models/             # Data models and schemas
├── services/           # Business logic and services
├── tests/              # Test suite
├── utils/              # Utility functions
├── main.py             # Application entry point
├── config.py           # Configuration management
└── requirements.txt    # Python dependencies
```

## Development

The project follows a layered architecture with clear separation between:
- API layer (FastAPI endpoints)
- Service layer (business logic)
- Data layer (models and storage)

See the design document in `.kiro/specs/pdf-qa-system/design.md` for detailed architecture information.