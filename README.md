# PDF Q&A System

A comprehensive system for uploading PDF documents and asking questions about their contents using retrieval-augmented generation (RAG). The system supports multilingual documents (Portuguese and English) and provides efficient semantic search capabilities.

## 🚀 Features

- **Document Processing**: Upload and process PDF documents with text extraction
- **Semantic Search**: Vector-based search using sentence transformers
- **Question Answering**: RAG-powered answers with source citations
- **Multilingual Support**: Handle Portuguese and English documents
- **RESTful API**: FastAPI-based API with automatic documentation
- **Health Monitoring**: Comprehensive health checks and monitoring
- **Production Ready**: Docker support, logging, error handling
- **Scalable Architecture**: Modular design with dependency injection

## 📋 Requirements

- Python 3.11+
- OpenAI API key (for question answering)
- 4GB+ RAM (for embedding models)
- 2GB+ disk space (for vector storage)

## 🛠️ Quick Start

### Option 1: Using the Startup Script (Recommended)

```bash
# Make the startup script executable
chmod +x start.sh

# Start development server (automatically sets up environment)
./start.sh dev

# Or start production server
./start.sh prod --workers 4
```

### Option 2: Manual Setup

1. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your configuration (especially OPENAI_API_KEY)
```

4. **Run the application:**
```bash
# Development
python main.py

# Production
python run.py server
```

### Option 3: Using Docker

```bash
# Development
docker-compose up --build

# Production
docker-compose --profile production up --build
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Required
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional (with defaults)
LLM_MODEL=amazon/nova-2-lite-v1:free
LLM_FALLBACK_MODEL=meta-llama/llama-3.3-70b-instruct:free
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=50
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### Advanced Configuration

See `.env.example` for all available configuration options including:
- Server settings (host, port, workers)
- Security settings (CORS, allowed hosts)
- Performance tuning (batch sizes, timeouts)
- Logging configuration

## 📚 API Documentation

Once running, visit:
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

### Key Endpoints

#### Upload Documents
```bash
POST /documents/
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/documents/" \
     -F "files=@document1.pdf" \
     -F "files=@document2.pdf"
```

#### Ask Questions
```bash
POST /question/
Content-Type: application/json

curl -X POST "http://localhost:8000/question/" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic discussed?"}'
```

#### Health Monitoring
```bash
GET /health              # Basic health check
GET /health/detailed     # Detailed component status
GET /health/ready        # Kubernetes readiness probe
GET /health/live         # Kubernetes liveness probe
```

## 🧪 Testing

### Run All Tests
```bash
# Using startup script
./start.sh test

# Manual
pytest tests/ -v --cov=. --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Property-Based Tests**: Correctness validation with Hypothesis

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   REST API      │    │   Health Checks │
│   Layer         │    │   & Monitoring  │
└─────────┬───────┘    └─────────────────┘
          │
┌─────────▼───────┐
│   Service       │
│   Layer         │
├─────────────────┤
│ • Document Svc  │
│ • Question Svc  │
│ • Embedding Svc │
│ • LLM Service   │
└─────────┬───────┘
          │
┌─────────▼───────┐
│   Data Layer    │
├─────────────────┤
│ • Vector Store  │
│ • File Storage  │
└─────────────────┘
```

### Key Components

- **Document Service**: PDF processing and text extraction
- **Embedding Service**: Vector generation using sentence transformers
- **Vector Store**: ChromaDB for semantic search
- **LLM Service**: OpenAI integration for answer generation
- **Question Service**: RAG pipeline orchestration

## 🚀 Deployment

### Production Deployment

1. **Using Docker (Recommended):**
```bash
docker-compose --profile production up -d
```

2. **Using the runner script:**
```bash
python run.py server
```

3. **Using Gunicorn:**
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Environment-Specific Settings

- **Development**: Debug mode, detailed logging, auto-reload
- **Production**: Optimized workers, structured logging, health checks
- **Docker**: Containerized with proper security and resource limits

## 📊 Monitoring

### Health Checks
- `/health` - Basic service status
- `/health/detailed` - Component-level diagnostics
- `/health/ready` - Readiness for traffic
- `/health/live` - Liveness check

### Logging
- Structured JSON logging in production
- Request/response logging with timing
- Error tracking with stack traces
- Performance metrics

## 🔍 Troubleshooting

### Common Issues

1. **"No OpenRouter API key"**
   - Set `OPENROUTER_API_KEY` in your `.env` file
   - Get your API key from [OpenRouter](https://openrouter.ai/)

2. **"Embedding model download fails"**
   - Ensure internet connection for initial model download
   - Check available disk space (models are ~500MB)

3. **"ChromaDB connection error"**
   - Verify `CHROMA_PERSIST_DIRECTORY` is writable
   - Check disk space for vector storage

4. **"Memory issues with large PDFs"**
   - Reduce `MAX_FILE_SIZE_MB` setting
   - Increase available system memory

### Debug Mode

Enable debug mode for detailed logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
python main.py
```

### Health Check

Check system health:
```bash
# Using the runner
python run.py health

# Using curl
curl http://localhost:8000/health/detailed
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 Project Structure

```
├── api/                 # API endpoints and controllers
├── models/             # Data models and schemas
├── services/           # Business logic and services
├── tests/              # Test suite
├── utils/              # Utility functions and helpers
├── case_files/         # Sample PDF documents for testing
├── chroma_db/          # Vector database storage
├── logs/               # Application logs
├── uploads/            # Temporary file uploads
├── main.py             # Development entry point
├── run.py              # Production application runner
├── start.sh            # Startup script for all modes
├── config.py           # Configuration management
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Container orchestration
└── requirements.txt    # Python dependencies
```

## 📖 Documentation

- **API Documentation**: Available at `/docs` when running
- **Design Document**: `.kiro/specs/pdf-qa-system/design.md`
- **Requirements**: `.kiro/specs/pdf-qa-system/requirements.md`
- **Implementation Tasks**: `.kiro/specs/pdf-qa-system/tasks.md`