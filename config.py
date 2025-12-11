"""
Configuration management for the PDF Q&A System
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application Configuration
    app_name: str = "PDF Q&A System"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Security Configuration
    allowed_hosts: str = "*"
    cors_origins: str = "*"
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    
    # LLM Configuration (OpenRouter)
    openrouter_api_key: Optional[str] = None
    llm_model: str = "amazon/nova-2-lite-v1:free"
    llm_fallback_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    max_context_tokens: int = 3000
    max_total_tokens: int = 4000
    
    # Legacy OpenAI Configuration (for backward compatibility)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_fallback_model: str = "gpt-3.5-turbo"
    
    # Application Configuration
    log_level: str = "INFO"
    log_format: str = "structured"
    log_file: Optional[str] = None
    max_file_size_mb: int = 50
    max_chunk_size: int = 1000
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # ChromaDB Configuration
    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "pdf_documents"
    
    # Performance Configuration
    embedding_batch_size: int = 32
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 300
    
    # Health Check Configuration
    health_check_interval: int = 60  # seconds
    health_check_timeout: int = 30   # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()