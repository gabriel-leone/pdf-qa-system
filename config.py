"""
Configuration management for the PDF Q&A System
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_fallback_model: str = "gpt-3.5-turbo"
    max_context_tokens: int = 3000
    max_total_tokens: int = 4000
    
    # Application Configuration
    log_level: str = "INFO"
    max_file_size_mb: int = 50
    max_chunk_size: int = 1000
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # ChromaDB Configuration
    chroma_persist_directory: str = "./chroma_db"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()