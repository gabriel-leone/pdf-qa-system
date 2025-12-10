"""
Data models for the PDF Q&A System
"""

from .document import Document, Chunk, ChunkMetadata, ProcessingStatus
from .question import Question, Answer, Reference
from .api import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    QuestionRequest,
    QuestionResponse,
    ErrorResponse
)

__all__ = [
    # Document models
    "Document",
    "Chunk", 
    "ChunkMetadata",
    "ProcessingStatus",
    
    # Question/Answer models
    "Question",
    "Answer",
    "Reference",
    
    # API models
    "DocumentUploadRequest",
    "DocumentUploadResponse", 
    "QuestionRequest",
    "QuestionResponse",
    "ErrorResponse"
]