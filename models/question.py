"""
Question and Answer related data models for the PDF Q&A System
"""
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
import uuid


class Question(BaseModel):
    """Question model for user queries"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "q-123e4567-e89b-12d3-a456-426614174000",
                "text": "What is the maximum operating temperature?",
                "timestamp": "2024-01-15T10:30:00Z",
                "embedding": None,
                "language": "en"
            }
        }
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique question identifier")
    text: str = Field(..., min_length=1, max_length=1000, description="The question text")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the question was asked")
    embedding: Optional[list[float]] = Field(None, description="Vector embedding of the question")
    language: str = Field(..., pattern="^(pt|en|auto)$", description="Language of the question")

    @field_validator('text')
    @classmethod
    def validate_question_text(cls, v):
        """Validate question text is meaningful"""
        stripped = v.strip()
        if not stripped:
            raise ValueError('Question text cannot be empty or only whitespace')
        if len(stripped) < 3:
            raise ValueError('Question must be at least 3 characters long')
        return stripped

    @field_validator('embedding')
    @classmethod
    def validate_embedding_dimensions(cls, v):
        """Validate embedding has expected dimensions (384 for multilingual MiniLM)"""
        if v is not None:
            expected_dim = 384
            if len(v) != expected_dim:
                raise ValueError(f'Embedding must have {expected_dim} dimensions, got {len(v)}')
        return v


class Reference(BaseModel):
    """Reference to a document chunk that supports an answer"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": "chunk-123e4567-e89b-12d3-a456-426614174000",
                "document_filename": "manual.pdf",
                "page_number": 5,
                "excerpt": "The maximum operating temperature is 85°C under normal conditions.",
                "relevance_score": 0.92
            }
        }
    )

    chunk_id: str = Field(..., description="ID of the referenced chunk")
    document_filename: str = Field(..., min_length=1, description="Filename of the source document")
    page_number: int = Field(..., ge=1, description="Page number where the reference appears")
    excerpt: str = Field(..., min_length=1, max_length=500, description="Relevant text excerpt from the chunk")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score for this reference")

    @field_validator('excerpt')
    @classmethod
    def validate_excerpt(cls, v):
        """Validate excerpt is meaningful"""
        stripped = v.strip()
        if not stripped:
            raise ValueError('Excerpt cannot be empty or only whitespace')
        return stripped


class Answer(BaseModel):
    """Answer model for system responses"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question_id": "q-123e4567-e89b-12d3-a456-426614174000",
                "text": "According to the manual, the maximum operating temperature is 85°C under normal conditions.",
                "confidence_score": 0.89,
                "references": [
                    {
                        "chunk_id": "chunk-123e4567-e89b-12d3-a456-426614174000",
                        "document_filename": "manual.pdf",
                        "page_number": 5,
                        "excerpt": "The maximum operating temperature is 85°C under normal conditions.",
                        "relevance_score": 0.92
                    }
                ],
                "processing_time_ms": 1250
            }
        }
    )

    question_id: str = Field(..., description="ID of the question being answered")
    text: str = Field(..., min_length=1, description="The generated answer text")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the answer")
    references: list[Reference] = Field(default_factory=list, description="Supporting references for the answer")
    processing_time_ms: int = Field(..., ge=0, description="Time taken to generate the answer in milliseconds")

    @field_validator('text')
    @classmethod
    def validate_answer_text(cls, v):
        """Validate answer text is meaningful"""
        stripped = v.strip()
        if not stripped:
            raise ValueError('Answer text cannot be empty or only whitespace')
        return stripped

    @field_validator('references')
    @classmethod
    def validate_references_limit(cls, v):
        """Validate reasonable number of references"""
        max_refs = 10
        if len(v) > max_refs:
            raise ValueError(f'Cannot have more than {max_refs} references')
        return v