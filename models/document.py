"""
Document-related data models for the PDF Q&A System
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
import uuid


class ProcessingStatus(str, Enum):
    """Status of document processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(BaseModel):
    """Document model representing an uploaded PDF file"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "manual.pdf",
                "upload_timestamp": "2024-01-15T10:30:00Z",
                "file_size": 1048576,
                "language": "pt",
                "processing_status": "completed",
                "chunk_count": 25,
                "file_path": "/uploads/123e4567-e89b-12d3-a456-426614174000.pdf"
            }
        }
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique document identifier")
    filename: str = Field(..., min_length=1, max_length=255, description="Original filename of the uploaded document")
    upload_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the document was uploaded")
    file_size: int = Field(..., gt=0, description="Size of the file in bytes")
    language: str = Field(..., pattern="^(pt|en|unknown)$", description="Detected language of the document")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Current processing status")
    chunk_count: int = Field(default=0, ge=0, description="Number of chunks created from this document")
    file_path: str = Field(..., min_length=1, description="Path where the file is stored")

    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v):
        """Validate filename has proper extension"""
        if not v.lower().endswith('.pdf'):
            raise ValueError('Filename must have .pdf extension')
        return v

    @field_validator('file_size')
    @classmethod
    def validate_file_size(cls, v):
        """Validate file size is within reasonable limits (max 100MB)"""
        max_size = 100 * 1024 * 1024  # 100MB
        if v > max_size:
            raise ValueError(f'File size cannot exceed {max_size} bytes')
        return v


class ChunkMetadata(BaseModel):
    """Metadata associated with a document chunk"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page_number": 1,
                "section_title": "Introduction",
                "chunk_index": 0,
                "language": "pt",
                "confidence_score": 0.95
            }
        }
    )

    page_number: int = Field(..., ge=1, description="Page number where the chunk appears")
    section_title: Optional[str] = Field(None, max_length=500, description="Title of the section containing this chunk")
    chunk_index: int = Field(..., ge=0, description="Sequential index of this chunk within the document")
    language: str = Field(..., pattern="^(pt|en|unknown)$", description="Language of the chunk content")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for language detection")


class Chunk(BaseModel):
    """Text chunk extracted from a document"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "chunk-123e4567-e89b-12d3-a456-426614174000",
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "content": "This is a sample chunk of text extracted from the PDF document.",
                "embedding": None,
                "metadata": {
                    "page_number": 1,
                    "section_title": "Introduction",
                    "chunk_index": 0,
                    "language": "en",
                    "confidence_score": 0.95
                },
                "start_position": 0,
                "end_position": 65
            }
        }
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique chunk identifier")
    document_id: str = Field(..., description="ID of the parent document")
    content: str = Field(..., min_length=1, max_length=10000, description="Text content of the chunk")
    embedding: Optional[list[float]] = Field(None, description="Vector embedding of the chunk content")
    metadata: ChunkMetadata = Field(..., description="Metadata about the chunk")
    start_position: int = Field(..., ge=0, description="Starting character position in the original document")
    end_position: int = Field(..., ge=0, description="Ending character position in the original document")

    @field_validator('end_position')
    @classmethod
    def validate_positions(cls, v, info):
        """Validate that end_position is greater than start_position"""
        if info.data.get('start_position') is not None and v <= info.data['start_position']:
            raise ValueError('end_position must be greater than start_position')
        return v

    @field_validator('embedding')
    @classmethod
    def validate_embedding_dimensions(cls, v):
        """Validate embedding has expected dimensions (384 for multilingual MiniLM)"""
        if v is not None:
            expected_dim = 384
            if len(v) != expected_dim:
                raise ValueError(f'Embedding must have {expected_dim} dimensions, got {len(v)}')
        return v