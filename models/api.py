"""
API request and response models for the PDF Q&A System
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DocumentUploadRequest(BaseModel):
    """Request model for document upload (used for validation)"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "language_hint": "pt"
            }
        }
    )

    language_hint: Optional[str] = Field(None, pattern="^(pt|en|auto)$", description="Language hint for processing")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Documents processed successfully",
                "documents_indexed": 2,
                "total_chunks": 45,
                "processing_time_ms": 3500
            }
        }
    )

    message: str = Field(..., description="Success message")
    documents_indexed: int = Field(..., ge=0, description="Number of documents successfully processed")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks created")
    processing_time_ms: int = Field(..., ge=0, description="Time taken to process documents in milliseconds")


class QuestionRequest(BaseModel):
    """Request model for question answering"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the maximum operating temperature?",
                "language": "en",
                "max_references": 5
            }
        }
    )

    question: str = Field(..., min_length=1, max_length=1000, description="The question to answer")
    language: Optional[str] = Field("auto", pattern="^(pt|en|auto)$", description="Language for the response")
    max_references: Optional[int] = Field(5, ge=1, le=10, description="Maximum number of references to include")
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        """Validate question is meaningful"""
        stripped = v.strip()
        if not stripped:
            raise ValueError('Question cannot be empty or only whitespace')
        if len(stripped) < 3:
            raise ValueError('Question must be at least 3 characters long')
        return stripped


class QuestionResponse(BaseModel):
    """Response model for question answering"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "According to the manual, the maximum operating temperature is 85°C under normal conditions.",
                "confidence_score": 0.89,
                "references": [
                    {
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

    answer: str = Field(..., description="The generated answer")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the answer")
    references: list[dict] = Field(default_factory=list, description="Supporting references")
    processing_time_ms: int = Field(..., ge=0, description="Time taken to generate answer in milliseconds")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "PROCESSING_FAILED",
                    "message": "Failed to process document: unsupported PDF version",
                    "details": {
                        "filename": "document.pdf",
                        "error_type": "PDF_VERSION_ERROR"
                    },
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }
    )

    error: dict = Field(..., description="Error details")