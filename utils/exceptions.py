"""
Custom exception classes for the PDF Q&A System

This module defines all custom exceptions used throughout the application,
providing structured error handling with proper error codes and messages.
"""
import time
from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(Enum):
    """Enumeration of error codes for consistent error handling"""
    
    # General errors
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    
    # File handling errors
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    EMPTY_FILE = "EMPTY_FILE"
    FILE_SAVE_FAILED = "FILE_SAVE_FAILED"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    
    # Document processing errors
    PDF_PROCESSING_FAILED = "PDF_PROCESSING_FAILED"
    TEXT_EXTRACTION_FAILED = "TEXT_EXTRACTION_FAILED"
    CHUNKING_FAILED = "CHUNKING_FAILED"
    EMBEDDING_GENERATION_FAILED = "EMBEDDING_GENERATION_FAILED"
    VECTOR_STORAGE_FAILED = "VECTOR_STORAGE_FAILED"
    DOCUMENT_PROCESSING_FAILED = "DOCUMENT_PROCESSING_FAILED"
    
    # Question answering errors
    INVALID_QUESTION = "INVALID_QUESTION"
    NO_DOCUMENTS_INDEXED = "NO_DOCUMENTS_INDEXED"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    LLM_SERVICE_UNAVAILABLE = "LLM_SERVICE_UNAVAILABLE"
    LLM_API_ERROR = "LLM_API_ERROR"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    QUESTION_PROCESSING_FAILED = "QUESTION_PROCESSING_FAILED"
    
    # External service errors
    OPENAI_API_ERROR = "OPENAI_API_ERROR"
    VECTOR_DB_CONNECTION_ERROR = "VECTOR_DB_CONNECTION_ERROR"
    EMBEDDING_SERVICE_ERROR = "EMBEDDING_SERVICE_ERROR"


class PDFQAException(Exception):
    """
    Base exception class for all PDF Q&A System errors
    
    Provides structured error information with error codes, messages,
    and optional details for debugging and user feedback.
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize the exception
        
        Args:
            message: Human-readable error message
            error_code: Structured error code for programmatic handling
            details: Optional dictionary with additional error details
            original_exception: Original exception that caused this error (if any)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary format for API responses
        
        Returns:
            Dictionary representation of the error
        """
        error_dict = {
            "error": {
                "code": self.error_code.value,
                "message": self.message,
                "timestamp": self.timestamp
            }
        }
        
        if self.details:
            error_dict["error"]["details"] = self.details
        
        return error_dict
    
    def __str__(self) -> str:
        """String representation of the exception"""
        return f"{self.error_code.value}: {self.message}"


class FileHandlingError(PDFQAException):
    """Exception for file handling operations"""
    
    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        file_size: Optional[int] = None,
        error_code: ErrorCode = ErrorCode.FILE_SAVE_FAILED,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if filename:
            details["filename"] = filename
        if file_size is not None:
            details["file_size"] = file_size
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            original_exception=original_exception
        )


class DocumentProcessingError(PDFQAException):
    """Exception for document processing operations"""
    
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        filename: Optional[str] = None,
        processing_stage: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.DOCUMENT_PROCESSING_FAILED,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if document_id:
            details["document_id"] = document_id
        if filename:
            details["filename"] = filename
        if processing_stage:
            details["processing_stage"] = processing_stage
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            original_exception=original_exception
        )


class PDFProcessingError(DocumentProcessingError):
    """Exception for PDF processing operations"""
    
    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        page_number: Optional[int] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if page_number is not None:
            details["page_number"] = page_number
        
        super().__init__(
            message=message,
            filename=filename,
            processing_stage="pdf_extraction",
            error_code=ErrorCode.PDF_PROCESSING_FAILED,
            original_exception=original_exception
        )
        
        if page_number is not None:
            self.details["page_number"] = page_number


class TextChunkingError(DocumentProcessingError):
    """Exception for text chunking operations"""
    
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        text_length: Optional[int] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if text_length is not None:
            details["text_length"] = text_length
        
        super().__init__(
            message=message,
            document_id=document_id,
            processing_stage="text_chunking",
            error_code=ErrorCode.CHUNKING_FAILED,
            original_exception=original_exception
        )
        
        if text_length is not None:
            self.details["text_length"] = text_length


class EmbeddingError(PDFQAException):
    """Exception for embedding generation operations"""
    
    def __init__(
        self,
        message: str,
        text_count: Optional[int] = None,
        model_name: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if text_count is not None:
            details["text_count"] = text_count
        if model_name:
            details["model_name"] = model_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.EMBEDDING_GENERATION_FAILED,
            details=details,
            original_exception=original_exception
        )


class VectorStoreError(PDFQAException):
    """Exception for vector storage operations"""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        collection_name: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if operation:
            details["operation"] = operation
        if collection_name:
            details["collection_name"] = collection_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.VECTOR_STORAGE_FAILED,
            details=details,
            original_exception=original_exception
        )


class QuestionProcessingError(PDFQAException):
    """Exception for question processing operations"""
    
    def __init__(
        self,
        message: str,
        question: Optional[str] = None,
        processing_stage: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.QUESTION_PROCESSING_FAILED,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if question:
            # Truncate question for security/privacy
            details["question"] = question[:100] + "..." if len(question) > 100 else question
        if processing_stage:
            details["processing_stage"] = processing_stage
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            original_exception=original_exception
        )


class LLMServiceError(QuestionProcessingError):
    """Exception for LLM service operations"""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        tokens_used: Optional[int] = None,
        error_code: ErrorCode = ErrorCode.LLM_API_ERROR,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if model_name:
            details["model_name"] = model_name
        if tokens_used is not None:
            details["tokens_used"] = tokens_used
        
        super().__init__(
            message=message,
            processing_stage="llm_generation",
            error_code=error_code,
            original_exception=original_exception
        )
        
        # Add LLM-specific details
        self.details.update(details)


class RetrievalError(QuestionProcessingError):
    """Exception for document retrieval operations"""
    
    def __init__(
        self,
        message: str,
        query_text: Optional[str] = None,
        top_k: Optional[int] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if top_k is not None:
            details["top_k"] = top_k
        
        super().__init__(
            message=message,
            question=query_text,
            processing_stage="document_retrieval",
            error_code=ErrorCode.RETRIEVAL_FAILED,
            original_exception=original_exception
        )
        
        if top_k is not None:
            self.details["top_k"] = top_k


class ServiceUnavailableError(PDFQAException):
    """Exception for service unavailability"""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        retry_after: Optional[int] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if service_name:
            details["service_name"] = service_name
        if retry_after is not None:
            details["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=message,
            error_code=ErrorCode.SERVICE_UNAVAILABLE,
            details=details,
            original_exception=original_exception
        )


class ValidationError(PDFQAException):
    """Exception for input validation errors"""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        validation_rule: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if field_name:
            details["field_name"] = field_name
        if field_value is not None:
            # Convert to string and truncate for safety
            value_str = str(field_value)
            details["field_value"] = value_str[:100] + "..." if len(value_str) > 100 else value_str
        if validation_rule:
            details["validation_rule"] = validation_rule
        
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=details,
            original_exception=original_exception
        )


# Convenience functions for creating common exceptions

def create_file_too_large_error(filename: str, file_size: int, max_size: int) -> FileHandlingError:
    """Create a file too large error"""
    return FileHandlingError(
        message=f"File '{filename}' exceeds maximum size limit of {max_size} bytes",
        filename=filename,
        file_size=file_size,
        error_code=ErrorCode.FILE_TOO_LARGE
    )


def create_invalid_file_type_error(filename: str, supported_types: list) -> FileHandlingError:
    """Create an invalid file type error"""
    return FileHandlingError(
        message=f"File '{filename}' is not supported. Supported types: {', '.join(supported_types)}",
        filename=filename,
        error_code=ErrorCode.INVALID_FILE_TYPE
    )


def create_no_documents_error() -> QuestionProcessingError:
    """Create a no documents indexed error"""
    return QuestionProcessingError(
        message="No documents have been uploaded and indexed yet. Please upload PDF documents first.",
        error_code=ErrorCode.NO_DOCUMENTS_INDEXED
    )


def create_llm_unavailable_error(service_name: str = "OpenAI") -> LLMServiceError:
    """Create an LLM service unavailable error"""
    return LLMServiceError(
        message=f"{service_name} service is currently unavailable. Please check your API configuration.",
        error_code=ErrorCode.LLM_SERVICE_UNAVAILABLE
    )