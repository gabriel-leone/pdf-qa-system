"""
Error handling middleware and utilities for the PDF Q&A System

This module provides comprehensive error handling middleware, exception handlers,
and utilities for consistent error responses across the application.
"""
import logging
import time
import traceback
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import openai

from utils.exceptions import (
    PDFQAException, ErrorCode, FileHandlingError, DocumentProcessingError,
    QuestionProcessingError, LLMServiceError, ServiceUnavailableError,
    ValidationError, VectorStoreError, EmbeddingError
)

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware:
    """
    Middleware for handling errors and providing consistent error responses
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """ASGI middleware implementation"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        try:
            await self.app(scope, receive, send)
        except Exception as e:
            response = await self.handle_error(request, e)
            await response(scope, receive, send)
    
    async def handle_error(self, request: Request, exc: Exception) -> JSONResponse:
        """
        Handle different types of exceptions and return appropriate responses
        
        Args:
            request: The FastAPI request object
            exc: The exception that occurred
            
        Returns:
            JSONResponse with error details
        """
        # Log the error with context
        self._log_error(request, exc)
        
        # Handle different exception types
        if isinstance(exc, PDFQAException):
            return self._handle_pdfqa_exception(exc)
        elif isinstance(exc, HTTPException):
            return self._handle_http_exception(exc)
        elif isinstance(exc, RequestValidationError):
            return self._handle_validation_exception(exc)
        elif isinstance(exc, openai.OpenAIError):
            return self._handle_openai_exception(exc)
        else:
            return self._handle_generic_exception(exc)
    
    def _log_error(self, request: Request, exc: Exception) -> None:
        """Log error with request context"""
        error_id = f"error_{int(time.time() * 1000)}"
        
        # Create error context
        context = {
            "error_id": error_id,
            "method": request.method,
            "url": str(request.url),
            "user_agent": request.headers.get("user-agent", "unknown"),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)
        }
        
        # Add client IP if available
        client_ip = request.client.host if request.client else "unknown"
        context["client_ip"] = client_ip
        
        # Log based on exception type
        if isinstance(exc, (PDFQAException, HTTPException)):
            logger.warning(f"Handled exception: {context}")
        else:
            logger.error(f"Unhandled exception: {context}")
            # Log full traceback for unexpected errors
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _handle_pdfqa_exception(self, exc: PDFQAException) -> JSONResponse:
        """Handle custom PDF Q&A exceptions"""
        status_code = self._get_status_code_for_error_code(exc.error_code)
        return JSONResponse(
            status_code=status_code,
            content=exc.to_dict()
        )
    
    def _handle_http_exception(self, exc: HTTPException) -> JSONResponse:
        """Handle FastAPI HTTP exceptions"""
        # If detail is already a dict (our custom format), return as-is
        if isinstance(exc.detail, dict):
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail
            )
        
        # Otherwise, format as standard error
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": str(exc.detail),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            }
        )
    
    def _handle_validation_exception(self, exc: RequestValidationError) -> JSONResponse:
        """Handle Pydantic validation errors"""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": ErrorCode.VALIDATION_ERROR.value,
                    "message": "Request validation failed",
                    "details": {
                        "validation_errors": exc.errors()
                    },
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            }
        )
    
    def _handle_openai_exception(self, exc: openai.OpenAIError) -> JSONResponse:
        """Handle OpenAI API exceptions"""
        if isinstance(exc, openai.RateLimitError):
            error_code = ErrorCode.LLM_RATE_LIMIT
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
            message = "Rate limit exceeded for LLM service. Please try again later."
        elif isinstance(exc, openai.APITimeoutError):
            error_code = ErrorCode.LLM_TIMEOUT
            status_code = status.HTTP_504_GATEWAY_TIMEOUT
            message = "LLM service request timed out. Please try again."
        elif isinstance(exc, openai.AuthenticationError):
            error_code = ErrorCode.LLM_SERVICE_UNAVAILABLE
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            message = "LLM service authentication failed. Please check configuration."
        else:
            error_code = ErrorCode.LLM_API_ERROR
            status_code = status.HTTP_502_BAD_GATEWAY
            message = f"LLM service error: {str(exc)}"
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": error_code.value,
                    "message": message,
                    "details": {
                        "service": "OpenAI",
                        "error_type": type(exc).__name__
                    },
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            }
        )
    
    def _handle_generic_exception(self, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions"""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": ErrorCode.INTERNAL_SERVER_ERROR.value,
                    "message": "An unexpected error occurred",
                    "details": {
                        "error_type": type(exc).__name__
                    },
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            }
        )
    
    def _get_status_code_for_error_code(self, error_code: ErrorCode) -> int:
        """Map error codes to HTTP status codes"""
        status_map = {
            ErrorCode.VALIDATION_ERROR: status.HTTP_400_BAD_REQUEST,
            ErrorCode.INVALID_QUESTION: status.HTTP_400_BAD_REQUEST,
            ErrorCode.FILE_TOO_LARGE: status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            ErrorCode.INVALID_FILE_TYPE: status.HTTP_400_BAD_REQUEST,
            ErrorCode.EMPTY_FILE: status.HTTP_400_BAD_REQUEST,
            ErrorCode.NO_DOCUMENTS_INDEXED: status.HTTP_404_NOT_FOUND,
            ErrorCode.FILE_NOT_FOUND: status.HTTP_404_NOT_FOUND,
            ErrorCode.SERVICE_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
            ErrorCode.LLM_SERVICE_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
            ErrorCode.LLM_RATE_LIMIT: status.HTTP_429_TOO_MANY_REQUESTS,
            ErrorCode.LLM_TIMEOUT: status.HTTP_504_GATEWAY_TIMEOUT,
            ErrorCode.DOCUMENT_PROCESSING_FAILED: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.PDF_PROCESSING_FAILED: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.CHUNKING_FAILED: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.EMBEDDING_GENERATION_FAILED: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.VECTOR_STORAGE_FAILED: status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.QUESTION_PROCESSING_FAILED: status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.RETRIEVAL_FAILED: status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.LLM_API_ERROR: status.HTTP_502_BAD_GATEWAY,
        }
        
        return status_map.get(error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external service calls
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func):
        """Decorator implementation"""
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise ServiceUnavailableError(
                        message=f"Service temporarily unavailable (circuit breaker open)",
                        service_name=func.__name__,
                        retry_after=self.recovery_timeout
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RetryHandler:
    """
    Retry handler with exponential backoff
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        retryable_exceptions: tuple = (Exception,)
    ):
        """
        Initialize retry handler
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Multiplier for exponential backoff
            retryable_exceptions: Tuple of exception types that should trigger retry
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retryable_exceptions = retryable_exceptions
    
    def __call__(self, func):
        """Decorator implementation"""
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except self.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == self.max_retries:
                        # Final attempt failed
                        logger.error(f"Function {func.__name__} failed after {self.max_retries} retries: {e}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper


# Utility functions for error handling

def log_processing_step(step_name: str, details: Optional[Dict[str, Any]] = None):
    """
    Log a processing step with optional details
    
    Args:
        step_name: Name of the processing step
        details: Optional dictionary with step details
    """
    log_message = f"Processing step: {step_name}"
    if details:
        log_message += f" - {details}"
    
    logger.info(log_message)


def log_performance_metric(operation: str, duration_ms: int, details: Optional[Dict[str, Any]] = None):
    """
    Log performance metrics for operations
    
    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        details: Optional dictionary with additional details
    """
    log_message = f"Performance: {operation} completed in {duration_ms}ms"
    if details:
        log_message += f" - {details}"
    
    logger.info(log_message)


def create_error_response(
    error_code: ErrorCode,
    message: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """
    Create a standardized error response
    
    Args:
        error_code: The error code
        message: Error message
        status_code: HTTP status code
        details: Optional error details
        
    Returns:
        JSONResponse with error information
    """
    error_dict = {
        "error": {
            "code": error_code.value,
            "message": message,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    }
    
    if details:
        error_dict["error"]["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=error_dict
    )


def handle_service_degradation(service_name: str, error: Exception) -> Dict[str, Any]:
    """
    Handle graceful service degradation
    
    Args:
        service_name: Name of the failing service
        error: The error that occurred
        
    Returns:
        Dictionary with degradation information
    """
    logger.warning(f"Service degradation detected for {service_name}: {error}")
    
    return {
        "service": service_name,
        "status": "degraded",
        "error": str(error),
        "fallback_available": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }