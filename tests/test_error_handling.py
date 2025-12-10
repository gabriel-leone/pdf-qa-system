"""
Tests for comprehensive error handling implementation
"""
import pytest
import time
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from main import app
from utils.exceptions import (
    PDFQAException, ErrorCode, FileHandlingError, DocumentProcessingError,
    LLMServiceError, ValidationError, create_file_too_large_error
)
from utils.error_handlers import CircuitBreaker, RetryHandler


class TestCustomExceptions:
    """Test custom exception classes"""
    
    def test_pdfqa_exception_creation(self):
        """Test basic PDFQAException creation"""
        exc = PDFQAException(
            message="Test error",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"field": "test"}
        )
        
        assert exc.message == "Test error"
        assert exc.error_code == ErrorCode.VALIDATION_ERROR
        assert exc.details == {"field": "test"}
        assert exc.timestamp is not None
    
    def test_pdfqa_exception_to_dict(self):
        """Test PDFQAException to_dict conversion"""
        exc = PDFQAException(
            message="Test error",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"field": "test"}
        )
        
        result = exc.to_dict()
        
        assert "error" in result
        assert result["error"]["code"] == ErrorCode.VALIDATION_ERROR.value
        assert result["error"]["message"] == "Test error"
        assert result["error"]["details"] == {"field": "test"}
        assert "timestamp" in result["error"]
    
    def test_file_handling_error(self):
        """Test FileHandlingError with file details"""
        exc = FileHandlingError(
            message="File too large",
            filename="test.pdf",
            file_size=1000000,
            error_code=ErrorCode.FILE_TOO_LARGE
        )
        
        assert exc.message == "File too large"
        assert exc.details["filename"] == "test.pdf"
        assert exc.details["file_size"] == 1000000
        assert exc.error_code == ErrorCode.FILE_TOO_LARGE
    
    def test_document_processing_error(self):
        """Test DocumentProcessingError with processing details"""
        exc = DocumentProcessingError(
            message="Processing failed",
            document_id="doc123",
            filename="test.pdf",
            processing_stage="text_extraction"
        )
        
        assert exc.message == "Processing failed"
        assert exc.details["document_id"] == "doc123"
        assert exc.details["filename"] == "test.pdf"
        assert exc.details["processing_stage"] == "text_extraction"
    
    def test_llm_service_error(self):
        """Test LLMServiceError with model details"""
        exc = LLMServiceError(
            message="API error",
            model_name="gpt-3.5-turbo",
            tokens_used=150,
            error_code=ErrorCode.LLM_API_ERROR
        )
        
        assert exc.message == "API error"
        assert exc.details["model_name"] == "gpt-3.5-turbo"
        assert exc.details["tokens_used"] == 150
        assert exc.error_code == ErrorCode.LLM_API_ERROR
    
    def test_validation_error(self):
        """Test ValidationError with field details"""
        exc = ValidationError(
            message="Invalid input",
            field_name="question",
            field_value="",
            validation_rule="non_empty"
        )
        
        assert exc.message == "Invalid input"
        assert exc.details["field_name"] == "question"
        assert exc.details["field_value"] == ""
        assert exc.details["validation_rule"] == "non_empty"
    
    def test_create_file_too_large_error(self):
        """Test convenience function for file too large error"""
        exc = create_file_too_large_error("test.pdf", 1000000, 500000)
        
        assert isinstance(exc, FileHandlingError)
        assert exc.error_code == ErrorCode.FILE_TOO_LARGE
        assert "exceeds maximum size limit" in exc.message
        assert exc.details["filename"] == "test.pdf"
        assert exc.details["file_size"] == 1000000


class TestErrorHandlers:
    """Test error handling utilities"""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state"""
        @CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        def test_function():
            return "success"
        
        # Should work normally in closed state
        result = test_function()
        assert result == "success"
    
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker opening after failures"""
        call_count = 0
        
        @CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Test failure")
        
        # First two calls should fail normally
        with pytest.raises(Exception, match="Test failure"):
            failing_function()
        
        with pytest.raises(Exception, match="Test failure"):
            failing_function()
        
        # Third call should trigger circuit breaker
        with pytest.raises(Exception, match="temporarily unavailable"):
            failing_function()
        
        # Should have only called the function twice
        assert call_count == 2
    
    def test_retry_handler_success(self):
        """Test retry handler with successful call"""
        call_count = 0
        
        @RetryHandler(max_retries=2, base_delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_handler_eventual_success(self):
        """Test retry handler with eventual success"""
        call_count = 0
        
        @RetryHandler(max_retries=2, base_delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 2
    
    def test_retry_handler_max_retries(self):
        """Test retry handler reaching max retries"""
        call_count = 0
        
        @RetryHandler(max_retries=2, base_delay=0.01)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent failure")
        
        with pytest.raises(Exception, match="Persistent failure"):
            always_failing_function()
        
        # Should have called 3 times (initial + 2 retries)
        assert call_count == 3


class TestAPIErrorHandling:
    """Test API error handling integration"""
    
    def setup_method(self):
        """Set up test client"""
        try:
            self.client = TestClient(app)
        except TypeError:
            # Handle compatibility issues with different FastAPI/Starlette versions
            pytest.skip("TestClient compatibility issue - skipping API tests")
    
    def test_health_endpoint(self):
        """Test basic health endpoint works"""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
    
    def test_validation_error_handling(self):
        """Test validation error handling in API"""
        # Test with invalid question request (missing required fields)
        response = self.client.post("/question/", json={})
        
        assert response.status_code == 422
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert "field_errors" in data["error"]["details"]
    
    def test_file_upload_validation(self):
        """Test file upload validation"""
        # Test with no files
        response = self.client.post("/documents/")
        
        assert response.status_code == 422
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
    
    def test_invalid_file_type_error(self):
        """Test invalid file type error handling"""
        # Create a fake non-PDF file
        fake_file = ("test.txt", b"This is not a PDF", "text/plain")
        
        response = self.client.post(
            "/documents/",
            files={"files": fake_file}
        )
        
        assert response.status_code == 400
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == "INVALID_FILE_TYPE"
        assert "test.txt" in data["error"]["message"]
    
    def test_empty_file_error(self):
        """Test empty file error handling"""
        # Create an empty PDF file
        empty_file = ("empty.pdf", b"", "application/pdf")
        
        response = self.client.post(
            "/documents/",
            files={"files": empty_file}
        )
        
        assert response.status_code == 400
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == "EMPTY_FILE"
        assert "empty.pdf" in data["error"]["message"]
    
    def test_file_too_large_error(self):
        """Test file too large error handling"""
        # Create a file that's too large (over 100MB)
        large_content = b"x" * (101 * 1024 * 1024)  # 101MB
        large_file = ("large.pdf", large_content, "application/pdf")
        
        response = self.client.post(
            "/documents/",
            files={"files": large_file}
        )
        
        assert response.status_code == 413
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == "FILE_TOO_LARGE"
        assert "large.pdf" in data["error"]["message"]
    
    @patch('api.dependencies.get_question_service')
    def test_service_unavailable_error(self, mock_get_service):
        """Test service unavailable error handling"""
        # Mock a service that's not ready
        mock_service = Mock()
        mock_service.is_service_ready.return_value = False
        mock_service.llm_service.is_available.return_value = False
        mock_get_service.return_value = mock_service
        
        response = self.client.post(
            "/question/",
            json={"question": "Test question"}
        )
        
        assert response.status_code == 503
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == "LLM_SERVICE_UNAVAILABLE"
    
    @patch('api.dependencies.get_question_service')
    @patch('api.dependencies.get_vector_store')
    def test_no_documents_error(self, mock_get_vector_store, mock_get_service):
        """Test no documents indexed error handling"""
        # Mock services
        mock_service = Mock()
        mock_service.is_service_ready.return_value = False
        mock_service.llm_service.is_available.return_value = True
        mock_get_service.return_value = mock_service
        
        mock_vector_store = Mock()
        mock_vector_store.get_collection_stats.return_value = {"total_chunks": 0}
        mock_get_vector_store.return_value = mock_vector_store
        
        response = self.client.post(
            "/question/",
            json={"question": "Test question"}
        )
        
        assert response.status_code == 404
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == "NO_DOCUMENTS_INDEXED"


class TestLoggingIntegration:
    """Test logging integration with error handling"""
    
    def test_structured_logging_format(self):
        """Test that structured logging is working"""
        from utils.logging import setup_logging
        
        logger = setup_logging(log_format="structured")
        assert logger is not None
    
    def test_performance_logger(self):
        """Test performance logging utility"""
        from utils.logging import get_performance_logger
        
        perf_logger = get_performance_logger()
        assert perf_logger is not None
        assert perf_logger.name == "performance"
    
    def test_security_logger(self):
        """Test security logging utility"""
        from utils.logging import get_security_logger, log_security_event
        
        security_logger = get_security_logger()
        assert security_logger is not None
        assert security_logger.name == "security"
        
        # Test logging a security event (should not raise)
        log_security_event(
            event_type="test_event",
            description="Test security event",
            severity="low"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])