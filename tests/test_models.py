"""
Tests for data models validation
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from models import (
    Document, Chunk, ChunkMetadata, ProcessingStatus,
    Question, Answer, Reference,
    DocumentUploadRequest, DocumentUploadResponse,
    QuestionRequest, QuestionResponse, ErrorResponse
)


class TestDocumentModels:
    """Test document-related models"""
    
    def test_document_creation_valid(self):
        """Test creating a valid document"""
        doc = Document(
            filename="test.pdf",
            file_size=1024,
            language="en",
            file_path="/uploads/test.pdf"
        )
        assert doc.filename == "test.pdf"
        assert doc.file_size == 1024
        assert doc.language == "en"
        assert doc.processing_status == ProcessingStatus.PENDING
        assert doc.chunk_count == 0
        assert isinstance(doc.upload_timestamp, datetime)
    
    def test_document_invalid_filename(self):
        """Test document with invalid filename"""
        with pytest.raises(ValidationError) as exc_info:
            Document(
                filename="test.txt",  # Not a PDF
                file_size=1024,
                language="en",
                file_path="/uploads/test.txt"
            )
        assert "Filename must have .pdf extension" in str(exc_info.value)
    
    def test_document_invalid_file_size(self):
        """Test document with invalid file size"""
        with pytest.raises(ValidationError) as exc_info:
            Document(
                filename="test.pdf",
                file_size=200 * 1024 * 1024,  # 200MB, exceeds limit
                language="en",
                file_path="/uploads/test.pdf"
            )
        assert "File size cannot exceed" in str(exc_info.value)
    
    def test_chunk_creation_valid(self):
        """Test creating a valid chunk"""
        metadata = ChunkMetadata(
            page_number=1,
            chunk_index=0,
            language="en",
            confidence_score=0.95
        )
        chunk = Chunk(
            document_id="doc-123",
            content="This is test content",
            metadata=metadata,
            start_position=0,
            end_position=20
        )
        assert chunk.document_id == "doc-123"
        assert chunk.content == "This is test content"
        assert chunk.start_position == 0
        assert chunk.end_position == 20
    
    def test_chunk_invalid_positions(self):
        """Test chunk with invalid positions"""
        metadata = ChunkMetadata(
            page_number=1,
            chunk_index=0,
            language="en",
            confidence_score=0.95
        )
        with pytest.raises(ValidationError) as exc_info:
            Chunk(
                document_id="doc-123",
                content="This is test content",
                metadata=metadata,
                start_position=20,
                end_position=10  # End before start
            )
        assert "end_position must be greater than start_position" in str(exc_info.value)


class TestQuestionModels:
    """Test question and answer models"""
    
    def test_question_creation_valid(self):
        """Test creating a valid question"""
        question = Question(
            text="What is the temperature?",
            language="en"
        )
        assert question.text == "What is the temperature?"
        assert question.language == "en"
        assert isinstance(question.timestamp, datetime)
    
    def test_question_invalid_text(self):
        """Test question with invalid text"""
        with pytest.raises(ValidationError) as exc_info:
            Question(
                text="  ",  # Only whitespace
                language="en"
            )
        assert "Question text cannot be empty or only whitespace" in str(exc_info.value)
    
    def test_answer_creation_valid(self):
        """Test creating a valid answer"""
        reference = Reference(
            chunk_id="chunk-123",
            document_filename="test.pdf",
            page_number=1,
            excerpt="Temperature is 85°C",
            relevance_score=0.9
        )
        answer = Answer(
            question_id="q-123",
            text="The temperature is 85°C",
            confidence_score=0.8,
            references=[reference],
            processing_time_ms=1000
        )
        assert answer.question_id == "q-123"
        assert answer.text == "The temperature is 85°C"
        assert len(answer.references) == 1
        assert answer.references[0].excerpt == "Temperature is 85°C"


class TestAPIModels:
    """Test API request/response models"""
    
    def test_question_request_valid(self):
        """Test creating a valid question request"""
        request = QuestionRequest(
            question="What is the maximum temperature?",
            language="en",
            max_references=3
        )
        assert request.question == "What is the maximum temperature?"
        assert request.language == "en"
        assert request.max_references == 3
    
    def test_question_request_invalid(self):
        """Test question request with invalid data"""
        with pytest.raises(ValidationError) as exc_info:
            QuestionRequest(
                question="  ",  # Only whitespace
                language="en"
            )
        assert "Question cannot be empty or only whitespace" in str(exc_info.value)
    
    def test_document_upload_response(self):
        """Test document upload response"""
        response = DocumentUploadResponse(
            message="Success",
            documents_indexed=2,
            total_chunks=50,
            processing_time_ms=2000
        )
        assert response.message == "Success"
        assert response.documents_indexed == 2
        assert response.total_chunks == 50
        assert response.processing_time_ms == 2000