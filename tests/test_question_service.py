"""
Tests for the QuestionService
"""
import pytest
from unittest.mock import Mock, MagicMock
from models.api import QuestionRequest, QuestionResponse
from models.question import Question, Reference
from services.question_service import QuestionService
from services.llm_service import LLMResponse, ContextChunk


@pytest.fixture
def mock_retrieval_service():
    """Create a mock retrieval service"""
    mock_service = Mock()
    mock_service.find_relevant_chunks.return_value = []
    mock_service.create_references.return_value = []
    mock_service.embedding_service = Mock()
    mock_service.embedding_service.generate_embedding.return_value = [0.1] * 384
    mock_service.vector_store = Mock()
    mock_service.vector_store.get_collection_stats.return_value = {"total_chunks": 10}
    mock_service.get_retrieval_stats.return_value = {"test": "stats"}
    return mock_service


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service"""
    mock_service = Mock()
    mock_service.is_available.return_value = True
    mock_service.get_model_info.return_value = {"model": "test"}
    mock_service.generate_answer.return_value = LLMResponse(
        answer="Test answer",
        tokens_used=100,
        processing_time_ms=500,
        model_used="test-model",
        confidence_score=0.8
    )
    mock_service.generate_simple_answer.return_value = LLMResponse(
        answer="No relevant information found",
        tokens_used=0,
        processing_time_ms=0,
        model_used="fallback",
        confidence_score=0.0
    )
    return mock_service


@pytest.fixture
def sample_references():
    """Create sample references"""
    return [
        Reference(
            chunk_id="chunk-1",
            document_filename="test.pdf",
            page_number=1,
            excerpt="This is a test excerpt",
            relevance_score=0.9
        ),
        Reference(
            chunk_id="chunk-2", 
            document_filename="test2.pdf",
            page_number=2,
            excerpt="Another test excerpt",
            relevance_score=0.7
        )
    ]


class TestQuestionService:
    """Test cases for QuestionService"""
    
    def test_initialization(self, mock_retrieval_service, mock_llm_service):
        """Test service initialization"""
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        assert service.retrieval_service == mock_retrieval_service
        assert service.llm_service == mock_llm_service
        assert service.default_max_references > 0
        assert service.min_chunks_for_answer > 0
    
    def test_answer_question_success(self, mock_retrieval_service, mock_llm_service, sample_references):
        """Test successful question answering"""
        # Setup mocks
        mock_chunk = Mock()
        mock_chunk.id = "chunk-1"
        mock_chunk.document_id = "doc-1"
        mock_chunk.content = "Test content"
        mock_chunk.metadata = Mock()
        mock_chunk.metadata.page_number = 1
        
        mock_retrieval_service.find_relevant_chunks.return_value = [(mock_chunk, 0.9)]
        mock_retrieval_service.create_references.return_value = sample_references
        
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        request = QuestionRequest(
            question="What is the test about?",
            language="en",
            max_references=5
        )
        
        response = service.answer_question(request)
        
        assert isinstance(response, QuestionResponse)
        assert response.answer == "Test answer"
        assert response.confidence_score == 0.8
        assert len(response.references) == 2
        assert response.processing_time_ms >= 0
        
        # Verify service calls
        mock_retrieval_service.find_relevant_chunks.assert_called_once()
        mock_llm_service.generate_answer.assert_called_once()
        mock_retrieval_service.create_references.assert_called_once()
    
    def test_answer_question_no_content(self, mock_retrieval_service, mock_llm_service):
        """Test question answering when no relevant content is found"""
        # Setup mocks for no content scenario
        mock_retrieval_service.find_relevant_chunks.return_value = []
        
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        request = QuestionRequest(question="What is this about?")
        response = service.answer_question(request)
        
        assert isinstance(response, QuestionResponse)
        assert response.answer == "No relevant information found"
        assert response.confidence_score == 0.0
        assert len(response.references) == 0
        
        # Should use simple answer generation
        mock_llm_service.generate_simple_answer.assert_called_once()
        mock_llm_service.generate_answer.assert_not_called()
    
    def test_answer_question_empty_question(self, mock_retrieval_service, mock_llm_service):
        """Test handling of empty question validation"""
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        # Test that Pydantic validation catches empty questions
        with pytest.raises(ValueError):
            QuestionRequest(question="")
    
    def test_answer_question_with_language_filter(self, mock_retrieval_service, mock_llm_service):
        """Test question answering with language filter"""
        mock_chunk = Mock()
        mock_chunk.id = "chunk-1"
        mock_chunk.document_id = "doc-1"
        mock_chunk.content = "Test content"
        mock_chunk.metadata = Mock()
        mock_chunk.metadata.page_number = 1
        
        mock_retrieval_service.find_relevant_chunks.return_value = [(mock_chunk, 0.9)]
        mock_retrieval_service.create_references.return_value = []
        
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        request = QuestionRequest(
            question="What is this?",
            language="pt",
            max_references=3
        )
        
        service.answer_question(request)
        
        # Check that language filter was passed
        call_args = mock_retrieval_service.find_relevant_chunks.call_args
        assert call_args[1]["language_filter"] == "pt"
    
    def test_convert_to_context_chunks(self, mock_retrieval_service, mock_llm_service):
        """Test conversion of chunks to context format"""
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        # Create mock chunk with document_filename attribute
        mock_chunk = Mock()
        mock_chunk.id = "chunk-1"
        mock_chunk.document_id = "test.pdf"
        mock_chunk.document_filename = "test.pdf"  # Set the attribute directly
        mock_chunk.content = "Test content for context"
        mock_chunk.metadata = Mock()
        mock_chunk.metadata.page_number = 5
        
        chunks_with_scores = [(mock_chunk, 0.85)]
        context_chunks = service._convert_to_context_chunks(chunks_with_scores)
        
        assert len(context_chunks) == 1
        assert isinstance(context_chunks[0], ContextChunk)
        assert context_chunks[0].content == "Test content for context"
        assert context_chunks[0].source == "test.pdf"
        assert context_chunks[0].page_number == 5
        assert context_chunks[0].relevance_score == 0.85
    
    def test_convert_to_context_chunks_filename_extraction(self, mock_retrieval_service, mock_llm_service):
        """Test filename extraction from document_id"""
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        # Create mock chunk with path-like document_id
        mock_chunk = Mock()
        mock_chunk.id = "chunk-1"
        mock_chunk.document_id = "path/to/document.pdf"
        mock_chunk.content = "Test content"
        mock_chunk.metadata = Mock()
        mock_chunk.metadata.page_number = 1
        
        # Mock getattr to return document_id (no separate filename)
        def mock_getattr(obj, attr, default=None):
            if attr == 'document_filename':
                return default
            return getattr(obj, attr, default)
        
        import builtins
        original_getattr = builtins.getattr
        builtins.getattr = mock_getattr
        
        try:
            context_chunks = service._convert_to_context_chunks([(mock_chunk, 0.9)])
            assert context_chunks[0].source == "document.pdf"
        finally:
            builtins.getattr = original_getattr
    
    def test_reference_to_dict(self, mock_retrieval_service, mock_llm_service):
        """Test reference conversion to dictionary"""
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        reference = Reference(
            chunk_id="chunk-1",
            document_filename="test.pdf",
            page_number=3,
            excerpt="Test excerpt text",
            relevance_score=0.75
        )
        
        ref_dict = service._reference_to_dict(reference)
        
        expected_keys = ["document_filename", "page_number", "excerpt", "relevance_score"]
        assert all(key in ref_dict for key in expected_keys)
        assert ref_dict["document_filename"] == "test.pdf"
        assert ref_dict["page_number"] == 3
        assert ref_dict["excerpt"] == "Test excerpt text"
        assert ref_dict["relevance_score"] == 0.75
    
    def test_create_question_object(self, mock_retrieval_service, mock_llm_service):
        """Test creating Question object with embedding"""
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        question_text = "What is the maximum temperature?"
        question = service.create_question_object(question_text, language="en")
        
        assert isinstance(question, Question)
        assert question.text == question_text
        assert question.language == "en"
        assert question.embedding is not None
        assert len(question.embedding) == 384
        
        mock_retrieval_service.embedding_service.generate_embedding.assert_called_once_with(question_text)
    
    def test_is_service_ready_true(self, mock_retrieval_service, mock_llm_service):
        """Test service readiness check when ready"""
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        assert service.is_service_ready() is True
        
        mock_llm_service.is_available.assert_called_once()
        mock_retrieval_service.vector_store.get_collection_stats.assert_called_once()
    
    def test_is_service_ready_llm_unavailable(self, mock_retrieval_service, mock_llm_service):
        """Test service readiness when LLM is unavailable"""
        mock_llm_service.is_available.return_value = False
        
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        assert service.is_service_ready() is False
    
    def test_is_service_ready_no_documents(self, mock_retrieval_service, mock_llm_service):
        """Test service readiness when no documents are indexed"""
        mock_retrieval_service.vector_store.get_collection_stats.return_value = {"total_chunks": 0}
        
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        assert service.is_service_ready() is False
    
    def test_get_service_stats(self, mock_retrieval_service, mock_llm_service):
        """Test getting service statistics"""
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        stats = service.get_service_stats()
        
        assert "service_ready" in stats
        assert "llm_available" in stats
        assert "llm_model_info" in stats
        assert "retrieval_stats" in stats
        assert "configuration" in stats
        
        # Verify all service calls were made
        mock_llm_service.is_available.assert_called()
        mock_llm_service.get_model_info.assert_called_once()
        mock_retrieval_service.get_retrieval_stats.assert_called_once()


class TestQuestionServiceIntegration:
    """Integration tests for QuestionService"""
    
    def test_full_question_answering_workflow(self, mock_retrieval_service, mock_llm_service, sample_references):
        """Test complete question answering workflow"""
        # Setup comprehensive mocks
        mock_chunk = Mock()
        mock_chunk.id = "chunk-1"
        mock_chunk.document_id = "manual.pdf"
        mock_chunk.content = "The maximum operating temperature is 85°C."
        mock_chunk.metadata = Mock()
        mock_chunk.metadata.page_number = 5
        
        mock_retrieval_service.find_relevant_chunks.return_value = [(mock_chunk, 0.92)]
        mock_retrieval_service.create_references.return_value = sample_references
        
        service = QuestionService(mock_retrieval_service, mock_llm_service)
        
        # Execute full workflow
        request = QuestionRequest(
            question="What is the maximum operating temperature?",
            language="en",
            max_references=3
        )
        
        response = service.answer_question(request)
        
        # Verify complete response
        assert isinstance(response, QuestionResponse)
        assert response.answer == "Test answer"
        assert response.confidence_score == 0.8
        assert len(response.references) == 2
        assert response.processing_time_ms >= 0
        
        # Verify all service interactions
        mock_retrieval_service.find_relevant_chunks.assert_called_once()
        mock_llm_service.generate_answer.assert_called_once()
        mock_retrieval_service.create_references.assert_called_once()
        
        # Verify LLM received proper context
        llm_call_args = mock_llm_service.generate_answer.call_args
        context_chunks = llm_call_args[0][1]
        assert len(context_chunks) == 1
        assert isinstance(context_chunks[0], ContextChunk)