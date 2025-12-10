"""
Tests for LLM service functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from services.llm_service import (
    LLMService, LLMResponse, ContextChunk, TokenCounter, PromptTemplate
)


class TestTokenCounter:
    """Test token counting utilities"""
    
    def test_estimate_tokens(self):
        """Test token estimation"""
        # Test empty string
        assert TokenCounter.estimate_tokens("") == 0
        
        # Test short text
        short_text = "Hello world"
        tokens = TokenCounter.estimate_tokens(short_text)
        assert tokens > 0
        assert tokens < len(short_text)  # Should be less than character count
        
        # Test longer text
        long_text = "This is a much longer text that should have more tokens " * 10
        long_tokens = TokenCounter.estimate_tokens(long_text)
        assert long_tokens > tokens
    
    def test_truncate_to_token_limit(self):
        """Test text truncation to token limits"""
        text = "This is a test text that needs to be truncated"
        
        # Test no truncation needed
        result = TokenCounter.truncate_to_token_limit(text, 1000)
        assert result == text
        
        # Test truncation needed
        result = TokenCounter.truncate_to_token_limit(text, 5)
        assert len(result) < len(text)
        assert TokenCounter.estimate_tokens(result) <= 5


class TestPromptTemplate:
    """Test prompt template functionality"""
    
    def test_format_context_empty(self):
        """Test formatting empty context"""
        result = PromptTemplate.format_context([])
        assert result == ""
    
    def test_format_context_with_chunks(self):
        """Test formatting context with chunks"""
        chunks = [
            ContextChunk(
                content="This is test content",
                source="test.pdf",
                page_number=1,
                relevance_score=0.9
            ),
            ContextChunk(
                content="More test content",
                source="test2.pdf", 
                page_number=2,
                relevance_score=0.8
            )
        ]
        
        result = PromptTemplate.format_context(chunks)
        assert "test.pdf" in result
        assert "test2.pdf" in result
        assert "This is test content" in result
        assert "More test content" in result
        assert "Page 1" in result
        assert "Page 2" in result
    
    def test_create_prompt_no_context(self):
        """Test prompt creation without context"""
        question = "What is the answer?"
        result = PromptTemplate.create_prompt(question, [])
        
        assert question in result
        assert "don't have any relevant information" in result
    
    def test_create_prompt_with_context(self):
        """Test prompt creation with context"""
        question = "What is the answer?"
        chunks = [
            ContextChunk(
                content="The answer is 42",
                source="guide.pdf",
                page_number=1,
                relevance_score=0.9
            )
        ]
        
        result = PromptTemplate.create_prompt(question, chunks)
        assert question in result
        assert "The answer is 42" in result
        assert "guide.pdf" in result


class TestLLMService:
    """Test LLM service functionality"""
    
    def test_initialization_without_api_key(self):
        """Test service initialization without API key"""
        service = LLMService(api_key=None)
        assert not service.is_available()
        assert service.client is None
    
    def test_initialization_with_api_key(self):
        """Test service initialization with API key"""
        with patch('services.llm_service.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            service = LLMService(api_key="test-key")
            assert service.is_available()
            assert service.client == mock_client
            mock_openai.assert_called_once_with(api_key="test-key")
    
    def test_select_context_chunks_empty(self):
        """Test context selection with empty chunks"""
        service = LLMService(api_key="test-key")
        result = service._select_context_chunks([], "test question")
        assert result == []
    
    def test_select_context_chunks_sorting(self):
        """Test context chunks are sorted by relevance"""
        service = LLMService(api_key="test-key")
        
        chunks = [
            ContextChunk("Low relevance", "test1.pdf", 1, 0.3),
            ContextChunk("High relevance", "test2.pdf", 1, 0.9),
            ContextChunk("Medium relevance", "test3.pdf", 1, 0.6)
        ]
        
        result = service._select_context_chunks(chunks, "short question")
        
        # Should be sorted by relevance (highest first)
        assert result[0].relevance_score == 0.9
        assert result[1].relevance_score == 0.6
        assert result[2].relevance_score == 0.3
    
    def test_select_context_chunks_token_limit(self):
        """Test context selection respects token limits"""
        service = LLMService(api_key="test-key")
        service.max_context_tokens = 100  # Very small limit for testing
        
        # Create chunks with long content
        long_content = "This is very long content " * 50
        chunks = [
            ContextChunk(long_content, "test1.pdf", 1, 0.9),
            ContextChunk(long_content, "test2.pdf", 1, 0.8)
        ]
        
        result = service._select_context_chunks(chunks, "question")
        
        # Should not include all chunks due to token limit
        total_tokens = sum(TokenCounter.estimate_tokens(chunk.content) for chunk in result)
        assert total_tokens < service.max_context_tokens
    
    @patch('services.llm_service.OpenAI')
    def test_generate_answer_success(self, mock_openai):
        """Test successful answer generation"""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is the answer"
        mock_response.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_response
        
        service = LLMService(api_key="test-key")
        
        chunks = [
            ContextChunk("Relevant content", "test.pdf", 1, 0.9)
        ]
        
        result = service.generate_answer("What is the answer?", chunks)
        
        assert isinstance(result, LLMResponse)
        assert result.answer == "This is the answer"
        assert result.tokens_used == 150
        assert result.model_used == "gpt-3.5-turbo"
        assert result.confidence_score > 0
        assert result.processing_time_ms >= 0
    
    @patch('services.llm_service.OpenAI')
    def test_generate_answer_with_fallback(self, mock_openai):
        """Test answer generation with fallback model"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Fallback answer"
        mock_response.usage.total_tokens = 100
        
        mock_client.chat.completions.create.side_effect = [
            Exception("Primary model failed"),
            mock_response
        ]
        
        service = LLMService(api_key="test-key", model="gpt-4")
        service.fallback_model = "gpt-3.5-turbo"
        
        chunks = [ContextChunk("Content", "test.pdf", 1, 0.9)]
        result = service.generate_answer("Question?", chunks)
        
        assert result.answer == "Fallback answer"
        assert result.model_used == "gpt-3.5-turbo"
    
    def test_generate_answer_no_api_key(self):
        """Test answer generation without API key"""
        service = LLMService(api_key=None)
        
        with pytest.raises(RuntimeError, match="not properly initialized"):
            service.generate_answer("Question?", [])
    
    @patch('services.llm_service.OpenAI')
    def test_generate_answer_empty_question(self, mock_openai):
        """Test answer generation with empty question"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        service = LLMService(api_key="test-key")
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            service.generate_answer("", [])
    
    def test_generate_simple_answer(self):
        """Test simple answer generation without context"""
        service = LLMService(api_key="test-key")
        
        result = service.generate_simple_answer("What is the answer?")
        
        assert isinstance(result, LLMResponse)
        assert "don't have any relevant information" in result.answer
        assert "What is the answer?" in result.answer
        assert result.tokens_used == 0
        assert result.model_used == "fallback"
        assert result.confidence_score == 0.0
    
    def test_get_model_info(self):
        """Test model information retrieval"""
        service = LLMService(api_key="test-key", model="gpt-4")
        
        info = service.get_model_info()
        
        assert info["primary_model"] == "gpt-4"
        assert info["fallback_model"] == "gpt-3.5-turbo"
        assert "available" in info


class TestContextChunk:
    """Test ContextChunk data class"""
    
    def test_context_chunk_creation(self):
        """Test creating a context chunk"""
        chunk = ContextChunk(
            content="Test content",
            source="test.pdf",
            page_number=1,
            relevance_score=0.85
        )
        
        assert chunk.content == "Test content"
        assert chunk.source == "test.pdf"
        assert chunk.page_number == 1
        assert chunk.relevance_score == 0.85


class TestLLMResponse:
    """Test LLMResponse data class"""
    
    def test_llm_response_creation(self):
        """Test creating an LLM response"""
        response = LLMResponse(
            answer="Test answer",
            tokens_used=100,
            processing_time_ms=500,
            model_used="gpt-3.5-turbo",
            confidence_score=0.8
        )
        
        assert response.answer == "Test answer"
        assert response.tokens_used == 100
        assert response.processing_time_ms == 500
        assert response.model_used == "gpt-3.5-turbo"
        assert response.confidence_score == 0.8