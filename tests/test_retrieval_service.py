"""
Tests for the RetrievalService
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from models.document import Chunk, ChunkMetadata
from models.question import Reference
from services.retrieval_service import RetrievalService


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store"""
    mock_store = Mock()
    mock_store.search_similar.return_value = []
    mock_store.get_collection_stats.return_value = {"total_chunks": 10}
    return mock_store


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service"""
    mock_service = Mock()
    mock_service.generate_embedding.return_value = np.random.rand(384)
    mock_service.get_cache_stats.return_value = {"cache_size": 5}
    return mock_service


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing"""
    chunk1 = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="This is about temperature limits in industrial equipment.",
        embedding=np.random.rand(384).tolist(),
        metadata=ChunkMetadata(
            page_number=1,
            section_title="Temperature Specifications",
            chunk_index=0,
            language="en",
            confidence_score=0.95
        ),
        start_position=0,
        end_position=50
    )
    
    chunk2 = Chunk(
        id="chunk-2", 
        document_id="doc-2",
        content="Este texto fala sobre limites de temperatura em equipamentos.",
        embedding=np.random.rand(384).tolist(),
        metadata=ChunkMetadata(
            page_number=2,
            section_title="Especificações",
            chunk_index=1,
            language="pt",
            confidence_score=0.88
        ),
        start_position=51,
        end_position=100
    )
    
    return [chunk1, chunk2]


class TestRetrievalService:
    """Test cases for RetrievalService"""
    
    def test_initialization(self, mock_vector_store, mock_embedding_service):
        """Test service initialization"""
        service = RetrievalService(mock_vector_store, mock_embedding_service)
        
        assert service.vector_store == mock_vector_store
        assert service.embedding_service == mock_embedding_service
        assert service.min_relevance_threshold > 0
        assert service.max_chunks_per_document > 0
    
    def test_find_relevant_chunks_success(self, mock_vector_store, mock_embedding_service, sample_chunks):
        """Test successful chunk retrieval"""
        # Setup mocks
        mock_embedding_service.generate_embedding.return_value = np.random.rand(384)
        mock_vector_store.search_similar.return_value = [
            (sample_chunks[0], 0.9),
            (sample_chunks[1], 0.7)
        ]
        
        service = RetrievalService(mock_vector_store, mock_embedding_service)
        
        # Test retrieval
        results = service.find_relevant_chunks("What are the temperature limits?", top_k=5)
        
        assert len(results) == 2
        assert results[0][1] >= results[1][1]  # Should be sorted by relevance
        mock_embedding_service.generate_embedding.assert_called_once()
        mock_vector_store.search_similar.assert_called_once()
    
    def test_find_relevant_chunks_with_language_filter(self, mock_vector_store, mock_embedding_service):
        """Test chunk retrieval with language filter"""
        service = RetrievalService(mock_vector_store, mock_embedding_service)
        
        service.find_relevant_chunks("Question?", top_k=5, language_filter="en")
        
        # Check that language filter was passed to vector store
        call_args = mock_vector_store.search_similar.call_args
        assert call_args[1]["metadata_filter"] == {"language": "en"}
    
    def test_find_relevant_chunks_empty_question(self, mock_vector_store, mock_embedding_service):
        """Test handling of empty question"""
        service = RetrievalService(mock_vector_store, mock_embedding_service)
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            service.find_relevant_chunks("")
    
    def test_find_relevant_chunks_no_results(self, mock_vector_store, mock_embedding_service):
        """Test handling when no chunks are found"""
        mock_vector_store.search_similar.return_value = []
        
        service = RetrievalService(mock_vector_store, mock_embedding_service)
        results = service.find_relevant_chunks("Question?")
        
        assert results == []
    
    def test_filter_and_rank_results_relevance_threshold(self, mock_vector_store, mock_embedding_service, sample_chunks):
        """Test filtering by relevance threshold"""
        service = RetrievalService(mock_vector_store, mock_embedding_service)
        service.min_relevance_threshold = 0.8
        
        # Create results with different relevance scores
        search_results = [
            (sample_chunks[0], 0.9),  # Above threshold
            (sample_chunks[1], 0.5)   # Below threshold
        ]
        
        filtered = service._filter_and_rank_results(search_results, top_k=5)
        
        assert len(filtered) == 1
        assert filtered[0][1] == 0.9
    
    def test_diversify_by_document(self, mock_vector_store, mock_embedding_service):
        """Test document diversification"""
        service = RetrievalService(mock_vector_store, mock_embedding_service)
        service.max_chunks_per_document = 1
        
        # Create chunks from same document
        chunk1 = Mock()
        chunk1.document_id = "doc-1"
        chunk2 = Mock()
        chunk2.document_id = "doc-1"
        chunk3 = Mock()
        chunk3.document_id = "doc-2"
        
        results = [(chunk1, 0.9), (chunk2, 0.8), (chunk3, 0.7)]
        diversified = service._diversify_by_document(results)
        
        # Should only include one chunk from doc-1
        assert len(diversified) == 2
        doc_ids = [chunk.document_id for chunk, _ in diversified]
        assert doc_ids.count("doc-1") == 1
        assert doc_ids.count("doc-2") == 1
    
    def test_create_references(self, mock_vector_store, mock_embedding_service, sample_chunks):
        """Test reference creation from chunks"""
        service = RetrievalService(mock_vector_store, mock_embedding_service)
        
        chunks_with_scores = [
            (sample_chunks[0], 0.9),
            (sample_chunks[1], 0.7)
        ]
        
        references = service.create_references(chunks_with_scores)
        
        assert len(references) == 2
        for ref in references:
            assert isinstance(ref, Reference)
            assert ref.chunk_id in ["chunk-1", "chunk-2"]
            assert ref.page_number > 0
            assert len(ref.excerpt) > 0
            assert 0 <= ref.relevance_score <= 1
    
    def test_create_references_long_excerpt(self, mock_vector_store, mock_embedding_service):
        """Test reference creation with long content"""
        service = RetrievalService(mock_vector_store, mock_embedding_service)
        
        # Create chunk with very long content
        long_content = "This is a very long piece of content " * 20
        chunk = Chunk(
            id="chunk-long",
            document_id="doc-1",
            content=long_content,
            embedding=[0.1] * 384,
            metadata=ChunkMetadata(
                page_number=1,
                section_title="Long Section",
                chunk_index=0,
                language="en",
                confidence_score=0.9
            ),
            start_position=0,
            end_position=len(long_content)
        )
        
        references = service.create_references([(chunk, 0.9)], max_excerpt_length=100)
        
        assert len(references) == 1
        assert len(references[0].excerpt) <= 103  # 100 + "..."
        assert references[0].excerpt.endswith("...")
    
    def test_get_retrieval_stats(self, mock_vector_store, mock_embedding_service):
        """Test retrieval statistics"""
        service = RetrievalService(mock_vector_store, mock_embedding_service)
        
        stats = service.get_retrieval_stats()
        
        assert "vector_store" in stats
        assert "embedding_cache" in stats
        assert "min_relevance_threshold" in stats
        assert "max_chunks_per_document" in stats
        
        mock_vector_store.get_collection_stats.assert_called_once()
        mock_embedding_service.get_cache_stats.assert_called_once()


class TestRetrievalServiceIntegration:
    """Integration tests for RetrievalService"""
    
    def test_full_retrieval_workflow(self, mock_vector_store, mock_embedding_service, sample_chunks):
        """Test complete retrieval workflow"""
        # Setup mocks for full workflow
        mock_embedding_service.generate_embedding.return_value = np.random.rand(384)
        mock_vector_store.search_similar.return_value = [
            (sample_chunks[0], 0.9),
            (sample_chunks[1], 0.7)
        ]
        
        service = RetrievalService(mock_vector_store, mock_embedding_service)
        
        # Execute full workflow
        question = "What are the temperature specifications?"
        chunks = service.find_relevant_chunks(question, top_k=3)
        references = service.create_references(chunks)
        
        # Verify results
        assert len(chunks) == 2
        assert len(references) == 2
        assert all(isinstance(ref, Reference) for ref in references)
        
        # Verify service calls
        mock_embedding_service.generate_embedding.assert_called_once_with(question)
        mock_vector_store.search_similar.assert_called_once()