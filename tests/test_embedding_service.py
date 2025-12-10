"""
Tests for the EmbeddingService
"""
import pytest
import numpy as np
from services.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test cases for EmbeddingService functionality"""
    
    @pytest.fixture
    def embedding_service(self):
        """Create an EmbeddingService instance for testing"""
        return EmbeddingService()
    
    def test_service_initialization(self, embedding_service):
        """Test that the service initializes correctly"""
        assert embedding_service.model is not None
        assert embedding_service.get_embedding_dimension() > 0
        assert isinstance(embedding_service.cache, dict)
    
    def test_single_embedding_generation(self, embedding_service):
        """Test generating a single embedding"""
        text = "This is a test sentence."
        embedding = embedding_service.generate_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedding_service.get_embedding_dimension(),)
        assert not np.all(embedding == 0)  # Should not be all zeros
    
    def test_batch_embedding_generation(self, embedding_service):
        """Test generating embeddings in batch"""
        texts = [
            "First test sentence.",
            "Segunda frase em português.",
            "Third sentence in English."
        ]
        embeddings = embedding_service.generate_embeddings_batch(texts)
        
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (embedding_service.get_embedding_dimension(),)
    
    def test_caching_functionality(self, embedding_service):
        """Test that caching works correctly"""
        text = "This sentence should be cached."
        
        # First generation
        embedding1 = embedding_service.generate_embedding(text)
        
        # Second generation (should use cache)
        embedding2 = embedding_service.generate_embedding(text)
        
        # Should be identical due to caching
        assert np.array_equal(embedding1, embedding2)
        
        # Check cache stats
        stats = embedding_service.get_cache_stats()
        assert stats["cache_size"] > 0
    
    def test_multilingual_support(self, embedding_service):
        """Test that the service handles multiple languages"""
        texts = [
            "English text for testing.",
            "Texto em português para teste.",
        ]
        
        embeddings = embedding_service.generate_embeddings_batch(texts)
        
        # Both should generate valid embeddings
        assert len(embeddings) == 2
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (embedding_service.get_embedding_dimension(),)
    
    def test_empty_text_handling(self, embedding_service):
        """Test handling of empty or invalid text"""
        with pytest.raises(ValueError):
            embedding_service.generate_embedding("")
        
        with pytest.raises(ValueError):
            embedding_service.generate_embedding("   ")  # Only whitespace
    
    def test_cache_management(self, embedding_service):
        """Test cache size management"""
        # Set a small cache size for testing
        embedding_service.cache_size = 2
        
        texts = ["Text 1", "Text 2", "Text 3"]
        
        for text in texts:
            embedding_service.generate_embedding(text)
        
        # Cache should not exceed the limit
        stats = embedding_service.get_cache_stats()
        assert stats["cache_size"] <= embedding_service.cache_size
    
    def test_clear_cache(self, embedding_service):
        """Test cache clearing functionality"""
        # Generate some embeddings to populate cache
        texts = ["Text 1", "Text 2"]
        for text in texts:
            embedding_service.generate_embedding(text)
        
        # Verify cache has content
        assert embedding_service.get_cache_stats()["cache_size"] > 0
        
        # Clear cache
        embedding_service.clear_cache()
        
        # Verify cache is empty
        assert embedding_service.get_cache_stats()["cache_size"] == 0