"""
Embedding generation service for the PDF Q&A System
"""
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers"""
    
    def __init__(self, model_name: Optional[str] = None, cache_size: int = 1000):
        """
        Initialize the embedding service
        
        Args:
            model_name: Name of the sentence-transformers model to use
            cache_size: Maximum number of embeddings to cache
        """
        self.model_name = model_name or settings.embedding_model
        self.model: Optional[SentenceTransformer] = None
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_size = cache_size
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence-transformers model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.get_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.get_sentence_embedding_dimension()
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _manage_cache_size(self) -> None:
        """Remove oldest entries if cache exceeds size limit"""
        if len(self.cache) >= self.cache_size:
            # Remove 20% of oldest entries (simple FIFO approach)
            num_to_remove = max(1, int(self.cache_size * 0.2))
            keys_to_remove = list(self.cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self.cache[key]
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return self.cache[cache_key]
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Cache the result
            self._manage_cache_size()
            self.cache[cache_key] = embedding
            
            logger.debug(f"Generated embedding for text: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with batch processing
        
        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of numpy arrays containing embedding vectors
        """
        if not texts:
            return []
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        embeddings = []
        cached_embeddings = {}
        texts_to_process = []
        text_indices = []
        
        # Check cache for existing embeddings
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
            
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                cached_embeddings[i] = self.cache[cache_key]
                logger.debug(f"Cache hit for batch text {i}: {text[:50]}...")
            else:
                texts_to_process.append(text)
                text_indices.append(i)
        
        # Process uncached texts in batches
        new_embeddings = []
        if texts_to_process:
            try:
                logger.info(f"Processing {len(texts_to_process)} texts in batches of {batch_size}")
                
                for i in range(0, len(texts_to_process), batch_size):
                    batch_texts = texts_to_process[i:i + batch_size]
                    batch_embeddings = self.model.encode(
                        batch_texts, 
                        convert_to_numpy=True,
                        show_progress_bar=len(texts_to_process) > 100
                    )
                    
                    # Cache the new embeddings
                    for j, embedding in enumerate(batch_embeddings):
                        text_idx = i + j
                        if text_idx < len(texts_to_process):
                            cache_key = self._get_cache_key(texts_to_process[text_idx])
                            self._manage_cache_size()
                            self.cache[cache_key] = embedding
                    
                    new_embeddings.extend(batch_embeddings)
                    
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                raise
        
        # Combine cached and new embeddings in original order
        for i in range(len(texts)):
            if i in cached_embeddings:
                embeddings.append(cached_embeddings[i])
            else:
                # Find the corresponding new embedding
                new_idx = text_indices.index(i)
                embeddings.append(new_embeddings[new_idx])
        
        logger.info(f"Generated {len(embeddings)} embeddings ({len(cached_embeddings)} from cache)")
        return embeddings
    
    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "cache_limit": self.cache_size
        }