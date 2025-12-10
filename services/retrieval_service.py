"""
Retrieval service for semantic search and ranking in the PDF Q&A System
"""
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from models.document import Chunk
from models.question import Reference
from services.vector_store import VectorStoreInterface
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for semantic search and ranking of document chunks"""
    
    def __init__(self, vector_store: VectorStoreInterface, embedding_service: EmbeddingService):
        """
        Initialize the retrieval service
        
        Args:
            vector_store: Vector store for similarity search
            embedding_service: Service for generating embeddings
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.min_relevance_threshold = 0.3  # Minimum relevance score to consider
        self.max_chunks_per_document = 3    # Max chunks from same document
    
    def find_relevant_chunks(self, question: str, top_k: int = 10, 
                           language_filter: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        """
        Find relevant chunks for a given question using semantic search
        
        Args:
            question: The user's question
            top_k: Maximum number of chunks to return
            language_filter: Optional language filter ("en", "pt", or None for all)
            
        Returns:
            List of tuples containing (Chunk, relevance_score) sorted by relevance
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            # Generate embedding for the question
            logger.info(f"Generating embedding for question: {question[:50]}...")
            question_embedding = self.embedding_service.generate_embedding(question.strip())
            
            # Prepare metadata filter
            metadata_filter = None
            if language_filter and language_filter in ["en", "pt"]:
                metadata_filter = {"language": language_filter}
            
            # Perform similarity search
            logger.info(f"Searching for similar chunks (top_k={top_k})")
            search_results = self.vector_store.search_similar(
                query_embedding=question_embedding,
                top_k=top_k * 2,  # Get more results for filtering
                metadata_filter=metadata_filter
            )
            
            if not search_results:
                logger.info("No chunks found in vector store")
                return []
            
            # Filter and rank results
            filtered_results = self._filter_and_rank_results(search_results, top_k)
            
            logger.info(f"Returning {len(filtered_results)} relevant chunks")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Failed to find relevant chunks: {e}")
            raise
    
    def _filter_and_rank_results(self, search_results: List[Tuple[Chunk, float]], 
                                top_k: int) -> List[Tuple[Chunk, float]]:
        """
        Filter and rank search results based on relevance and diversity
        
        Args:
            search_results: Raw search results from vector store
            top_k: Maximum number of results to return
            
        Returns:
            Filtered and ranked results
        """
        # Filter by minimum relevance threshold
        relevant_results = [
            (chunk, score) for chunk, score in search_results 
            if score >= self.min_relevance_threshold
        ]
        
        if not relevant_results:
            logger.info("No chunks meet minimum relevance threshold")
            return []
        
        # Diversify results by limiting chunks per document
        diversified_results = self._diversify_by_document(relevant_results)
        
        # Sort by relevance score (highest first) and limit to top_k
        final_results = sorted(diversified_results, key=lambda x: x[1], reverse=True)[:top_k]
        
        logger.info(f"Filtered {len(search_results)} -> {len(relevant_results)} -> {len(final_results)} chunks")
        return final_results
    
    def _diversify_by_document(self, results: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        """
        Diversify results by limiting chunks per document
        
        Args:
            results: List of (chunk, score) tuples
            
        Returns:
            Diversified results
        """
        document_counts = {}
        diversified = []
        
        for chunk, score in results:
            doc_id = chunk.document_id
            current_count = document_counts.get(doc_id, 0)
            
            if current_count < self.max_chunks_per_document:
                diversified.append((chunk, score))
                document_counts[doc_id] = current_count + 1
        
        return diversified
    
    def create_references(self, chunks_with_scores: List[Tuple[Chunk, float]], 
                         max_excerpt_length: int = 200) -> List[Reference]:
        """
        Create reference objects from chunks with scores
        
        Args:
            chunks_with_scores: List of (chunk, relevance_score) tuples
            max_excerpt_length: Maximum length of excerpt text
            
        Returns:
            List of Reference objects
        """
        references = []
        
        for chunk, relevance_score in chunks_with_scores:
            try:
                # Create excerpt (truncate if too long)
                excerpt = chunk.content
                if len(excerpt) > max_excerpt_length:
                    excerpt = excerpt[:max_excerpt_length].rsplit(' ', 1)[0] + "..."
                
                # Get document filename from chunk metadata or use document_id as fallback
                document_filename = getattr(chunk, 'document_filename', None)
                if not document_filename:
                    # Try to extract filename from document_id if it looks like a path
                    if '/' in chunk.document_id:
                        document_filename = chunk.document_id.split('/')[-1]
                    elif not chunk.document_id.endswith('.pdf'):
                        document_filename = f"{chunk.document_id}.pdf"
                    else:
                        document_filename = chunk.document_id
                
                reference = Reference(
                    chunk_id=chunk.id,
                    document_filename=document_filename,
                    page_number=chunk.metadata.page_number,
                    excerpt=excerpt,
                    relevance_score=relevance_score
                )
                
                references.append(reference)
                
            except Exception as e:
                logger.warning(f"Failed to create reference for chunk {chunk.id}: {e}")
                continue
        
        return references
    
    def get_retrieval_stats(self) -> Dict[str, any]:
        """
        Get statistics about the retrieval service
        
        Returns:
            Dictionary containing retrieval statistics
        """
        try:
            vector_stats = self.vector_store.get_collection_stats()
            embedding_stats = self.embedding_service.get_cache_stats()
            
            return {
                "vector_store": vector_stats,
                "embedding_cache": embedding_stats,
                "min_relevance_threshold": self.min_relevance_threshold,
                "max_chunks_per_document": self.max_chunks_per_document
            }
        except Exception as e:
            logger.error(f"Failed to get retrieval stats: {e}")
            return {"error": str(e)}