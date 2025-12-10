"""
Vector storage and retrieval service for the PDF Q&A System
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from models.document import Chunk, ChunkMetadata
from config import settings

logger = logging.getLogger(__name__)


class VectorStoreInterface(ABC):
    """Abstract interface for vector storage operations"""
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> bool:
        """Add chunks with their embeddings to the vector store"""
        pass
    
    @abstractmethod
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10, 
                      metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks based on embedding similarity"""
        pass
    
    @abstractmethod
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a specific chunk by its ID"""
        pass
    
    @abstractmethod
    def delete_chunks_by_document(self, document_id: str) -> bool:
        """Delete all chunks belonging to a specific document"""
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        pass


class ChromaVectorStore(VectorStoreInterface):
    """ChromaDB implementation of vector storage"""
    
    def __init__(self, collection_name: str = "pdf_chunks", persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and collection"""
        try:
            logger.info(f"Initializing ChromaDB client with persist directory: {self.persist_directory}")
            
            # Create ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document chunks with embeddings"}
            )
            
            logger.info(f"ChromaDB collection '{self.collection_name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_chunks(self, chunks: List[Chunk]) -> bool:
        """
        Add chunks with their embeddings to ChromaDB
        
        Args:
            chunks: List of chunks to add
            
        Returns:
            True if successful, False otherwise
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return True
        
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                    continue
                
                ids.append(chunk.id)
                embeddings.append(chunk.embedding)
                documents.append(chunk.content)
                
                # Prepare metadata (ChromaDB requires flat dictionary)
                metadata = {
                    "document_id": chunk.document_id,
                    "page_number": chunk.metadata.page_number,
                    "section_title": chunk.metadata.section_title or "",
                    "chunk_index": chunk.metadata.chunk_index,
                    "language": chunk.metadata.language,
                    "confidence_score": chunk.metadata.confidence_score,
                    "start_position": chunk.start_position,
                    "end_position": chunk.end_position
                }
                metadatas.append(metadata)
            
            if not ids:
                logger.warning("No valid chunks with embeddings to add")
                return True
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(ids)} chunks to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10, 
                      metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks based on embedding similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            metadata_filter: Optional metadata filters
            
        Returns:
            List of tuples containing (Chunk, similarity_score)
        """
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        
        try:
            # Convert numpy array to list for ChromaDB
            query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            # Prepare where clause for metadata filtering
            where_clause = None
            if metadata_filter:
                where_clause = {}
                for key, value in metadata_filter.items():
                    if isinstance(value, list):
                        # Handle list values (e.g., language in ["en", "pt"])
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, 100),  # ChromaDB has limits
                where=where_clause,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            
            # Convert results back to Chunk objects
            chunks_with_scores = []
            
            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    chunk_id = results["ids"][0][i]
                    document_content = results["documents"][0][i]
                    metadata_dict = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    embedding = results["embeddings"][0][i] if results["embeddings"] else None
                    
                    # Convert distance to similarity score (ChromaDB returns distances)
                    similarity_score = 1.0 / (1.0 + distance)
                    
                    # Reconstruct ChunkMetadata
                    chunk_metadata = ChunkMetadata(
                        page_number=metadata_dict["page_number"],
                        section_title=metadata_dict["section_title"] if metadata_dict["section_title"] else None,
                        chunk_index=metadata_dict["chunk_index"],
                        language=metadata_dict["language"],
                        confidence_score=metadata_dict["confidence_score"]
                    )
                    
                    # Reconstruct Chunk
                    chunk = Chunk(
                        id=chunk_id,
                        document_id=metadata_dict["document_id"],
                        content=document_content,
                        embedding=embedding,
                        metadata=chunk_metadata,
                        start_position=metadata_dict["start_position"],
                        end_position=metadata_dict["end_position"]
                    )
                    
                    chunks_with_scores.append((chunk, similarity_score))
            
            logger.info(f"Found {len(chunks_with_scores)} similar chunks")
            return chunks_with_scores
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """
        Retrieve a specific chunk by its ID
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Chunk object if found, None otherwise
        """
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if not results["ids"]:
                return None
            
            # Reconstruct chunk from results
            metadata_dict = results["metadatas"][0]
            document_content = results["documents"][0]
            embedding = results["embeddings"][0].tolist() if results["embeddings"] is not None and len(results["embeddings"]) > 0 else None
            
            chunk_metadata = ChunkMetadata(
                page_number=metadata_dict["page_number"],
                section_title=metadata_dict["section_title"] if metadata_dict["section_title"] else None,
                chunk_index=metadata_dict["chunk_index"],
                language=metadata_dict["language"],
                confidence_score=metadata_dict["confidence_score"]
            )
            
            chunk = Chunk(
                id=chunk_id,
                document_id=metadata_dict["document_id"],
                content=document_content,
                embedding=embedding,
                metadata=chunk_metadata,
                start_position=metadata_dict["start_position"],
                end_position=metadata_dict["end_position"]
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to get chunk by ID {chunk_id}: {e}")
            return None
    
    def delete_chunks_by_document(self, document_id: str) -> bool:
        """
        Delete all chunks belonging to a specific document
        
        Args:
            document_id: ID of the document whose chunks should be deleted
            
        Returns:
            True if successful, False otherwise
        """
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        
        try:
            # First, find all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents"]
            )
            
            if not results["ids"]:
                logger.info(f"No chunks found for document {document_id}")
                return True
            
            # Delete the chunks
            self.collection.delete(ids=results["ids"])
            
            logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ChromaDB collection
        
        Returns:
            Dictionary containing collection statistics
        """
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze languages
            sample_results = self.collection.get(
                limit=min(100, count) if count > 0 else 0,
                include=["metadatas"]
            )
            
            language_counts = {}
            document_counts = {}
            
            if sample_results["metadatas"]:
                for metadata in sample_results["metadatas"]:
                    # Count languages
                    lang = metadata.get("language", "unknown")
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                    
                    # Count documents
                    doc_id = metadata.get("document_id", "unknown")
                    document_counts[doc_id] = document_counts.get(doc_id, 0) + 1
            
            stats = {
                "total_chunks": count,
                "unique_documents": len(document_counts),
                "language_distribution": language_counts,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def reset_collection(self) -> bool:
        """
        Reset the collection (delete all data) - useful for testing
        
        Returns:
            True if successful, False otherwise
        """
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document chunks with embeddings"}
            )
            
            logger.info(f"Reset ChromaDB collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False


# Factory function to create vector store instances
def create_vector_store(store_type: str = "chroma", **kwargs) -> VectorStoreInterface:
    """
    Factory function to create vector store instances
    
    Args:
        store_type: Type of vector store ("chroma")
        **kwargs: Additional arguments for the vector store
        
    Returns:
        VectorStore instance
    """
    if store_type.lower() == "chroma":
        return ChromaVectorStore(**kwargs)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")