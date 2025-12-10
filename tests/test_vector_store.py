"""
Tests for the vector storage service
"""
import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from models.document import Chunk, ChunkMetadata
from services.vector_store import ChromaVectorStore, create_vector_store


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing"""
    chunks = []
    
    # Create sample embeddings (384 dimensions for multilingual MiniLM)
    embedding1 = np.random.rand(384).tolist()
    embedding2 = np.random.rand(384).tolist()
    
    chunk1 = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="This is the first test chunk about temperature limits.",
        embedding=embedding1,
        metadata=ChunkMetadata(
            page_number=1,
            section_title="Introduction",
            chunk_index=0,
            language="en",
            confidence_score=0.95
        ),
        start_position=0,
        end_position=50
    )
    
    chunk2 = Chunk(
        id="chunk-2",
        document_id="doc-1",
        content="Este é o segundo chunk de teste sobre limites de temperatura.",
        embedding=embedding2,
        metadata=ChunkMetadata(
            page_number=2,
            section_title="Especificações",
            chunk_index=1,
            language="pt",
            confidence_score=0.92
        ),
        start_position=51,
        end_position=110
    )
    
    chunks.extend([chunk1, chunk2])
    return chunks


class TestChromaVectorStore:
    """Test ChromaDB vector store implementation"""
    
    def test_initialization(self, temp_chroma_dir):
        """Test ChromaDB initialization"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_chroma_dir
        )
        
        assert store.collection is not None
        assert store.collection_name == "test_collection"
        assert store.persist_directory == temp_chroma_dir
    
    def test_add_chunks(self, temp_chroma_dir, sample_chunks):
        """Test adding chunks to the vector store"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_chroma_dir
        )
        
        # Add chunks
        result = store.add_chunks(sample_chunks)
        assert result is True
        
        # Verify collection stats
        stats = store.get_collection_stats()
        assert stats["total_chunks"] == 2
        assert stats["unique_documents"] == 1
    
    def test_search_similar(self, temp_chroma_dir, sample_chunks):
        """Test similarity search"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_chroma_dir
        )
        
        # Add chunks first
        store.add_chunks(sample_chunks)
        
        # Search with the first chunk's embedding
        query_embedding = np.array(sample_chunks[0].embedding)
        results = store.search_similar(query_embedding, top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(result[0], Chunk) for result in results)
        assert all(isinstance(result[1], float) for result in results)
        
        # First result should be the most similar (same embedding)
        assert results[0][0].id == sample_chunks[0].id
        assert results[0][1] > results[1][1]  # Higher similarity score
    
    def test_search_with_metadata_filter(self, temp_chroma_dir, sample_chunks):
        """Test similarity search with metadata filtering"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_chroma_dir
        )
        
        # Add chunks first
        store.add_chunks(sample_chunks)
        
        # Search only for English chunks
        query_embedding = np.array(sample_chunks[0].embedding)
        results = store.search_similar(
            query_embedding, 
            top_k=10, 
            metadata_filter={"language": "en"}
        )
        
        assert len(results) == 1
        assert results[0][0].metadata.language == "en"
    
    def test_get_chunk_by_id(self, temp_chroma_dir, sample_chunks):
        """Test retrieving chunk by ID"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_chroma_dir
        )
        
        # Add chunks first
        store.add_chunks(sample_chunks)
        
        # Retrieve by ID
        chunk = store.get_chunk_by_id("chunk-1")
        assert chunk is not None
        assert chunk.id == "chunk-1"
        assert chunk.content == sample_chunks[0].content
        
        # Test non-existent ID
        chunk = store.get_chunk_by_id("non-existent")
        assert chunk is None
    
    def test_delete_chunks_by_document(self, temp_chroma_dir, sample_chunks):
        """Test deleting chunks by document ID"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_chroma_dir
        )
        
        # Add chunks first
        store.add_chunks(sample_chunks)
        
        # Verify chunks exist
        stats = store.get_collection_stats()
        assert stats["total_chunks"] == 2
        
        # Delete chunks for document
        result = store.delete_chunks_by_document("doc-1")
        assert result is True
        
        # Verify chunks are deleted
        stats = store.get_collection_stats()
        assert stats["total_chunks"] == 0
    
    def test_empty_operations(self, temp_chroma_dir):
        """Test operations with empty data"""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_chroma_dir
        )
        
        # Test adding empty list
        result = store.add_chunks([])
        assert result is True
        
        # Test search on empty collection
        query_embedding = np.random.rand(384)
        results = store.search_similar(query_embedding)
        assert len(results) == 0
        
        # Test stats on empty collection
        stats = store.get_collection_stats()
        assert stats["total_chunks"] == 0


class TestVectorStoreFactory:
    """Test vector store factory function"""
    
    def test_create_chroma_store(self, temp_chroma_dir):
        """Test creating ChromaDB store via factory"""
        store = create_vector_store(
            store_type="chroma",
            collection_name="factory_test",
            persist_directory=temp_chroma_dir
        )
        
        assert isinstance(store, ChromaVectorStore)
        assert store.collection_name == "factory_test"
    
    def test_unsupported_store_type(self):
        """Test creating unsupported store type"""
        with pytest.raises(ValueError, match="Unsupported vector store type"):
            create_vector_store(store_type="unsupported")


class TestVectorStoreIntegration:
    """Integration tests for vector store operations"""
    
    def test_full_workflow(self, temp_chroma_dir, sample_chunks):
        """Test complete workflow: add, search, retrieve, delete"""
        store = ChromaVectorStore(
            collection_name="integration_test",
            persist_directory=temp_chroma_dir
        )
        
        # 1. Add chunks
        result = store.add_chunks(sample_chunks)
        assert result is True
        
        # 2. Search for similar chunks
        query_embedding = np.array(sample_chunks[0].embedding)
        search_results = store.search_similar(query_embedding, top_k=1)
        assert len(search_results) == 1
        
        # 3. Retrieve specific chunk
        chunk = store.get_chunk_by_id(search_results[0][0].id)
        assert chunk is not None
        
        # 4. Check stats
        stats = store.get_collection_stats()
        assert stats["total_chunks"] == 2
        
        # 5. Delete chunks
        result = store.delete_chunks_by_document("doc-1")
        assert result is True
        
        # 6. Verify deletion
        final_stats = store.get_collection_stats()
        assert final_stats["total_chunks"] == 0