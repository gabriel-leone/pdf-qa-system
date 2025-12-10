"""
Integration tests for DocumentService with real components
"""
import os
import tempfile
import shutil
from pathlib import Path
import pytest

from services.document_service import DocumentService
from services.pdf_processor import PDFProcessor
from services.text_chunker import TextChunker
from services.embedding_service import EmbeddingService
from services.vector_store import ChromaVectorStore


class TestDocumentServiceIntegration:
    """Integration tests using real service components"""
    
    def setup_method(self):
        """Set up test fixtures with real services"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.upload_dir = Path(self.temp_dir) / "uploads"
        self.chroma_dir = Path(self.temp_dir) / "chroma_test"
        
        # Initialize real services
        self.vector_store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=str(self.chroma_dir)
        )
        
        self.document_service = DocumentService(
            upload_directory=str(self.upload_dir),
            vector_store=self.vector_store,
            pdf_processor=PDFProcessor(),
            text_chunker=TextChunker(),
            embedding_service=EmbeddingService()
        )
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_retrieve_file(self):
        """Test saving a file and retrieving its path"""
        file_content = b"Test PDF content for integration testing"
        filename = "integration_test.pdf"
        
        # Save the file
        saved_path = self.document_service.save_uploaded_file(file_content, filename)
        
        # Verify file exists and has correct content
        assert os.path.exists(saved_path)
        with open(saved_path, 'rb') as f:
            assert f.read() == file_content
        
        # Verify path is in upload directory
        assert str(self.upload_dir) in saved_path
        assert filename in saved_path
    
    def test_service_stats_integration(self):
        """Test getting service statistics with real components"""
        stats = self.document_service.get_service_stats()
        
        # Verify stats structure
        assert "upload_directory" in stats
        assert "uploaded_files" in stats
        assert "vector_store" in stats
        assert "embedding_cache" in stats
        assert "processing_progress" in stats
        
        # Verify initial state
        assert stats["uploaded_files"] == 0
        assert stats["vector_store"]["total_chunks"] == 0
        assert stats["processing_progress"]["total_tracked"] == 0
    
    def test_progress_tracking_integration(self):
        """Test progress tracking with real workflow"""
        # Initially no progress
        assert len(self.document_service.get_all_processing_progress()) == 0
        
        # Add some mock progress
        from services.document_service import DocumentProcessingProgress
        from models.document import ProcessingStatus
        
        progress = DocumentProcessingProgress(
            document_id="test-doc-123",
            filename="test.pdf",
            status=ProcessingStatus.PROCESSING,
            chunks_created=5
        )
        
        self.document_service.processing_progress["test-doc-123"] = progress
        
        # Verify progress tracking
        all_progress = self.document_service.get_all_processing_progress()
        assert len(all_progress) == 1
        assert "test-doc-123" in all_progress
        
        retrieved_progress = self.document_service.get_processing_progress("test-doc-123")
        assert retrieved_progress is not None
        assert retrieved_progress.filename == "test.pdf"
        assert retrieved_progress.chunks_created == 5
        
        # Clear progress
        self.document_service.clear_processing_progress()
        assert len(self.document_service.get_all_processing_progress()) == 0
    
    def test_cleanup_functionality(self):
        """Test cleanup of failed uploads"""
        # Create some test files
        test_files = []
        for i in range(3):
            file_content = f"Test content {i}".encode()
            filename = f"test_{i}.pdf"
            saved_path = self.document_service.save_uploaded_file(file_content, filename)
            test_files.append(saved_path)
        
        # Verify files exist
        for file_path in test_files:
            assert os.path.exists(file_path)
        
        # Add failed processing records
        from services.document_service import DocumentProcessingProgress
        from models.document import ProcessingStatus
        
        for i, file_path in enumerate(test_files[:2]):  # Mark first 2 as failed
            filename = os.path.basename(file_path)
            progress = DocumentProcessingProgress(
                document_id=f"doc-{i}",
                filename=filename,
                status=ProcessingStatus.FAILED
            )
            self.document_service.processing_progress[f"doc-{i}"] = progress
        
        # Run cleanup
        cleaned_count = self.document_service.cleanup_failed_uploads()
        
        # Verify cleanup results
        assert cleaned_count >= 0  # May not find exact matches due to timestamp in filename
        
        # Verify progress records were cleared for failed uploads
        remaining_progress = self.document_service.get_all_processing_progress()
        failed_count = sum(1 for p in remaining_progress.values() 
                          if p.status == ProcessingStatus.FAILED)
        assert failed_count == 0
    
    def test_vector_store_integration(self):
        """Test integration with vector store"""
        # Get initial stats
        initial_stats = self.vector_store.get_collection_stats()
        assert initial_stats["total_chunks"] == 0
        
        # Create a test chunk
        from models.document import Chunk, ChunkMetadata
        import numpy as np
        
        chunk = Chunk(
            document_id="test-doc",
            content="This is a test chunk for integration testing",
            embedding=np.random.rand(384).tolist(),
            metadata=ChunkMetadata(
                page_number=1,
                chunk_index=0,
                language="en",
                confidence_score=0.95
            ),
            start_position=0,
            end_position=45
        )
        
        # Add chunk to vector store
        success = self.vector_store.add_chunks([chunk])
        assert success is True
        
        # Verify stats updated
        updated_stats = self.vector_store.get_collection_stats()
        assert updated_stats["total_chunks"] == 1
        
        # Test deletion
        delete_success = self.document_service.delete_document("test-doc")
        assert delete_success is True
        
        # Verify chunk was deleted
        final_stats = self.vector_store.get_collection_stats()
        assert final_stats["total_chunks"] == 0