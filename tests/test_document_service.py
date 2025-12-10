"""
Tests for the DocumentService class
"""
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from services.document_service import DocumentService, DocumentProcessingError, ProcessingResult
from services.pdf_processor import PDFProcessor
from services.text_chunker import TextChunker
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStoreInterface
from models.document import ProcessingStatus


class TestDocumentService:
    """Test cases for DocumentService"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary directory for uploads
        self.temp_dir = tempfile.mkdtemp()
        self.upload_dir = Path(self.temp_dir) / "uploads"
        
        # Create mock services
        self.mock_vector_store = Mock(spec=VectorStoreInterface)
        self.mock_pdf_processor = Mock(spec=PDFProcessor)
        self.mock_text_chunker = Mock(spec=TextChunker)
        self.mock_embedding_service = Mock(spec=EmbeddingService)
        
        # Initialize DocumentService with mocks
        self.document_service = DocumentService(
            upload_directory=str(self.upload_dir),
            vector_store=self.mock_vector_store,
            pdf_processor=self.mock_pdf_processor,
            text_chunker=self.mock_text_chunker,
            embedding_service=self.mock_embedding_service
        )
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_service_initialization(self):
        """Test DocumentService initialization"""
        assert self.document_service.upload_directory == self.upload_dir
        assert self.upload_dir.exists()
        assert self.document_service.vector_store == self.mock_vector_store
        assert self.document_service.pdf_processor == self.mock_pdf_processor
        assert self.document_service.text_chunker == self.mock_text_chunker
        assert self.document_service.embedding_service == self.mock_embedding_service
    
    def test_save_uploaded_file(self):
        """Test saving uploaded file content"""
        file_content = b"Test PDF content"
        filename = "test.pdf"
        
        saved_path = self.document_service.save_uploaded_file(file_content, filename)
        
        assert os.path.exists(saved_path)
        with open(saved_path, 'rb') as f:
            assert f.read() == file_content
        
        # Check filename includes timestamp
        assert "test.pdf" in saved_path
        assert saved_path.startswith(str(self.upload_dir))
    
    def test_save_uploaded_file_error(self):
        """Test error handling in file saving"""
        # Mock the file writing to raise an exception
        with patch('builtins.open', side_effect=OSError("Permission denied")):
            with pytest.raises(DocumentProcessingError):
                self.document_service.save_uploaded_file(b"content", "test.pdf")
    
    def test_process_single_document_success(self):
        """Test successful processing of a single document"""
        # Create test file
        test_file = self.upload_dir / "test.pdf"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(b"Test PDF content")
        
        # Mock service responses
        self.mock_pdf_processor.process_pdf.return_value = ("Extracted text", "en", 0.95)
        
        # Mock chunk creation
        from models.document import Chunk, ChunkMetadata
        mock_chunk = Chunk(
            document_id="test-doc",
            content="Extracted text",
            metadata=ChunkMetadata(
                page_number=1,
                chunk_index=0,
                language="en",
                confidence_score=0.95
            ),
            start_position=0,
            end_position=14
        )
        self.mock_text_chunker.chunk_text.return_value = [mock_chunk]
        
        # Mock embedding generation
        import numpy as np
        mock_embedding = np.random.rand(384)
        self.mock_embedding_service.generate_embeddings_batch.return_value = [mock_embedding]
        
        # Mock vector store
        self.mock_vector_store.add_chunks.return_value = True
        
        # Process the document
        result = self.document_service._process_single_document(str(test_file))
        
        # Verify result
        assert result.success is True
        assert result.documents_processed == 1
        assert result.total_chunks == 1
        assert len(result.failed_documents) == 0
        
        # Verify service calls
        self.mock_pdf_processor.process_pdf.assert_called_once_with(str(test_file))
        self.mock_text_chunker.chunk_text.assert_called_once()
        self.mock_embedding_service.generate_embeddings_batch.assert_called_once()
        self.mock_vector_store.add_chunks.assert_called_once()
    
    def test_process_single_document_pdf_error(self):
        """Test handling of PDF processing errors"""
        # Create test file
        test_file = self.upload_dir / "test.pdf"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(b"Invalid PDF content")
        
        # Mock PDF processing error
        from services.pdf_processor import PDFProcessingError
        self.mock_pdf_processor.process_pdf.side_effect = PDFProcessingError("Invalid PDF")
        
        # Process the document
        result = self.document_service._process_single_document(str(test_file))
        
        # Verify error handling
        assert result.success is False
        assert result.documents_processed == 0
        assert result.total_chunks == 0
        assert len(result.failed_documents) == 1
        assert "test.pdf" in result.failed_documents[0]
        assert "Invalid PDF" in result.error_message
    
    def test_process_documents_batch(self):
        """Test batch processing of multiple documents"""
        # Create test files
        test_files = []
        for i in range(3):
            test_file = self.upload_dir / f"test_{i}.pdf"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_bytes(f"Test PDF content {i}".encode())
            test_files.append(str(test_file))
        
        # Mock successful processing for all files
        self.mock_pdf_processor.process_pdf.return_value = ("Extracted text", "en", 0.95)
        
        from models.document import Chunk, ChunkMetadata
        mock_chunk = Chunk(
            document_id="test-doc",
            content="Extracted text",
            metadata=ChunkMetadata(
                page_number=1,
                chunk_index=0,
                language="en",
                confidence_score=0.95
            ),
            start_position=0,
            end_position=14
        )
        self.mock_text_chunker.chunk_text.return_value = [mock_chunk]
        
        import numpy as np
        mock_embedding = np.random.rand(384)
        self.mock_embedding_service.generate_embeddings_batch.return_value = [mock_embedding]
        self.mock_vector_store.add_chunks.return_value = True
        
        # Process batch
        result = self.document_service.process_documents(test_files)
        
        # Verify batch result
        assert result.success is True
        assert result.documents_processed == 3
        assert result.total_chunks == 3
        assert len(result.failed_documents) == 0
        
        # Verify service was called for each file
        assert self.mock_pdf_processor.process_pdf.call_count == 3
        assert self.mock_text_chunker.chunk_text.call_count == 3
        assert self.mock_embedding_service.generate_embeddings_batch.call_count == 3
        assert self.mock_vector_store.add_chunks.call_count == 3
    
    def test_process_documents_partial_failure(self):
        """Test batch processing with some failures"""
        # Create test files
        test_files = []
        for i in range(3):
            test_file = self.upload_dir / f"test_{i}.pdf"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_bytes(f"Test PDF content {i}".encode())
            test_files.append(str(test_file))
        
        # Mock mixed success/failure
        from services.pdf_processor import PDFProcessingError
        
        def mock_process_pdf(file_path):
            if "test_1.pdf" in file_path:
                raise PDFProcessingError("Processing failed")
            return ("Extracted text", "en", 0.95)
        
        self.mock_pdf_processor.process_pdf.side_effect = mock_process_pdf
        
        from models.document import Chunk, ChunkMetadata
        mock_chunk = Chunk(
            document_id="test-doc",
            content="Extracted text",
            metadata=ChunkMetadata(
                page_number=1,
                chunk_index=0,
                language="en",
                confidence_score=0.95
            ),
            start_position=0,
            end_position=14
        )
        self.mock_text_chunker.chunk_text.return_value = [mock_chunk]
        
        import numpy as np
        mock_embedding = np.random.rand(384)
        self.mock_embedding_service.generate_embeddings_batch.return_value = [mock_embedding]
        self.mock_vector_store.add_chunks.return_value = True
        
        # Process batch
        result = self.document_service.process_documents(test_files)
        
        # Verify partial failure
        assert result.success is False  # Overall failure due to one failed document
        assert result.documents_processed == 2  # Two successful
        assert result.total_chunks == 2
        assert len(result.failed_documents) == 1
        assert any("test_1.pdf" in path for path in result.failed_documents)
    
    def test_progress_tracking(self):
        """Test progress tracking functionality"""
        # Initially no progress
        assert len(self.document_service.get_all_processing_progress()) == 0
        
        # Create test file and start processing
        test_file = self.upload_dir / "test.pdf"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(b"Test PDF content")
        
        # Mock successful processing
        self.mock_pdf_processor.process_pdf.return_value = ("Extracted text", "en", 0.95)
        
        from models.document import Chunk, ChunkMetadata
        mock_chunk = Chunk(
            document_id="test-doc",
            content="Extracted text",
            metadata=ChunkMetadata(
                page_number=1,
                chunk_index=0,
                language="en",
                confidence_score=0.95
            ),
            start_position=0,
            end_position=14
        )
        self.mock_text_chunker.chunk_text.return_value = [mock_chunk]
        
        import numpy as np
        mock_embedding = np.random.rand(384)
        self.mock_embedding_service.generate_embeddings_batch.return_value = [mock_embedding]
        self.mock_vector_store.add_chunks.return_value = True
        
        # Process document
        result = self.document_service._process_single_document(str(test_file))
        
        # Check progress was tracked
        progress_dict = self.document_service.get_all_processing_progress()
        assert len(progress_dict) == 1
        
        progress = list(progress_dict.values())[0]
        assert progress.filename == "test.pdf"
        assert progress.status == ProcessingStatus.COMPLETED
        assert progress.chunks_created == 1
        assert progress.processing_start_time is not None
        assert progress.processing_end_time is not None
    
    def test_clear_progress(self):
        """Test clearing progress tracking"""
        # Add some mock progress
        from services.document_service import DocumentProcessingProgress
        progress = DocumentProcessingProgress(
            document_id="test-doc",
            filename="test.pdf",
            status=ProcessingStatus.COMPLETED
        )
        self.document_service.processing_progress["test-doc"] = progress
        
        # Verify progress exists
        assert len(self.document_service.get_all_processing_progress()) == 1
        
        # Clear progress
        self.document_service.clear_processing_progress()
        
        # Verify progress cleared
        assert len(self.document_service.get_all_processing_progress()) == 0
    
    def test_get_service_stats(self):
        """Test service statistics"""
        # Mock vector store stats
        self.mock_vector_store.get_collection_stats.return_value = {
            "total_chunks": 10,
            "unique_documents": 2
        }
        
        # Mock embedding service stats
        self.mock_embedding_service.get_cache_stats.return_value = {
            "cache_size": 5,
            "cache_limit": 100
        }
        
        # Get stats
        stats = self.document_service.get_service_stats()
        
        # Verify stats structure
        assert "upload_directory" in stats
        assert "uploaded_files" in stats
        assert "vector_store" in stats
        assert "embedding_cache" in stats
        assert "processing_progress" in stats
        
        assert stats["vector_store"]["total_chunks"] == 10
        assert stats["embedding_cache"]["cache_size"] == 5