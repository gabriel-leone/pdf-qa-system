"""
Document processing service for the PDF Q&A System

This service orchestrates the complete document processing pipeline:
1. PDF text extraction
2. Text chunking
3. Embedding generation
4. Vector storage

Includes progress tracking and error recovery for batch processing.
"""
import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from models.document import Document, ProcessingStatus
from services.pdf_processor import PDFProcessor
from services.text_chunker import TextChunker
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStoreInterface, create_vector_store
from services.language_validator import LanguageValidator
from config import settings
from utils.exceptions import (
    DocumentProcessingError, PDFProcessingError, TextChunkingError,
    EmbeddingError, VectorStoreError, FileHandlingError
)
from utils.error_handlers import log_processing_step, log_performance_metric

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of document processing operation"""
    success: bool
    documents_processed: int
    total_chunks: int
    failed_documents: List[str]
    processing_time_seconds: float
    error_message: Optional[str] = None


@dataclass
class DocumentProcessingProgress:
    """Progress tracking for document processing"""
    document_id: str
    filename: str
    status: ProcessingStatus
    chunks_created: int = 0
    error_message: Optional[str] = None
    processing_start_time: Optional[datetime] = None
    processing_end_time: Optional[datetime] = None


# DocumentProcessingError is now imported from utils.exceptions


class DocumentService:
    """
    Service for orchestrating the complete document processing pipeline.
    
    Handles PDF upload, text extraction, chunking, embedding generation,
    and storage with progress tracking and error recovery.
    """
    
    def __init__(
        self,
        upload_directory: str = "./uploads",
        vector_store: Optional[VectorStoreInterface] = None,
        pdf_processor: Optional[PDFProcessor] = None,
        text_chunker: Optional[TextChunker] = None,
        embedding_service: Optional[EmbeddingService] = None,
        language_validator: Optional[LanguageValidator] = None
    ):
        """
        Initialize the document service with all required components.
        
        Args:
            upload_directory: Directory to store uploaded files
            vector_store: Vector storage service instance
            pdf_processor: PDF processing service instance
            text_chunker: Text chunking service instance
            embedding_service: Embedding generation service instance
        """
        self.upload_directory = Path(upload_directory)
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize services with dependency injection or defaults
        self.vector_store = vector_store or create_vector_store("chroma")
        self.pdf_processor = pdf_processor or PDFProcessor()
        self.text_chunker = text_chunker or TextChunker()
        self.embedding_service = embedding_service or EmbeddingService()
        self.language_validator = language_validator or LanguageValidator()
        
        # Progress tracking
        self.processing_progress: Dict[str, DocumentProcessingProgress] = {}
        
        logger.info(f"DocumentService initialized with upload directory: {self.upload_directory}")
    
    def process_documents(self, file_paths: List[str]) -> ProcessingResult:
        """
        Process multiple PDF documents through the complete pipeline.
        
        Args:
            file_paths: List of paths to PDF files to process
            
        Returns:
            ProcessingResult with summary of processing operation
        """
        start_time = datetime.now()
        documents_processed = 0
        total_chunks = 0
        failed_documents = []
        
        logger.info(f"Starting batch processing of {len(file_paths)} documents")
        
        try:
            for file_path in file_paths:
                try:
                    # Process single document
                    result = self._process_single_document(file_path)
                    
                    if result.success:
                        documents_processed += 1
                        total_chunks += result.total_chunks
                        logger.info(f"Successfully processed {file_path}: {result.total_chunks} chunks")
                    else:
                        failed_documents.append(file_path)
                        logger.error(f"Failed to process {file_path}: {result.error_message}")
                        
                except Exception as e:
                    failed_documents.append(file_path)
                    logger.error(f"Unexpected error processing {file_path}: {e}")
                    continue
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            success = len(failed_documents) == 0
            error_message = None if success else f"Failed to process {len(failed_documents)} documents"
            
            result = ProcessingResult(
                success=success,
                documents_processed=documents_processed,
                total_chunks=total_chunks,
                failed_documents=failed_documents,
                processing_time_seconds=processing_time,
                error_message=error_message
            )
            
            logger.info(f"Batch processing completed: {documents_processed}/{len(file_paths)} documents, "
                       f"{total_chunks} total chunks, {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Batch processing failed: {e}")
            
            return ProcessingResult(
                success=False,
                documents_processed=documents_processed,
                total_chunks=total_chunks,
                failed_documents=failed_documents + file_paths[documents_processed:],
                processing_time_seconds=processing_time,
                error_message=str(e)
            )
    
    def _process_single_document(self, file_path: str) -> ProcessingResult:
        """
        Process a single PDF document through the complete pipeline.
        
        Args:
            file_path: Path to the PDF file to process
            
        Returns:
            ProcessingResult for the single document
        """
        start_time = datetime.now()
        filename = os.path.basename(file_path)
        
        # Create document record
        document = Document(
            filename=filename,
            file_size=os.path.getsize(file_path),
            language="unknown",  # Will be updated after processing
            processing_status=ProcessingStatus.PENDING,
            file_path=file_path
        )
        
        # Initialize progress tracking
        progress = DocumentProcessingProgress(
            document_id=document.id,
            filename=filename,
            status=ProcessingStatus.PROCESSING,
            processing_start_time=start_time
        )
        self.processing_progress[document.id] = progress
        
        try:
            # Step 1: Extract text from PDF
            logger.info(f"Extracting text from {filename}")
            progress.status = ProcessingStatus.PROCESSING
            
            text, language, confidence = self.pdf_processor.process_pdf(file_path)
            
            # Validate detected language
            validation_result = self.language_validator.validate_document_language(text, language)
            
            if not validation_result.is_valid:
                logger.warning(f"Language validation failed for {filename}: {validation_result.validation_errors}")
                # Continue processing but log the issues
                for error in validation_result.validation_errors:
                    logger.warning(f"Language validation error: {error}")
            
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(f"Language validation warning: {warning}")
            
            # Update document with validated language
            document.language = validation_result.detected_language
            document.processing_status = ProcessingStatus.PROCESSING
            
            logger.info(f"Extracted {len(text)} characters from {filename}, "
                       f"language: {validation_result.detected_language} "
                       f"(confidence: {validation_result.confidence_score:.2f})")
            
            # Step 2: Chunk the text
            logger.info(f"Chunking text for {filename}")
            
            chunks = self.text_chunker.chunk_text(
                text=text,
                document_id=document.id,
                page_number=1,  # TODO: Implement page-aware chunking
                language=language,
                confidence_score=confidence
            )
            
            if not chunks:
                raise DocumentProcessingError(f"No chunks created from {filename}")
            
            progress.chunks_created = len(chunks)
            document.chunk_count = len(chunks)
            
            logger.info(f"Created {len(chunks)} chunks from {filename}")
            
            # Step 3: Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks from {filename}")
            
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.generate_embeddings_batch(chunk_texts)
            
            # Attach embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()
            
            # Validate embedding language consistency
            chunk_texts = [chunk.content for chunk in chunks]
            chunk_embeddings = [chunk.embedding for chunk in chunks]
            embedding_validation = self.language_validator.validate_embedding_language_consistency(
                chunk_texts, chunk_embeddings
            )
            
            if not embedding_validation.is_valid:
                logger.warning(f"Embedding validation issues for {filename}: {embedding_validation.validation_errors}")
            
            if embedding_validation.warnings:
                for warning in embedding_validation.warnings:
                    logger.warning(f"Embedding validation warning: {warning}")
            
            # Validate chunk language consistency
            chunk_consistency = self.language_validator.validate_chunk_language_consistency(chunks)
            if not chunk_consistency.is_valid:
                logger.warning(f"Chunk language consistency issues for {filename}: {chunk_consistency.validation_errors}")
            
            logger.info(f"Generated embeddings for {len(chunks)} chunks from {filename} "
                       f"(embedding validation confidence: {embedding_validation.confidence_score:.2f})")
            
            # Step 4: Store in vector database
            logger.info(f"Storing {len(chunks)} chunks in vector store for {filename}")
            
            storage_success = self.vector_store.add_chunks(chunks)
            
            if not storage_success:
                raise DocumentProcessingError(f"Failed to store chunks for {filename}")
            
            logger.info(f"Successfully stored {len(chunks)} chunks for {filename}")
            
            # Update final status
            document.processing_status = ProcessingStatus.COMPLETED
            progress.status = ProcessingStatus.COMPLETED
            progress.processing_end_time = datetime.now()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                documents_processed=1,
                total_chunks=len(chunks),
                failed_documents=[],
                processing_time_seconds=processing_time
            )
            
        except (PDFProcessingError, TextChunkingError, DocumentProcessingError) as e:
            # Handle known processing errors
            error_msg = f"Processing failed for {filename}: {str(e)}"
            logger.error(error_msg)
            
            document.processing_status = ProcessingStatus.FAILED
            progress.status = ProcessingStatus.FAILED
            progress.error_message = str(e)
            progress.processing_end_time = datetime.now()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=False,
                documents_processed=0,
                total_chunks=0,
                failed_documents=[filename],
                processing_time_seconds=processing_time,
                error_message=error_msg
            )
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error processing {filename}: {str(e)}"
            logger.error(error_msg)
            
            document.processing_status = ProcessingStatus.FAILED
            progress.status = ProcessingStatus.FAILED
            progress.error_message = str(e)
            progress.processing_end_time = datetime.now()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=False,
                documents_processed=0,
                total_chunks=0,
                failed_documents=[filename],
                processing_time_seconds=processing_time,
                error_message=error_msg
            )
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """
        Save uploaded file content to the upload directory.
        
        Args:
            file_content: Binary content of the uploaded file
            filename: Original filename
            
        Returns:
            Path to the saved file
            
        Raises:
            DocumentProcessingError: If file saving fails
        """
        try:
            # Generate unique filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(filename)
            unique_filename = f"{timestamp}_{name}{ext}"
            
            file_path = self.upload_directory / unique_filename
            
            # Write file content
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Saved uploaded file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to save uploaded file {filename}: {e}")
    
    def get_processing_progress(self, document_id: str) -> Optional[DocumentProcessingProgress]:
        """
        Get processing progress for a specific document.
        
        Args:
            document_id: ID of the document to check
            
        Returns:
            DocumentProcessingProgress if found, None otherwise
        """
        return self.processing_progress.get(document_id)
    
    def get_all_processing_progress(self) -> Dict[str, DocumentProcessingProgress]:
        """
        Get processing progress for all documents.
        
        Returns:
            Dictionary mapping document IDs to their progress
        """
        return self.processing_progress.copy()
    
    def clear_processing_progress(self) -> None:
        """Clear all processing progress records."""
        self.processing_progress.clear()
        logger.info("Cleared all processing progress records")
    
    def delete_document(self, document_id: str, file_path: Optional[str] = None) -> bool:
        """
        Delete a document and all its associated data.
        
        Args:
            document_id: ID of the document to delete
            file_path: Optional path to the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete chunks from vector store
            vector_success = self.vector_store.delete_chunks_by_document(document_id)
            
            # Delete file if path provided
            file_success = True
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {e}")
                    file_success = False
            
            # Remove from progress tracking
            if document_id in self.processing_progress:
                del self.processing_progress[document_id]
            
            success = vector_success and file_success
            logger.info(f"Document deletion {'successful' if success else 'partially failed'}: {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document service.
        
        Returns:
            Dictionary containing service statistics
        """
        try:
            vector_stats = self.vector_store.get_collection_stats()
            embedding_stats = self.embedding_service.get_cache_stats()
            
            # Count files in upload directory
            upload_files = list(self.upload_directory.glob("*.pdf"))
            
            # Processing progress stats
            progress_stats = {
                "total_tracked": len(self.processing_progress),
                "completed": sum(1 for p in self.processing_progress.values() 
                               if p.status == ProcessingStatus.COMPLETED),
                "processing": sum(1 for p in self.processing_progress.values() 
                                if p.status == ProcessingStatus.PROCESSING),
                "failed": sum(1 for p in self.processing_progress.values() 
                            if p.status == ProcessingStatus.FAILED)
            }
            
            return {
                "upload_directory": str(self.upload_directory),
                "uploaded_files": len(upload_files),
                "vector_store": vector_stats,
                "embedding_cache": embedding_stats,
                "processing_progress": progress_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {"error": str(e)}
    
    def cleanup_failed_uploads(self) -> int:
        """
        Clean up files from failed processing attempts.
        
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        
        try:
            # Find failed processing records
            failed_docs = [p for p in self.processing_progress.values() 
                          if p.status == ProcessingStatus.FAILED]
            
            for progress in failed_docs:
                # Try to find and remove the associated file
                for file_path in self.upload_directory.glob(f"*{progress.filename}"):
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                        logger.info(f"Cleaned up failed upload: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {file_path}: {e}")
                
                # Remove from progress tracking
                if progress.document_id in self.processing_progress:
                    del self.processing_progress[progress.document_id]
            
            logger.info(f"Cleanup completed: {cleaned_count} files removed")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return cleaned_count