"""
Document upload controller for the PDF Q&A System REST API
"""
import logging
import time
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form
from fastapi.responses import JSONResponse

from models.api import DocumentUploadResponse, ErrorResponse
from services.document_service import DocumentService
from utils.exceptions import (
    FileHandlingError, DocumentProcessingError, ValidationError, ErrorCode,
    create_file_too_large_error, create_invalid_file_type_error
)

logger = logging.getLogger(__name__)

# Create router for document endpoints
router = APIRouter(prefix="/documents", tags=["documents"])

# Import dependencies
from api.dependencies import DocumentServiceDep


@router.post(
    "/",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload PDF documents for processing",
    description="Upload one or more PDF files to be processed and indexed for question answering"
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="PDF files to upload and process"),
    language_hint: str = Form("auto", pattern="^(pt|en|auto)$", description="Language hint for processing"),
    document_service: DocumentServiceDep = None
) -> DocumentUploadResponse:
    """
    Upload and process PDF documents.
    
    This endpoint accepts multiple PDF files, processes them through the complete pipeline
    (text extraction, chunking, embedding generation, and vector storage), and returns
    a summary of the processing results.
    
    Args:
        files: List of PDF files to upload
        language_hint: Optional language hint for processing (pt, en, or auto)
        
    Returns:
        DocumentUploadResponse with processing summary
        
    Raises:
        HTTPException: For various error conditions (400, 413, 422, 500)
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not files:
            raise ValidationError(
                message="No files were provided for upload",
                field_name="files"
            )
        
        # Validate file types and sizes
        validated_files = []
        for file in files:
            # Check file type
            if not file.filename or not file.filename.lower().endswith('.pdf'):
                raise create_invalid_file_type_error(file.filename or "unknown", [".pdf"])
            
            # Read file content to check size
            content = await file.read()
            file_size = len(content)
            
            # Check file size (max 100MB)
            max_size = 100 * 1024 * 1024  # 100MB
            if file_size > max_size:
                raise create_file_too_large_error(file.filename, file_size, max_size)
            
            # Check for empty files
            if file_size == 0:
                raise FileHandlingError(
                    message=f"File '{file.filename}' is empty",
                    filename=file.filename,
                    file_size=file_size,
                    error_code=ErrorCode.EMPTY_FILE
                )
            
            validated_files.append((file.filename, content))
        
        logger.info(f"Processing {len(validated_files)} uploaded files")
        
        # Save uploaded files and get their paths
        file_paths = []
        for filename, content in validated_files:
            try:
                file_path = document_service.save_uploaded_file(content, filename)
                file_paths.append(file_path)
            except DocumentProcessingError as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": {
                            "code": "FILE_SAVE_FAILED",
                            "message": f"Failed to save uploaded file: {str(e)}",
                            "details": {"filename": filename},
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        }
                    }
                )
        
        # Process documents through the pipeline
        try:
            processing_result = document_service.process_documents(file_paths)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Handle partial failures
            if processing_result.failed_documents:
                logger.warning(f"Some documents failed processing: {processing_result.failed_documents}")
                
                # If all documents failed, return error
                if processing_result.documents_processed == 0:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "error": {
                                "code": "ALL_DOCUMENTS_FAILED",
                                "message": "All uploaded documents failed to process",
                                "details": {
                                    "failed_documents": processing_result.failed_documents,
                                    "error_message": processing_result.error_message
                                },
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                            }
                        }
                    )
            
            # Create success response
            response = DocumentUploadResponse(
                message="Documents processed successfully" if processing_result.success 
                       else f"Processed {processing_result.documents_processed} documents with some failures",
                documents_indexed=processing_result.documents_processed,
                total_chunks=processing_result.total_chunks,
                processing_time_ms=processing_time_ms
            )
            
            logger.info(f"Document upload completed: {processing_result.documents_processed} documents, "
                       f"{processing_result.total_chunks} chunks, {processing_time_ms}ms")
            
            return response
            
        except DocumentProcessingError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "PROCESSING_FAILED",
                        "message": f"Document processing failed: {str(e)}",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                }
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in document upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred while processing documents",
                    "details": {"error_type": type(e).__name__},
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            }
        )


@router.get(
    "/stats",
    summary="Get document processing statistics",
    description="Get statistics about uploaded documents and processing status"
)
async def get_document_stats(document_service: DocumentServiceDep = None):
    """
    Get statistics about the document service.
    
    Returns:
        Dictionary containing service statistics
    """
    try:
        stats = document_service.get_service_stats()
        return {"status": "success", "data": stats}
        
    except Exception as e:
        logger.error(f"Failed to get document stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "STATS_RETRIEVAL_FAILED",
                    "message": "Failed to retrieve document statistics",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            }
        )


@router.post(
    "/cleanup",
    summary="Clean up failed uploads",
    description="Remove files from failed processing attempts"
)
async def cleanup_failed_uploads(document_service: DocumentServiceDep = None):
    """
    Clean up files from failed processing attempts.
    
    Returns:
        Summary of cleanup operation
    """
    try:
        cleaned_count = document_service.cleanup_failed_uploads()
        return {
            "status": "success",
            "message": f"Cleaned up {cleaned_count} failed uploads",
            "files_removed": cleaned_count
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup uploads: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "CLEANUP_FAILED",
                    "message": "Failed to cleanup failed uploads",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            }
        )