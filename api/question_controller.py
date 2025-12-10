"""
Question answering controller for the PDF Q&A System REST API
"""
import logging
import time
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from models.api import QuestionRequest, QuestionResponse, ErrorResponse
from utils.exceptions import (
    ServiceUnavailableError, QuestionProcessingError, ValidationError,
    create_no_documents_error, create_llm_unavailable_error, ErrorCode
)


logger = logging.getLogger(__name__)

# Create router for question endpoints
router = APIRouter(prefix="/question", tags=["questions"])

# Import dependencies
from api.dependencies import QuestionServiceDep, VectorStoreDep


@router.post(
    "/",
    response_model=QuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about uploaded documents",
    description="Submit a question to get an answer based on the content of uploaded PDF documents"
)
async def ask_question(
    request: QuestionRequest,
    question_service: QuestionServiceDep = None,
    vector_store: VectorStoreDep = None
) -> QuestionResponse:
    """
    Answer a question based on uploaded document content.
    
    This endpoint processes a natural language question, searches for relevant content
    in the uploaded documents using semantic search, and generates an answer using
    a large language model with the retrieved context.
    
    Args:
        request: QuestionRequest containing the question and optional parameters
        
    Returns:
        QuestionResponse with answer, confidence score, and supporting references
        
    Raises:
        HTTPException: For various error conditions (400, 404, 500, 503)
    """
    try:
        # Validate that the service is ready
        if not question_service.is_service_ready():
            # Check specific issues
            if not question_service.llm_service.is_available():
                raise create_llm_unavailable_error()
            
            # Check if documents are indexed
            stats = vector_store.get_collection_stats()
            if stats.get("total_chunks", 0) == 0:
                raise create_no_documents_error()
            
            # Generic service unavailable
            raise ServiceUnavailableError(
                message="Question answering service is not ready. Please try again later.",
                service_name="question_service"
            )
        
        logger.info(f"Processing question: {request.question[:100]}...")
        
        # Process the question
        try:
            response = question_service.answer_question(request)
            
            logger.info(f"Question answered successfully in {response.processing_time_ms}ms")
            return response
            
        except ValueError as e:
            # Handle validation errors
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "INVALID_QUESTION",
                        "message": f"Invalid question format: {str(e)}",
                        "details": {"question": request.question},
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                }
            )
            
        except Exception as e:
            # Handle processing errors
            logger.error(f"Error processing question: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "code": "QUESTION_PROCESSING_FAILED",
                        "message": f"Failed to process question: {str(e)}",
                        "details": {"error_type": type(e).__name__},
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                }
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in question processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred while processing the question",
                    "details": {"error_type": type(e).__name__},
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            }
        )


@router.get(
    "/stats",
    summary="Get question service statistics",
    description="Get statistics about the question answering service and its readiness"
)
async def get_question_stats(question_service: QuestionServiceDep = None):
    """
    Get statistics about the question service.
    
    Returns:
        Dictionary containing service statistics and readiness information
    """
    try:
        stats = question_service.get_service_stats()
        return {"status": "success", "data": stats}
        
    except Exception as e:
        logger.error(f"Failed to get question stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "STATS_RETRIEVAL_FAILED",
                    "message": "Failed to retrieve question service statistics",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            }
        )


@router.get(
    "/health",
    summary="Check question service health",
    description="Check if the question answering service is ready to process questions"
)
async def check_question_health(
    question_service: QuestionServiceDep = None,
    vector_store: VectorStoreDep = None
):
    """
    Check the health and readiness of the question service.
    
    Returns:
        Health status information
    """
    try:
        is_ready = question_service.is_service_ready()
        llm_available = question_service.llm_service.is_available()
        
        # Get vector store stats
        vector_stats = vector_store.get_collection_stats()
        has_documents = vector_stats.get("total_chunks", 0) > 0
        
        health_status = {
            "service_ready": is_ready,
            "llm_available": llm_available,
            "documents_indexed": has_documents,
            "total_chunks": vector_stats.get("total_chunks", 0),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        # Determine overall status
        if is_ready:
            status_code = status.HTTP_200_OK
            health_status["status"] = "healthy"
            health_status["message"] = "Question service is ready to process questions"
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            health_status["status"] = "unhealthy"
            
            # Provide specific reason
            if not llm_available:
                health_status["message"] = "LLM service is not available"
            elif not has_documents:
                health_status["message"] = "No documents indexed"
            else:
                health_status["message"] = "Service is not ready"
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
        
    except Exception as e:
        logger.error(f"Failed to check question health: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Failed to check service health",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        )