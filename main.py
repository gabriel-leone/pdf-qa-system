"""
PDF Q&A System - Main application entry point
"""
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import os
from dotenv import load_dotenv

# Import API routers
from api.document_controller import router as document_router
from api.question_controller import router as question_router

# Import enhanced error handling
from utils.error_handlers import ErrorHandlingMiddleware
from utils.logging import setup_logging
from utils.exceptions import PDFQAException, ErrorCode

# Load environment variables
load_dotenv()

# Configure enhanced logging
logger = setup_logging()

app = FastAPI(
    title="PDF Q&A System",
    description="A system for uploading PDF documents and asking questions about their contents using retrieval-augmented generation (RAG)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add comprehensive error handling middleware
app.add_middleware(ErrorHandlingMiddleware)


# Custom exception handlers for specific cases
@app.exception_handler(PDFQAException)
async def pdfqa_exception_handler(request: Request, exc: PDFQAException):
    """
    Handle custom PDF Q&A exceptions with structured error responses
    """
    logger.warning(f"PDF Q&A exception in {request.method} {request.url}: {exc}")
    
    # Map error codes to HTTP status codes
    status_code_map = {
        ErrorCode.VALIDATION_ERROR: 400,
        ErrorCode.INVALID_QUESTION: 400,
        ErrorCode.FILE_TOO_LARGE: 413,
        ErrorCode.INVALID_FILE_TYPE: 400,
        ErrorCode.EMPTY_FILE: 400,
        ErrorCode.NO_DOCUMENTS_INDEXED: 404,
        ErrorCode.SERVICE_UNAVAILABLE: 503,
        ErrorCode.LLM_SERVICE_UNAVAILABLE: 503,
        ErrorCode.LLM_RATE_LIMIT: 429,
        ErrorCode.LLM_TIMEOUT: 504,
        ErrorCode.DOCUMENT_PROCESSING_FAILED: 422,
        ErrorCode.QUESTION_PROCESSING_FAILED: 500,
    }
    
    status_code = status_code_map.get(exc.error_code, 500)
    
    return JSONResponse(
        status_code=status_code,
        content=exc.to_dict()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with enhanced formatting
    """
    logger.warning(f"Validation error in {request.method} {request.url}: {exc}")
    
    # Extract field-specific errors for better user experience
    field_errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        field_errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": "Request validation failed",
                "details": {
                    "field_errors": field_errors,
                    "raw_errors": exc.errors()
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with consistent formatting
    """
    # If detail is already a dict (our custom format), return as-is
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    
    # Otherwise, format as standard error
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        }
    )


# Include API routers
app.include_router(document_router)
app.include_router(question_router)


@app.get("/health")
async def health_check():
    """
    Basic health check endpoint for load balancers and monitoring
    """
    return {
        "status": "healthy",
        "message": "PDF Q&A System is running",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check endpoint with component status
    """
    try:
        from utils.health_check import HealthChecker
        from api.dependencies import get_llm_service, get_embedding_service, get_vector_store
        
        # Get service instances (this is a simplified approach)
        # In a real implementation, you'd inject these properly
        health_checker = HealthChecker()
        
        system_health = await health_checker.check_system_health(include_details=True)
        
        # Convert to dict for JSON response
        response = {
            "status": system_health.status.value,
            "message": system_health.message,
            "timestamp": system_health.timestamp,
            "uptime_seconds": system_health.uptime_seconds,
            "components": [
                {
                    "name": comp.name,
                    "status": comp.status.value,
                    "message": comp.message,
                    "details": comp.details,
                    "response_time_ms": comp.response_time_ms,
                    "last_check": comp.last_check
                }
                for comp in system_health.components
            ]
        }
        
        # Return appropriate HTTP status based on health
        if system_health.status.value == "healthy":
            status_code = 200
        elif system_health.status.value == "degraded":
            status_code = 200  # Still operational
        else:
            status_code = 503  # Service unavailable
        
        return JSONResponse(status_code=status_code, content=response)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        )


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Welcome to the PDF Q&A System API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "upload_documents": "POST /documents/",
            "ask_question": "POST /question/",
            "health_check": "GET /health"
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)