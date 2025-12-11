"""
PDF Q&A System - Main application entry point
"""
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Import configuration and dependencies
from config import settings
from api.dependencies import (
    get_vector_store, 
    get_document_service, 
    get_question_service,
    get_llm_service,
    get_embedding_service
)

# Import API routers
from api.document_controller import router as document_router
from api.question_controller import router as question_router

# Import enhanced error handling and utilities
from utils.error_handlers import ErrorHandlingMiddleware
from utils.logging import setup_logging, log_api_request
from utils.exceptions import PDFQAException, ErrorCode
from utils.health_check import HealthChecker

# Configure enhanced logging
logger = setup_logging()

# Global application state
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events
    """
    # Startup
    start_time = time.time()
    app_state["start_time"] = start_time
    
    logger.info(f"Starting {settings.app_name} v{settings.app_version}...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    try:
        # Initialize core services
        logger.info("Initializing core services...")
        
        # Initialize vector store
        vector_store = get_vector_store()
        app_state["vector_store"] = vector_store
        logger.info("Vector store initialized")
        
        # Initialize embedding service
        embedding_service = get_embedding_service()
        app_state["embedding_service"] = embedding_service
        logger.info("Embedding service initialized")
        
        # Initialize LLM service
        llm_service = get_llm_service()
        app_state["llm_service"] = llm_service
        logger.info("LLM service initialized")
        
        # Initialize document service
        document_service = get_document_service()
        app_state["document_service"] = document_service
        logger.info("Document service initialized")
        
        # Initialize question service
        question_service = get_question_service()
        app_state["question_service"] = question_service
        logger.info("Question service initialized")
        
        # Initialize health checker
        health_checker = HealthChecker(
            llm_service=llm_service,
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        app_state["health_checker"] = health_checker
        logger.info("Health checker initialized")
        
        # Perform initial health check
        system_health = await health_checker.check_system_health()
        logger.info(f"Initial system health check: {system_health.status.value}")
        
        # Log startup completion
        startup_time = time.time() - start_time
        logger.info(f"{settings.app_name} startup completed successfully in {startup_time:.2f} seconds")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start PDF Q&A System: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down PDF Q&A System...")
    
    try:
        # Cleanup services
        if "vector_store" in app_state:
            # Perform any necessary cleanup for vector store
            logger.info("Cleaning up vector store...")
        
        if "embedding_service" in app_state:
            # Cleanup embedding service cache if needed
            logger.info("Cleaning up embedding service...")
        
        # Clear application state
        app_state.clear()
        
        logger.info("PDF Q&A System shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application with lifespan manager
app = FastAPI(
    title="PDF Q&A System",
    description="A system for uploading PDF documents and asking questions about their contents using retrieval-augmented generation (RAG)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure middleware
def configure_middleware():
    """Configure all application middleware"""
    
    # Security middleware - Trusted Host
    if hasattr(settings, 'allowed_hosts') and settings.allowed_hosts:
        app.add_middleware(
            TrustedHostMiddleware, 
            allowed_hosts=settings.allowed_hosts.split(",")
        )
    
    # CORS middleware
    cors_origins = getattr(settings, 'cors_origins', "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        max_age=3600,  # Cache preflight requests for 1 hour
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # Get client information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        try:
            response = await call_next(request)
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log the request
            log_api_request(
                method=request.method,
                path=str(request.url.path),
                status_code=response.status_code,
                duration_ms=duration_ms,
                user_agent=user_agent,
                client_ip=client_ip
            )
            
            # Add response headers
            response.headers["X-Response-Time"] = f"{duration_ms}ms"
            response.headers["X-Request-ID"] = getattr(request.state, "request_id", "unknown")
            
            return response
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Request failed: {request.method} {request.url.path} - {str(e)}")
            
            # Log failed request
            log_api_request(
                method=request.method,
                path=str(request.url.path),
                status_code=500,
                duration_ms=duration_ms,
                user_agent=user_agent,
                client_ip=client_ip
            )
            
            raise
    
    # Add comprehensive error handling middleware
    app.add_middleware(ErrorHandlingMiddleware)


# Configure all middleware
configure_middleware()


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
        # Get health checker from application state
        health_checker = app_state.get("health_checker")
        
        if not health_checker:
            # Fallback: create a new health checker
            health_checker = HealthChecker(
                llm_service=app_state.get("llm_service"),
                embedding_service=app_state.get("embedding_service"),
                vector_store=app_state.get("vector_store")
            )
        
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


@app.get("/health/ready")
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes/container orchestration
    """
    try:
        # Check if all required services are ready
        vector_store = app_state.get("vector_store")
        llm_service = app_state.get("llm_service")
        
        if not vector_store or not llm_service:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "message": "Required services not initialized",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            )
        
        # Check if LLM service is available
        if not llm_service.is_available():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "message": "LLM service not available",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            )
        
        return {
            "status": "ready",
            "message": "Service is ready to accept requests",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "message": f"Readiness check failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        )


@app.get("/health/live")
async def liveness_check():
    """
    Liveness check endpoint for Kubernetes/container orchestration
    """
    return {
        "status": "alive",
        "message": "Service is alive",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uptime_seconds": int(time.time() - app_state.get("start_time", time.time()))
    }


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": f"Welcome to the {settings.app_name} API",
        "version": settings.app_version,
        "environment": settings.environment,
        "documentation": "/docs",
        "endpoints": {
            "upload_documents": "POST /documents/",
            "ask_question": "POST /question/",
            "health_check": "GET /health",
            "detailed_health": "GET /health/detailed",
            "readiness": "GET /health/ready",
            "liveness": "GET /health/live"
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


@app.get("/info")
async def application_info():
    """
    Application information endpoint
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "configuration": {
            "max_file_size_mb": settings.max_file_size_mb,
            "max_chunk_size": settings.max_chunk_size,
            "embedding_model": settings.embedding_model,
            "openai_model": settings.openai_model,
            "log_level": settings.log_level
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


def create_app() -> FastAPI:
    """
    Application factory function
    """
    return app


if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn logging
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["default"],
        },
    }
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers if not settings.debug else 1,
        reload=settings.debug,
        log_config=log_config,
        access_log=True,
        server_header=False,
        date_header=False
    )