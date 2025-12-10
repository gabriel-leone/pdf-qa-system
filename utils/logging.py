"""
Enhanced logging configuration for the PDF Q&A System
"""
import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from config import settings


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs for better parsing
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add extra fields if present
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, default=str)


class ContextFilter(logging.Filter):
    """
    Filter that adds contextual information to log records
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to log record"""
        # Add service context
        record.service = "pdf-qa-system"
        record.version = "1.0.0"
        
        # Add process information
        record.pid = os.getpid()
        
        return True


def setup_logging(
    log_level: Optional[str] = None,
    log_format: str = "structured",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure comprehensive application logging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("structured" for JSON, "simple" for text)
        log_file: Optional file path for file logging
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured root logger
    """
    # Determine log level
    if log_level is None:
        log_level = getattr(settings, 'log_level', 'INFO')
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    if log_format == "structured":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(service)s - %(name)s - %(levelname)s - %(message)s - '
            '[%(module)s:%(funcName)s:%(lineno)d] [PID:%(pid)d]'
        )
    
    # Add context filter
    context_filter = ContextFilter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    configure_service_loggers(level)
    
    # Log startup information
    root_logger.info(
        "Logging configured",
        extra={
            "log_level": log_level,
            "log_format": log_format,
            "log_file": log_file,
            "handlers": len(root_logger.handlers)
        }
    )
    
    return root_logger


def configure_service_loggers(level: int):
    """Configure logging for specific services and libraries"""
    
    # Set appropriate levels for third-party libraries
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
    
    # Configure service-specific loggers
    service_loggers = [
        "services.document_service",
        "services.question_service", 
        "services.llm_service",
        "services.embedding_service",
        "services.vector_store",
        "services.pdf_processor",
        "services.text_chunker",
        "services.retrieval_service",
        "api.document_controller",
        "api.question_controller"
    ]
    
    for logger_name in service_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)


def get_performance_logger() -> logging.Logger:
    """Get a dedicated logger for performance metrics"""
    perf_logger = logging.getLogger("performance")
    
    # Add a separate handler for performance logs if needed
    if not perf_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s'
        )
        handler.setFormatter(formatter)
        perf_logger.addHandler(handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False  # Don't propagate to root logger
    
    return perf_logger


def get_security_logger() -> logging.Logger:
    """Get a dedicated logger for security events"""
    security_logger = logging.getLogger("security")
    
    # Add a separate handler for security logs if needed
    if not security_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        security_logger.addHandler(handler)
        security_logger.setLevel(logging.WARNING)
        security_logger.propagate = False  # Don't propagate to root logger
    
    return security_logger


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: int,
    user_agent: Optional[str] = None,
    client_ip: Optional[str] = None
):
    """
    Log API request information
    
    Args:
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        user_agent: User agent string
        client_ip: Client IP address
    """
    logger = logging.getLogger("api")
    
    extra = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "user_agent": user_agent,
        "client_ip": client_ip
    }
    
    if status_code >= 500:
        logger.error(f"API request failed: {method} {path}", extra=extra)
    elif status_code >= 400:
        logger.warning(f"API request error: {method} {path}", extra=extra)
    else:
        logger.info(f"API request: {method} {path}", extra=extra)


def log_security_event(
    event_type: str,
    description: str,
    severity: str = "medium",
    client_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    additional_data: Optional[Dict[str, Any]] = None
):
    """
    Log security-related events
    
    Args:
        event_type: Type of security event
        description: Description of the event
        severity: Severity level (low, medium, high, critical)
        client_ip: Client IP address
        user_agent: User agent string
        additional_data: Additional event data
    """
    security_logger = get_security_logger()
    
    extra = {
        "event_type": event_type,
        "severity": severity,
        "client_ip": client_ip,
        "user_agent": user_agent
    }
    
    if additional_data:
        extra.update(additional_data)
    
    if severity in ["high", "critical"]:
        security_logger.error(f"Security event: {description}", extra=extra)
    elif severity == "medium":
        security_logger.warning(f"Security event: {description}", extra=extra)
    else:
        security_logger.info(f"Security event: {description}", extra=extra)


# Initialize logging with default configuration
logger = setup_logging()