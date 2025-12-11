#!/usr/bin/env python3
"""
Production application runner for the PDF Q&A System
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import settings


def setup_production_logging():
    """Setup production-grade logging"""
    from utils.logging import setup_logging
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Setup logging with file output
    log_file = logs_dir / "app.log"
    setup_logging(
        log_level=settings.log_level,
        log_format=settings.log_format,
        log_file=str(log_file)
    )


def run_server():
    """Run the application server"""
    import uvicorn
    from main import app
    
    # Setup production logging
    setup_production_logging()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Host: {settings.host}:{settings.port}")
    logger.info(f"Workers: {settings.workers}")
    
    # Configure uvicorn for production
    uvicorn_config = {
        "app": app,
        "host": settings.host,
        "port": settings.port,
        "workers": settings.workers,
        "log_level": settings.log_level.lower(),
        "access_log": True,
        "server_header": False,
        "date_header": False,
        "proxy_headers": True,
        "forwarded_allow_ips": "*"
    }
    
    # Add SSL configuration if certificates are available
    ssl_keyfile = os.getenv("SSL_KEYFILE")
    ssl_certfile = os.getenv("SSL_CERTFILE")
    
    if ssl_keyfile and ssl_certfile:
        uvicorn_config.update({
            "ssl_keyfile": ssl_keyfile,
            "ssl_certfile": ssl_certfile
        })
        logger.info("SSL/TLS enabled")
    
    # Run the server
    uvicorn.run(**uvicorn_config)


def run_health_check():
    """Run a health check against the running service"""
    import requests
    import json
    
    base_url = f"http://{settings.host}:{settings.port}"
    
    try:
        # Basic health check
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health Check Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        
        # Detailed health check
        response = requests.get(f"{base_url}/health/detailed", timeout=30)
        print(f"\nDetailed Health Check Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        
        return response.status_code == 200
        
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False


def run_migration():
    """Run database migrations or setup tasks"""
    logger = logging.getLogger(__name__)
    logger.info("Running migration/setup tasks...")
    
    try:
        # Initialize vector store
        from api.dependencies import get_vector_store
        vector_store = get_vector_store()
        
        # Perform any necessary setup
        stats = vector_store.get_collection_stats()
        logger.info(f"Vector store initialized with {stats.get('total_chunks', 0)} chunks")
        
        logger.info("Migration/setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Migration/setup failed: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PDF Q&A System Runner")
    parser.add_argument(
        "command",
        choices=["server", "health", "migrate"],
        help="Command to run"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load custom configuration if provided
    if args.config:
        os.environ["ENV_FILE"] = args.config
    
    # Execute the requested command
    if args.command == "server":
        run_server()
    elif args.command == "health":
        success = run_health_check()
        sys.exit(0 if success else 1)
    elif args.command == "migrate":
        success = run_migration()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()