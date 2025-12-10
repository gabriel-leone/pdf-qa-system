"""
Health check utilities for the PDF Q&A System

This module provides comprehensive health checking for all system components,
including external services, database connections, and service readiness.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

from services.llm_service import LLMService
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStoreInterface
from utils.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a system component"""
    name: str
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[int] = None
    last_check: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health information"""
    status: HealthStatus
    message: str
    components: List[ComponentHealth]
    timestamp: str
    uptime_seconds: Optional[int] = None


class HealthChecker:
    """
    Comprehensive health checker for all system components
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreInterface] = None
    ):
        """
        Initialize health checker with system components
        
        Args:
            llm_service: LLM service instance
            embedding_service: Embedding service instance
            vector_store: Vector store instance
        """
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.start_time = time.time()
    
    async def check_system_health(self, include_details: bool = True) -> SystemHealth:
        """
        Check the health of all system components
        
        Args:
            include_details: Whether to include detailed component information
            
        Returns:
            SystemHealth object with overall status and component details
        """
        components = []
        
        # Check all components
        if self.llm_service:
            components.append(await self._check_llm_service())
        
        if self.embedding_service:
            components.append(await self._check_embedding_service())
        
        if self.vector_store:
            components.append(await self._check_vector_store())
        
        # Add basic system checks
        components.append(self._check_memory_usage())
        components.append(self._check_disk_space())
        
        # Determine overall system status
        overall_status = self._determine_overall_status(components)
        
        # Create system health summary
        system_health = SystemHealth(
            status=overall_status,
            message=self._get_status_message(overall_status, components),
            components=components if include_details else [],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            uptime_seconds=int(time.time() - self.start_time)
        )
        
        return system_health
    
    async def _check_llm_service(self) -> ComponentHealth:
        """Check LLM service health"""
        start_time = time.time()
        
        try:
            if not self.llm_service.is_available():
                return ComponentHealth(
                    name="llm_service",
                    status=HealthStatus.UNHEALTHY,
                    message="LLM service is not available - missing API key or client not initialized",
                    response_time_ms=int((time.time() - start_time) * 1000),
                    last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                )
            
            # Try a simple test call
            test_chunks = []  # Empty context for basic test
            try:
                # This will test the service without making an actual API call
                model_info = self.llm_service.get_model_info()
                
                return ComponentHealth(
                    name="llm_service",
                    status=HealthStatus.HEALTHY,
                    message="LLM service is available and configured",
                    details=model_info,
                    response_time_ms=int((time.time() - start_time) * 1000),
                    last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                )
            except Exception as e:
                return ComponentHealth(
                    name="llm_service",
                    status=HealthStatus.DEGRADED,
                    message=f"LLM service configuration issue: {str(e)}",
                    response_time_ms=int((time.time() - start_time) * 1000),
                    last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                )
                
        except Exception as e:
            return ComponentHealth(
                name="llm_service",
                status=HealthStatus.UNHEALTHY,
                message=f"LLM service check failed: {str(e)}",
                response_time_ms=int((time.time() - start_time) * 1000),
                last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
    
    async def _check_embedding_service(self) -> ComponentHealth:
        """Check embedding service health"""
        start_time = time.time()
        
        try:
            # Test with a simple text
            test_text = "health check test"
            embedding = self.embedding_service.generate_embedding(test_text)
            
            if embedding is not None and len(embedding) > 0:
                cache_stats = self.embedding_service.get_cache_stats()
                
                return ComponentHealth(
                    name="embedding_service",
                    status=HealthStatus.HEALTHY,
                    message="Embedding service is working correctly",
                    details={
                        "model_name": getattr(self.embedding_service, 'model_name', 'unknown'),
                        "embedding_dimension": len(embedding),
                        "cache_stats": cache_stats
                    },
                    response_time_ms=int((time.time() - start_time) * 1000),
                    last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                )
            else:
                return ComponentHealth(
                    name="embedding_service",
                    status=HealthStatus.UNHEALTHY,
                    message="Embedding service returned invalid result",
                    response_time_ms=int((time.time() - start_time) * 1000),
                    last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                )
                
        except Exception as e:
            return ComponentHealth(
                name="embedding_service",
                status=HealthStatus.UNHEALTHY,
                message=f"Embedding service check failed: {str(e)}",
                response_time_ms=int((time.time() - start_time) * 1000),
                last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
    
    async def _check_vector_store(self) -> ComponentHealth:
        """Check vector store health"""
        start_time = time.time()
        
        try:
            # Get collection stats
            stats = self.vector_store.get_collection_stats()
            
            # Check if we can connect and get basic info
            if isinstance(stats, dict):
                return ComponentHealth(
                    name="vector_store",
                    status=HealthStatus.HEALTHY,
                    message="Vector store is accessible",
                    details=stats,
                    response_time_ms=int((time.time() - start_time) * 1000),
                    last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                )
            else:
                return ComponentHealth(
                    name="vector_store",
                    status=HealthStatus.DEGRADED,
                    message="Vector store returned unexpected response",
                    response_time_ms=int((time.time() - start_time) * 1000),
                    last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                )
                
        except Exception as e:
            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.UNHEALTHY,
                message=f"Vector store check failed: {str(e)}",
                response_time_ms=int((time.time() - start_time) * 1000),
                last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
    
    def _check_memory_usage(self) -> ComponentHealth:
        """Check system memory usage"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Memory usage is normal ({memory_percent:.1f}%)"
            elif memory_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"Memory usage is high ({memory_percent:.1f}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage is critical ({memory_percent:.1f}%)"
            
            return ComponentHealth(
                name="memory",
                status=status,
                message=message,
                details={
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": memory_percent
                },
                last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
            
        except ImportError:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="Memory monitoring not available (psutil not installed)",
                last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {str(e)}",
                last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
    
    def _check_disk_space(self) -> ComponentHealth:
        """Check disk space usage"""
        try:
            import psutil
            
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Disk usage is normal ({disk_percent:.1f}%)"
            elif disk_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"Disk usage is high ({disk_percent:.1f}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Disk usage is critical ({disk_percent:.1f}%)"
            
            return ComponentHealth(
                name="disk",
                status=status,
                message=message,
                details={
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": disk_percent
                },
                last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
            
        except ImportError:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message="Disk monitoring not available (psutil not installed)",
                last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
        except Exception as e:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message=f"Disk check failed: {str(e)}",
                last_check=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
    
    def _determine_overall_status(self, components: List[ComponentHealth]) -> HealthStatus:
        """Determine overall system status based on component health"""
        if not components:
            return HealthStatus.UNKNOWN
        
        # Count status types
        unhealthy_count = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)
        healthy_count = sum(1 for c in components if c.status == HealthStatus.HEALTHY)
        
        # Determine overall status
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        elif healthy_count > 0:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _get_status_message(self, status: HealthStatus, components: List[ComponentHealth]) -> str:
        """Get a descriptive message for the overall status"""
        total_components = len(components)
        
        if status == HealthStatus.HEALTHY:
            return f"All {total_components} system components are healthy"
        elif status == HealthStatus.DEGRADED:
            degraded_components = [c.name for c in components if c.status == HealthStatus.DEGRADED]
            return f"System is degraded - issues with: {', '.join(degraded_components)}"
        elif status == HealthStatus.UNHEALTHY:
            unhealthy_components = [c.name for c in components if c.status == HealthStatus.UNHEALTHY]
            return f"System is unhealthy - critical issues with: {', '.join(unhealthy_components)}"
        else:
            return "System status is unknown"


# Utility functions for health checking

async def quick_health_check(
    llm_service: Optional[LLMService] = None,
    vector_store: Optional[VectorStoreInterface] = None
) -> Dict[str, Any]:
    """
    Perform a quick health check of critical components
    
    Args:
        llm_service: LLM service instance
        vector_store: Vector store instance
        
    Returns:
        Dictionary with basic health information
    """
    health_info = {
        "status": "unknown",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "services": {}
    }
    
    try:
        # Check LLM service
        if llm_service:
            health_info["services"]["llm"] = {
                "available": llm_service.is_available(),
                "model_info": llm_service.get_model_info()
            }
        
        # Check vector store
        if vector_store:
            try:
                stats = vector_store.get_collection_stats()
                health_info["services"]["vector_store"] = {
                    "available": True,
                    "stats": stats
                }
            except Exception as e:
                health_info["services"]["vector_store"] = {
                    "available": False,
                    "error": str(e)
                }
        
        # Determine overall status
        all_available = all(
            service.get("available", False) 
            for service in health_info["services"].values()
        )
        
        health_info["status"] = "healthy" if all_available else "degraded"
        
    except Exception as e:
        health_info["status"] = "unhealthy"
        health_info["error"] = str(e)
    
    return health_info


def is_service_ready(
    llm_service: Optional[LLMService] = None,
    vector_store: Optional[VectorStoreInterface] = None,
    require_documents: bool = True
) -> bool:
    """
    Check if the service is ready to handle requests
    
    Args:
        llm_service: LLM service instance
        vector_store: Vector store instance
        require_documents: Whether to require documents to be indexed
        
    Returns:
        True if service is ready, False otherwise
    """
    try:
        # Check LLM service
        if llm_service and not llm_service.is_available():
            return False
        
        # Check vector store and documents
        if vector_store:
            try:
                stats = vector_store.get_collection_stats()
                if require_documents and stats.get("total_chunks", 0) == 0:
                    return False
            except Exception:
                return False
        
        return True
        
    except Exception:
        return False