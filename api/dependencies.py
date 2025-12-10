"""
Dependency injection for the PDF Q&A System API
"""
import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from services.document_service import DocumentService
from services.question_service import QuestionService
from services.retrieval_service import RetrievalService
from services.llm_service import LLMService
from services.vector_store import create_vector_store
from config import settings

logger = logging.getLogger(__name__)


@lru_cache()
def get_vector_store():
    """
    Get vector store instance (cached singleton)
    """
    return create_vector_store("chroma")


@lru_cache()
def get_document_service():
    """
    Get document service instance (cached singleton)
    """
    vector_store = get_vector_store()
    embedding_service = get_embedding_service()
    return DocumentService(vector_store=vector_store, embedding_service=embedding_service)


@lru_cache()
def get_llm_service():
    """
    Get LLM service instance (cached singleton)
    """
    return LLMService()


@lru_cache()
def get_embedding_service():
    """
    Get embedding service instance (cached singleton)
    """
    from services.embedding_service import EmbeddingService
    return EmbeddingService()


@lru_cache()
def get_retrieval_service():
    """
    Get retrieval service instance (cached singleton)
    """
    vector_store = get_vector_store()
    embedding_service = get_embedding_service()
    return RetrievalService(vector_store, embedding_service)


@lru_cache()
def get_question_service():
    """
    Get question service instance (cached singleton)
    """
    retrieval_service = get_retrieval_service()
    llm_service = get_llm_service()
    return QuestionService(retrieval_service, llm_service)


# Type annotations for dependency injection
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
QuestionServiceDep = Annotated[QuestionService, Depends(get_question_service)]
VectorStoreDep = Annotated[object, Depends(get_vector_store)]