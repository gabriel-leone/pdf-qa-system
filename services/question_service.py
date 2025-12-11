"""
Question answering service for the PDF Q&A System
"""
import logging
import time
from typing import List, Optional
from models.question import Question, Answer, Reference
from models.api import QuestionRequest, QuestionResponse
from services.retrieval_service import RetrievalService
from services.llm_service import LLMService, ContextChunk
from services.embedding_service import EmbeddingService
from services.language_validator import LanguageValidator
from utils.exceptions import (
    QuestionProcessingError, ValidationError, ServiceUnavailableError,
    ErrorCode, create_no_documents_error, create_llm_unavailable_error
)
from utils.error_handlers import log_processing_step, log_performance_metric

logger = logging.getLogger(__name__)


class QuestionService:
    """Service for orchestrating question answering pipeline"""
    
    def __init__(self, retrieval_service: RetrievalService, llm_service: LLMService, 
                 language_validator: Optional[LanguageValidator] = None):
        """
        Initialize the question service
        
        Args:
            retrieval_service: Service for retrieving relevant chunks
            llm_service: Service for generating answers
            language_validator: Service for language validation
        """
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service
        self.language_validator = language_validator or LanguageValidator()
        self.default_max_references = 5
        self.min_chunks_for_answer = 1  # Minimum chunks needed to generate answer
    
    def answer_question(self, request: QuestionRequest) -> QuestionResponse:
        """
        Answer a question using the retrieval-augmented generation pipeline
        
        Args:
            request: Question request containing question text and parameters
            
        Returns:
            QuestionResponse containing answer and references
        """
        start_time = time.time()
        
        try:
            # Validate request
            if not request.question or not request.question.strip():
                raise ValidationError(
                    message="Question cannot be empty",
                    field_name="question",
                    field_value=request.question
                )
            
            # Validate question language
            question_validation = self.language_validator.validate_document_language(
                request.question, request.language
            )
            
            if not question_validation.is_valid:
                logger.warning(f"Question language validation failed: {question_validation.validation_errors}")
            
            if question_validation.warnings:
                for warning in question_validation.warnings:
                    logger.warning(f"Question language warning: {warning}")
            
            # Determine language filter based on validation
            language_filter = None
            detected_question_lang = question_validation.detected_language
            
            if request.language and request.language != "auto":
                if self.language_validator.is_language_supported(request.language):
                    language_filter = request.language
                else:
                    logger.warning(f"Requested language '{request.language}' not supported, using auto-detection")
                    if detected_question_lang in self.language_validator.get_supported_languages():
                        language_filter = detected_question_lang
            
            # Determine max references
            max_references = request.max_references or self.default_max_references
            
            logger.info(f"Processing question: {request.question[:100]}...")
            
            # Step 1: Retrieve relevant chunks
            chunks_with_scores = self.retrieval_service.find_relevant_chunks(
                question=request.question,
                top_k=max_references * 2,  # Get more chunks for better context selection
                language_filter=language_filter
            )
            
            # Validate cross-language search capabilities
            if chunks_with_scores:
                cross_lang_validation = self.language_validator.validate_cross_language_search(
                    detected_question_lang, chunks_with_scores
                )
                
                if not cross_lang_validation.search_successful:
                    logger.warning(f"Cross-language search issues: {cross_lang_validation.issues}")
                
                logger.info(f"Cross-language search found languages: {cross_lang_validation.languages_found}, "
                           f"consistency score: {cross_lang_validation.consistency_score:.2f}")
            
            # Step 2: Check if we have enough relevant content
            if not chunks_with_scores or len(chunks_with_scores) < self.min_chunks_for_answer:
                logger.info("No relevant content found for question")
                return self._create_no_content_response(request.question, start_time)
            
            # Step 3: Convert chunks to context format for LLM
            context_chunks = self._convert_to_context_chunks(chunks_with_scores)
            
            # Step 4: Generate answer using LLM
            llm_response = self.llm_service.generate_answer(request.question, context_chunks)
            
            # Step 5: Create references from the most relevant chunks
            references = self.retrieval_service.create_references(
                chunks_with_scores[:max_references]
            )
            
            # Step 6: Create response
            processing_time = int((time.time() - start_time) * 1000)
            
            response = QuestionResponse(
                answer=llm_response.answer,
                confidence_score=llm_response.confidence_score,
                references=[self._reference_to_dict(ref) for ref in references],
                processing_time_ms=processing_time
            )
            
            logger.info(f"Successfully answered question in {processing_time}ms")
            return response
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            # Return error response
            processing_time = int((time.time() - start_time) * 1000)
            return QuestionResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                confidence_score=0.0,
                references=[],
                processing_time_ms=processing_time
            )
    
    def _create_no_content_response(self, question: str, start_time: float) -> QuestionResponse:
        """
        Create a response when no relevant content is found
        
        Args:
            question: The original question
            start_time: When processing started
            
        Returns:
            QuestionResponse indicating no content found
        """
        processing_time = int((time.time() - start_time) * 1000)
        
        # Use LLM service to generate a proper "no content" response
        no_content_response = self.llm_service.generate_simple_answer(question)
        
        return QuestionResponse(
            answer=no_content_response.answer,
            confidence_score=0.0,
            references=[],
            processing_time_ms=processing_time
        )
    
    def _convert_to_context_chunks(self, chunks_with_scores: List[tuple]) -> List[ContextChunk]:
        """
        Convert retrieval results to LLM context chunks
        
        Args:
            chunks_with_scores: List of (Chunk, relevance_score) tuples
            
        Returns:
            List of ContextChunk objects for LLM processing
        """
        context_chunks = []
        
        for chunk, relevance_score in chunks_with_scores:
            try:
                # Get document filename
                document_filename = getattr(chunk, 'document_filename', None)
                if not document_filename:
                    if '/' in chunk.document_id:
                        document_filename = chunk.document_id.split('/')[-1]
                    elif not chunk.document_id.endswith('.pdf'):
                        document_filename = f"{chunk.document_id}.pdf"
                    else:
                        document_filename = chunk.document_id
                
                context_chunk = ContextChunk(
                    content=chunk.content,
                    source=document_filename,
                    page_number=chunk.metadata.page_number,
                    relevance_score=relevance_score
                )
                
                context_chunks.append(context_chunk)
                
            except Exception as e:
                logger.warning(f"Failed to convert chunk {chunk.id} to context: {e}")
                continue
        
        return context_chunks
    
    def _reference_to_dict(self, reference: Reference) -> dict:
        """
        Convert Reference object to dictionary for API response
        
        Args:
            reference: Reference object
            
        Returns:
            Dictionary representation of reference
        """
        return {
            "document_filename": reference.document_filename,
            "page_number": reference.page_number,
            "excerpt": reference.excerpt,
            "relevance_score": reference.relevance_score
        }
    
    def create_question_object(self, question_text: str, language: str = "auto") -> Question:
        """
        Create a Question object with embedding
        
        Args:
            question_text: The question text
            language: Language of the question
            
        Returns:
            Question object with generated embedding
        """
        try:
            # Generate embedding for the question
            embedding = self.retrieval_service.embedding_service.generate_embedding(question_text)
            
            # Convert to list if it's a numpy array
            if hasattr(embedding, 'tolist'):
                embedding_list = embedding.tolist()
            else:
                embedding_list = embedding
            
            # Validate the question language if not auto
            if language != "auto":
                validation_result = self.language_validator.validate_document_language(question_text, language)
                if not validation_result.is_valid:
                    logger.warning(f"Question language validation failed: {validation_result.validation_errors}")
                # Use detected language if validation suggests a different one
                final_language = validation_result.detected_language if validation_result.confidence_score > 0.7 else language
            else:
                final_language = language
            
            question = Question(
                text=question_text,
                embedding=embedding_list,
                language=final_language
            )
            
            return question
            
        except Exception as e:
            logger.error(f"Failed to create question object: {e}")
            raise
    
    def is_service_ready(self) -> bool:
        """
        Check if the question service is ready to process questions
        
        Returns:
            True if all required services are available
        """
        try:
            # Check if LLM service is available
            if not self.llm_service.is_available():
                logger.warning("LLM service is not available")
                return False
            
            # Check if vector store has content
            stats = self.retrieval_service.vector_store.get_collection_stats()
            if stats.get("total_chunks", 0) == 0:
                logger.warning("No documents indexed in vector store")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking service readiness: {e}")
            return False
    
    def get_service_stats(self) -> dict:
        """
        Get statistics about the question service and its components
        
        Returns:
            Dictionary containing service statistics
        """
        try:
            stats = {
                "service_ready": self.is_service_ready(),
                "llm_available": self.llm_service.is_available(),
                "llm_model_info": self.llm_service.get_model_info(),
                "retrieval_stats": self.retrieval_service.get_retrieval_stats(),
                "configuration": {
                    "default_max_references": self.default_max_references,
                    "min_chunks_for_answer": self.min_chunks_for_answer
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {"error": str(e)}