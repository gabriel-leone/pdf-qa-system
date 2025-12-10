"""
LLM integration service for the PDF Q&A System
"""
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import openai
from openai import OpenAI
from config import settings
from utils.exceptions import LLMServiceError, ErrorCode, ServiceUnavailableError
from utils.error_handlers import CircuitBreaker, RetryHandler, log_performance_metric

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM service"""
    answer: str
    tokens_used: int
    processing_time_ms: int
    model_used: str
    confidence_score: float = 0.0


@dataclass
class ContextChunk:
    """Context chunk for LLM processing"""
    content: str
    source: str
    page_number: int
    relevance_score: float


class TokenCounter:
    """Utility class for counting tokens"""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count using a simple heuristic
        OpenAI's tiktoken would be more accurate but this provides a reasonable approximation
        """
        # Rough approximation: 1 token ≈ 4 characters for English text
        # For multilingual content, we use a slightly more conservative estimate
        return len(text) // 3
    
    @staticmethod
    def truncate_to_token_limit(text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        if TokenCounter.estimate_tokens(text) <= max_tokens:
            return text
        
        # Binary search to find the right length
        left, right = 0, len(text)
        while left < right:
            mid = (left + right + 1) // 2
            if TokenCounter.estimate_tokens(text[:mid]) <= max_tokens:
                left = mid
            else:
                right = mid - 1
        
        return text[:left]


class PromptTemplate:
    """Template for generating prompts"""
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided document context. 
Follow these guidelines:
1. Answer only based on the provided context
2. If the context doesn't contain relevant information, say so clearly
3. Maintain the same language as the question (Portuguese or English)
4. Provide specific references to the source material when possible
5. Be concise but comprehensive in your answers"""

    QUESTION_TEMPLATE = """Context from documents:
{context}

Question: {question}

Please provide a clear and accurate answer based on the context above. If the context doesn't contain enough information to answer the question, please state that clearly."""

    NO_CONTEXT_TEMPLATE = """I don't have any relevant information in the uploaded documents to answer your question: "{question}". 
Please make sure you have uploaded documents that contain information related to your question."""

    @classmethod
    def format_context(cls, chunks: List[ContextChunk]) -> str:
        """Format context chunks into a readable string"""
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_part = f"[Source {i}: {chunk.source}, Page {chunk.page_number}]\n{chunk.content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    @classmethod
    def create_prompt(cls, question: str, chunks: List[ContextChunk]) -> str:
        """Create a complete prompt for the LLM"""
        if not chunks:
            return cls.NO_CONTEXT_TEMPLATE.format(question=question)
        
        context = cls.format_context(chunks)
        return cls.QUESTION_TEMPLATE.format(context=context, question=question)


class LLMService:
    """Service for LLM integration with OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM service
        
        Args:
            api_key: OpenAI API key (if None, will use settings.openai_api_key)
            model: OpenAI model to use
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model
        self.fallback_model = "gpt-3.5-turbo"  # Fallback if primary model fails
        self.max_context_tokens = 3000  # Conservative limit for context
        self.max_total_tokens = 4000   # Total token limit including response
        self.client: Optional[OpenAI] = None
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the OpenAI client"""
        try:
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized with model: {self.model}")
            else:
                self.client = None
                logger.warning("No API key provided, client not initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def _select_context_chunks(self, chunks: List[ContextChunk], question: str) -> List[ContextChunk]:
        """
        Select and truncate context chunks to fit within token limits
        
        Args:
            chunks: Available context chunks
            question: The user's question
            
        Returns:
            Selected chunks that fit within token limits
        """
        if not chunks:
            return []
        
        # Reserve tokens for question, system prompt, and response
        question_tokens = TokenCounter.estimate_tokens(question)
        system_tokens = TokenCounter.estimate_tokens(PromptTemplate.SYSTEM_PROMPT)
        template_tokens = TokenCounter.estimate_tokens(PromptTemplate.QUESTION_TEMPLATE) - 20  # Subtract placeholders
        
        available_tokens = self.max_context_tokens - question_tokens - system_tokens - template_tokens
        
        if available_tokens <= 0:
            logger.warning("Question too long, no tokens available for context")
            return []
        
        # Sort chunks by relevance score (highest first)
        sorted_chunks = sorted(chunks, key=lambda x: x.relevance_score, reverse=True)
        
        selected_chunks = []
        used_tokens = 0
        
        for chunk in sorted_chunks:
            chunk_tokens = TokenCounter.estimate_tokens(chunk.content)
            
            if used_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                used_tokens += chunk_tokens
            else:
                # Try to fit a truncated version of the chunk
                remaining_tokens = available_tokens - used_tokens
                if remaining_tokens > 100:  # Only if we have reasonable space left
                    truncated_content = TokenCounter.truncate_to_token_limit(
                        chunk.content, remaining_tokens
                    )
                    if truncated_content:
                        truncated_chunk = ContextChunk(
                            content=truncated_content,
                            source=chunk.source,
                            page_number=chunk.page_number,
                            relevance_score=chunk.relevance_score
                        )
                        selected_chunks.append(truncated_chunk)
                break
        
        logger.info(f"Selected {len(selected_chunks)} chunks using ~{used_tokens} tokens")
        return selected_chunks
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=60, expected_exception=openai.OpenAIError)
    @RetryHandler(max_retries=2, retryable_exceptions=(openai.APITimeoutError, openai.APIConnectionError))
    def _make_api_call(self, prompt: str, model: str) -> Tuple[str, int]:
        """
        Make API call to OpenAI with circuit breaker and retry logic
        
        Args:
            prompt: The formatted prompt
            model: Model to use
            
        Returns:
            Tuple of (response_text, tokens_used)
        """
        if not self.client:
            raise LLMServiceError(
                message="OpenAI client not initialized - missing API key",
                error_code=ErrorCode.LLM_SERVICE_UNAVAILABLE
            )
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PromptTemplate.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1,  # Low temperature for more consistent answers
                timeout=30.0
            )
            
            # Log performance
            duration_ms = int((time.time() - start_time) * 1000)
            log_performance_metric("llm_api_call", duration_ms, {"model": model})
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return answer, tokens_used
            
        except openai.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise LLMServiceError(
                message="Rate limit exceeded for LLM service. Please try again later.",
                model_name=model,
                error_code=ErrorCode.LLM_RATE_LIMIT,
                original_exception=e
            )
        except openai.APITimeoutError as e:
            logger.error(f"API timeout: {e}")
            raise LLMServiceError(
                message="LLM service request timed out. Please try again.",
                model_name=model,
                error_code=ErrorCode.LLM_TIMEOUT,
                original_exception=e
            )
        except openai.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise LLMServiceError(
                message="LLM service authentication failed. Please check API key configuration.",
                model_name=model,
                error_code=ErrorCode.LLM_SERVICE_UNAVAILABLE,
                original_exception=e
            )
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMServiceError(
                message=f"LLM service API error: {str(e)}",
                model_name=model,
                error_code=ErrorCode.LLM_API_ERROR,
                original_exception=e
            )
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            raise LLMServiceError(
                message=f"Unexpected error in LLM service: {str(e)}",
                model_name=model,
                error_code=ErrorCode.LLM_API_ERROR,
                original_exception=e
            )
    
    def generate_answer(self, question: str, context_chunks: List[ContextChunk]) -> LLMResponse:
        """
        Generate an answer to a question using provided context
        
        Args:
            question: The user's question
            context_chunks: Relevant context chunks from documents
            
        Returns:
            LLMResponse containing the generated answer and metadata
        """
        start_time = time.time()
        
        if not self.client:
            raise LLMServiceError(
                message="LLM service not properly initialized - missing API key",
                error_code=ErrorCode.LLM_SERVICE_UNAVAILABLE
            )
        
        if not question or not question.strip():
            raise LLMServiceError(
                message="Question cannot be empty",
                error_code=ErrorCode.INVALID_QUESTION
            )
        
        # Select appropriate context chunks within token limits
        selected_chunks = self._select_context_chunks(context_chunks, question)
        
        # Create the prompt
        prompt = PromptTemplate.create_prompt(question, selected_chunks)
        
        # Try primary model first, then fallback
        models_to_try = [self.model]
        if self.fallback_model != self.model:
            models_to_try.append(self.fallback_model)
        
        last_error = None
        for model in models_to_try:
            try:
                logger.info(f"Attempting to generate answer using model: {model}")
                answer, tokens_used = self._make_api_call(prompt, model)
                
                processing_time = int((time.time() - start_time) * 1000)
                
                # Calculate confidence score based on context availability
                confidence_score = min(1.0, len(selected_chunks) * 0.3) if selected_chunks else 0.1
                
                logger.info(f"Successfully generated answer using {model} in {processing_time}ms")
                
                return LLMResponse(
                    answer=answer,
                    tokens_used=tokens_used,
                    processing_time_ms=processing_time,
                    model_used=model,
                    confidence_score=confidence_score
                )
                
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to generate answer with {model}: {e}")
                continue
        
        # If all models failed, raise the last error
        logger.error(f"All LLM models failed. Last error: {last_error}")
        raise last_error
    
    def generate_simple_answer(self, question: str) -> LLMResponse:
        """
        Generate an answer without context (fallback mode)
        
        Args:
            question: The user's question
            
        Returns:
            LLMResponse with a no-context answer
        """
        no_context_answer = PromptTemplate.NO_CONTEXT_TEMPLATE.format(question=question)
        
        return LLMResponse(
            answer=no_context_answer,
            tokens_used=0,
            processing_time_ms=0,
            model_used="fallback",
            confidence_score=0.0
        )
    
    def is_available(self) -> bool:
        """Check if the LLM service is available"""
        return self.client is not None and self.api_key is not None
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured models"""
        return {
            "primary_model": self.model,
            "fallback_model": self.fallback_model,
            "available": str(self.is_available())
        }