"""
LLM integration service for the PDF Q&A System using OpenRouter
"""
import logging
import time
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests
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
    """Service for LLM integration with OpenRouter API"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM service
        
        Args:
            api_key: OpenRouter API key (if None, will use settings.openrouter_api_key)
            model: Model to use (if None, will use settings.llm_model)
        """
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.llm_model
        self.fallback_model = settings.llm_fallback_model
        self.max_context_tokens = settings.max_context_tokens
        self.max_total_tokens = settings.max_total_tokens
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout = 30.0
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the OpenRouter client"""
        try:
            if self.api_key:
                logger.info(f"OpenRouter client initialized with model: {self.model}")
            else:
                logger.warning("No OpenRouter API key provided, client not initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
    
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
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=60, expected_exception=Exception)
    @RetryHandler(max_retries=2, retryable_exceptions=(requests.exceptions.Timeout, requests.exceptions.ConnectionError))
    def _make_api_call(self, prompt: str, model: str) -> Tuple[str, int]:
        """
        Make API call to OpenRouter with circuit breaker and retry logic
        
        Args:
            prompt: The formatted prompt
            model: Model to use
            
        Returns:
            Tuple of (response_text, tokens_used)
        """
        if not self.api_key:
            raise LLMServiceError(
                message="OpenRouter API key not configured",
                error_code=ErrorCode.LLM_SERVICE_UNAVAILABLE
            )
        
        try:
            start_time = time.time()
            
            # Prepare the request payload
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": PromptTemplate.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.1,  # Low temperature for more consistent answers
            }
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/pdf-qa-system",  # Optional: for OpenRouter analytics
                "X-Title": "PDF Q&A System"  # Optional: for OpenRouter analytics
            }
            
            # Make the API call
            response = requests.post(
                url=self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            # Log performance
            duration_ms = int((time.time() - start_time) * 1000)
            log_performance_metric("llm_api_call", duration_ms, {"model": model})
            
            # Handle response
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data["choices"][0]["message"]["content"]
                tokens_used = response_data.get("usage", {}).get("total_tokens", 0)
                return answer, tokens_used
            else:
                # Handle error responses
                error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
                
                if response.status_code == 429:
                    logger.error(f"Rate limit exceeded: {error_message}")
                    raise LLMServiceError(
                        message="Rate limit exceeded for LLM service. Please try again later.",
                        model_name=model,
                        error_code=ErrorCode.LLM_RATE_LIMIT
                    )
                elif response.status_code == 401:
                    logger.error(f"Authentication error: {error_message}")
                    raise LLMServiceError(
                        message="LLM service authentication failed. Please check OpenRouter API key configuration.",
                        model_name=model,
                        error_code=ErrorCode.LLM_SERVICE_UNAVAILABLE
                    )
                else:
                    logger.error(f"OpenRouter API error: {error_message}")
                    raise LLMServiceError(
                        message=f"LLM service API error: {error_message}",
                        model_name=model,
                        error_code=ErrorCode.LLM_API_ERROR
                    )
            
        except requests.exceptions.Timeout as e:
            logger.error(f"API timeout: {e}")
            raise LLMServiceError(
                message="LLM service request timed out. Please try again.",
                model_name=model,
                error_code=ErrorCode.LLM_TIMEOUT,
                original_exception=e
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise LLMServiceError(
                message="Failed to connect to LLM service. Please check your internet connection.",
                model_name=model,
                error_code=ErrorCode.LLM_SERVICE_UNAVAILABLE,
                original_exception=e
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise LLMServiceError(
                message="Invalid response from LLM service.",
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
        
        if not self.api_key:
            raise LLMServiceError(
                message="LLM service not properly initialized - missing OpenRouter API key",
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
        return self.api_key is not None
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured models"""
        return {
            "primary_model": self.model,
            "fallback_model": self.fallback_model,
            "available": str(self.is_available())
        }