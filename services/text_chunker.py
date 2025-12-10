"""
Text chunking service for the PDF Q&A System

This module provides semantic text chunking functionality with overlap strategy
and optimal chunk sizing for embedding generation and retrieval.
"""
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from models.document import Chunk, ChunkMetadata


@dataclass
class ChunkingConfig:
    """Configuration for text chunking parameters"""
    min_chunk_size: int = 100  # Minimum tokens per chunk
    max_chunk_size: int = 1000  # Maximum tokens per chunk
    target_chunk_size: int = 500  # Target tokens per chunk
    overlap_size: int = 50  # Overlap tokens between chunks
    sentence_boundary_preference: bool = True  # Prefer sentence boundaries
    paragraph_boundary_preference: bool = True  # Prefer paragraph boundaries


class TextChunkingError(Exception):
    """Exception raised when text chunking fails"""
    pass


class TextChunker:
    """
    Semantic text chunker that creates optimally-sized chunks with overlap
    for efficient embedding generation and retrieval.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the text chunker with configuration.
        
        Args:
            config: Chunking configuration parameters
        """
        self.config = config or ChunkingConfig()
        
        # Initialize tokenizer lazily to avoid import issues during testing
        self._tokenizer = None
        
        # Sentence and paragraph boundary patterns
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
    
    @property
    def tokenizer(self):
        """Lazy initialization of tokenizer"""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                )
            except Exception:
                # Fallback to None if tokenizer can't be loaded
                self._tokenizer = None
        return self._tokenizer
    
    def chunk_text(
        self, 
        text: str, 
        document_id: str,
        page_number: int = 1,
        section_title: Optional[str] = None,
        language: str = "unknown",
        confidence_score: float = 1.0
    ) -> List[Chunk]:
        """
        Chunk text into semantically meaningful segments with overlap.
        
        Args:
            text: The text content to chunk
            document_id: ID of the parent document
            page_number: Page number where this text appears
            section_title: Optional section title for context
            language: Language of the text content
            confidence_score: Confidence score for language detection
            
        Returns:
            List of Chunk objects with metadata
            
        Raises:
            TextChunkingError: If chunking fails
        """
        if not text or not text.strip():
            return []
        
        try:
            # Clean and normalize the text
            cleaned_text = self._clean_text(text)
            
            # Get token count for the entire text
            total_tokens = self._count_tokens(cleaned_text)
            
            # If text is smaller than target chunk size, return as single chunk
            if total_tokens <= self.config.target_chunk_size:
                return [self._create_chunk(
                    content=cleaned_text,
                    document_id=document_id,
                    chunk_index=0,
                    start_position=0,
                    end_position=len(cleaned_text),
                    page_number=page_number,
                    section_title=section_title,
                    language=language,
                    confidence_score=confidence_score
                )]
            
            # Perform semantic chunking with overlap
            chunks = self._semantic_chunk_with_overlap(
                text=cleaned_text,
                document_id=document_id,
                page_number=page_number,
                section_title=section_title,
                language=language,
                confidence_score=confidence_score
            )
            
            return chunks
            
        except Exception as e:
            raise TextChunkingError(f"Failed to chunk text: {str(e)}") from e
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for chunking.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace while preserving paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n[ \t]*\n', '\n\n', text)  # Clean paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the embedding model tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception:
                pass
        
        # Fallback to approximate word-based counting
        # Rough approximation: 1 token ≈ 0.75 words for multilingual text
        words = len(text.split())
        return int(words * 1.33)  # Convert words to approximate tokens
    
    def _semantic_chunk_with_overlap(
        self,
        text: str,
        document_id: str,
        page_number: int,
        section_title: Optional[str],
        language: str,
        confidence_score: float
    ) -> List[Chunk]:
        """
        Perform semantic chunking with overlap strategy.
        
        Args:
            text: Text to chunk
            document_id: Parent document ID
            page_number: Page number
            section_title: Section title
            language: Text language
            confidence_score: Language confidence
            
        Returns:
            List of chunks with overlap
        """
        chunks = []
        chunk_index = 0
        start_pos = 0
        
        while start_pos < len(text):
            # Find the end position for this chunk
            end_pos, actual_end_pos = self._find_chunk_boundary(
                text, start_pos, self.config.target_chunk_size
            )
            
            # Extract chunk content
            chunk_content = text[start_pos:end_pos].strip()
            
            if chunk_content:  # Only create non-empty chunks
                chunk = self._create_chunk(
                    content=chunk_content,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    start_position=start_pos,
                    end_position=end_pos,
                    page_number=page_number,
                    section_title=section_title,
                    language=language,
                    confidence_score=confidence_score
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Calculate next start position with overlap
            if actual_end_pos >= len(text):
                break
                
            # Find overlap start position
            overlap_start = self._find_overlap_position(
                text, end_pos, self.config.overlap_size
            )
            start_pos = overlap_start
        
        return chunks
    
    def _find_chunk_boundary(
        self, 
        text: str, 
        start_pos: int, 
        target_size: int
    ) -> Tuple[int, int]:
        """
        Find optimal chunk boundary respecting semantic boundaries.
        
        Args:
            text: Full text
            start_pos: Starting position
            target_size: Target chunk size in tokens
            
        Returns:
            Tuple of (chunk_end_position, actual_boundary_position)
        """
        # Get approximate character position for target token count
        remaining_text = text[start_pos:]
        
        # Estimate characters per token (rough approximation)
        chars_per_token = 4  # Average for multilingual text
        target_chars = target_size * chars_per_token
        
        # Don't exceed text length
        if target_chars >= len(remaining_text):
            return len(text), len(text)
        
        # Find the approximate end position
        approx_end = start_pos + target_chars
        
        # Ensure we don't exceed max chunk size in tokens
        max_chars = self.config.max_chunk_size * chars_per_token
        max_end = start_pos + max_chars
        
        if approx_end > max_end:
            approx_end = max_end
        
        # Find the best boundary within a reasonable range
        search_start = max(start_pos + self.config.min_chunk_size * chars_per_token, start_pos)
        search_end = min(approx_end + 200, len(text))  # Allow some flexibility
        
        # Look for paragraph boundary first (highest priority)
        if self.config.paragraph_boundary_preference:
            para_boundary = self._find_paragraph_boundary(text, search_start, search_end)
            if para_boundary != -1:
                return para_boundary, para_boundary
        
        # Look for sentence boundary (medium priority)
        if self.config.sentence_boundary_preference:
            sent_boundary = self._find_sentence_boundary(text, search_start, search_end)
            if sent_boundary != -1:
                return sent_boundary, sent_boundary
        
        # Fall back to word boundary (lowest priority)
        word_boundary = self._find_word_boundary(text, approx_end)
        return word_boundary, word_boundary
    
    def _find_paragraph_boundary(self, text: str, start: int, end: int) -> int:
        """Find paragraph boundary within range."""
        search_text = text[start:end]
        matches = list(self.paragraph_pattern.finditer(search_text))
        
        if matches:
            # Prefer boundaries closer to the target
            target_pos = (start + end) // 2
            best_match = min(matches, key=lambda m: abs((start + m.end()) - target_pos))
            return start + best_match.start()
        
        return -1
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find sentence boundary within range."""
        search_text = text[start:end]
        matches = list(self.sentence_pattern.finditer(search_text))
        
        if matches:
            # Prefer boundaries closer to the target
            target_pos = (start + end) // 2
            best_match = min(matches, key=lambda m: abs((start + m.end()) - target_pos))
            return start + best_match.end()
        
        return -1
    
    def _find_word_boundary(self, text: str, approx_pos: int) -> int:
        """Find word boundary near the approximate position."""
        if approx_pos >= len(text):
            return len(text)
        
        # Look for whitespace around the position
        for offset in range(50):  # Search within 50 characters
            # Try forward
            if approx_pos + offset < len(text) and text[approx_pos + offset].isspace():
                return approx_pos + offset
            
            # Try backward
            if approx_pos - offset > 0 and text[approx_pos - offset].isspace():
                return approx_pos - offset
        
        # If no whitespace found, use the approximate position
        return approx_pos
    
    def _find_overlap_position(self, text: str, end_pos: int, overlap_size: int) -> int:
        """
        Find the starting position for overlap with the next chunk.
        
        Args:
            text: Full text
            end_pos: End position of current chunk
            overlap_size: Desired overlap size in tokens
            
        Returns:
            Starting position for next chunk
        """
        # Estimate character position for overlap
        chars_per_token = 4
        overlap_chars = overlap_size * chars_per_token
        
        overlap_start = max(0, end_pos - overlap_chars)
        
        # Try to find a good boundary for overlap start
        # Look for sentence boundary within the overlap region
        search_start = max(0, end_pos - overlap_chars - 50)
        search_end = end_pos
        
        sent_boundary = self._find_sentence_boundary(text, search_start, search_end)
        if sent_boundary != -1 and sent_boundary > overlap_start:
            return sent_boundary
        
        # Fall back to word boundary
        word_boundary = self._find_word_boundary(text, overlap_start)
        return word_boundary
    
    def _create_chunk(
        self,
        content: str,
        document_id: str,
        chunk_index: int,
        start_position: int,
        end_position: int,
        page_number: int,
        section_title: Optional[str],
        language: str,
        confidence_score: float
    ) -> Chunk:
        """
        Create a Chunk object with metadata.
        
        Args:
            content: Chunk text content
            document_id: Parent document ID
            chunk_index: Index of this chunk
            start_position: Start position in original text
            end_position: End position in original text
            page_number: Page number
            section_title: Section title
            language: Text language
            confidence_score: Language confidence
            
        Returns:
            Chunk object with metadata
        """
        metadata = ChunkMetadata(
            page_number=page_number,
            section_title=section_title,
            chunk_index=chunk_index,
            language=language,
            confidence_score=confidence_score
        )
        
        return Chunk(
            document_id=document_id,
            content=content,
            metadata=metadata,
            start_position=start_position,
            end_position=end_position
        )
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> dict:
        """
        Get statistics about the generated chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size_chars": 0,
                "avg_chunk_size_tokens": 0,
                "min_chunk_size_chars": 0,
                "max_chunk_size_chars": 0,
                "total_characters": 0,
                "total_tokens": 0
            }
        
        chunk_sizes_chars = [len(chunk.content) for chunk in chunks]
        chunk_sizes_tokens = [self._count_tokens(chunk.content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size_chars": sum(chunk_sizes_chars) / len(chunks),
            "avg_chunk_size_tokens": sum(chunk_sizes_tokens) / len(chunks),
            "min_chunk_size_chars": min(chunk_sizes_chars),
            "max_chunk_size_chars": max(chunk_sizes_chars),
            "min_chunk_size_tokens": min(chunk_sizes_tokens),
            "max_chunk_size_tokens": max(chunk_sizes_tokens),
            "total_characters": sum(chunk_sizes_chars),
            "total_tokens": sum(chunk_sizes_tokens)
        }