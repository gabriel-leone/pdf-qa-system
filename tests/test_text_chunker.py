"""
Tests for the TextChunker service
"""
import pytest
from services.text_chunker import TextChunker, ChunkingConfig, TextChunkingError


class TestTextChunker:
    """Test the TextChunker functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.chunker = TextChunker()
        # Create a longer text that will definitely require multiple chunks
        self.sample_text = """
        This is the first paragraph of a sample document. It contains multiple sentences 
        that should be processed correctly by the chunking system. The text is in English 
        and should be detected as such. This paragraph needs to be long enough to trigger 
        the chunking mechanism properly. We need to add more content here to ensure that 
        the total token count exceeds the target chunk size of 500 tokens.
        
        This is the second paragraph. It continues the document with more content that 
        will help test the chunking boundaries and overlap functionality. The chunker 
        should respect paragraph boundaries when possible. This paragraph also needs to 
        be substantial in length to contribute to the overall token count. We want to 
        make sure that the text processing system works correctly with longer documents.
        
        Here is a third paragraph with even more content to ensure we have enough text 
        to create multiple chunks. This will help us test the overlap strategy and 
        verify that chunks are created with appropriate sizes and metadata. The chunking 
        system should handle this text efficiently and create well-structured chunks 
        that maintain semantic coherence while respecting the configured size limits.
        
        This is a fourth paragraph that adds even more content to our test document. 
        The purpose of this additional content is to ensure that we have sufficient 
        text to trigger the multi-chunk processing logic. Each paragraph should contribute 
        to the overall structure and help validate that the chunking system works 
        correctly with documents of various sizes and complexities.
        
        Finally, this fifth paragraph completes our comprehensive test document. It 
        provides the final portion of content needed to thoroughly test the text 
        chunking functionality. The system should be able to process this entire 
        document and create multiple chunks with appropriate overlap and metadata 
        tracking throughout the entire process.
        """ * 2  # Double the content to ensure it's long enough
    
    def test_chunker_initialization(self):
        """Test TextChunker initialization"""
        chunker = TextChunker()
        assert chunker.config is not None
        assert chunker.config.target_chunk_size == 500
        assert chunker.config.overlap_size == 50
        
        # Test with custom config
        config = ChunkingConfig(target_chunk_size=300, overlap_size=30)
        custom_chunker = TextChunker(config)
        assert custom_chunker.config.target_chunk_size == 300
        assert custom_chunker.config.overlap_size == 30
    
    def test_chunk_empty_text(self):
        """Test chunking empty or whitespace-only text"""
        # Empty string
        chunks = self.chunker.chunk_text("", "doc-123")
        assert len(chunks) == 0
        
        # Whitespace only
        chunks = self.chunker.chunk_text("   \n\t  ", "doc-123")
        assert len(chunks) == 0
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than target chunk size"""
        short_text = "This is a short text that should fit in one chunk."
        chunks = self.chunker.chunk_text(short_text, "doc-123")
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.document_id == "doc-123"
        assert chunk.content.strip() == short_text.strip()
        assert chunk.metadata.chunk_index == 0
        assert chunk.start_position == 0
        assert chunk.end_position == len(short_text.strip())
    
    def test_chunk_long_text(self):
        """Test chunking longer text that requires multiple chunks"""
        chunks = self.chunker.chunk_text(self.sample_text, "doc-456", page_number=2)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Verify chunk properties
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == "doc-456"
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.page_number == 2
            assert len(chunk.content.strip()) > 0
            assert chunk.start_position < chunk.end_position
    
    def test_chunk_metadata(self):
        """Test chunk metadata is properly set"""
        chunks = self.chunker.chunk_text(
            self.sample_text,
            "doc-789",
            page_number=3,
            section_title="Test Section",
            language="en",
            confidence_score=0.95
        )
        
        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.metadata.page_number == 3
        assert chunk.metadata.section_title == "Test Section"
        assert chunk.metadata.language == "en"
        assert chunk.metadata.confidence_score == 0.95
    
    def test_chunk_statistics(self):
        """Test chunk statistics calculation"""
        chunks = self.chunker.chunk_text(self.sample_text, "doc-stats")
        stats = self.chunker.get_chunk_statistics(chunks)
        
        assert stats["total_chunks"] == len(chunks)
        assert stats["avg_chunk_size_chars"] > 0
        assert stats["avg_chunk_size_tokens"] > 0
        assert stats["min_chunk_size_chars"] > 0
        assert stats["max_chunk_size_chars"] > 0
        assert stats["total_characters"] > 0
        assert stats["total_tokens"] > 0
        
        # Test empty chunks
        empty_stats = self.chunker.get_chunk_statistics([])
        assert empty_stats["total_chunks"] == 0
        assert empty_stats["avg_chunk_size_chars"] == 0
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        messy_text = "  This   has    excessive   whitespace.  \n\n\n\n  And multiple newlines.  "
        chunks = self.chunker.chunk_text(messy_text, "doc-clean")
        
        assert len(chunks) == 1
        cleaned_content = chunks[0].content
        
        # Should not have excessive whitespace
        assert "   " not in cleaned_content  # No triple spaces
        assert not cleaned_content.startswith(" ")  # No leading space
        assert not cleaned_content.endswith(" ")  # No trailing space
    
    def test_chunking_config_validation(self):
        """Test different chunking configurations"""
        # Small chunks
        small_config = ChunkingConfig(
            target_chunk_size=100,
            max_chunk_size=150,
            overlap_size=20
        )
        small_chunker = TextChunker(small_config)
        small_chunks = small_chunker.chunk_text(self.sample_text, "doc-small")
        
        # Large chunks  
        large_config = ChunkingConfig(
            target_chunk_size=800,
            max_chunk_size=1000,
            overlap_size=100
        )
        large_chunker = TextChunker(large_config)
        large_chunks = large_chunker.chunk_text(self.sample_text, "doc-large")
        
        # Small config should create more chunks
        assert len(small_chunks) >= len(large_chunks)
    
    def test_portuguese_text_chunking(self):
        """Test chunking Portuguese text"""
        portuguese_text = """
        Este é o primeiro parágrafo de um documento de exemplo em português. 
        Contém múltiplas frases que devem ser processadas corretamente pelo 
        sistema de chunking. O texto está em português e deve ser detectado como tal.
        
        Este é o segundo parágrafo. Continua o documento com mais conteúdo que 
        ajudará a testar os limites de chunking e funcionalidade de sobreposição.
        """
        
        chunks = self.chunker.chunk_text(
            portuguese_text, 
            "doc-pt", 
            language="pt",
            confidence_score=0.9
        )
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.language == "pt"
            assert chunk.metadata.confidence_score == 0.9
            assert len(chunk.content.strip()) > 0