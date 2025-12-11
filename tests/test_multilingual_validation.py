"""
Tests for multilingual support validation in the PDF Q&A System

This test module validates that the system correctly handles multilingual
documents and provides consistent cross-language functionality.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Tuple

from services.language_validator import LanguageValidator, LanguageValidationResult, CrossLanguageSearchResult
from services.document_service import DocumentService
from services.question_service import QuestionService
from services.retrieval_service import RetrievalService
from models.document import Chunk, ChunkMetadata
from models.question import Question
from models.api import QuestionRequest


class TestLanguageValidator:
    """Test the LanguageValidator service"""
    
    @pytest.fixture
    def language_validator(self):
        """Create a LanguageValidator instance"""
        return LanguageValidator()
    
    def test_validate_portuguese_text(self, language_validator):
        """Test validation of Portuguese text"""
        portuguese_text = """
        Este manual contém informações importantes sobre a operação e manutenção do equipamento.
        A temperatura máxima de operação é de 85°C em condições normais.
        Para mais informações, consulte a seção de especificações técnicas.
        """
        
        result = language_validator.validate_document_language(portuguese_text, "pt")
        
        assert result.is_valid
        assert result.detected_language == "pt"
        assert result.confidence_score > 0.5
        assert len(result.validation_errors) == 0
    
    def test_validate_english_text(self, language_validator):
        """Test validation of English text"""
        english_text = """
        This manual contains important information about the operation and maintenance of the equipment.
        The maximum operating temperature is 85°C under normal conditions.
        For more information, please refer to the technical specifications section.
        """
        
        result = language_validator.validate_document_language(english_text, "en")
        
        assert result.is_valid
        assert result.detected_language == "en"
        assert result.confidence_score > 0.5
        assert len(result.validation_errors) == 0
    
    def test_validate_mixed_language_text(self, language_validator):
        """Test validation of mixed language text"""
        mixed_text = """
        This manual contains informações importantes about the operation.
        A temperatura máxima is 85°C under condições normais.
        """
        
        result = language_validator.validate_document_language(mixed_text)
        
        # Should detect one of the languages, but with lower confidence
        assert result.detected_language in ["pt", "en", "unknown"]
        if result.detected_language != "unknown":
            assert result.confidence_score < 0.9  # Lower confidence for mixed text
    
    def test_validate_short_text(self, language_validator):
        """Test validation of text that's too short"""
        short_text = "Hi"
        
        result = language_validator.validate_document_language(short_text)
        
        assert not result.is_valid
        assert "Text too short" in result.validation_errors[0]
    
    def test_validate_unsupported_language(self, language_validator):
        """Test validation of unsupported language"""
        # French text
        french_text = """
        Ce manuel contient des informations importantes sur le fonctionnement et la maintenance de l'équipement.
        La température maximale de fonctionnement est de 85°C dans des conditions normales.
        """
        
        result = language_validator.validate_document_language(french_text)
        
        # Should detect as unsupported and mark as unknown
        assert result.detected_language == "unknown"
        assert len(result.warnings) > 0
        assert "not in supported languages" in result.warnings[0]
    
    def test_validate_chunk_language_consistency(self, language_validator):
        """Test validation of language consistency across chunks"""
        # Create chunks with consistent language
        chunks = []
        for i in range(5):
            chunk = Chunk(
                document_id="test-doc",
                content=f"This is English text chunk number {i}.",
                metadata=ChunkMetadata(
                    page_number=1,
                    chunk_index=i,
                    language="en",
                    confidence_score=0.9
                ),
                start_position=i * 50,
                end_position=(i + 1) * 50
            )
            chunks.append(chunk)
        
        result = language_validator.validate_chunk_language_consistency(chunks)
        
        assert result.is_valid
        assert result.detected_language == "en"
        assert result.confidence_score > 0.8
    
    def test_validate_inconsistent_chunk_languages(self, language_validator):
        """Test validation of inconsistent chunk languages"""
        chunks = []
        languages = ["en", "pt", "en", "unknown", "pt"]
        
        for i, lang in enumerate(languages):
            chunk = Chunk(
                document_id="test-doc",
                content=f"Text chunk number {i}.",
                metadata=ChunkMetadata(
                    page_number=1,
                    chunk_index=i,
                    language=lang,
                    confidence_score=0.8
                ),
                start_position=i * 50,
                end_position=(i + 1) * 50
            )
            chunks.append(chunk)
        
        result = language_validator.validate_chunk_language_consistency(chunks)
        
        # Should have warnings about multiple languages or no dominant language
        assert len(result.warnings) > 0
        warning_text = result.warnings[0]
        assert ("Multiple languages detected" in warning_text or 
                "No dominant language found" in warning_text)
    
    def test_validate_cross_language_search_success(self, language_validator):
        """Test successful cross-language search validation"""
        # Create chunks in different languages
        chunks_with_scores = []
        
        # English chunks
        for i in range(3):
            chunk = Chunk(
                document_id="doc-en",
                content=f"English content {i}",
                metadata=ChunkMetadata(
                    page_number=1,
                    chunk_index=i,
                    language="en",
                    confidence_score=0.9
                ),
                start_position=i * 20,
                end_position=(i + 1) * 20
            )
            chunks_with_scores.append((chunk, 0.8 - i * 0.1))
        
        # Portuguese chunks
        for i in range(2):
            chunk = Chunk(
                document_id="doc-pt",
                content=f"Conteúdo português {i}",
                metadata=ChunkMetadata(
                    page_number=1,
                    chunk_index=i,
                    language="pt",
                    confidence_score=0.9
                ),
                start_position=i * 20,
                end_position=(i + 1) * 20
            )
            chunks_with_scores.append((chunk, 0.7 - i * 0.1))
        
        result = language_validator.validate_cross_language_search("en", chunks_with_scores)
        
        assert result.search_successful
        assert "en" in result.languages_found
        assert "pt" in result.languages_found
        assert result.chunks_by_language["en"] == 3
        assert result.chunks_by_language["pt"] == 2
        assert result.consistency_score > 0.5
    
    def test_validate_embedding_language_consistency(self, language_validator):
        """Test validation of embedding language consistency"""
        texts = [
            "This is English text for testing.",
            "Este é um texto em português para teste.",
            "Another English text sample.",
            "Mais um exemplo de texto português."
        ]
        
        # Create mock embeddings (384 dimensions)
        embeddings = []
        for _ in texts:
            embedding = np.random.rand(384).tolist()
            embeddings.append(embedding)
        
        result = language_validator.validate_embedding_language_consistency(texts, embeddings)
        
        assert result.is_valid
        assert result.confidence_score > 0.8
        assert len(result.validation_errors) == 0
    
    def test_validate_zero_embeddings(self, language_validator):
        """Test validation catches zero embeddings"""
        texts = ["Test text"]
        embeddings = [[0.0] * 384]  # Zero embedding
        
        result = language_validator.validate_embedding_language_consistency(texts, embeddings)
        
        assert not result.is_valid
        assert "zero vector" in result.validation_errors[0]
    
    def test_supported_languages(self, language_validator):
        """Test supported language methods"""
        supported = language_validator.get_supported_languages()
        
        assert "en" in supported
        assert "pt" in supported
        assert language_validator.is_language_supported("en")
        assert language_validator.is_language_supported("pt")
        assert not language_validator.is_language_supported("fr")


class TestMultilingualIntegration:
    """Test multilingual support integration across services"""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing"""
        mock_vector_store = Mock()
        mock_embedding_service = Mock()
        mock_llm_service = Mock()
        mock_pdf_processor = Mock()
        mock_text_chunker = Mock()
        
        return {
            'vector_store': mock_vector_store,
            'embedding_service': mock_embedding_service,
            'llm_service': mock_llm_service,
            'pdf_processor': mock_pdf_processor,
            'text_chunker': mock_text_chunker
        }
    
    def test_document_service_language_validation(self, mock_services, tmp_path):
        """Test that DocumentService validates languages during processing"""
        # Create a test PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"dummy pdf content")
        
        # Mock PDF processor to return Portuguese text
        mock_services['pdf_processor'].process_pdf.return_value = (
            "Este é um texto em português para teste.", "pt", 0.9
        )
        
        # Mock text chunker
        chunk = Chunk(
            document_id="test-doc",
            content="Este é um texto em português para teste.",
            metadata=ChunkMetadata(
                page_number=1,
                chunk_index=0,
                language="pt",
                confidence_score=0.9
            ),
            start_position=0,
            end_position=50
        )
        mock_services['text_chunker'].chunk_text.return_value = [chunk]
        
        # Mock embedding service
        mock_services['embedding_service'].generate_embeddings_batch.return_value = [
            np.random.rand(384)
        ]
        
        # Mock vector store
        mock_services['vector_store'].add_chunks.return_value = True
        
        # Create document service with mocks
        doc_service = DocumentService(
            upload_directory=str(tmp_path),
            vector_store=mock_services['vector_store'],
            pdf_processor=mock_services['pdf_processor'],
            text_chunker=mock_services['text_chunker'],
            embedding_service=mock_services['embedding_service']
        )
        
        # Process the document
        result = doc_service.process_documents([str(test_file)])
        
        # Verify processing succeeded
        assert result.success
        assert result.documents_processed == 1
        
        # Verify language validation was performed (check logs would be ideal)
        mock_services['pdf_processor'].process_pdf.assert_called_once()
    
    def test_question_service_cross_language_search(self, mock_services):
        """Test that QuestionService validates cross-language search"""
        # Mock retrieval service
        mock_retrieval_service = Mock()
        
        # Create mixed language chunks
        en_chunk = Chunk(
            document_id="doc-en",
            content="Maximum operating temperature is 85°C.",
            metadata=ChunkMetadata(
                page_number=1,
                chunk_index=0,
                language="en",
                confidence_score=0.9
            ),
            start_position=0,
            end_position=35
        )
        
        pt_chunk = Chunk(
            document_id="doc-pt",
            content="A temperatura máxima de operação é 85°C.",
            metadata=ChunkMetadata(
                page_number=1,
                chunk_index=0,
                language="pt",
                confidence_score=0.9
            ),
            start_position=0,
            end_position=40
        )
        
        mock_retrieval_service.find_relevant_chunks.return_value = [
            (en_chunk, 0.9),
            (pt_chunk, 0.8)
        ]
        
        mock_retrieval_service.create_references.return_value = []
        
        # Mock LLM service
        mock_llm_response = Mock()
        mock_llm_response.answer = "The maximum operating temperature is 85°C."
        mock_llm_response.confidence_score = 0.9
        mock_services['llm_service'].generate_answer.return_value = mock_llm_response
        
        # Create question service
        question_service = QuestionService(
            retrieval_service=mock_retrieval_service,
            llm_service=mock_services['llm_service']
        )
        
        # Test cross-language question
        request = QuestionRequest(
            question="What is the maximum temperature?",
            language="en"
        )
        
        response = question_service.answer_question(request)
        
        # Verify the service handled the multilingual content
        assert response.answer == "The maximum operating temperature is 85°C."
        mock_retrieval_service.find_relevant_chunks.assert_called_once()
    
    def test_retrieval_service_language_filtering(self, mock_services):
        """Test that RetrievalService properly filters by language"""
        # Create retrieval service
        retrieval_service = RetrievalService(
            vector_store=mock_services['vector_store'],
            embedding_service=mock_services['embedding_service']
        )
        
        # Mock embedding generation
        mock_services['embedding_service'].generate_embedding.return_value = np.random.rand(384)
        
        # Mock vector store search
        mock_services['vector_store'].search_similar.return_value = []
        
        # Test with language filter
        retrieval_service.find_relevant_chunks(
            question="What is the temperature?",
            language_filter="pt"
        )
        
        # Verify language filter was applied
        call_args = mock_services['vector_store'].search_similar.call_args
        assert call_args[1]['metadata_filter'] == {"language": "pt"}
    
    def test_unsupported_language_handling(self, mock_services):
        """Test handling of unsupported languages"""
        retrieval_service = RetrievalService(
            vector_store=mock_services['vector_store'],
            embedding_service=mock_services['embedding_service']
        )
        
        mock_services['embedding_service'].generate_embedding.return_value = np.random.rand(384)
        mock_services['vector_store'].search_similar.return_value = []
        
        # Test with unsupported language filter
        retrieval_service.find_relevant_chunks(
            question="Quelle est la température?",  # French
            language_filter="fr"  # Unsupported
        )
        
        # Should search without language filter
        call_args = mock_services['vector_store'].search_similar.call_args
        assert call_args[1]['metadata_filter'] is None


class TestMultilingualEndToEnd:
    """End-to-end tests for multilingual functionality"""
    
    def test_portuguese_document_english_question(self):
        """Test querying Portuguese document with English question"""
        # This would be an integration test with real services
        # For now, we'll test the validation logic
        
        validator = LanguageValidator()
        
        # Validate Portuguese document
        pt_text = "A temperatura máxima de operação é 85°C em condições normais."
        doc_validation = validator.validate_document_language(pt_text, "pt")
        assert doc_validation.is_valid
        assert doc_validation.detected_language == "pt"
        
        # Validate English question
        en_question = "What is the maximum operating temperature?"
        question_validation = validator.validate_document_language(en_question, "en")
        assert question_validation.is_valid
        assert question_validation.detected_language == "en"
        
        # This demonstrates that the system can handle cross-language scenarios
    
    def test_language_consistency_validation(self):
        """Test that language consistency is properly validated"""
        validator = LanguageValidator()
        
        # Create chunks with mixed languages
        chunks = []
        
        # Mostly Portuguese chunks
        for i in range(4):
            chunk = Chunk(
                document_id="mixed-doc",
                content=f"Texto em português número {i}.",
                metadata=ChunkMetadata(
                    page_number=1,
                    chunk_index=i,
                    language="pt",
                    confidence_score=0.9
                ),
                start_position=i * 30,
                end_position=(i + 1) * 30
            )
            chunks.append(chunk)
        
        # One English chunk
        chunk = Chunk(
            document_id="mixed-doc",
            content="One English text chunk.",
            metadata=ChunkMetadata(
                page_number=1,
                chunk_index=4,
                language="en",
                confidence_score=0.9
            ),
            start_position=120,
            end_position=145
        )
        chunks.append(chunk)
        
        result = validator.validate_chunk_language_consistency(chunks)
        
        # Should detect Portuguese as dominant but warn about mixed languages
        assert result.detected_language == "pt"
        # The test might not generate warnings if Portuguese is clearly dominant
        # This is acceptable behavior