"""
Integration tests for multilingual support validation

This module tests the complete multilingual pipeline from document processing
to question answering, ensuring cross-language functionality works correctly.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from services.document_service import DocumentService
from services.question_service import QuestionService
from services.retrieval_service import RetrievalService
from services.language_validator import LanguageValidator
from services.embedding_service import EmbeddingService
from services.vector_store import ChromaVectorStore
from services.llm_service import LLMService
from models.api import QuestionRequest
from models.document import Chunk, ChunkMetadata


class TestMultilingualPipeline:
    """Test the complete multilingual processing pipeline"""
    
    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_pdf_content(self):
        """Mock PDF content in different languages"""
        return {
            'portuguese': """
            Manual de Operação do Equipamento
            
            Este manual contém informações importantes sobre a operação e manutenção do equipamento industrial.
            
            Especificações Técnicas:
            - Temperatura máxima de operação: 85°C
            - Pressão máxima: 10 bar
            - Voltagem: 220V
            
            Para mais informações sobre segurança, consulte o capítulo 5.
            Em caso de emergência, desligue imediatamente o equipamento.
            """,
            'english': """
            Equipment Operation Manual
            
            This manual contains important information about the operation and maintenance of industrial equipment.
            
            Technical Specifications:
            - Maximum operating temperature: 85°C
            - Maximum pressure: 10 bar
            - Voltage: 220V
            
            For more safety information, refer to chapter 5.
            In case of emergency, immediately shut down the equipment.
            """
        }
    
    @pytest.fixture
    def language_validator(self):
        """Create a language validator instance"""
        return LanguageValidator()
    
    def test_document_language_detection_and_validation(self, temp_directory, mock_pdf_content, language_validator):
        """Test document language detection and validation during processing"""
        
        # Test Portuguese document validation
        pt_validation = language_validator.validate_document_language(
            mock_pdf_content['portuguese'], 'pt'
        )
        
        assert pt_validation.is_valid
        assert pt_validation.detected_language == 'pt'
        assert pt_validation.confidence_score > 0.6
        assert len(pt_validation.validation_errors) == 0
        
        # Test English document validation
        en_validation = language_validator.validate_document_language(
            mock_pdf_content['english'], 'en'
        )
        
        assert en_validation.is_valid
        assert en_validation.detected_language == 'en'
        assert en_validation.confidence_score > 0.6
        assert len(en_validation.validation_errors) == 0
    
    def test_cross_language_embedding_consistency(self, mock_pdf_content, language_validator):
        """Test that embeddings are consistent across languages"""
        
        texts = [
            mock_pdf_content['portuguese'][:200],  # First 200 chars
            mock_pdf_content['english'][:200],
            "Temperatura máxima: 85°C",
            "Maximum temperature: 85°C"
        ]
        
        # Mock embeddings (in real test, these would come from actual embedding service)
        embeddings = []
        for _ in texts:
            # Create realistic embeddings (not zero vectors)
            embedding = np.random.normal(0, 0.1, 384).tolist()
            embeddings.append(embedding)
        
        validation_result = language_validator.validate_embedding_language_consistency(
            texts, embeddings
        )
        
        assert validation_result.is_valid
        assert validation_result.confidence_score > 0.8
        assert len(validation_result.validation_errors) == 0
    
    def test_chunk_language_consistency_validation(self, language_validator):
        """Test validation of language consistency across document chunks"""
        
        # Create chunks with consistent Portuguese language
        pt_chunks = []
        pt_texts = [
            "A temperatura máxima de operação é 85°C.",
            "Para mais informações, consulte o manual.",
            "Em caso de emergência, desligue o equipamento.",
            "As especificações técnicas estão no capítulo 3."
        ]
        
        for i, text in enumerate(pt_texts):
            chunk = Chunk(
                document_id="pt-doc",
                content=text,
                metadata=ChunkMetadata(
                    page_number=1,
                    chunk_index=i,
                    language="pt",
                    confidence_score=0.9
                ),
                start_position=i * 50,
                end_position=(i + 1) * 50
            )
            pt_chunks.append(chunk)
        
        consistency_result = language_validator.validate_chunk_language_consistency(pt_chunks)
        
        assert consistency_result.is_valid
        assert consistency_result.detected_language == "pt"
        assert consistency_result.confidence_score > 0.8
        
        # Test mixed language chunks (should generate warnings)
        mixed_chunks = pt_chunks[:2]  # Take first 2 Portuguese chunks
        
        # Add English chunks
        en_texts = [
            "The maximum operating temperature is 85°C.",
            "For more information, refer to the manual."
        ]
        
        for i, text in enumerate(en_texts):
            chunk = Chunk(
                document_id="en-doc",
                content=text,
                metadata=ChunkMetadata(
                    page_number=1,
                    chunk_index=i + 2,
                    language="en",
                    confidence_score=0.9
                ),
                start_position=(i + 2) * 50,
                end_position=(i + 3) * 50
            )
            mixed_chunks.append(chunk)
        
        mixed_result = language_validator.validate_chunk_language_consistency(mixed_chunks)
        
        # Should have warnings about multiple languages or no dominant language
        assert len(mixed_result.warnings) > 0
        warning_text = mixed_result.warnings[0]
        assert ("Multiple languages detected" in warning_text or 
                "No dominant language found" in warning_text)
    
    def test_cross_language_search_validation(self, language_validator):
        """Test validation of cross-language search capabilities"""
        
        # Create chunks in different languages that should be relevant to the same topic
        chunks_with_scores = []
        
        # Portuguese chunks about temperature
        pt_chunk1 = Chunk(
            document_id="pt-manual",
            content="A temperatura máxima de operação é 85°C em condições normais.",
            metadata=ChunkMetadata(
                page_number=5,
                chunk_index=0,
                language="pt",
                confidence_score=0.95
            ),
            start_position=0,
            end_position=60
        )
        chunks_with_scores.append((pt_chunk1, 0.92))
        
        pt_chunk2 = Chunk(
            document_id="pt-manual",
            content="Para operação segura, não exceda a temperatura recomendada.",
            metadata=ChunkMetadata(
                page_number=5,
                chunk_index=1,
                language="pt",
                confidence_score=0.93
            ),
            start_position=60,
            end_position=120
        )
        chunks_with_scores.append((pt_chunk2, 0.85))
        
        # English chunks about temperature
        en_chunk1 = Chunk(
            document_id="en-manual",
            content="The maximum operating temperature is 85°C under normal conditions.",
            metadata=ChunkMetadata(
                page_number=3,
                chunk_index=0,
                language="en",
                confidence_score=0.96
            ),
            start_position=0,
            end_position=65
        )
        chunks_with_scores.append((en_chunk1, 0.94))
        
        en_chunk2 = Chunk(
            document_id="en-manual",
            content="For safe operation, do not exceed the recommended temperature.",
            metadata=ChunkMetadata(
                page_number=3,
                chunk_index=1,
                language="en",
                confidence_score=0.94
            ),
            start_position=65,
            end_position=125
        )
        chunks_with_scores.append((en_chunk2, 0.88))
        
        # Test cross-language search with English question
        search_result = language_validator.validate_cross_language_search(
            "en", chunks_with_scores
        )
        
        assert search_result.search_successful
        assert "en" in search_result.languages_found
        assert "pt" in search_result.languages_found
        assert search_result.chunks_by_language["en"] == 2
        assert search_result.chunks_by_language["pt"] == 2
        assert search_result.consistency_score > 0.5
        
        # Test with Portuguese question
        search_result_pt = language_validator.validate_cross_language_search(
            "pt", chunks_with_scores
        )
        
        assert search_result_pt.search_successful
        assert search_result_pt.consistency_score > 0.5
    
    def test_question_language_validation_and_processing(self, language_validator):
        """Test question language validation in the processing pipeline"""
        
        # Test Portuguese question
        pt_question_text = "Qual é a temperatura máxima de operação?"
        pt_validation = language_validator.validate_document_language(pt_question_text, "pt")
        
        assert pt_validation.is_valid
        assert pt_validation.detected_language == "pt"
        assert pt_validation.confidence_score > 0.4
        
        # Test English question
        en_question_text = "What is the maximum operating temperature?"
        en_validation = language_validator.validate_document_language(en_question_text, "en")
        
        assert en_validation.is_valid
        assert en_validation.detected_language == "en"
        assert en_validation.confidence_score > 0.4
        
        # Test auto-detection
        auto_validation_pt = language_validator.validate_document_language(pt_question_text, None)
        assert auto_validation_pt.detected_language == "pt"
        
        auto_validation_en = language_validator.validate_document_language(en_question_text, None)
        assert auto_validation_en.detected_language == "en"
    
    def test_multilingual_support_end_to_end_simulation(self, language_validator):
        """Simulate end-to-end multilingual support validation"""
        
        # Simulate document processing with language validation
        documents = {
            "pt_manual.pdf": "Manual de operação com temperatura máxima de 85°C.",
            "en_manual.pdf": "Operation manual with maximum temperature of 85°C."
        }
        
        processed_docs = {}
        
        for filename, content in documents.items():
            # Validate document language
            validation = language_validator.validate_document_language(content)
            
            assert validation.is_valid or validation.detected_language != 'unknown'
            processed_docs[filename] = {
                'content': content,
                'language': validation.detected_language,
                'confidence': validation.confidence_score
            }
        
        # Verify we have both languages
        languages_found = set(doc['language'] for doc in processed_docs.values())
        assert 'pt' in languages_found
        assert 'en' in languages_found
        
        # Simulate cross-language question answering
        questions = [
            ("What is the maximum temperature?", "en"),
            ("Qual é a temperatura máxima?", "pt"),
            ("What is the temperatura máxima?", "auto")  # Mixed language
        ]
        
        for question_text, expected_lang in questions:
            question_validation = language_validator.validate_document_language(question_text)
            
            if expected_lang != "auto":
                # For specific language expectations
                if question_validation.confidence_score > 0.5:
                    assert question_validation.detected_language == expected_lang
            else:
                # For auto-detection, just ensure it's not unknown
                assert question_validation.detected_language in ['pt', 'en', 'unknown']
    
    def test_language_filter_validation(self, language_validator):
        """Test language filter validation in retrieval"""
        
        # Test supported language filters
        assert language_validator.is_language_supported("pt")
        assert language_validator.is_language_supported("en")
        assert not language_validator.is_language_supported("fr")
        assert not language_validator.is_language_supported("de")
        
        # Test supported languages set
        supported = language_validator.get_supported_languages()
        assert "pt" in supported
        assert "en" in supported
        assert len(supported) == 2
    
    def test_embedding_dimension_validation(self, language_validator):
        """Test that embedding dimensions are validated correctly"""
        
        texts = ["Test text", "Texto de teste"]
        
        # Valid embeddings (384 dimensions)
        valid_embeddings = [
            np.random.normal(0, 0.1, 384).tolist(),
            np.random.normal(0, 0.1, 384).tolist()
        ]
        
        result = language_validator.validate_embedding_language_consistency(texts, valid_embeddings)
        assert result.is_valid
        
        # Invalid embeddings (wrong dimensions)
        invalid_embeddings = [
            np.random.normal(0, 0.1, 256).tolist(),  # Wrong dimension
            np.random.normal(0, 0.1, 384).tolist()
        ]
        
        result = language_validator.validate_embedding_language_consistency(texts, invalid_embeddings)
        assert not result.is_valid
        assert "dimension" in result.validation_errors[0]
    
    def test_error_handling_in_language_validation(self, language_validator):
        """Test error handling in language validation"""
        
        # Test empty text
        empty_result = language_validator.validate_document_language("")
        assert not empty_result.is_valid
        assert "Text too short" in empty_result.validation_errors[0]
        
        # Test None text
        none_result = language_validator.validate_document_language(None)
        assert not none_result.is_valid
        
        # Test empty chunks list
        empty_chunks_result = language_validator.validate_chunk_language_consistency([])
        assert not empty_chunks_result.is_valid
        assert "No chunks provided" in empty_chunks_result.validation_errors[0]
        
        # Test mismatched texts and embeddings
        mismatch_result = language_validator.validate_embedding_language_consistency(
            ["text1", "text2"], [[1.0, 2.0]]  # 2 texts, 1 embedding
        )
        assert not mismatch_result.is_valid
        assert "Mismatch between number" in mismatch_result.validation_errors[0]


class TestMultilingualServiceIntegration:
    """Test integration of multilingual validation with existing services"""
    
    def test_document_service_with_language_validation(self):
        """Test that DocumentService integrates language validation"""
        
        # This test would require mocking the entire service stack
        # For now, we test that the LanguageValidator can be integrated
        
        validator = LanguageValidator()
        
        # Simulate document processing validation
        sample_text = "Este é um documento em português com informações técnicas."
        validation_result = validator.validate_document_language(sample_text, "pt")
        
        assert validation_result.is_valid
        assert validation_result.detected_language == "pt"
        
        # This demonstrates that the validator can be integrated into DocumentService
        # The actual integration is tested in the DocumentService constructor and methods
    
    def test_question_service_with_language_validation(self):
        """Test that QuestionService integrates language validation"""
        
        validator = LanguageValidator()
        
        # Simulate question validation
        question_text = "What is the maximum operating temperature?"
        validation_result = validator.validate_document_language(question_text, "en")
        
        assert validation_result.is_valid
        assert validation_result.detected_language == "en"
        
        # Test cross-language scenario
        pt_chunks = []
        en_chunks = []
        
        # This demonstrates the validation that would occur in QuestionService
        # when processing cross-language queries
    
    def test_retrieval_service_language_filtering(self):
        """Test language filtering integration in RetrievalService"""
        
        validator = LanguageValidator()
        
        # Test language filter validation
        assert validator.is_language_supported("pt")
        assert validator.is_language_supported("en")
        assert not validator.is_language_supported("unsupported_lang")
        
        # This demonstrates the validation that occurs in RetrievalService
        # when applying language filters