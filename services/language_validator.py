"""
Language validation service for the PDF Q&A System

This service provides comprehensive language detection, validation, and consistency
checking throughout the document processing and question answering pipeline.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from models.document import Chunk
from models.question import Question
from utils.exceptions import LanguageValidationError

# Set seed for consistent language detection results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


@dataclass
class LanguageValidationResult:
    """Result of language validation"""
    is_valid: bool
    detected_language: str
    confidence_score: float
    validation_errors: List[str]
    warnings: List[str]


@dataclass
class CrossLanguageSearchResult:
    """Result of cross-language search validation"""
    search_successful: bool
    languages_found: Set[str]
    chunks_by_language: Dict[str, int]
    consistency_score: float
    issues: List[str]


class LanguageValidator:
    """
    Service for validating language consistency and multilingual support
    throughout the PDF Q&A system pipeline.
    """
    
    def __init__(self):
        """Initialize the language validator"""
        self.supported_languages = {'pt', 'en'}
        self.language_patterns = {
            'pt': [
                r'\b(de|da|do|para|com|em|por|que|não|são|está|foram|será|uma|mais|sobre|informações|operação|manutenção|equipamento|condições|normais|técnicas)\b',
                r'\b(ção|são|ões|mente|idade|izar|ável|ões|ção)\b',
                r'\b(português|brasil|temperatura|especificação|manual|máxima|consulte|seção)\b'
            ],
            'en': [
                r'\b(the|and|of|to|in|for|with|that|not|are|is|was|will|this|about|information|operation|maintenance|equipment|conditions|normal|technical)\b',
                r'\b(tion|ing|ed|ly|ness|ment|able|ful|tion)\b',
                r'\b(english|temperature|specification|manual|operating|maximum|refer|section)\b'
            ]
        }
        
    def validate_document_language(self, text: str, expected_language: Optional[str] = None) -> LanguageValidationResult:
        """
        Validate the language of document text
        
        Args:
            text: Text content to validate
            expected_language: Expected language code (optional)
            
        Returns:
            LanguageValidationResult with validation details
        """
        errors = []
        warnings = []
        
        if not text or len(text.strip()) < 10:
            errors.append("Text too short for reliable language detection")
            return LanguageValidationResult(
                is_valid=False,
                detected_language='unknown',
                confidence_score=0.0,
                validation_errors=errors,
                warnings=warnings
            )
        
        try:
            # Detect language using langdetect
            detected_lang = detect(text[:1000])  # Use first 1000 chars for efficiency
            
            # Calculate confidence using pattern matching
            confidence = self._calculate_language_confidence(text, detected_lang)
            
            # Validate against supported languages
            if detected_lang not in self.supported_languages:
                warnings.append(f"Detected language '{detected_lang}' is not in supported languages {self.supported_languages}")
                detected_lang = 'unknown'
                confidence = max(0.1, confidence * 0.5)  # Reduce confidence for unsupported languages
            
            # Check against expected language if provided
            if expected_language and expected_language != 'auto' and expected_language != detected_lang:
                if expected_language in self.supported_languages:
                    warnings.append(f"Expected language '{expected_language}' differs from detected '{detected_lang}'")
                else:
                    errors.append(f"Expected language '{expected_language}' is not supported")
            
            # Validate confidence threshold
            min_confidence = 0.6
            if confidence < min_confidence:
                warnings.append(f"Language detection confidence ({confidence:.2f}) below threshold ({min_confidence})")
            
            is_valid = len(errors) == 0 and confidence >= 0.3  # Lower threshold for validity
            
            return LanguageValidationResult(
                is_valid=is_valid,
                detected_language=detected_lang,
                confidence_score=confidence,
                validation_errors=errors,
                warnings=warnings
            )
            
        except LangDetectException as e:
            errors.append(f"Language detection failed: {str(e)}")
            return LanguageValidationResult(
                is_valid=False,
                detected_language='unknown',
                confidence_score=0.0,
                validation_errors=errors,
                warnings=warnings
            )
    
    def _calculate_language_confidence(self, text: str, detected_lang: str) -> float:
        """
        Calculate confidence score for language detection using pattern matching
        
        Args:
            text: Text to analyze
            detected_lang: Detected language code
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if detected_lang not in self.language_patterns:
            return 0.5  # Default confidence for unknown languages
        
        patterns = self.language_patterns[detected_lang]
        text_lower = text.lower()
        
        total_matches = 0
        total_patterns = len(patterns)
        
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
            # Normalize by text length (per 100 characters)
            normalized_matches = min(1.0, matches / max(1, len(text) / 100))
            total_matches += normalized_matches
        
        # Calculate base confidence
        base_confidence = min(1.0, total_matches / total_patterns)
        
        # Adjust based on text length
        length_factor = min(1.0, len(text) / 500)  # Full confidence at 500+ chars
        
        final_confidence = base_confidence * (0.7 + 0.3 * length_factor)
        return round(final_confidence, 2)
    
    def validate_question_language(self, question: Question) -> LanguageValidationResult:
        """
        Validate the language of a question
        
        Args:
            question: Question object to validate
            
        Returns:
            LanguageValidationResult with validation details
        """
        return self.validate_document_language(question.text, question.language)
    
    def validate_chunk_language_consistency(self, chunks: List[Chunk]) -> LanguageValidationResult:
        """
        Validate language consistency across multiple chunks
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            LanguageValidationResult with consistency validation
        """
        if not chunks:
            return LanguageValidationResult(
                is_valid=False,
                detected_language='unknown',
                confidence_score=0.0,
                validation_errors=["No chunks provided for validation"],
                warnings=[]
            )
        
        language_counts = {}
        total_chunks = len(chunks)
        errors = []
        warnings = []
        
        # Count languages across chunks
        for chunk in chunks:
            lang = chunk.metadata.language
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Find dominant language
        dominant_lang = max(language_counts.keys(), key=lambda k: language_counts[k])
        dominant_count = language_counts[dominant_lang]
        dominant_ratio = dominant_count / total_chunks
        
        # Validate consistency
        if len(language_counts) > 2:
            warnings.append(f"Multiple languages detected: {list(language_counts.keys())}")
        
        if dominant_ratio < 0.7:
            warnings.append(f"No dominant language found. Distribution: {language_counts}")
        
        # Check for unknown languages
        unknown_count = language_counts.get('unknown', 0)
        if unknown_count > 0:
            unknown_ratio = unknown_count / total_chunks
            if unknown_ratio > 0.3:
                errors.append(f"Too many chunks with unknown language: {unknown_count}/{total_chunks}")
            else:
                warnings.append(f"Some chunks have unknown language: {unknown_count}/{total_chunks}")
        
        # Calculate overall confidence
        confidence = dominant_ratio * (1.0 - unknown_count / total_chunks)
        
        is_valid = len(errors) == 0 and confidence >= 0.5
        
        return LanguageValidationResult(
            is_valid=is_valid,
            detected_language=dominant_lang,
            confidence_score=confidence,
            validation_errors=errors,
            warnings=warnings
        )
    
    def validate_cross_language_search(self, question_lang: str, retrieved_chunks: List[Tuple[Chunk, float]]) -> CrossLanguageSearchResult:
        """
        Validate that cross-language search is working correctly
        
        Args:
            question_lang: Language of the question
            retrieved_chunks: List of (chunk, score) tuples from search
            
        Returns:
            CrossLanguageSearchResult with validation details
        """
        if not retrieved_chunks:
            return CrossLanguageSearchResult(
                search_successful=False,
                languages_found=set(),
                chunks_by_language={},
                consistency_score=0.0,
                issues=["No chunks retrieved from search"]
            )
        
        languages_found = set()
        chunks_by_language = {}
        issues = []
        
        # Analyze retrieved chunks
        for chunk, score in retrieved_chunks:
            chunk_lang = chunk.metadata.language
            languages_found.add(chunk_lang)
            chunks_by_language[chunk_lang] = chunks_by_language.get(chunk_lang, 0) + 1
        
        # Validate cross-language capability
        if question_lang != 'auto' and question_lang in self.supported_languages:
            # Check if we can find content in the same language as the question
            same_lang_chunks = chunks_by_language.get(question_lang, 0)
            if same_lang_chunks == 0 and len(retrieved_chunks) > 0:
                # This might be okay if there's no content in that language
                issues.append(f"No chunks found in question language '{question_lang}', but found in: {list(languages_found)}")
        
        # Check for multilingual retrieval
        if len(languages_found) > 1:
            # Good - system can retrieve across languages
            pass
        elif len(languages_found) == 1 and 'unknown' not in languages_found:
            # Single language retrieval - check if it makes sense
            single_lang = list(languages_found)[0]
            if question_lang != 'auto' and question_lang != single_lang:
                issues.append(f"Only found chunks in '{single_lang}' for question in '{question_lang}'")
        
        # Calculate consistency score
        total_chunks = len(retrieved_chunks)
        if question_lang in chunks_by_language:
            same_lang_ratio = chunks_by_language[question_lang] / total_chunks
        else:
            same_lang_ratio = 0.0
        
        # Score based on language diversity and relevance
        diversity_score = min(1.0, len(languages_found) / 2)  # Max score for 2+ languages
        relevance_score = same_lang_ratio if question_lang != 'auto' else 1.0
        consistency_score = (diversity_score + relevance_score) / 2
        
        search_successful = len(issues) == 0 or (len(issues) == 1 and "No chunks found in question language" in issues[0])
        
        return CrossLanguageSearchResult(
            search_successful=search_successful,
            languages_found=languages_found,
            chunks_by_language=chunks_by_language,
            consistency_score=consistency_score,
            issues=issues
        )
    
    def validate_embedding_language_consistency(self, texts: List[str], embeddings: List[List[float]]) -> LanguageValidationResult:
        """
        Validate that embeddings are consistent across different languages
        
        Args:
            texts: List of text strings
            embeddings: List of corresponding embeddings
            
        Returns:
            LanguageValidationResult with embedding validation
        """
        if len(texts) != len(embeddings):
            return LanguageValidationResult(
                is_valid=False,
                detected_language='unknown',
                confidence_score=0.0,
                validation_errors=["Mismatch between number of texts and embeddings"],
                warnings=[]
            )
        
        if not texts or not embeddings:
            return LanguageValidationResult(
                is_valid=False,
                detected_language='unknown',
                confidence_score=0.0,
                validation_errors=["No texts or embeddings provided"],
                warnings=[]
            )
        
        errors = []
        warnings = []
        
        # Check embedding dimensions consistency
        expected_dim = len(embeddings[0]) if embeddings else 0
        for i, embedding in enumerate(embeddings):
            if len(embedding) != expected_dim:
                errors.append(f"Embedding {i} has dimension {len(embedding)}, expected {expected_dim}")
        
        # Detect languages in texts
        languages_detected = []
        for i, text in enumerate(texts):
            try:
                lang = detect(text) if len(text.strip()) > 10 else 'unknown'
                languages_detected.append(lang)
            except:
                languages_detected.append('unknown')
        
        # Check for multilingual support
        unique_languages = set(lang for lang in languages_detected if lang != 'unknown')
        if len(unique_languages) > 1:
            # Good - multilingual embeddings
            pass
        elif len(unique_languages) == 0:
            warnings.append("All texts have unknown language")
        
        # Validate embedding quality (basic checks)
        for i, embedding in enumerate(embeddings):
            # Check for zero vectors (usually indicates problems)
            if all(abs(x) < 1e-6 for x in embedding):
                errors.append(f"Embedding {i} appears to be zero vector")
            
            # Check for reasonable magnitude
            magnitude = sum(x*x for x in embedding) ** 0.5
            if magnitude < 0.1 or magnitude > 10.0:
                warnings.append(f"Embedding {i} has unusual magnitude: {magnitude:.3f}")
        
        # Calculate overall confidence
        valid_embeddings = len([e for e in embeddings if not all(abs(x) < 1e-6 for x in e)])
        confidence = valid_embeddings / len(embeddings) if embeddings else 0.0
        
        # Determine dominant language
        if languages_detected:
            lang_counts = {}
            for lang in languages_detected:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            dominant_lang = max(lang_counts.keys(), key=lambda k: lang_counts[k])
        else:
            dominant_lang = 'unknown'
        
        is_valid = len(errors) == 0 and confidence >= 0.8
        
        return LanguageValidationResult(
            is_valid=is_valid,
            detected_language=dominant_lang,
            confidence_score=confidence,
            validation_errors=errors,
            warnings=warnings
        )
    
    def get_supported_languages(self) -> Set[str]:
        """
        Get the set of supported languages
        
        Returns:
            Set of supported language codes
        """
        return self.supported_languages.copy()
    
    def is_language_supported(self, language_code: str) -> bool:
        """
        Check if a language is supported
        
        Args:
            language_code: Language code to check
            
        Returns:
            True if language is supported
        """
        return language_code in self.supported_languages