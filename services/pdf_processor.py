"""
PDF processing service for text extraction and language detection
"""
import logging
import re
from pathlib import Path
from typing import Optional, Tuple
import PyPDF2
import pdfplumber
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from utils.exceptions import PDFProcessingError
from utils.error_handlers import log_processing_step, RetryHandler

# Set seed for consistent language detection results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Service for extracting text from PDF files and detecting language.
    Supports multiple extraction methods with fallback mechanisms.
    """
    
    def __init__(self):
        """Initialize the PDF processor"""
        self.supported_languages = {'pt', 'en'}
        
    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            PDFProcessingError: If text extraction fails
        """
        if not Path(file_path).exists():
            raise PDFProcessingError(f"File not found: {file_path}")
            
        if not file_path.lower().endswith('.pdf'):
            raise PDFProcessingError(f"File is not a PDF: {file_path}")
            
        # Try pdfplumber first (better for complex layouts)
        try:
            text = self._extract_with_pdfplumber(file_path)
            if text.strip():
                logger.info(f"Successfully extracted text using pdfplumber from {file_path}")
                return text
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed for {file_path}: {e}")
            
        # Fallback to PyPDF2
        try:
            text = self._extract_with_pypdf2(file_path)
            if text.strip():
                logger.info(f"Successfully extracted text using PyPDF2 from {file_path}")
                return text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed for {file_path}: {e}")
            
        raise PDFProcessingError(f"Failed to extract text from {file_path} using all available methods")
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """
        Extract text using pdfplumber library.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        text_parts = []
        
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                raise PDFProcessingError("PDF contains no pages")
                
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        # Clean up the text
                        cleaned_text = self._clean_text(page_text)
                        if cleaned_text.strip():
                            text_parts.append(cleaned_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
                    
        return '\n\n'.join(text_parts)
    
    def _extract_with_pypdf2(self, file_path: str) -> str:
        """
        Extract text using PyPDF2 library.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        text_parts = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            if len(pdf_reader.pages) == 0:
                raise PDFProcessingError("PDF contains no pages")
                
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        # Clean up the text
                        cleaned_text = self._clean_text(page_text)
                        if cleaned_text.strip():
                            text_parts.append(cleaned_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
                    
        return '\n\n'.join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing excessive whitespace and formatting artifacts.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to analyze for language detection
            
        Returns:
            Tuple of (language_code, confidence_score)
            Language codes: 'pt' for Portuguese, 'en' for English, 'unknown' for others
        """
        if not text or len(text.strip()) < 10:
            return 'unknown', 0.0
            
        try:
            # Use a sample of text for detection (first 1000 chars for efficiency)
            sample_text = text[:1000].strip()
            
            detected_lang = detect(sample_text)
            
            # Map detected language to supported languages
            if detected_lang in self.supported_languages:
                # Calculate confidence based on text length and language patterns
                confidence = self._calculate_confidence(sample_text, detected_lang)
                return detected_lang, confidence
            else:
                # For unsupported languages, return unknown
                return 'unknown', 0.5
                
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
            return 'unknown', 0.0
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {e}")
            return 'unknown', 0.0
    
    def _calculate_confidence(self, text: str, detected_lang: str) -> float:
        """
        Calculate confidence score for language detection.
        
        Args:
            text: Text sample used for detection
            detected_lang: Detected language code
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.7  # Base confidence for successful detection
        
        # Adjust confidence based on text length
        text_length = len(text.strip())
        if text_length >= 500:
            length_bonus = 0.2
        elif text_length >= 100:
            length_bonus = 0.1
        else:
            length_bonus = 0.0
            
        # Adjust confidence based on language-specific patterns
        pattern_bonus = 0.0
        if detected_lang == 'pt':
            # Portuguese-specific patterns
            pt_patterns = [r'\bde\b', r'\bda\b', r'\bdo\b', r'\bpara\b', r'\bcom\b', r'\bção\b']
            matches = sum(1 for pattern in pt_patterns if re.search(pattern, text, re.IGNORECASE))
            pattern_bonus = min(0.1, matches * 0.02)
        elif detected_lang == 'en':
            # English-specific patterns
            en_patterns = [r'\bthe\b', r'\band\b', r'\bof\b', r'\bto\b', r'\bin\b', r'\btion\b']
            matches = sum(1 for pattern in en_patterns if re.search(pattern, text, re.IGNORECASE))
            pattern_bonus = min(0.1, matches * 0.02)
            
        final_confidence = min(1.0, base_confidence + length_bonus + pattern_bonus)
        return round(final_confidence, 2)
    
    def process_pdf(self, file_path: str) -> Tuple[str, str, float]:
        """
        Complete PDF processing: extract text and detect language.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, language_code, confidence_score)
            
        Raises:
            PDFProcessingError: If processing fails
        """
        try:
            # Extract text
            text = self.extract_text(file_path)
            
            if not text.strip():
                raise PDFProcessingError(f"No text content found in {file_path}")
            
            # Detect language
            language, confidence = self.detect_language(text)
            
            logger.info(f"Processed PDF {file_path}: {len(text)} chars, language={language}, confidence={confidence}")
            
            return text, language, confidence
            
        except PDFProcessingError:
            raise
        except Exception as e:
            raise PDFProcessingError(f"Unexpected error processing {file_path}: {e}")