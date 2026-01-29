"""
Abstract Base Class for OCR Services
All OCR service implementations must inherit from this class
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseOCRService(ABC):
    """Abstract base class for all OCR service implementations"""
    
    @abstractmethod
    def extract_text(self, image_path: str, config: Optional[Dict] = None) -> str:
        """
        Extract text from an image file
        
        Args:
            image_path: Path to the image file
            config: Optional configuration dictionary with OCR settings
            
        Returns:
            Extracted text as string
        """
        pass
    
    @abstractmethod
    def extract_text_with_confidence(self, image_path: str, config: Optional[Dict] = None) -> Dict:
        """
        Extract text with confidence scores from an image file
        
        Args:
            image_path: Path to the image file
            config: Optional configuration dictionary with OCR settings
            
        Returns:
            Dictionary with 'text' and 'confidence' keys
        """
        pass
    
    @abstractmethod
    def supports_language(self, language_code: str) -> bool:
        """
        Check if the OCR service supports a specific language
        
        Args:
            language_code: Language code (e.g., 'eng', 'hin', 'eng+hin')
            
        Returns:
            True if language is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> list:
        """
        Get list of supported language codes
        
        Returns:
            List of supported language codes
        """
        pass

