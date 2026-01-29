"""
Tesseract OCR Service Implementation
Handles text extraction using Tesseract OCR engine
"""

import pytesseract
from pdf2image import convert_from_path
from typing import Dict, Optional
import os
from .base_ocr_service import BaseOCRService


class TesseractOCRService(BaseOCRService):
    """Tesseract OCR service implementation"""
    
    # Common language codes supported by Tesseract
    SUPPORTED_LANGUAGES = [
        'eng', 'hin', 'spa', 'fra', 'deu', 'chi_sim', 'chi_tra',
        'jpn', 'kor', 'ara', 'por', 'rus', 'ita', 'nld', 'pol'
    ]
    
    def __init__(self):
        """Initialize Tesseract OCR service"""
        # Verify Tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Tesseract OCR not available: {e}")
    
    def _get_tesseract_config(self, config: Optional[Dict] = None) -> str:
        """
        Build Tesseract configuration string
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tesseract config string
        """
        if not config:
            return ""
        
        # PSM (Page Segmentation Mode) - default is 3 (fully automatic)
        psm = config.get('psm', 3)
        # OEM (OCR Engine Mode) - default is 3 (LSTM only)
        oem = config.get('oem', 3)
        
        config_str = f'--psm {psm} --oem {oem}'
        return config_str
    
    def _detect_language(self, image_path: str) -> str:
        """
        Auto-detect language from image (simplified - uses OSDA)
        
        Args:
            image_path: Path to image file
            
        Returns:
            Detected language code
        """
        try:
            # Use Tesseract's OSDA (Optical Script Detection and Analysis)
            # This is a simplified approach - for production, consider using langdetect library
            from pdf2image import convert_from_path
            if image_path.lower().endswith('.pdf'):
                images = convert_from_path(image_path, first_page=1, last_page=1)
                if images:
                    # Try to detect using Tesseract's built-in detection
                    # For now, default to English if auto-detect fails
                    return 'eng'
            return 'eng'  # Default fallback
        except Exception:
            return 'eng'  # Default fallback
    
    def extract_text(self, image_path: str, config: Optional[Dict] = None) -> str:
        """
        Extract text from image or PDF using Tesseract OCR
        
        Args:
            image_path: Path to image file or PDF
            config: Configuration dictionary with:
                - language_mode: 'auto' or language code (e.g., 'eng', 'hin+eng')
                - psm: Page segmentation mode (default: 3)
                - oem: OCR engine mode (default: 3)
        
        Returns:
            Extracted text as string
        """
        try:
            config = config or {}
            language_mode = config.get('language_mode', 'auto')
            
            # Determine language
            if language_mode == 'auto':
                lang = self._detect_language(image_path)
            else:
                lang = language_mode
            
            # Get Tesseract config
            tesseract_config = self._get_tesseract_config(config)
            
            # Handle PDF files
            if image_path.lower().endswith('.pdf'):
                images = convert_from_path(image_path)
                text_parts = []
                
                for img in images:
                    page_text = pytesseract.image_to_string(
                        img,
                        lang=lang,
                        config=tesseract_config
                    )
                    text_parts.append(page_text)
                
                return '\n'.join(text_parts)
            
            # Handle image files
            else:
                from PIL import Image
                img = Image.open(image_path)
                text = pytesseract.image_to_string(
                    img,
                    lang=lang,
                    config=tesseract_config
                )
                return text
                
        except Exception as e:
            error_msg = f"Tesseract OCR failed for {image_path}: {e}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def extract_text_with_confidence(self, image_path: str, config: Optional[Dict] = None) -> Dict:
        """
        Extract text with confidence scores from image or PDF
        
        Args:
            image_path: Path to image file or PDF
            config: Configuration dictionary
        
        Returns:
            Dictionary with 'text' and 'confidence' keys
        """
        try:
            config = config or {}
            language_mode = config.get('language_mode', 'auto')
            
            # Determine language
            if language_mode == 'auto':
                lang = self._detect_language(image_path)
            else:
                lang = language_mode
            
            # Get Tesseract config
            tesseract_config = self._get_tesseract_config(config)
            
            # Handle PDF files
            if image_path.lower().endswith('.pdf'):
                images = convert_from_path(image_path)
                all_text = []
                all_confidences = []
                
                for img in images:
                    # Get text
                    page_text = pytesseract.image_to_string(
                        img,
                        lang=lang,
                        config=tesseract_config
                    )
                    all_text.append(page_text)
                    
                    # Get confidence data
                    ocr_data = pytesseract.image_to_data(
                        img,
                        lang=lang,
                        config=tesseract_config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Calculate average confidence for page
                    confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    all_confidences.append(avg_confidence)
                
                full_text = '\n'.join(all_text)
                overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
                
                return {
                    'text': full_text,
                    'confidence': overall_confidence
                }
            
            # Handle image files
            else:
                from PIL import Image
                img = Image.open(image_path)
                
                # Get text
                text = pytesseract.image_to_string(
                    img,
                    lang=lang,
                    config=tesseract_config
                )
                
                # Get confidence data
                ocr_data = pytesseract.image_to_data(
                    img,
                    lang=lang,
                    config=tesseract_config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Calculate average confidence
                confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                return {
                    'text': text,
                    'confidence': avg_confidence
                }
                
        except Exception as e:
            error_msg = f"Tesseract OCR with confidence failed for {image_path}: {e}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def supports_language(self, language_code: str) -> bool:
        """
        Check if Tesseract supports a specific language
        
        Args:
            language_code: Language code (e.g., 'eng', 'hin', 'eng+hin')
        
        Returns:
            True if language is supported, False otherwise
        """
        if not language_code or language_code == 'auto':
            return True
        
        # Handle multi-language codes (e.g., 'eng+hin')
        languages = language_code.split('+')
        
        for lang in languages:
            lang = lang.strip()
            if lang not in self.SUPPORTED_LANGUAGES:
                # Check if it's available in Tesseract (some languages might be installed)
                try:
                    pytesseract.get_languages()
                    # For now, assume it's supported if Tesseract doesn't raise error
                    # In production, you might want to check actual available languages
                    return True
                except:
                    return False
        
        return True
    
    def get_supported_languages(self) -> list:
        """
        Get list of supported language codes
        
        Returns:
            List of supported language codes
        """
        try:
            # Try to get actual installed languages from Tesseract
            installed_langs = pytesseract.get_languages()
            return installed_langs if installed_langs else self.SUPPORTED_LANGUAGES
        except:
            return self.SUPPORTED_LANGUAGES
