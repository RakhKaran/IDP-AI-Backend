"""
PaddleOCR-First Service - Uses PaddleOCR as primary with Tesseract fallback
Optimized for poor quality documents where PaddleOCR performs better
"""

import os
import gc
import time
import logging
import signal
import multiprocessing
from typing import Dict, Optional, List
from contextlib import contextmanager

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

from .base_ocr_service import BaseOCRService

logger = logging.getLogger(__name__)


class PaddleFirstOCRService(BaseOCRService):
    """
    OCR service that prioritizes PaddleOCR for better accuracy on poor quality documents
    Falls back to Tesseract if PaddleOCR fails or produces poor results
    """
    
    def __init__(self, 
                 paddle_timeout: int = 45,
                 tesseract_timeout: int = 30,
                 min_confidence_threshold: float = 30.0,
                 enable_preprocessing: bool = True):
        """
        Initialize PaddleOCR-first service
        
        Args:
            paddle_timeout: Timeout for PaddleOCR processing (seconds)
            tesseract_timeout: Timeout for Tesseract processing (seconds)
            min_confidence_threshold: Minimum confidence to accept PaddleOCR result
            enable_preprocessing: Enable image preprocessing for better OCR
        """
        self.paddle_timeout = paddle_timeout
        self.tesseract_timeout = tesseract_timeout
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_preprocessing = enable_preprocessing
        
        # Check availability
        self.paddle_available = self._check_paddle_availability()
        self.tesseract_available = self._check_tesseract_availability()
        
        # PaddleOCR instance (lazy loaded)
        self._paddle_ocr = None
        self._paddle_init_attempted = False
        
        # Supported languages
        self.SUPPORTED_LANGUAGES = ['eng', 'hin', 'spa', 'fra', 'deu', 'chi_sim', 'ara']
        
        logger.info(f"PaddleFirstOCR initialized: paddle={self.paddle_available}, tesseract={self.tesseract_available}")
    
    def _check_paddle_availability(self) -> bool:
        """Check if PaddleOCR can be imported"""
        try:
            import paddleocr
            logger.info("PaddleOCR import successful")
            return True
        except Exception as e:
            logger.warning(f"PaddleOCR not available: {e}")
            return False
    
    def _check_tesseract_availability(self) -> bool:
        """Check if Tesseract is available"""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract available: {version}")
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False
    
    def _get_paddle_ocr(self):
        """Lazy load PaddleOCR with robust error handling"""
        if self._paddle_ocr is not None:
            return self._paddle_ocr
        
        if self._paddle_init_attempted:
            return None
        
        self._paddle_init_attempted = True
        
        if not self.paddle_available:
            return None
        
        try:
            from paddleocr import PaddleOCR
            
            # Initialize with safe settings to avoid segfaults
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=True,  # Enable for better accuracy on rotated text
                lang='en',
                show_log=False,
                use_gpu=False,  # CPU mode for stability
                enable_mkldnn=False,  # Disable for compatibility
                cpu_threads=1,  # Single thread to avoid race conditions
                max_text_length=25,  # Limit text length to avoid memory issues
                det_limit_side_len=960,  # Limit detection resolution
                det_limit_type='min',  # Use minimum limit
                rec_batch_num=6,  # Smaller batch size
            )
            
            logger.info("PaddleOCR initialized successfully")
            return self._paddle_ocr
            
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            self._paddle_ocr = None
            return None
    
    @contextmanager
    def _timeout_context(self, seconds: int):
        """Context manager for operation timeout with proper cleanup"""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Clean up
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
    
    def _preprocess_image_for_paddle(self, image: Image.Image) -> Image.Image:
        """Enhanced preprocessing specifically for PaddleOCR on poor quality documents"""
        if not self.enable_preprocessing:
            return image
        
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Enhance for poor quality documents
            
            # 1. Upscale for better detail recognition
            height, width = gray.shape
            if width < 1000:  # Only upscale if image is small
                scale_factor = min(2.0, 1000 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 2. Noise reduction for scanned documents
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # 3. Contrast enhancement using CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # 4. Sharpening for blurry text
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # 5. Adaptive thresholding for varying lighting
            binary = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 31, 11
            )
            
            # Convert back to RGB for PaddleOCR
            rgb_result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            
            return Image.fromarray(rgb_result)
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {e}")
            return image
    
    def _preprocess_image_for_tesseract(self, image: Image.Image) -> Image.Image:
        """Light preprocessing for Tesseract"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convert to numpy for processing
            img_array = np.array(image)
            
            # Light enhancement
            img_array = cv2.convertScaleAbs(img_array, alpha=1.2, beta=10)
            
            # Simple denoising
            img_array = cv2.medianBlur(img_array, 3)
            
            # Simple thresholding
            _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return Image.fromarray(img_array)
            
        except Exception as e:
            logger.warning(f"Tesseract preprocessing failed, using original: {e}")
            return image
    
    def _extract_with_paddle(self, image: Image.Image, config: Optional[Dict] = None) -> Dict:
        """Extract text using PaddleOCR with robust error handling"""
        
        paddle_ocr = self._get_paddle_ocr()
        if paddle_ocr is None:
            raise RuntimeError("PaddleOCR not available")
        
        try:
            # Preprocess image for better OCR on poor quality documents
            processed_image = self._preprocess_image_for_paddle(image)
            
            # Convert to numpy array
            if isinstance(processed_image, Image.Image):
                image_array = np.array(processed_image)
            else:
                image_array = processed_image
            
            # Process with timeout protection
            with self._timeout_context(self.paddle_timeout):
                result = paddle_ocr.ocr(image_array, cls=True)  # Enable classification for rotated text
            
            # Clean up processed image
            if processed_image != image:
                processed_image.close()
            
            if not result or not result[0]:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'engine': 'paddle',
                    'success': False
                }
            
            texts = []
            confidences = []
            
            for line in result[0]:
                try:
                    if len(line) >= 2 and len(line[1]) >= 2:
                        text = line[1][0].strip()
                        conf = float(line[1][1])
                        if text:
                            texts.append(text)
                            confidences.append(conf)
                except (IndexError, ValueError, TypeError) as e:
                    logger.debug(f"Skipping malformed OCR result: {e}")
                    continue
            
            if not texts:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'engine': 'paddle',
                    'success': False
                }
            
            avg_confidence = sum(confidences) / len(confidences) * 100  # Convert to percentage
            final_text = '\n'.join(texts)
            
            # Check if result meets quality threshold
            success = (
                len(final_text.strip()) > 3 and  # Minimum text length
                avg_confidence >= self.min_confidence_threshold and  # Minimum confidence
                len([t for t in texts if len(t.strip()) > 2]) >= 1  # At least one meaningful word
            )
            
            return {
                'text': final_text,
                'confidence': avg_confidence,
                'engine': 'paddle',
                'success': success,
                'word_count': len(texts)
            }
            
        except TimeoutError:
            logger.warning(f"PaddleOCR timed out after {self.paddle_timeout}s")
            raise RuntimeError("PaddleOCR processing timeout")
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            raise RuntimeError(f"PaddleOCR processing failed: {e}")
        finally:
            # Force garbage collection to prevent memory leaks
            gc.collect()
    
    def _extract_with_tesseract(self, image: Image.Image, config: Optional[Dict] = None) -> Dict:
        """Extract text using Tesseract as fallback"""
        
        if not self.tesseract_available:
            raise RuntimeError("Tesseract not available")
        
        config = config or {}
        
        try:
            # Preprocess for Tesseract
            processed_image = self._preprocess_image_for_tesseract(image)
            
            # Safe language handling
            lang = config.get('language_mode', 'eng')
            if lang == 'auto' or not lang or lang not in self.SUPPORTED_LANGUAGES:
                lang = 'eng'
            
            # Conservative PSM settings for poor quality documents
            psm = config.get('psm', 6)  # Uniform text block
            oem = config.get('oem', 3)  # LSTM only
            
            tesseract_config = f'--psm {psm} --oem {oem}'
            
            # Extract text with timeout
            with self._timeout_context(self.tesseract_timeout):
                text = pytesseract.image_to_string(processed_image, lang=lang, config=tesseract_config)
            
            # Get confidence if possible
            confidence = 60  # Default confidence for Tesseract
            try:
                with self._timeout_context(15):  # Shorter timeout for confidence
                    data = pytesseract.image_to_data(
                        processed_image, lang=lang, config=tesseract_config,
                        output_type=pytesseract.Output.DICT
                    )
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    if confidences:
                        confidence = sum(confidences) / len(confidences)
            except Exception as e:
                logger.debug(f"Could not get Tesseract confidence: {e}")
            
            # Clean up processed image
            if processed_image != image:
                processed_image.close()
            
            final_text = text.strip()
            success = len(final_text) > 2  # Basic success criteria
            
            return {
                'text': final_text,
                'confidence': confidence,
                'engine': 'tesseract',
                'success': success
            }
            
        except TimeoutError:
            logger.warning(f"Tesseract timed out after {self.tesseract_timeout}s")
            raise RuntimeError("Tesseract processing timeout")
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            raise RuntimeError(f"Tesseract processing failed: {e}")
    
    def _extract_with_fallback(self, image: Image.Image, config: Optional[Dict] = None) -> Dict:
        """Extract text with PaddleOCR first, Tesseract fallback strategy"""
        
        paddle_result = None
        tesseract_result = None
        
        # Try PaddleOCR first (better for poor quality documents)
        if self.paddle_available:
            try:
                logger.debug("Attempting PaddleOCR extraction...")
                paddle_result = self._extract_with_paddle(image, config)
                
                if paddle_result['success']:
                    logger.info(f"PaddleOCR successful: conf={paddle_result['confidence']:.1f}%, "
                              f"chars={len(paddle_result['text'])}")
                    return paddle_result
                else:
                    logger.info(f"PaddleOCR result poor: conf={paddle_result['confidence']:.1f}%, "
                              f"chars={len(paddle_result['text'])}, trying Tesseract...")
                
            except Exception as e:
                logger.warning(f"PaddleOCR failed: {e}")
                paddle_result = {
                    'text': '',
                    'confidence': 0.0,
                    'engine': 'paddle',
                    'success': False,
                    'error': str(e)
                }
        
        # Try Tesseract fallback
        if self.tesseract_available:
            try:
                logger.debug("Attempting Tesseract fallback...")
                tesseract_result = self._extract_with_tesseract(image, config)
                
                if tesseract_result['success']:
                    logger.info(f"Tesseract fallback successful: conf={tesseract_result['confidence']:.1f}%, "
                              f"chars={len(tesseract_result['text'])}")
                    return tesseract_result
                else:
                    logger.warning("Tesseract fallback also produced poor results")
                
            except Exception as e:
                logger.warning(f"Tesseract fallback failed: {e}")
                tesseract_result = {
                    'text': '',
                    'confidence': 0.0,
                    'engine': 'tesseract',
                    'success': False,
                    'error': str(e)
                }
        
        # Return best available result or failure
        if paddle_result and paddle_result['text']:
            logger.info("Returning PaddleOCR result despite low confidence")
            return paddle_result
        elif tesseract_result and tesseract_result['text']:
            logger.info("Returning Tesseract result despite low confidence")
            return tesseract_result
        else:
            logger.error("Both OCR engines failed")
            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'failed',
                'success': False,
                'paddle_error': paddle_result.get('error') if paddle_result else 'Not attempted',
                'tesseract_error': tesseract_result.get('error') if tesseract_result else 'Not attempted'
            }
    
    def extract_text(self, image_path: str, config: Optional[Dict] = None) -> str:
        """Extract text from image or PDF"""
        result = self.extract_text_with_confidence(image_path, config)
        return result.get('text', '')
    
    def extract_text_with_confidence(self, image_path: str, config: Optional[Dict] = None) -> Dict:
        """
        Extract text with confidence using PaddleOCR-first strategy
        
        Args:
            image_path: Path to image or PDF file
            config: Configuration dictionary
        
        Returns:
            Dictionary with text, confidence, and processing info
        """
        config = config or {}
        
        try:
            if image_path.lower().endswith('.pdf'):
                return self._process_pdf(image_path, config)
            else:
                return self._process_image(image_path, config)
                
        except Exception as e:
            logger.error(f"OCR processing failed for {image_path}: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'failed',
                'success': False,
                'error': str(e)
            }
    
    def _process_pdf(self, pdf_path: str, config: Dict) -> Dict:
        """Process PDF with PaddleOCR-first strategy"""
        
        # Conservative settings for stability
        dpi = config.get('dpi', 200)  # Higher DPI for better PaddleOCR performance
        max_pages = config.get('max_pages', 15)  # Reasonable limit
        
        try:
            convert_kwargs = {'dpi': dpi}
            if max_pages:
                convert_kwargs['last_page'] = max_pages
            
            images = convert_from_path(pdf_path, **convert_kwargs)
            
            if not images:
                return {'text': '', 'confidence': 0.0, 'engine': 'failed', 'success': False}
            
            # Process pages sequentially for stability
            all_texts = []
            all_confidences = []
            engines_used = []
            successful_pages = 0
            
            for i, img in enumerate(images):
                try:
                    logger.info(f"Processing page {i+1}/{len(images)} with PaddleOCR-first strategy")
                    
                    page_result = self._extract_with_fallback(img, config)
                    
                    if page_result['success'] and page_result['text']:
                        all_texts.append(page_result['text'])
                        all_confidences.append(page_result['confidence'])
                        engines_used.append(page_result['engine'])
                        successful_pages += 1
                    
                    # Clean up
                    img.close()
                    
                except Exception as e:
                    logger.warning(f"Failed to process page {i+1}: {e}")
                    continue
            
            # Clean up images
            for img in images:
                try:
                    img.close()
                except:
                    pass
            del images
            gc.collect()
            
            # Aggregate results
            if all_texts:
                final_text = '\n\n'.join(all_texts)
                avg_confidence = sum(all_confidences) / len(all_confidences)
                primary_engine = max(set(engines_used), key=engines_used.count) if engines_used else 'unknown'
                
                return {
                    'text': final_text,
                    'confidence': avg_confidence,
                    'pages_processed': successful_pages,
                    'total_pages': len(images) if 'images' in locals() else 0,
                    'engines_used': dict(zip(*np.unique(engines_used, return_counts=True))) if engines_used else {},
                    'primary_engine': primary_engine,
                    'engine': primary_engine,
                    'success': True
                }
            else:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'pages_processed': 0,
                    'engine': 'failed',
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'failed',
                'success': False,
                'error': str(e)
            }
    
    def _process_image(self, image_path: str, config: Dict) -> Dict:
        """Process single image file"""
        
        try:
            image = Image.open(image_path)
            result = self._extract_with_fallback(image, config)
            image.close()
            return result
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'failed',
                'success': False,
                'error': str(e)
            }
    
    # BaseOCRService interface methods
    def supports_language(self, language_code: str) -> bool:
        """Check if language is supported"""
        if not language_code or language_code == 'auto':
            return True
        return language_code in self.SUPPORTED_LANGUAGES
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        return self.SUPPORTED_LANGUAGES.copy()
    
    def cleanup(self):
        """Clean up resources"""
        if self._paddle_ocr is not None:
            try:
                del self._paddle_ocr
                self._paddle_ocr = None
                gc.collect()
                logger.info("PaddleOCR resources cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up PaddleOCR: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()