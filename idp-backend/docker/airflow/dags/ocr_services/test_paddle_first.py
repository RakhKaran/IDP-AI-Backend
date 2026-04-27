#!/usr/bin/env python3
"""
Test script for PaddleOCR-first service
Tests both PaddleOCR and Tesseract fallback functionality
"""

import os
import sys
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add the dags directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_services.paddle_first_ocr_service import PaddleFirstOCRService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_image(text: str, quality: str = "good") -> Image.Image:
    """Create a test image with specified text and quality"""
    
    # Image dimensions
    width, height = 400, 150
    
    # Create image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Draw text
    text_position = (20, 50)
    draw.text(text_position, text, fill='black', font=font)
    
    # Apply quality degradation
    if quality == "poor":
        # Convert to numpy for degradation
        img_array = np.array(img)
        
        # Add noise
        noise = np.random.normal(0, 25, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Blur
        import cv2
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        # Reduce contrast
        img_array = cv2.convertScaleAbs(img_array, alpha=0.7, beta=30)
        
        img = Image.fromarray(img_array)
    
    elif quality == "very_poor":
        # Severe degradation
        img_array = np.array(img)
        
        # Heavy noise
        noise = np.random.normal(0, 40, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Heavy blur
        import cv2
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
        
        # Poor contrast
        img_array = cv2.convertScaleAbs(img_array, alpha=0.5, beta=50)
        
        # Compression artifacts simulation
        img_array = cv2.medianBlur(img_array, 3)
        
        img = Image.fromarray(img_array)
    
    return img


def test_paddle_first_service():
    """Test the PaddleOCR-first service"""
    
    logger.info("🧪 Testing PaddleOCR-First Service")
    logger.info("=" * 50)
    
    # Initialize service
    try:
        ocr_service = PaddleFirstOCRService(
            paddle_timeout=30,
            tesseract_timeout=20,
            min_confidence_threshold=25.0,
            enable_preprocessing=True
        )
        logger.info(f"✅ Service initialized: paddle={ocr_service.paddle_available}, tesseract={ocr_service.tesseract_available}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}")
        return False
    
    # Test cases
    test_cases = [
        ("Hello World Test", "good"),
        ("Poor Quality Document", "poor"),
        ("Very Bad Scan Quality", "very_poor"),
        ("Invoice #12345 Date: 2024-01-15", "poor"),
        ("TOTAL AMOUNT: $1,234.56", "very_poor")
    ]
    
    results = []
    
    for i, (text, quality) in enumerate(test_cases, 1):
        logger.info(f"\n📄 Test {i}: '{text}' (quality: {quality})")
        
        try:
            # Create test image
            test_img = create_test_image(text, quality)
            
            # Save test image for debugging
            test_img_path = f"/tmp/test_ocr_{i}_{quality}.png"
            test_img.save(test_img_path)
            logger.info(f"Test image saved: {test_img_path}")
            
            # Test OCR
            result = ocr_service._extract_with_fallback(test_img, {})
            
            # Analyze result
            extracted_text = result.get('text', '').strip()
            confidence = result.get('confidence', 0)
            engine_used = result.get('engine', 'unknown')
            success = result.get('success', False)
            
            # Check accuracy
            original_words = text.lower().split()
            extracted_words = extracted_text.lower().split()
            
            # Simple word matching
            matches = sum(1 for word in original_words if any(word in ext_word for ext_word in extracted_words))
            accuracy = (matches / len(original_words)) * 100 if original_words else 0
            
            logger.info(f"   Engine used: {engine_used}")
            logger.info(f"   Confidence: {confidence:.1f}%")
            logger.info(f"   Success: {success}")
            logger.info(f"   Extracted: '{extracted_text}'")
            logger.info(f"   Accuracy: {accuracy:.1f}%")
            
            if success and accuracy > 50:
                logger.info("   ✅ PASS")
                status = "PASS"
            else:
                logger.info("   ⚠️ POOR")
                status = "POOR"
            
            results.append({
                'test': f"Test {i}",
                'original': text,
                'quality': quality,
                'extracted': extracted_text,
                'engine': engine_used,
                'confidence': confidence,
                'accuracy': accuracy,
                'status': status
            })
            
            # Clean up
            test_img.close()
            
        except Exception as e:
            logger.error(f"   ❌ FAIL: {e}")
            results.append({
                'test': f"Test {i}",
                'original': text,
                'quality': quality,
                'extracted': '',
                'engine': 'failed',
                'confidence': 0,
                'accuracy': 0,
                'status': 'FAIL',
                'error': str(e)
            })
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    poor = sum(1 for r in results if r['status'] == 'POOR')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Poor results: {poor}")
    logger.info(f"Failed: {failed}")
    
    # Engine usage
    paddle_used = sum(1 for r in results if r['engine'] == 'paddle')
    tesseract_used = sum(1 for r in results if r['engine'] == 'tesseract')
    
    logger.info(f"\nEngine Usage:")
    logger.info(f"PaddleOCR: {paddle_used}")
    logger.info(f"Tesseract: {tesseract_used}")
    
    # Average confidence by engine
    if paddle_used > 0:
        paddle_conf = np.mean([r['confidence'] for r in results if r['engine'] == 'paddle'])
        logger.info(f"Average PaddleOCR confidence: {paddle_conf:.1f}%")
    
    if tesseract_used > 0:
        tesseract_conf = np.mean([r['confidence'] for r in results if r['engine'] == 'tesseract'])
        logger.info(f"Average Tesseract confidence: {tesseract_conf:.1f}%")
    
    # Recommendations
    logger.info(f"\n💡 RECOMMENDATIONS:")
    
    if paddle_used == 0:
        logger.info("- PaddleOCR was not used. Check if it's properly installed.")
    elif paddle_used > tesseract_used:
        logger.info("- PaddleOCR is working well as primary engine.")
    else:
        logger.info("- PaddleOCR is falling back to Tesseract frequently. Consider adjusting confidence threshold.")
    
    if failed > 0:
        logger.info("- Some tests failed completely. Check error logs above.")
    
    if poor > passed:
        logger.info("- Many tests had poor results. Consider:")
        logger.info("  * Increasing image DPI")
        logger.info("  * Enabling preprocessing")
        logger.info("  * Adjusting confidence thresholds")
    
    # Overall result
    success_rate = (passed / len(results)) * 100
    logger.info(f"\n🎯 Overall Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        logger.info("🎉 Excellent! PaddleOCR-first service is working well.")
        return True
    elif success_rate >= 60:
        logger.info("👍 Good! Service is functional but could be improved.")
        return True
    else:
        logger.info("⚠️ Poor performance. Consider using safe mode or adjusting settings.")
        return False


def test_pdf_processing():
    """Test PDF processing if a sample PDF is available"""
    
    logger.info("\n📄 Testing PDF Processing")
    logger.info("=" * 30)
    
    # Look for sample PDFs
    sample_paths = [
        "/opt/airflow/downloaded_docs",
        "/tmp",
        "."
    ]
    
    sample_pdf = None
    for path in sample_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.lower().endswith('.pdf'):
                    sample_pdf = os.path.join(path, file)
                    break
            if sample_pdf:
                break
    
    if not sample_pdf:
        logger.info("No sample PDF found for testing")
        return True
    
    logger.info(f"Testing with: {sample_pdf}")
    
    try:
        ocr_service = PaddleFirstOCRService()
        
        result = ocr_service.extract_text_with_confidence(sample_pdf, {
            'dpi': 200,
            'max_pages': 2
        })
        
        logger.info(f"Pages processed: {result.get('pages_processed', 0)}")
        logger.info(f"Primary engine: {result.get('primary_engine', 'unknown')}")
        logger.info(f"Confidence: {result.get('confidence', 0):.1f}%")
        logger.info(f"Text length: {len(result.get('text', ''))}")
        logger.info(f"Success: {result.get('success', False)}")
        
        if result.get('success') and len(result.get('text', '')) > 50:
            logger.info("✅ PDF processing successful")
            return True
        else:
            logger.info("⚠️ PDF processing had poor results")
            return False
            
    except Exception as e:
        logger.error(f"❌ PDF processing failed: {e}")
        return False


def main():
    """Main test function"""
    
    logger.info("🚀 Starting PaddleOCR-First Service Tests")
    
    # Test 1: Basic functionality
    basic_test_passed = test_paddle_first_service()
    
    # Test 2: PDF processing
    pdf_test_passed = test_pdf_processing()
    
    # Overall result
    logger.info("\n" + "=" * 60)
    logger.info("🏁 FINAL RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"Basic functionality: {'✅ PASS' if basic_test_passed else '❌ FAIL'}")
    logger.info(f"PDF processing: {'✅ PASS' if pdf_test_passed else '❌ FAIL'}")
    
    if basic_test_passed and pdf_test_passed:
        logger.info("\n🎉 All tests passed! PaddleOCR-first service is ready for production.")
        logger.info("\nTo use in your DAGs:")
        logger.info('ocr_engine = "paddle_first"')
        return True
    elif basic_test_passed:
        logger.info("\n👍 Basic tests passed. PDF processing needs attention.")
        logger.info("\nService is usable but monitor PDF processing carefully.")
        return True
    else:
        logger.info("\n⚠️ Tests failed. Consider using safe mode:")
        logger.info('ocr_engine = "safe"')
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)