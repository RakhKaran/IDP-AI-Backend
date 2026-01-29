"""
AI Text Cleanup Service
Uses OpenAI GPT to clean and improve OCR-extracted text
"""

from openai import OpenAI
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class TextCleanupService:
    """AI-powered text cleanup service using OpenAI"""
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        """
        Initialize text cleanup service
        
        Args:
            openai_client: Optional OpenAI client instance. If not provided, creates new one.
        """
        if openai_client:
            self.client = openai_client
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            self.client = OpenAI(api_key=api_key)
    
    def cleanup_text(self, raw_text: str, model: str = "gpt-4o") -> str:
        """
        Clean and improve OCR-extracted text using AI
        
        Args:
            raw_text: Raw text extracted from OCR
            model: OpenAI model to use (default: "gpt-4o")
        
        Returns:
            Cleaned and improved text
        """
        if not raw_text or not raw_text.strip():
            return raw_text
        
        try:
            prompt = f"""You are a text cleaning assistant. The following text was extracted from a document using OCR (Optical Character Recognition). 

Please clean and improve this text by:
1. Fixing obvious OCR errors and typos
2. Correcting spacing and formatting issues
3. Removing artifacts and noise
4. Preserving the original meaning and structure
5. Keeping numbers, dates, and proper nouns intact
6. Maintaining paragraph structure where possible

Do NOT:
- Add information that wasn't in the original text
- Change the meaning or content
- Remove important details

Return only the cleaned text, without any explanations or additional commentary.

OCR Text:
{raw_text[:8000]}  # Limit to avoid token limits
"""
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful text cleaning assistant that fixes OCR errors while preserving original content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for consistent, accurate cleaning
                max_tokens=4000,
                timeout=60
            )
            
            cleaned_text = response.choices[0].message.content.strip()
            return cleaned_text
            
        except Exception as e:
            error_msg = f"AI text cleanup failed: {e}"
            print(f"⚠️ {error_msg}")
            # Return original text if cleanup fails
            return raw_text
    
    def cleanup_text_batch(self, texts: list, model: str = "gpt-4o") -> list:
        """
        Clean multiple texts in batch
        
        Args:
            texts: List of raw text strings
            model: OpenAI model to use
        
        Returns:
            List of cleaned text strings
        """
        cleaned_texts = []
        for text in texts:
            cleaned = self.cleanup_text(text, model)
            cleaned_texts.append(cleaned)
        return cleaned_texts

