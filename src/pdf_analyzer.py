# PDF Analysis module for finding anchor pages and relevant sections

import fitz  # PyMuPDF
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class AnchorPage:
    page_num: int
    keywords_found: List[str]
    text_snippet: str
    confidence: float

class PDFAnalyzer:
    """Finds anchor pages (ESRS/GRI indexes) in sustainability reports"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.total_pages = len(self.doc)
        self.toc = self.extract_toc()
        
    def extract_toc(self) -> List[Tuple]:
        """Extract Table of Contents"""
        return self.doc.get_toc()
    
    def find_anchor_pages(self, keywords: List[str]) -> List[AnchorPage]:
        """
        Find pages containing anchor keywords (ESRS Index, GRI Index, etc.)
        Returns pages where sustainability data is likely indexed
        """
        anchor_pages = []
        
        print(f"Scanning {self.total_pages} pages for anchor keywords...")
        
        for page_num in tqdm(range(self.total_pages)):
            page = self.doc[page_num]
            text = page.get_text().lower()
            
            keywords_found = [kw for kw in keywords if kw.lower() in text]
            
            if keywords_found:
                # Calculate confidence based on keyword density
                confidence = self._calculate_anchor_confidence(text, keywords_found)
                
                anchor_pages.append(AnchorPage(
                    page_num=page_num,
                    keywords_found=keywords_found,
                    text_snippet=text[:500],
                    confidence=confidence
                ))
        
        # Sort by confidence
        anchor_pages.sort(key=lambda x: x.confidence, reverse=True)
        
        print(f"Found {len(anchor_pages)} anchor pages")
        return anchor_pages
    
    def _calculate_anchor_confidence(self, text: str, keywords: List[str]) -> float:
        """Calculate confidence score for anchor page"""
        confidence = 0.0
        
        # More keywords = higher confidence
        confidence += len(keywords) * 0.2
        
        # Check for table-like structures
        if "page" in text and any(char.isdigit() for char in text):
            confidence += 0.3
        
        # Check for ESRS/GRI specific patterns
        if re.search(r'esrs\s+[e|s|g]\d', text, re.IGNORECASE):
            confidence += 0.3
        
        if "disclosure" in text or "requirement" in text:
            confidence += 0.2
            
        return min(1.0, confidence)
    
    def find_section_pages(self, section_keywords: List[str], 
                           max_pages: int = 50) -> List[int]:
        """
        Find pages containing specific section keywords
        Used for targeted extraction after anchor pages identified
        """
        relevant_pages = []
        
        # First try ToC
        toc_pages = self._search_toc(section_keywords)
        if toc_pages:
            return toc_pages[:max_pages]
        
        # Fallback: keyword search
        for page_num in range(min(self.total_pages, 500)):  # Limit search
            page = self.doc[page_num]
            text = page.get_text().lower()
            
            if any(kw.lower() in text for kw in section_keywords):
                relevant_pages.append(page_num)
                
            if len(relevant_pages) >= max_pages:
                break
        
        return relevant_pages
    
    def _search_toc(self, keywords: List[str]) -> List[int]:
        """Search Table of Contents for keywords"""
        pages = []
        for level, title, page_num in self.toc:
            if any(kw.lower() in title.lower() for kw in keywords):
                pages.append(page_num - 1)  # fitz uses 0-indexing
        return pages
    
    def extract_text_from_pages(self, page_numbers: List[int]) -> Dict[int, str]:
        """Extract text from specific pages"""
        text_dict = {}
        for page_num in page_numbers:
            if 0 <= page_num < self.total_pages:
                page = self.doc[page_num]
                text_dict[page_num] = page.get_text()
        return text_dict
    
    def get_page_context(self, page_num: int, context_pages: int = 2) -> str:
        """Get text from page plus surrounding context"""
        start = max(0, page_num - context_pages)
        end = min(self.total_pages, page_num + context_pages + 1)
        
        context = ""
        for p in range(start, end):
            context += f"\\n--- Page {p} ---\\n"
            context += self.doc[p].get_text()
        
        return context
    
    def close(self):
        """Close PDF document"""
        self.doc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()