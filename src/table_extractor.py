import pdfplumber
from typing import List, Dict, Optional
import re
from dataclasses import dataclass

@dataclass
class ExtractedTable:
    page_num: int
    table_data: List[List[str]]
    table_markdown: str
    contains_keywords: List[str]

class TableExtractor:
    """Extract and process tables from PDF reports"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf = pdfplumber.open(pdf_path)
    
    def extract_tables_from_pages(self, page_numbers: List[int], 
                                    keywords: List[str]) -> List[ExtractedTable]:
        """
        Extract tables from specific pages that contain relevant keywords
        """
        extracted_tables = []
        
        for page_num in page_numbers:
            if page_num >= len(self.pdf.pages):
                continue
                
            page = self.pdf.pages[page_num]
            page_text = page.extract_text() or ""
            
            # Check if page contains relevant keywords
            found_keywords = [kw for kw in keywords if kw.lower() in page_text.lower()]
            
            if not found_keywords:
                continue
            
            # Extract all tables from this page
            tables = page.extract_tables()
            
            for table in tables:
                if table and len(table) > 1:  # Must have header + data
                    # Convert to markdown for LLM
                    markdown = self._table_to_markdown(table)
                    
                    # Check if table contains keywords in cells
                    table_text = " ".join([" ".join([str(cell) for cell in row if cell]) 
                                          for row in table])
                    
                    table_keywords = [kw for kw in keywords 
                                     if kw.lower() in table_text.lower()]
                    
                    if table_keywords:
                        extracted_tables.append(ExtractedTable(
                            page_num=page_num,
                            table_data=table,
                            table_markdown=markdown,
                            contains_keywords=table_keywords
                        ))
        
        return extracted_tables
    
    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """Convert table to markdown format for LLM processing"""
        if not table:
            return ""
        
        # Clean cells
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_table.append(cleaned_row)
        
        # Header
        header = cleaned_table[0]
        markdown = "| " + " | ".join(header) + " |\\n"
        markdown += "|" + "|".join(["---" for _ in header]) + "|\\n"
        
        # Rows
        for row in cleaned_table[1:]:
            markdown += "| " + " | ".join(row) + " |\\n"
        
        return markdown
    
    def find_table_with_indicator(self, indicator_name: str, 
                                   search_keywords: List[str],
                                   max_pages_to_check: int = 100) -> Optional[ExtractedTable]:
        """
        Search for a specific indicator in tables
        Returns the best matching table
        """
        best_match = None
        best_score = 0
        
        for page_num in range(min(len(self.pdf.pages), max_pages_to_check)):
            page = self.pdf.pages[page_num]
            tables = page.extract_tables()
            
            if not tables:
                continue
            
            for table in tables:
                if not table or len(table) < 2:
                    continue
                
                # Score the table
                score = self._score_table_relevance(table, indicator_name, search_keywords)
                
                if score > best_score:
                    best_score = score
                    best_match = ExtractedTable(
                        page_num=page_num,
                        table_data=table,
                        table_markdown=self._table_to_markdown(table),
                        contains_keywords=search_keywords
                    )
        
        return best_match if best_score > 0.3 else None
    
    def _score_table_relevance(self, table: List[List[str]], 
                               indicator_name: str, 
                               keywords: List[str]) -> float:
        """Score how relevant a table is to the indicator"""
        score = 0.0
        
        # Convert table to text
        table_text = " ".join([" ".join([str(cell) for cell in row if cell]) 
                              for row in table]).lower()
        
        # Check keyword presence
        for keyword in keywords:
            if keyword.lower() in table_text:
                score += 0.3
        
        # Check for numerical data (indicators usually have numbers)
        if re.search(r'\\d+[,.]\\d+', table_text):
            score += 0.2
        
        # Check for year 2024
        if "2024" in table_text:
            score += 0.2
        
        # Check for units
        units = ["tco2e", "mwh", "gj", "%", "fte", "hours", "days", "million"]
        if any(unit in table_text for unit in units):
            score += 0.3
        
        return min(1.0, score)
    
    def extract_value_from_table(self, table_markdown: str, 
                                  indicator_name: str, 
                                  unit: str) -> Optional[Dict]:
        """
        Parse a table markdown to find specific value
        Returns dict with value, unit, confidence
        """
        # This is a simple pattern-based extraction
        # In practice, we'll use LLM for this, but this provides fallback
        
        lines = table_markdown.split("\\n")
        
        for line in lines:
            if any(keyword in line.lower() for keyword in indicator_name.lower().split()):
                # Extract numbers
                numbers = re.findall(r'\\d+[,.]?\\d*', line)
                if numbers:
                    return {
                        "value": numbers[-1].replace(",", ""),  # Last number usually the total
                        "unit": unit,
                        "confidence": 0.6
                    }
        
        return None
    
    def close(self):
        """Close PDF file"""
        self.pdf.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()