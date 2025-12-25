from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import json

class IndicatorExtraction(BaseModel):
    """Structured output model for indicator extraction"""
    value: Optional[float] = Field(description="Numeric value of the indicator")
    unit: str = Field(description="Unit of measurement")
    source_quote: str = Field(description="Exact text from document supporting this extraction")
    page_number: int = Field(description="Page number where data was found")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
    extraction_notes: str = Field(description="Any caveats or clarifications about the extraction")
    data_quality: str = Field(description="Quality assessment: 'high', 'medium', or 'low'")

class LLMProcessor:
    """Handles LLM-based extraction with verification"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        
        if "gpt" in model_name:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=4000
            )
        elif "claude" in model_name:
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=4000
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.parser = PydanticOutputParser(pydantic_object=IndicatorExtraction)
    
    def extract_from_table(self, table_markdown: str, indicator_name: str, 
                          expected_unit: str, esrs_ref: str, 
                          page_number: int) -> Optional[IndicatorExtraction]:
        """
        Extract indicator value from table markdown using LLM
        """
        prompt_template = """You are an expert sustainability data analyst extracting ESRS indicators from bank reports.

Extract the following indicator from this table:

Indicator: {indicator_name}
ESRS Reference: {esrs_ref}
Expected Unit: {expected_unit}
Source Page: {page_number}

Table Data (in markdown format):
{table_markdown}

CRITICAL INSTRUCTIONS:
1. Extract ONLY the 2024 value (or most recent year if 2024 not available)
2. If multiple values exist (e.g., different scopes or boundaries), extract the CONSOLIDATED/TOTAL value
3. Return the EXACT quote from the table that contains this value
4. Assign confidence score:
   - 1.0 = Clear table cell with explicit label and value
   - 0.8 = Value found but label is ambiguous
   - 0.6 = Value inferred from context
   - 0.4 = Multiple possible values, best guess
   - 0.0 = Cannot find value

5. If the value cannot be found, return null for value and confidence_score of 0.0

{format_instructions}

Return ONLY valid JSON, no additional text."""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        formatted_prompt = prompt.format(
            indicator_name=indicator_name,
            esrs_ref=esrs_ref,
            expected_unit=expected_unit,
            page_number=page_number,
            table_markdown=table_markdown[:8000],  # Limit context
            format_instructions=self.parser.get_format_instructions()
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            result = self.parser.parse(response.content)
            
            # Validate result
            result = self._validate_extraction(result, expected_unit)
            
            return result
        except Exception as e:
            print(f"Error in table extraction: {e}")
            return None
    
    def extract_from_narrative(self, text_context: str, indicator_name: str,
                               expected_unit: str, esrs_ref: str,
                               page_number: int, 
                               search_keywords: List[str]) -> Optional[IndicatorExtraction]:
        """
        Extract indicator from narrative text using LLM
        Used for governance indicators and qualitative data
        """
        prompt_template = """You are an expert sustainability analyst extracting governance and qualitative indicators from bank reports.

Extract the following indicator from this narrative text:

Indicator: {indicator_name}
ESRS Reference: {esrs_ref}
Expected Unit: {expected_unit}
Keywords to look for: {keywords}
Source Page: {page_number}

Text Context:
{text_context}

CRITICAL INSTRUCTIONS:
1. Look for explicit statements about {indicator_name}
2. Extract the 2024 value or most recent disclosed value
3. Return the EXACT SENTENCE that contains this information as source_quote
4. For qualitative targets (e.g., "net zero by 2050"), extract the numeric part (2050)
5. Assign confidence based on clarity of disclosure:
   - 1.0 = Explicit numerical statement
   - 0.7 = Clearly stated but requires interpretation
   - 0.5 = Implied from context
   - 0.0 = Not found

{format_instructions}

Return ONLY valid JSON."""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        formatted_prompt = prompt.format(
            indicator_name=indicator_name,
            esrs_ref=esrs_ref,
            expected_unit=expected_unit,
            keywords=", ".join(search_keywords),
            page_number=page_number,
            text_context=text_context[:10000],  # Larger context for narrative
            format_instructions=self.parser.get_format_instructions()
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            result = self.parser.parse(response.content)
            
            result = self._validate_extraction(result, expected_unit)
            
            return result
        except Exception as e:
            print(f"Error in narrative extraction: {e}")
            return None
    
    def verify_extraction(self, extraction: IndicatorExtraction, 
                         original_text: str) -> float:
        """
        Verify an extraction by checking if source_quote exists in original text
        Returns adjusted confidence score
        """
        if not extraction.source_quote:
            return extraction.confidence_score * 0.5
        
        # Check if quote exists in original (fuzzy match)
        quote_lower = extraction.source_quote.lower()
        text_lower = original_text.lower()
        
        if quote_lower in text_lower:
            return extraction.confidence_score  # Full confidence
        
        # Check for partial match (at least 50% of words)
        quote_words = set(quote_lower.split())
        text_words = set(text_lower.split())
        
        overlap = len(quote_words.intersection(text_words)) / len(quote_words) if quote_words else 0
        
        if overlap > 0.5:
            return extraction.confidence_score * 0.9  # Minor penalty
        else:
            return extraction.confidence_score * 0.6  # Significant penalty
    
    def _validate_extraction(self, extraction: IndicatorExtraction, 
                            expected_unit: str) -> IndicatorExtraction:
        """Validate and clean extraction result"""
        
        # Check unit consistency
        if extraction.unit != expected_unit:
            extraction.extraction_notes += f" | Unit mismatch: expected {expected_unit}, got {extraction.unit}"
            extraction.confidence_score *= 0.8
        
        # Validate confidence range
        extraction.confidence_score = max(0.0, min(1.0, extraction.confidence_score))
        
        # Assign data quality
        if extraction.confidence_score >= 0.8:
            extraction.data_quality = "high"
        elif extraction.confidence_score >= 0.5:
            extraction.data_quality = "medium"
        else:
            extraction.data_quality = "low"
        
        return extraction
    
    def batch_extract(self, contexts: List[Dict[str, Any]], 
                     indicator_name: str, expected_unit: str,
                     esrs_ref: str) -> List[IndicatorExtraction]:
        """
        Batch extraction for multiple contexts
        Returns list of extractions sorted by confidence
        """
        results = []
        
        for ctx in contexts:
            if ctx.get("type") == "table":
                result = self.extract_from_table(
                    ctx["content"], indicator_name, expected_unit,
                    esrs_ref, ctx["page_number"]
                )
            else:  # narrative
                result = self.extract_from_narrative(
                    ctx["content"], indicator_name, expected_unit,
                    esrs_ref, ctx["page_number"], ctx.get("keywords", [])
                )
            
            if result and result.value is not None:
                results.append(result)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return results