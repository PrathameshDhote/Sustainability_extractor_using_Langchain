import os
from typing import Dict, List
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4o")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# Extraction Settings
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
MAX_PAGES_PER_SEARCH = int(os.getenv("MAX_PAGES_PER_SEARCH", "50"))
ENABLE_VERIFICATION = os.getenv("ENABLE_VERIFICATION", "true").lower() == "true"

# Database Configuration
DB_PATH = "data/sustainability_data.db"

# Report Configuration
BANKS = {
    "AIB": {
        "name": "Allied Irish Banks",
        "country": "Ireland",
        "report_year": 2024,
        "filename": "AIB_2024_Annual_Report.pdf"
    },
    "BBVA": {
        "name": "BBVA",
        "country": "Spain", 
        "report_year": 2024,
        "filename": "BBVA_2024_Management_Report.pdf"
    },
    "BPCE": {
        "name": "Groupe BPCE",
        "country": "France",
        "report_year": 2024,
        "filename": "BPCE_2024_Registration_Document.pdf"
    }
}

# Indicator Definitions
class Indicator(BaseModel):
    id: int
    name: str
    category: str
    esrs_reference: str
    unit: str
    extraction_method: str  # "table" or "narrative"
    search_keywords: List[str]
    validation_range: tuple = None

INDICATORS = [
    # Environmental Indicators (ESRS E1)
    Indicator(
        id=1, name="Total Scope 1 GHG Emissions", category="Environmental",
        esrs_reference="ESRS E1", unit="tCO2e", extraction_method="table",
        search_keywords=["scope 1", "direct emissions", "GHG emissions", "CO2 emissions"],
        validation_range=(0, 10000000)
    ),
    Indicator(
        id=2, name="Total Scope 2 GHG Emissions", category="Environmental",
        esrs_reference="ESRS E1", unit="tCO2e", extraction_method="table",
        search_keywords=["scope 2", "indirect emissions", "electricity emissions"],
        validation_range=(0, 10000000)
    ),
    Indicator(
        id=3, name="Total Scope 3 GHG Emissions", category="Environmental",
        esrs_reference="ESRS E1", unit="tCO2e", extraction_method="table",
        search_keywords=["scope 3", "value chain", "financed emissions"],
        validation_range=(0, 100000000)
    ),
    Indicator(
        id=4, name="GHG Emissions Intensity", category="Environmental",
        esrs_reference="ESRS E1", unit="tCO2e per €M revenue", extraction_method="table",
        search_keywords=["emissions intensity", "carbon intensity", "GHG intensity"],
        validation_range=(0, 1000)
    ),
    Indicator(
        id=5, name="Total Energy Consumption", category="Environmental",
        esrs_reference="ESRS E1", unit="MWh", extraction_method="table",
        search_keywords=["energy consumption", "total energy", "energy use"],
        validation_range=(0, 10000000)
    ),
    Indicator(
        id=6, name="Renewable Energy Percentage", category="Environmental",
        esrs_reference="ESRS E1", unit="%", extraction_method="table",
        search_keywords=["renewable energy", "green energy", "renewable percentage"],
        validation_range=(0, 100)
    ),
    Indicator(
        id=7, name="Net Zero Target Year", category="Environmental",
        esrs_reference="ESRS E1", unit="year", extraction_method="narrative",
        search_keywords=["net zero", "carbon neutral", "climate neutral", "2050"],
        validation_range=(2024, 2060)
    ),
    Indicator(
        id=8, name="Green Financing Volume", category="Environmental",
        esrs_reference="ESRS E1", unit="€ millions", extraction_method="table",
        search_keywords=["green financing", "sustainable financing", "green loans"],
        validation_range=(0, 1000000)
    ),
    
    # Social Indicators (ESRS S1)
    Indicator(
        id=9, name="Total Employees", category="Social",
        esrs_reference="ESRS S1", unit="FTE", extraction_method="table",
        search_keywords=["total employees", "headcount", "workforce", "FTE"],
        validation_range=(0, 200000)
    ),
    Indicator(
        id=10, name="Female Employees", category="Social",
        esrs_reference="ESRS S1", unit="%", extraction_method="table",
        search_keywords=["female employees", "women", "gender diversity"],
        validation_range=(0, 100)
    ),
    Indicator(
        id=11, name="Gender Pay Gap", category="Social",
        esrs_reference="ESRS S1", unit="%", extraction_method="table",
        search_keywords=["gender pay gap", "pay equity", "wage gap"],
        validation_range=(0, 50)
    ),
    Indicator(
        id=12, name="Training Hours per Employee", category="Social",
        esrs_reference="ESRS S1", unit="hours", extraction_method="table",
        search_keywords=["training hours", "employee training", "development hours"],
        validation_range=(0, 200)
    ),
    Indicator(
        id=13, name="Employee Turnover Rate", category="Social",
        esrs_reference="ESRS S1", unit="%", extraction_method="table",
        search_keywords=["turnover rate", "attrition", "employee retention"],
        validation_range=(0, 50)
    ),
    Indicator(
        id=14, name="Work-Related Accidents", category="Social",
        esrs_reference="ESRS S1", unit="count", extraction_method="table",
        search_keywords=["work accidents", "workplace injuries", "accident rate"],
        validation_range=(0, 10000)
    ),
    Indicator(
        id=15, name="Collective Bargaining Coverage", category="Social",
        esrs_reference="ESRS S1", unit="%", extraction_method="table",
        search_keywords=["collective bargaining", "union coverage", "collective agreements"],
        validation_range=(0, 100)
    ),
    
    # Governance Indicators (ESRS G1)
    Indicator(
        id=16, name="Board Female Representation", category="Governance",
        esrs_reference="ESRS G1", unit="%", extraction_method="narrative",
        search_keywords=["board", "female", "women directors", "gender diversity"],
        validation_range=(0, 100)
    ),
    Indicator(
        id=17, name="Board Meetings", category="Governance",
        esrs_reference="ESRS G1", unit="count/year", extraction_method="narrative",
        search_keywords=["board meetings", "meetings held", "board sessions"],
        validation_range=(0, 100)
    ),
    Indicator(
        id=18, name="Corruption Incidents", category="Governance",
        esrs_reference="ESRS G1", unit="count", extraction_method="narrative",
        search_keywords=["corruption", "bribery", "fraud incidents"],
        validation_range=(0, 100)
    ),
    Indicator(
        id=19, name="Avg Payment Period to Suppliers", category="Governance",
        esrs_reference="ESRS G1", unit="days", extraction_method="narrative",
        search_keywords=["payment period", "payment terms", "supplier payment"],
        validation_range=(0, 365)
    ),
    Indicator(
        id=20, name="Suppliers Screened for ESG", category="Governance",
        esrs_reference="ESRS G1", unit="%", extraction_method="narrative",
        search_keywords=["supplier screening", "ESG assessment", "supplier evaluation"],
        validation_range=(0, 100)
    )
]

# Anchor keywords for finding index pages
ANCHOR_KEYWORDS = [
    "ESRS Index",
    "ESRS Content Index",
    "GRI Content Index",
    "Sustainability Index",
    "Non-financial statement",
    "Climate-related disclosures",
    "TCFD Index",
    "ESG Index",
    "Reporting Index"
]