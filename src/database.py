import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

class SustainabilityDatabase:
    """Manages SQLite database for sustainability extractions"""
    
    def __init__(self, db_path: str = "data/sustainability_data.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        """Create database schema"""
        cursor = self.conn.cursor()
        
        # Main extractions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS extractions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            report_year INTEGER NOT NULL,
            indicator_id INTEGER NOT NULL,
            indicator_name TEXT NOT NULL,
            indicator_category TEXT NOT NULL,
            value REAL,
            unit TEXT NOT NULL,
            confidence REAL NOT NULL,
            source_page INTEGER,
            source_section TEXT,
            source_quote TEXT,
            extraction_notes TEXT,
            data_quality TEXT,
            extraction_method TEXT,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(company, report_year, indicator_id)
        )
        """)
        
        # Audit log for tracking extraction attempts
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS extraction_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            indicator_name TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        self.conn.commit()
    
    def insert_extraction(self, extraction_data: Dict) -> int:
        """
        Insert or update extraction result
        Returns row ID
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        INSERT OR REPLACE INTO extractions (
            company, report_year, indicator_id, indicator_name, indicator_category,
            value, unit, confidence, source_page, source_section, source_quote,
            extraction_notes, data_quality, extraction_method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            extraction_data["company"],
            extraction_data["report_year"],
            extraction_data["indicator_id"],
            extraction_data["indicator_name"],
            extraction_data["indicator_category"],
            extraction_data.get("value"),
            extraction_data["unit"],
            extraction_data["confidence"],
            extraction_data.get("source_page"),
            extraction_data.get("source_section"),
            extraction_data.get("source_quote"),
            extraction_data.get("extraction_notes"),
            extraction_data.get("data_quality"),
            extraction_data.get("extraction_method")
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def log_extraction_attempt(self, company: str, indicator_name: str, 
                               status: str, error_message: str = None):
        """Log extraction attempt for debugging"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        INSERT INTO extraction_audit (company, indicator_name, status, error_message)
        VALUES (?, ?, ?, ?)
        """, (company, indicator_name, status, error_message))
        
        self.conn.commit()
    
    def get_all_extractions(self) -> pd.DataFrame:
        """Get all extractions as DataFrame"""
        return pd.read_sql_query("""
            SELECT 
                company,
                report_year,
                indicator_name,
                value,
                unit,
                confidence,
                source_page,
                source_section,
                extraction_notes,
                data_quality
            FROM extractions
            ORDER BY company, indicator_id
        """, self.conn)
    
    def export_to_csv(self, output_path: str = "data/output/sustainability_extractions.csv"):
        """Export data to CSV format required by assignment"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        df = self.get_all_extractions()
        
        # Rename columns to match required format
        df = df.rename(columns={
            "source_section": "notes"  # Combine with extraction_notes if needed
        })
        
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} records to {output_path}")
        
        return output_path
    
    def get_extraction_stats(self) -> Dict:
        """Get statistics about extractions"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total extractions
        cursor.execute("SELECT COUNT(*) FROM extractions")
        stats["total_extractions"] = cursor.fetchone()[0]
        
        # By company
        cursor.execute("""
            SELECT company, COUNT(*) as count
            FROM extractions
            GROUP BY company
        """)
        stats["by_company"] = dict(cursor.fetchall())
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM extractions WHERE value IS NOT NULL")
        stats["avg_confidence"] = cursor.fetchone()[0]
        
        # By quality
        cursor.execute("""
            SELECT data_quality, COUNT(*) as count
            FROM extractions
            GROUP BY data_quality
        """)
        stats["by_quality"] = dict(cursor.fetchall())
        
        # Missing extractions
        cursor.execute("SELECT COUNT(*) FROM extractions WHERE value IS NULL")
        stats["missing_extractions"] = cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()