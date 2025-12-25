from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path


class RAGExtractor:
    """
    RAG system optimized for processing massive sustainability reports (ESRS/CSRD).
    
    Key Optimizations:
    - Batch upload with proper sizing (500 chunks per batch)
    - Direct delete with where filter for faster cleanup
    - Enhanced hybrid search with keyword boosting (40% weight)
    - Larger chunk size (1200) to capture full table rows
    - Cosine similarity for better semantic ranking
    
    FIXED: ChromaDB filter syntax using $and operator
    """
    
    def __init__(self, collection_name: str = "sustainability_docs", 
                 persist_directory: str = "data/chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Persistent Client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Consistent Embedding Function (OpenAI v3-small is cost-effective)
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        # Get/Create collection with Cosine similarity for better semantic ranking
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.openai_ef,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Optimal chunking for sustainability tables and narrative
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250,
            length_function=len,
            separators=["\\n\\n", "\\n", ". ", " ", ""]
        )
    
    def index_document(self, company: str, text_pages: Dict[int, str], 
                       report_year: int = 2024):
        """
        Indexes document chunks with rich metadata for targeted retrieval.
        """
        print(f"Indexing {company} ({len(text_pages)} pages)...")
        
        documents, metadatas, ids = [], [], []
        
        for page_num, page_text in text_pages.items():
            if not page_text or len(page_text.strip()) < 100:
                continue
            
            chunks = self.text_splitter.split_text(page_text)
            
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"{company}_{report_year}_p{page_num}_c{chunk_idx}"
                
                documents.append(chunk)
                metadatas.append({
                    "company": company,
                    "page_num": int(page_num),
                    "chunk_idx": chunk_idx,
                    "report_year": report_year
                })
                ids.append(doc_id)
        
        # Enhanced Batch Upload
        batch_size = 500
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        print(f"  Uploading {len(documents)} chunks in {total_batches} batches...")
        
        for i in range(0, len(documents), batch_size):
            batch_num = i // batch_size + 1
            try:
                self.collection.add(
                    documents=documents[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                    ids=ids[i:i + batch_size]
                )
                print(f"    Batch {batch_num}/{total_batches} uploaded")
            except Exception as e:
                print(f"    ✗ Error in batch {batch_num}: {e}")
                continue
        
        print(f"  ✓ Successfully indexed {len(documents)} chunks.")
    
    def retrieve_relevant_sections(self, query: str, company: str,
                                   n_results: int = 5,
                                   report_year: int = 2024) -> List[Dict]:
        """
        Retrieve most relevant document sections using semantic search.
        
        FIXED: ChromaDB filter syntax with $and operator
        """
        # FIXED: Proper ChromaDB filter syntax for multiple conditions
        where_filter = {
            "$and": [
                {"company": {"$eq": company}},
                {"report_year": {"$eq": report_year}}
            ]
        }
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 2,
                where=where_filter
            )
        except Exception as e:
            print(f"  ⚠ Query error: {e}")
            # Fallback: try with just company filter
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results * 2,
                    where={"company": {"$eq": company}}
                )
            except Exception as e2:
                print(f"  ⚠ Fallback query also failed: {e2}")
                return []
        
        if not results or not results["documents"][0]:
            return []
        
        formatted_results = []
        seen_pages = set()
        
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            page_num = metadata["page_num"]
            
            if page_num in seen_pages:
                continue
            
            seen_pages.add(page_num)
            
            formatted_results.append({
                "text": doc,
                "page_num": page_num,
                "score": 1.0 - distance,
                "metadata": metadata
            })
            
            if len(formatted_results) >= n_results:
                break
        
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        
        return formatted_results
    
    def hybrid_search(self, query: str, keywords: List[str], 
                      company: str, n_results: int = 3,
                      report_year: int = 2024) -> List[Dict]:
        """
        OPTIMIZED: Combines semantic search with keyword boosting.
        
        FIXED: ChromaDB filter syntax with $and operator
        """
        # FIXED: Proper ChromaDB filter syntax
        where_filter = {
            "$and": [
                {"company": {"$eq": company}},
                {"report_year": {"$eq": report_year}}
            ]
        }
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 3,
                where=where_filter
            )
        except Exception as e:
            print(f"  ⚠ Hybrid search error: {e}")
            # Fallback: try with just company filter
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results * 3,
                    where={"company": {"$eq": company}}
                )
            except Exception as e2:
                print(f"  ⚠ Fallback hybrid search failed: {e2}")
                return []
        
        if not results or not results["documents"][0]:
            return []
        
        formatted_results = []
        seen_pages = set()
        
        for doc, metadata, distance in zip(
            results["documents"][0], 
            results["metadatas"][0], 
            results["distances"][0]
        ):
            page_num = metadata["page_num"]
            
            if page_num in seen_pages:
                continue
            
            seen_pages.add(page_num)
            
            # KEYWORD BOOSTING
            keyword_match_count = sum(
                1 for kw in keywords 
                if kw.lower() in doc.lower()
            )
            keyword_score = keyword_match_count / len(keywords) if keywords else 0
            
            semantic_score = 1.0 - distance
            
            # COMBINED SCORE: 60% semantic + 40% keyword
            combined_score = (semantic_score * 0.6) + (keyword_score * 0.4)
            
            formatted_results.append({
                "text": doc,
                "page_num": page_num,
                "score": combined_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "metadata": metadata
            })
        
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        
        return formatted_results[:n_results]
    
    def search_for_indicator(self, indicator_name: str, 
                            search_keywords: List[str],
                            company: str,
                            n_results: int = 3,
                            report_year: int = 2024) -> List[Dict]:
        """
        Specialized search optimized for ESRS indicators.
        """
        query = f"{indicator_name} {' '.join(search_keywords)}"
        
        results = self.hybrid_search(
            query, 
            search_keywords, 
            company, 
            n_results,
            report_year
        )
        
        expanded_results = []
        for result in results:
            expanded_context = self.get_expanded_context(
                company, 
                result["page_num"],
                context_pages=1,
                report_year=report_year
            )
            
            expanded_results.append({
                "text": expanded_context,
                "page_num": result["page_num"],
                "score": result["score"],
                "snippet": result["text"],
                "semantic_score": result.get("semantic_score", 0),
                "keyword_score": result.get("keyword_score", 0)
            })
        
        return expanded_results
    
    def get_expanded_context(self, company: str, page_num: int,
                            context_pages: int = 1,
                            report_year: int = 2024) -> str:
        """
        Get expanded context around a specific page.
        
        FIXED: ChromaDB filter syntax with $and operator
        """
        page_range = list(range(
            max(0, page_num - context_pages),
            page_num + context_pages + 1
        ))
        
        contexts = []
        
        for page in page_range:
            # FIXED: Proper filter syntax
            where_filter = {
                "$and": [
                    {"company": {"$eq": company}},
                    {"page_num": {"$eq": page}},
                    {"report_year": {"$eq": report_year}}
                ]
            }
            
            try:
                results = self.collection.get(
                    where=where_filter,
                    include=["documents", "metadatas"]
                )
            except Exception as e:
                # Fallback: try simpler filter
                try:
                    results = self.collection.get(
                        where={
                            "$and": [
                                {"company": {"$eq": company}},
                                {"page_num": {"$eq": page}}
                            ]
                        },
                        include=["documents", "metadatas"]
                    )
                except:
                    continue
            
            if results and results["documents"]:
                chunks_with_idx = list(zip(
                    results["documents"],
                    results["metadatas"]
                ))
                chunks_with_idx.sort(key=lambda x: x[1]["chunk_idx"])
                
                page_text = " ".join([chunk for chunk, _ in chunks_with_idx])
                contexts.append(f"--- Page {page} ---\\n{page_text}")
        
        return "\\n\\n".join(contexts)
    
    def clear_company_data(self, company: str):
        """
        Delete data by filter.
        
        FIXED: ChromaDB filter syntax
        """
        try:
            # FIXED: Proper delete filter syntax
            self.collection.delete(
                where={"company": {"$eq": company}}
            )
            print(f"  ✓ Cleared all entries for {company}")
        except Exception as e:
            print(f"  ✗ Error clearing {company}: {e}")
    
    def clear_all_data(self):
        """Clear entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"  ✓ Cleared collection: {self.collection_name}")
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.openai_ef,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"  ✗ Error clearing collection: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the indexed collection"""
        try:
            count = self.collection.count()
            
            sample = self.collection.peek(limit=50)
            
            companies = set()
            years = set()
            if sample and sample["metadatas"]:
                companies = {meta["company"] for meta in sample["metadatas"]}
                years = {meta.get("report_year", 2024) for meta in sample["metadatas"]}
            
            return {
                "total_chunks": count,
                "companies_indexed": list(companies),
                "report_years": list(years),
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"error": str(e)}
    
    def close(self):
        """Cleanup"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SimpleRAGExtractor:
    """
    Simplified RAG extractor - no vector DB required.
    
    RECOMMENDED FOR 3-DAY TIMELINE:
    - No ChromaDB setup issues
    - No embedding API costs
    - Fast and reliable
    - Good enough for governance indicators
    """
    
    def __init__(self):
        self.documents = {}  # {company: {page_num: text}}
    
    def index_document(self, company: str, text_pages: Dict[int, str], 
                       report_year: int = 2024):
        """Store document pages in memory"""
        if company not in self.documents:
            self.documents[company] = {}
        
        self.documents[company] = text_pages
        print(f"  ✓ Indexed {company}: {len(text_pages)} pages (in-memory)")
    
    def search_for_indicator(self, indicator_name: str,
                            search_keywords: List[str],
                            company: str,
                            n_results: int = 3,
                            report_year: int = 2024) -> List[Dict]:
        """
        Simple keyword-based search with scoring.
        """
        if company not in self.documents:
            return []
        
        text_pages = self.documents[company]
        scored_pages = []
        
        for page_num, text in text_pages.items():
            text_lower = text.lower()
            
            keyword_matches = sum(1 for kw in search_keywords if kw.lower() in text_lower)
            
            if keyword_matches > 0:
                score = keyword_matches / len(search_keywords)
                
                scored_pages.append({
                    "page_num": page_num,
                    "score": score,
                    "text": text,
                    "keyword_matches": keyword_matches
                })
        
        scored_pages.sort(key=lambda x: (x["score"], x["keyword_matches"]), reverse=True)
        
        results = []
        for page_data in scored_pages[:n_results]:
            context_text = ""
            for offset in [-1, 0, 1]:
                context_page = page_data["page_num"] + offset
                if context_page in text_pages:
                    context_text += f"\\n--- Page {context_page} ---\\n"
                    context_text += text_pages[context_page]
            
            results.append({
                "text": context_text,
                "page_num": page_data["page_num"],
                "score": page_data["score"],
                "snippet": page_data["text"][:500],
                "keyword_score": page_data["score"],
                "semantic_score": 0.0
            })
        
        return results
    
    def clear_company_data(self, company: str):
        """Remove company data from memory"""
        if company in self.documents:
            del self.documents[company]
            print(f"  ✓ Cleared {company} from memory")
    
    def clear_all_data(self):
        """Clear all indexed data"""
        self.documents.clear()
        print("  ✓ Cleared all indexed documents")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about indexed documents"""
        return {
            "companies_indexed": list(self.documents.keys()),
            "total_pages": sum(len(pages) for pages in self.documents.values()),
            "storage_type": "in-memory"
        }
    
    def close(self):
        """Cleanup"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()